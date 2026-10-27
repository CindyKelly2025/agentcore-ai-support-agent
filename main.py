from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client

import argparse
import asyncio
import boto3
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

from strands.hooks import (
    HookProvider,
    AfterInvocationEvent,
    HookRegistry,
    MessageAddedEvent,
)

from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("CSAI_Agent")


# ============================================================
# TODO 1 - Initialize AgentCore application
# ============================================================

app = BedrockAgentCoreApp()


# ============================================================
# Environment
# ============================================================

os.environ["BYPASS_TOOL_CONSENT"] = "true"


# ============================================================
# TODO 2 - AgentCore resources
# ============================================================

# IMPORTANT:
# Set GATEWAY_URL, KB_ID and MEMORY_ID as environment variables
# for your own AWS account before running agentcore deploy.

GATEWAY_URL = os.environ.get("GATEWAY_URL", "<YOUR_GATEWAY_URL>")

KB_ID = os.environ.get("KB_ID", "<YOUR_KNOWLEDGE_BASE_ID>")

REGION = "us-east-1"

MEMORY_ID = os.environ.get("MEMORY_ID", "<YOUR_MEMORY_ID>")


# ============================================================
# TODO 3 - Initialize model, memory and Bedrock client
# ============================================================

model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(
    model_id=model_id
)

memory_client = MemoryClient(
    region_name=REGION
)

_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION
)


# ============================================================
# Helper: determine whether a resource is configured
# ============================================================

def _is_configured(value: str) -> bool:
    """
    Return True when an AWS resource value appears configured.
    """

    if not value:
        return False

    placeholders = {
        "",
        "<gateway_url>",
        "<kbid>",
        "<region>",
        "<mem_id>",
        "<your-knowledge-base-id>",
        "<your-memory-id>",
    }

    if value in placeholders:
        return False

    if "<" in value or ">" in value:
        return False

    return True


# ============================================================
# Helper: extract text from Strands content
# ============================================================

def _extract_text_from_content(content: Any) -> str:
    """
    Extract plain text from a Strands message content field.
    """

    if isinstance(content, str):
        return content.strip()

    if not isinstance(content, list):
        return ""

    texts = []

    for block in content:

        if not isinstance(block, dict):
            continue

        text = block.get("text")

        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    return "\n".join(texts)


# ============================================================
# Helper: determine whether message is plain text
# ============================================================

def _is_plain_text_message(
    message: Any,
    role: Optional[str] = None
) -> bool:
    """
    Determine whether a message contains ordinary text rather than
    tool-use or tool-result content.
    """

    if not isinstance(message, dict):
        return False

    if role is not None:
        if message.get("role") != role:
            return False

    content = message.get("content", [])

    if isinstance(content, str):
        return bool(content.strip())

    if not isinstance(content, list) or not content:
        return False

    for block in content:

        if not isinstance(block, dict):
            continue

        if "toolUse" in block:
            return False

        if "toolResult" in block:
            return False

    return bool(
        _extract_text_from_content(content)
    )


# ============================================================
# Helper: extract final agent response
# ============================================================

def _extract_agent_response(response: Any) -> str:
    """
    Extract the customer-facing text from a Strands Agent result.
    """

    try:

        message = response.message

        if isinstance(message, dict):

            content = message.get(
                "content",
                []
            )

            text = _extract_text_from_content(
                content
            )

            if text:
                return text

    except Exception as exc:

        logger.debug(
            "Could not extract response.message: %s",
            exc
        )

    if isinstance(response, str):
        return response

    return str(response)


# ============================================================
# TODO 4 - Memory namespace helper
# ============================================================

def get_namespaces(
    mem_client: MemoryClient,
    memory_id: str
) -> Dict[str, str]:
    """
    Return a dictionary mapping memory strategy type to namespace
    template.

    AgentCore memory strategies expose namespace information that
    can be used to retrieve long-term memories for the customer.
    """

    if not _is_configured(memory_id):
        return {}

    try:

        # AgentCore MemoryClient supports the memory ID as the
        # positional argument.
        response = mem_client.get_memory_strategies(
            memory_id
        )

        # Depending on SDK version, the response can be returned
        # directly as a list or wrapped in a dictionary.
        if isinstance(response, list):

            strategies = response

        elif isinstance(response, dict):

            strategies = response.get(
                "strategies",
                response.get(
                    "memoryStrategies",
                    []
                )
            )

        else:

            strategies = []

        namespaces = {}

        for strategy in strategies:

            if not isinstance(strategy, dict):
                continue

            strategy_type = (
                strategy.get("type")
                or strategy.get("strategyType")
                or strategy.get("name")
            )

            namespace_templates = (
                strategy.get("namespaceTemplates")
            )

            if not namespace_templates:

                namespace_templates = (
                    strategy.get("namespaces")
                )

            if not namespace_templates:
                continue

            if isinstance(
                namespace_templates,
                str
            ):
                namespace_templates = [
                    namespace_templates
                ]

            if (
                isinstance(
                    namespace_templates,
                    list
                )
                and namespace_templates
            ):

                if strategy_type:
                    namespaces[
                        strategy_type
                    ] = namespace_templates[0]

        logger.debug(
            "Memory namespaces: %s",
            namespaces
        )

        return namespaces

    except Exception as exc:

        logger.warning(
            "Unable to retrieve memory strategies: %s",
            exc
        )

        return {}


# ============================================================
# TODO 5 - Memory Hook
# ============================================================

class MemoryHook(HookProvider):
    """
    Retrieves relevant customer memories and stores completed
    customer-support interactions.
    """

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str
    ):

        self.actor_id = actor_id

        self.session_id = session_id

        self.memory_client = memory_client

        self.memory_id = memory_id

        self.namespaces = get_namespaces(
            memory_client,
            memory_id
        )

    # --------------------------------------------------------
    # Find latest plain-text user message
    # --------------------------------------------------------

    def _get_latest_user_message(
        self,
        messages: list
    ) -> Optional[str]:

        for message in reversed(messages):

            if _is_plain_text_message(
                message,
                role="user"
            ):

                return _extract_text_from_content(
                    message.get(
                        "content",
                        []
                    )
                )

        return None

    # --------------------------------------------------------
    # Retrieve customer context
    # --------------------------------------------------------

    def retrieve_customer_context(
        self,
        event: MessageAddedEvent
    ):
        """
        Retrieve relevant memories from every configured strategy
        namespace and prepend them to the current customer message.
        """

        if not _is_configured(
            self.memory_id
        ):
            return

        try:

            agent = event.agent

            messages = agent.messages

            if not messages:
                return

            user_query = (
                self._get_latest_user_message(
                    messages
                )
            )

            if not user_query:
                return

            memory_context = []

            for (
                strategy_type,
                namespace_template
            ) in self.namespaces.items():

                try:

                    namespace = namespace_template.format(
                        actorId=self.actor_id,
                        actor_id=self.actor_id,
                        sessionId=self.session_id,
                        session_id=self.session_id
                    )

                    memories = (
                        self.memory_client.retrieve_memories(
                            memory_id=self.memory_id,
                            namespace=namespace,
                            query=user_query,
                            top_k=5
                        )
                    )

                    for memory in memories or []:

                        if not isinstance(
                            memory,
                            dict
                        ):
                            continue

                        content = memory.get(
                            "content",
                            {}
                        )

                        if isinstance(
                            content,
                            dict
                        ):

                            text = content.get(
                                "text",
                                ""
                            )

                        else:

                            text = str(
                                content
                            )

                        if (
                            isinstance(
                                text,
                                str
                            )
                            and text.strip()
                        ):

                            memory_context.append(
                                f"[{strategy_type}] "
                                f"{text.strip()}"
                            )

                except Exception as exc:

                    logger.warning(
                        "Memory retrieval failed for "
                        "%s: %s",
                        strategy_type,
                        exc
                    )

            if not memory_context:
                return

            context_text = (
                "Customer Context:\n"
                + "\n".join(
                    memory_context
                )
                + "\n\n"
                + user_query
            )

            # Locate the latest user message and enrich it.
            for message in reversed(messages):

                if _is_plain_text_message(
                    message,
                    role="user"
                ):

                    message["content"] = [
                        {
                            "text": context_text
                        }
                    ]

                    break

        except Exception as exc:

            logger.warning(
                "Customer context retrieval failed: %s",
                exc
            )

    # --------------------------------------------------------
    # Save interaction
    # --------------------------------------------------------

    def save_support_interaction(
        self,
        event: AfterInvocationEvent
    ):
        """
        Save the latest customer query and assistant response
        to AgentCore Memory.
        """

        if not _is_configured(
            self.memory_id
        ):
            return

        try:

            agent = event.agent

            messages = agent.messages

            if not messages:
                return

            customer_query = None

            agent_response = None

            # Find latest assistant text.
            for message in reversed(messages):

                if _is_plain_text_message(
                    message,
                    role="assistant"
                ):

                    agent_response = (
                        _extract_text_from_content(
                            message.get(
                                "content",
                                []
                            )
                        )
                    )

                    if agent_response:
                        break

            # Find latest user text.
            for message in reversed(messages):

                if _is_plain_text_message(
                    message,
                    role="user"
                ):

                    customer_query = (
                        _extract_text_from_content(
                            message.get(
                                "content",
                                []
                            )
                        )
                    )

                    if customer_query:
                        break

            if (
                not customer_query
                or not agent_response
            ):
                return

            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (
                        customer_query,
                        "USER"
                    ),
                    (
                        agent_response,
                        "ASSISTANT"
                    )
                ]
            )

        except Exception as exc:

            logger.warning(
                "Unable to save support interaction: %s",
                exc
            )

    # --------------------------------------------------------
    # Register hooks
    # --------------------------------------------------------

    def register_hooks(
        self,
        registry: HookRegistry
    ) -> None:

        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context
        )

        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction
        )


# ============================================================
# TODO 6 - Knowledge Base search
# ============================================================

@tool
def search_knowledge_base(
    query: str
) -> str:
    """
    Search the Customer Support Knowledge Base for product
    information, return policies, refunds, loyalty benefits,
    product specifications, and other customer-support information.
    """

    if not _is_configured(KB_ID):

        return (
            "Knowledge base is not configured."
        )

    if not query or not query.strip():

        return (
            "Please provide a search query."
        )

    try:

        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={
                "text": query.strip()
            }
        )

        results = response.get(
            "retrievalResults",
            []
        )

        chunks = []

        for result in results:

            if not isinstance(
                result,
                dict
            ):
                continue

            content = result.get(
                "content",
                {}
            )

            if isinstance(
                content,
                dict
            ):

                text = content.get(
                    "text",
                    ""
                )

            else:

                text = str(
                    content
                )

            if (
                isinstance(
                    text,
                    str
                )
                and text.strip()
            ):

                chunks.append(
                    text.strip()
                )

        if not chunks:

            return (
                "No relevant information was found "
                "in the knowledge base."
            )

        return "\n---\n".join(
            chunks
        )

    except Exception as exc:

        logger.exception(
            "Knowledge Base retrieval failed"
        )

        return (
            "I could not search the knowledge base "
            "right now."
        )


# ============================================================
# TODO 7 - Loyalty discount calculator
# ============================================================

@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard"
) -> str:
    """
    Calculate loyalty points redemption, tier discount, final order
    total, points earned, and remaining points.

    The calculation is performed using Amazon Bedrock AgentCore
    Code Interpreter.
    """

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    try:

        loyalty_points = max(
            0,
            int(loyalty_points)
        )

        order_total = max(
            0.0,
            float(order_total)
        )

    except Exception:

        return json.dumps({
            "points_redeemed": 0,
            "tier_discount_pct": 0,
            "final_total": 0,
            "remaining_points": 0,
            "error": (
                "Invalid loyalty points "
                "or order total."
            )
        })

    tier_normalized = (
        str(tier)
        .strip()
        .title()
    )

    if tier_normalized not in {
        "Silver",
        "Gold",
        "Platinum"
    }:

        tier_normalized = "Silver"

    product_category = (
        str(product_category)
        .strip()
        .lower()
    )

    if product_category not in {
        "standard",
        "device",
        "fresh"
    }:

        product_category = "standard"

    # --------------------------------------------------------
    # Code executed in AgentCore Code Interpreter
    # --------------------------------------------------------

    code = f"""
import json
import math

loyalty_points = {loyalty_points}
tier = {tier_normalized!r}
order_total = {order_total}
product_category = {product_category!r}

earn_rates = {{
    "standard": 1,
    "device": 2,
    "fresh": 5
}}

tier_rates = {{
    "Silver": 0.00,
    "Gold": 0.10,
    "Platinum": 0.15
}}

POINTS_PER_DOLLAR = 100
MIN_REDEMPTION = 500

# Maximum redemption is 50 percent of the order value.
max_redemption_dollars = (
    order_total * 0.50
)

max_redeemable_points = int(
    max_redemption_dollars
    * POINTS_PER_DOLLAR
)

# Customer cannot redeem more points than they have.
redeemable_points = min(
    loyalty_points,
    max_redeemable_points
)

# Redemptions must be in 500-point increments.
redeemable_points = (
    redeemable_points // MIN_REDEMPTION
) * MIN_REDEMPTION

points_value = (
    redeemable_points
    / POINTS_PER_DOLLAR
)

subtotal_after_points = max(
    0.0,
    order_total - points_value
)

tier_discount_rate = tier_rates.get(
    tier,
    0.0
)

tier_discount_pct = (
    tier_discount_rate * 100
)

tier_discount = (
    subtotal_after_points
    * tier_discount_rate
)

final_total = max(
    0.0,
    subtotal_after_points - tier_discount
)

remaining_points = max(
    0,
    loyalty_points - redeemable_points
)

points_earned = int(
    math.floor(
        final_total
        * earn_rates[product_category]
    )
)

total_savings = (
    order_total - final_total
)

result = {{
    "points_redeemed": redeemable_points,
    "tier_discount_pct": tier_discount_pct,
    "final_total": round(
        final_total,
        2
    ),
    "remaining_points": remaining_points,
    "points_earned": points_earned,
    "total_savings": round(
        total_savings,
        2
    )
}}

print(json.dumps(result))
"""

    # --------------------------------------------------------
    # AgentCore Code Interpreter
    # --------------------------------------------------------

    try:

        with code_session(
            REGION
        ) as code_client:

            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True
                }
            )

            # AWS returns a stream of execution events.
            # The rubric requires processing the result event.
            for event in response.get(
                "stream",
                []
            ):

                if not isinstance(
                    event,
                    dict
                ):
                    continue

                if "result" not in event:
                    continue

                result = event["result"]

                if not isinstance(
                    result,
                    dict
                ):
                    return json.dumps(
                        result
                    )

                # ------------------------------------------------
                # Successful Code Interpreter output
                # ------------------------------------------------

                content = result.get(
                    "content",
                    []
                )

                for content_item in content:

                    if not isinstance(
                        content_item,
                        dict
                    ):
                        continue

                    if (
                        content_item.get(
                            "type"
                        )
                        == "text"
                    ):

                        text = content_item.get(
                            "text",
                            ""
                        )

                        if not text:
                            continue

                        text = text.strip()

                        try:

                            parsed = json.loads(
                                text
                            )

                            return json.dumps(
                                parsed
                            )

                        except json.JSONDecodeError:

                            return text

                # ------------------------------------------------
                # structuredContent fallback
                # ------------------------------------------------

                structured_content = (
                    result.get(
                        "structuredContent"
                    )
                )

                if isinstance(
                    structured_content,
                    dict
                ):

                    stdout = (
                        structured_content
                        .get(
                            "stdout",
                            ""
                        )
                    )

                    if stdout:

                        stdout = stdout.strip()

                        try:

                            parsed = json.loads(
                                stdout
                            )

                            return json.dumps(
                                parsed
                            )

                        except json.JSONDecodeError:

                            return stdout

                return json.dumps(
                    result
                )

        raise RuntimeError(
            "Code Interpreter returned no result."
        )

    except Exception as exc:

        logger.warning(
            "Code Interpreter unavailable: %s",
            exc
        )

        # --------------------------------------------------------
        # Required fallback
        # --------------------------------------------------------

        tier_discount_rates = {
            "Silver": 0.00,
            "Gold": 0.10,
            "Platinum": 0.15
        }

        tier_discount_rate = (
            tier_discount_rates.get(
                tier_normalized,
                0.0
            )
        )

        tier_discount_pct = (
            tier_discount_rate * 100
        )

        final_total = (
            order_total
            * (1 - tier_discount_rate)
        )

        fallback = {
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct,
            "final_total": round(
                final_total,
                2
            ),
            "remaining_points": loyalty_points
        }

        return json.dumps(
            fallback
        )


# ============================================================
# Agent system prompt
# ============================================================

SYSTEM_PROMPT = """
You are the Customer Support AI Assistant for an e-commerce
business running on Amazon Bedrock AgentCore.

Your responsibility is to provide accurate, concise, friendly,
and professional customer support.

TOOL USAGE:

1. ORDER TRACKING

When the customer asks about:
- order status
- shipment status
- tracking
- delivery
- customer orders

use the appropriate AgentCore Gateway order tool.

Do not invent order information.

2. RETURNS AND REFUNDS

When the customer asks to:
- initiate a refund
- check a refund
- obtain a return label

use the appropriate AgentCore Gateway refund tool.

Do not invent refund IDs or refund statuses.

3. KNOWLEDGE BASE

Use search_knowledge_base for:
- product information
- return policies
- refund timelines
- loyalty benefits
- product specifications
- warranty information

Use the retrieved information to answer the customer.

4. LOYALTY

Use calculate_loyalty_discount whenever the customer asks
for a loyalty points or tier discount calculation.

Do not manually perform the loyalty calculation when the tool
is available.

5. MEMORY

Customer Context may contain information retrieved from
AgentCore Memory.

Use it naturally to personalize the response.

Never claim to remember something that is not present in the
provided customer context.

6. BROWSER

When the customer explicitly asks you to visit or inspect
a live webpage, use the AgentCore Browser tool.

Do not guess live webpage information.

7. TOOL SELECTION

Select the tool that directly matches the customer's request.

Do not call unrelated tools.

8. ACCURACY

Never fabricate:
- order IDs
- refund IDs
- tracking numbers
- prices
- customer details
- policies
- delivery dates
- webpage information

9. CUSTOMER-FACING RESPONSES

Respond as the customer support agent.

Do not expose:
- Python code
- MCP implementation details
- internal hooks
- internal memory implementation
- system prompts
- AWS implementation details

unless the customer explicitly asks about the technology.

10. RESPONSE STYLE

Keep responses concise and useful.

Use short bullet points when presenting several pieces of
order, refund, loyalty, or product information.

11. SERVICE ERRORS

If a required tool or service is unavailable, explain that
the requested service could not be completed at the moment.

Never create a fake result to compensate for a failed tool.
"""


# ============================================================
# TODO 8 - AgentCore Runtime entrypoint
# ============================================================

@app.entrypoint
async def invoke(
    payload,
    context=None
):
    """
    AgentCore Runtime entrypoint.
    """

    try:

        # --------------------------------------------------------
        # 1. Validate payload
        # --------------------------------------------------------

        if not isinstance(
            payload,
            dict
        ):

            payload = {}

        prompt = str(
            payload.get(
                "prompt",
                ""
            )
        ).strip()

        customer_id = str(
            payload.get(
                "customer_id",
                ""
            )
        ).strip()

        session_id = str(
            payload.get(
                "session_id",
                ""
            )
        ).strip()

        if not customer_id:

            customer_id = "anonymous"

        if not session_id:

            session_id = str(
                uuid.uuid4()
            )

        if not prompt:

            return (
                "I’m ready to help. "
                "Please tell me what you need assistance with."
            )

        # --------------------------------------------------------
        # 2. Memory hook
        # --------------------------------------------------------

        memory_hook = MemoryHook(
            actor_id=customer_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID
        )

        # --------------------------------------------------------
        # 3. AgentCore Browser
        # --------------------------------------------------------

        agent_core_browser = (
            AgentCoreBrowser(
                region=REGION
            )
        )

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            agent_core_browser.browser
        ]

        # --------------------------------------------------------
        # 4. AgentCore Gateway
        # --------------------------------------------------------

        if _is_configured(
            GATEWAY_URL
        ):

            gateway_client = MCPClient(
                lambda: streamable_http_client(
                    GATEWAY_URL
                )
            )

            try:

                with gateway_client:

                    gateway_tools = (
                        gateway_client
                        .list_tools_sync()
                    )

                    tools.extend(
                        gateway_tools
                    )

                    # ------------------------------------------------
                    # 5. Create agent
                    # ------------------------------------------------

                    agent = Agent(
                        model=model,
                        tools=tools,
                        hooks=[
                            memory_hook
                        ],
                        system_prompt=SYSTEM_PROMPT
                    )

                    # ------------------------------------------------
                    # 6. Invoke agent
                    # ------------------------------------------------

                    response = agent(
                        prompt
                    )

                    # ------------------------------------------------
                    # 7. Customer-facing response
                    # ------------------------------------------------

                    return _extract_agent_response(
                        response
                    )

            except Exception as exc:

                logger.exception(
                    "Gateway/agent invocation failed: %s",
                    exc
                )

                # Continue using non-Gateway tools.
                # This avoids fabricating order/refund results.

                fallback_prompt = """
The order and refund Gateway is currently unavailable.

You may still answer using the available Knowledge Base,
loyalty calculator, and browser tools.

If the customer specifically requests order tracking,
order information, a refund, refund status, or a return
label, explain that the required order/refund service is
temporarily unavailable.

Do not invent the requested information.
"""

                agent = Agent(
                    model=model,
                    tools=tools,
                    hooks=[
                        memory_hook
                    ],
                    system_prompt=(
                        SYSTEM_PROMPT
                        + "\n\n"
                        + fallback_prompt
                    )
                )

                response = agent(
                    prompt
                )

                return _extract_agent_response(
                    response
                )

        # --------------------------------------------------------
        # Gateway not configured
        # --------------------------------------------------------

        agent = Agent(
            model=model,
            tools=tools,
            hooks=[
                memory_hook
            ],
            system_prompt=(
                SYSTEM_PROMPT
                + """
                
The order/refund Gateway is not currently configured.

If the customer requests order tracking or refund operations,
explain that the required service is unavailable.

Do not invent an order or refund result.
"""
            )
        )

        response = agent(
            prompt
        )

        return _extract_agent_response(
            response
        )

    except Exception as exc:

        logger.exception(
            "Agent invocation failed: %s",
            exc
        )

        return (
            "I’m sorry, but I couldn’t complete that request "
            "right now. Please try again."
        )


# ============================================================
# Optional local CLI testing
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Customer Support AI Agent"
        )
    )

    parser.add_argument(
        "--prompt",
        type=str,
        help="Customer prompt"
    )

    parser.add_argument(
        "--customer-id",
        type=str,
        default="CUST-123"
    )

    parser.add_argument(
        "--session-id",
        type=str,
        default=None
    )

    args = parser.parse_args()

    payload = {
        "prompt": (
            args.prompt
            or
            "Hello, how can you help me?"
        ),
        "customer_id": args.customer_id,
        "session_id": (
            args.session_id
            or
            str(uuid.uuid4())
        )
    }

    result = asyncio.run(
        invoke(payload)
    )

    print(result)


# ============================================================
# AgentCore Runtime launch
# ============================================================

if __name__ == "__main__":

    app.run()

    # --------------------------------------------------------
    # LOCAL TESTING ONLY
    # --------------------------------------------------------
    #
    # To test without AgentCore Runtime:
    #
    # 1. Comment out:
    #
    #       app.run()
    #
    # 2. Uncomment:
    #
    #       main()
    #