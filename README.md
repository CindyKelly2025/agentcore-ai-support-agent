# AI Customer Support Agent with Strands & Amazon Bedrock AgentCore


## Summary
A customer-support AI agent built with the Strands Agents SDK and deployed on Amazon Bedrock AgentCore Runtime. The agent decides which capability fits each customer request:
- **AgentCore Gateway (MCP)**: order tracking and refunds
- **Bedrock Knowledge Base**: questions about products and policies (`search_knowledge_base`)
- **AgentCore Code Interpreter**: loyalty points and discount calculation (`calculate_loyalty_discount`)
- **AgentCore Browser**: working with live web pages
- **AgentCore Memory**: remembers customer facts and preferences across sessions

## Design highlights (from my project reflection)
- **Memory through Strands hooks.** A `MemoryHook` loads relevant memories before the agent responds and saves each interaction afterward. The customer ID serves as the actor ID, so things like a customer's name and preferred contact method follow them from one session to the next. Two memory strategies are used: semantic facts and user preferences.
- **One agent with many tools.** Each AgentCore service is exposed as a tool, so the agent picks the right one for the request and no separate decision rules are needed for each case.
- **Production next steps.** Add authentication and authorization to the Gateway, which ran unauthenticated in this project, and add structured logging and monitoring for tool failures, latency, memory activity and session outcomes.

## How to run
```bash
pip install -r requirements.txt
export GATEWAY_URL=<YOUR_GATEWAY_URL> KB_ID=<YOUR_KNOWLEDGE_BASE_ID> MEMORY_ID=<YOUR_MEMORY_ID>
agentcore deploy
agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
```
The model is `global.amazon.nova-2-lite-v1:0` in us-east-1.

## Test scenarios covered
Order tracking, refund, knowledge-base answer, saving and recalling memory, loyalty calculation (Code Interpreter), and browser use.

## Results
**Results: rerun in progress — real results from my AWS account coming soon.**

## Skills demonstrated
Agentic AI design, the Strands Agents SDK, MCP tools, RAG with Bedrock Knowledge Bases, long-term agent memory, sandboxed code execution, and serverless agent deployment.
