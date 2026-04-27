import streamlit as st
import streamlit as st
from duckduckgo_search import DDGS
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# ─── AGENTS SETUP (outside button - runs once) ───


@tool
def search_web(query: str) -> str:
    """Search for real information about a topic in the internet and gather information and return conclusion on the question"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
        return str(results)


@tool
def format_response(content: str) -> str:
    """Format and structure content nicely and return the content which is about the topic the user asked about"""
    return f"## Formatted Response\n{content}"


model = ChatOllama(model="llama3.2")

research_agent = create_react_agent(model=model, tools=[search_web])
writer_agent = create_react_agent(
    model=model,
    tools=[format_response],
    prompt="""You are a writer agent. 
    RULES:
    - Always respond in plain text
    - Never use JSON format
    - Never use code blocks
    - Use bullet points and headers
    - Be clear and professional
    - Never explain what you're doing, just do it
    """,
)

st.title("Ai research assistant")
st.write("powerd by multi agent ai")


question = st.text_input("What do you want to research about?")

if st.button("research now"):
    if question:
        with st.spinner("Researching..."):
            research_result = research_agent.invoke(
                {"messages": [HumanMessage(content=question)]}
            )
            research_output = research_result["messages"][-1].content
            writer_result = writer_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=f"format this nicely: {research_output}")
                    ]
                }
            )
            final_output = writer_result["messages"][-1].content
        st.success("Research Complete!")
        st.markdown(final_output)
        st.subheader("Sources searched:")
        st.write(research_output)  # show raw research too

    else:
        st.write("please enter a question to research about")
