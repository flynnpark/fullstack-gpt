from typing import TypedDict

import streamlit as st
from bs4 import Tag
from consts import answers_prompt_template, choose_prompt_messages
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class GetAnswerInput(TypedDict):
    docs: list[Document]
    question: RunnablePassthrough


class ChooseAnswerInput(TypedDict):
    answers: list[dict[str, str]]
    question: str


class Chain:

    def __init__(self, url: str):
        if not url.endswith(".xml"):
            raise ValueError("The URL must end with '.xml'")
        self.llm = ChatOpenAI(
            temperature=0.1,
            api_key=st.session_state["openai_key"],
        )
        self.url = url
        self.answers_prompt = ChatPromptTemplate.from_template(answers_prompt_template)
        self.choose_prompt = ChatPromptTemplate.from_messages(choose_prompt_messages)

        self.chain = self.make_chain()

    def __parse_page(self, soup: Tag):
        header = soup.find("header")
        footer = soup.find("footer")
        if header and isinstance(header, Tag):
            header.decompose()
        if footer and isinstance(footer, Tag):
            footer.decompose()
        return (
            soup.get_text()
            .replace("\n", " ")
            .replace("\xa0", " ")
            .replace("CodeSearch Submit Blog", "")
        )

    def __get_answers(self, input: GetAnswerInput):
        docs = input["docs"]
        question = input["question"]
        answers_chain = self.answers_prompt | self.llm
        return {
            "question": question,
            "answers": [
                {
                    "answer": answers_chain.invoke(
                        {"question": question, "context": doc.page_content}
                    ).content,
                    "source": doc.metadata["source"],
                    "date": doc.metadata["lastmod"],
                }
                for doc in docs
            ],
        }

    def __choose_answer(self, input: ChooseAnswerInput):
        answers = input["answers"]
        question = input["question"]
        choose_chain = self.choose_prompt | self.llm
        condensed = "\n\n".join(
            f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
            for answer in answers
        )
        return choose_chain.invoke(
            {
                "question": question,
                "answers": condensed,
            }
        )

    @st.cache_resource(show_spinner="Loading websites...")
    def _load_website(_self):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
        )
        loader = SitemapLoader(
            web_path=_self.url,
            parsing_function=_self.__parse_page,
            filter_urls=[r"^(.*\/(ai-gateway|vectorize|workers-ai)\/).*"],
        )
        loader.requests_per_second = 2
        docs = loader.load_and_split(text_splitter=splitter)
        vector_store = FAISS.from_documents(
            docs,
            OpenAIEmbeddings(
                api_key=st.session_state["openai_key"],
            ),
        )
        return vector_store.as_retriever()

    def make_chain(self):
        retriever = self._load_website()
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.__get_answers)
            | RunnableLambda(self.__choose_answer)
        )
        return chain

    def invoke(self, question: str):
        result = self.chain.invoke(question)
        return result


def main():
    st.set_page_config(
        page_title="SiteGPT",
        page_icon="🧊",
    )
    st.title("SiteGPT")
    st.text(
        "First, input your OpenAI API key in to the sidebar. Then, write the URL of the website."
    )
    st.text("Finally, Ask questions about the content of a website.")

    with st.sidebar:
        st.link_button(
            "Github Repo",
            "https://github.com/flynnpark/fullstack-gpt/tree/assignment17",
        )
        st.session_state["openai_key"] = st.text_input("OpenAI API Key")
        url = st.text_input("Wirte down a URL", placeholder="https://www.example.com")

    if len(url) > 1 and not url.endswith(".xml"):
        st.error("The URL must end with '.xml'")
        st.stop()

    elif url and st.session_state.get("openai_key"):
        question = st.text_input("Ask a question")
        chain = Chain(url=url)
        if question:
            result = chain.invoke(question)
            st.markdown(result.content.replace("$", "\$"))  # type: ignore


if __name__ == "__main__":
    main()
