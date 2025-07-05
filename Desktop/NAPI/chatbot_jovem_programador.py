import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
from bs4 import BeautifulSoup

# Carrega as variáveis de ambiente
load_dotenv()

class JovemProgramadorChatbot:
    def __init__(self):
        # Inicializa os modelos
        self.groq_model = ChatGroq(
            temperature=0.7,
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        self.gemini_model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Configura o sistema de embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Carrega e processa o conteúdo do site
        self.vectorstore = self._load_website_content()
        
        # Configura a cadeia de processamento
        self.chain = self._setup_chain()

    def _load_website_content(self):
        # URL base do site Jovem Programador
        base_url = "https://www.jovemprogramador.com.br/"
        
        # Páginas importantes para extrair conteúdo
        pages = [
            "",
            "blog/",
            "cursos/",
            "sobre/",
            "contato/"
        ]
        
        documents = []
        
        for page in pages:
            url = base_url + page
            try:
                # Carrega o conteúdo da página
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                # Processa o HTML para extrair conteúdo relevante
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove elementos indesejados
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                # Atualiza o conteúdo do documento
                docs[0].page_content = soup.get_text(separator="\n", strip=True)
                documents.extend(docs)
                
                print(f"Conteúdo carregado com sucesso: {url}")
            except Exception as e:
                print(f"Erro ao carregar {url}: {str(e)}")
        
        # Divide os documentos em chunks menores
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Cria o vectorstore com os embeddings
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        return vectorstore

    def _setup_chain(self):
        # Template do prompt
        template = """
        Você é um assistente especializado em ajudar usuários com informações sobre o site Jovem Programador.
        O site Jovem Programador (https://www.jovemprogramador.com.br/) é uma plataforma educacional que oferece cursos, artigos e recursos para programadores iniciantes.
        
        Use o seguinte contexto para responder à pergunta. Se não souber a resposta, diga que não sabe, não invente informações.
        
        Contexto:
        {context}
        
        Pergunta: {question}
        
        Responda de forma clara, concisa e útil, mantendo um tom amigável e encorajador, adequado para programadores iniciantes.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Configura a cadeia de processamento
        retriever = self.vectorstore.as_retriever()
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.groq_model  # Você pode alternar para self.gemini_model se preferir
            | StrOutputParser()
        )
        
        return chain

    def chat(self, question):
        try:
            response = self.chain.invoke(question)
            return response
        except Exception as e:
            return f"Ocorreu um erro ao processar sua pergunta: {str(e)}"

def main():
    print("Inicializando o chatbot Jovem Programador...")
    chatbot = JovemProgramadorChatbot()
    
    print("\nChatbot pronto! Digite 'sair' para encerrar.")
    print("Pergunte sobre cursos, artigos ou qualquer informação disponível no site Jovem Programador.\n")
    
    while True:
        user_input = input("Você: ")
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("Chatbot: Até logo! Bons estudos na sua jornada de programação!")
            break
            
        response = chatbot.chat(user_input)
        print("\nChatbot:", response)
        print()

if __name__ == "__main__":
    main()