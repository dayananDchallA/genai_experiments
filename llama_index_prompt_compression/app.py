from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate, download_loader
from llama_index.llms import openai
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
documents = loader.load_data(pages=['Nicolas Cage'])

index = VectorStoreIndex.from_documents(documents)

retriever = index.as_retriever(similarity_top_k=2)

question = "Where did Nicolas Cage go to school?"

contexts = retriever.retrieve(question)

# Expected answer:  Beverly Hills High School

# The contexts list carries NodeWithScore data entities with metadata and relationship information with other nodes. 
# For now, we are only interested in the content.

context_list = [n.get_content() for n in contexts]
print(context_list)


# We combine these relevant chunks with the original query to create a prompt. 
# We will use a prompt template instead of just a f-string because we want to reuse it down the line.
# We then feed this prompt into gpt-3.5-turbo-16k to generate a response.

# The response from original prompt

llm = openai.OpenAI(model="gpt-3.5-turbo-16k")

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

qa_template = PromptTemplate(template)

# you can create text prompt (for completion API)
prompt = qa_template.format(context_str="\n\n".join(context_list), query_str=question)

response = llm.complete(prompt)
print(str(response))

# Now, let’s measure the RAG performance after using different prompt compression techniques.

# Selective Context
# We will use a reduce_ratio of 0.5 and see how the model does. 
# If the compression keeps the information we are interested in, we will lower the value in order to compress more text.

# from selective_context import SelectiveContext
# sc = SelectiveContext(model_type='gpt2', lang='en')
# context_string = "\n\n".join(context_list)
# context, reduced_content = sc(context_string, reduce_ratio = 0.5,reduce_level="sent")
# prompt = qa_template.format(context_str="\n\n".join(reduced_content), query_str=question)
# response = llm.complete(prompt)
# print(str(response))


# Setup LLMLingua
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
node_postprocessor = LongLLMLinguaPostprocessor(
    instruction_str="Given the context, please answer the final question",
    target_token=300,
    rank_method="longllmlingua",
    additional_compress_kwargs={
        "condition_compare": True,
        "condition_in_question": "after",
        "context_budget": "+100",
        "reorder_context": "sort",  # enable document reorder,
        "dynamic_context_compression_ratio": 0.3,
    },
)
retrieved_nodes = retriever.retrieve(question)
synthesizer = CompactAndRefine()

#The postprocess_nodes function is the one we care about the most because it shortens the node text given the query.

from llama_index.core.indices.query.schema import QueryBundle

new_retrieved_nodes = node_postprocessor.postprocess_nodes(
    retrieved_nodes, query_bundle=QueryBundle(query_str=question)
)

original_contexts = "\n\n".join([n.get_content() for n in retrieved_nodes])
compressed_contexts = "\n\n".join([n.get_content() for n in new_retrieved_nodes])
original_tokens = node_postprocessor._llm_lingua.get_token_length(original_contexts)
compressed_tokens = node_postprocessor._llm_lingua.get_token_length(compressed_contexts)
print(compressed_contexts)
print()
print("Original Tokens:", original_tokens)
print("Compressed Tokens:", compressed_tokens)
print("Compressed Ratio:", f"{original_tokens/(compressed_tokens + 1e-5):.2f}x")