import argparse
import time
from common import helper


class SimplePdfRag:
    """
    A simple implementation of a PDF-based Retrieval-Augmented Generation (RAG) system.
    This class handles encoding a PDF document into vectorized chunks and retrieving relevant chunks
    based on a given query.
    """

    def __init__(self, path: str, chunk_size: int = 1000, chunk_overlap: int = 200, n_retrieved: int = 2):
        """
        Initializes the SimplePdfRag retriever by encoding the PDF and setting up the retriever.

        :param path: Path to the PDF file.
        :param chunk_size: Size of each text chunk (default: 1000).
        :param chunk_overlap: Overlap between consecutive chunks (default: 200).
        :param n_retrieved: Number of chunks to retrieve per query (default: 2).
        """
        print("\n----- Initializing Simple Pdf Rag Retriever")

        start_time = time.time()
        self.vector_store = helper.encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.time_records = {'chunking': time.time() - start_time}

        print(f"Chunking time: {self.time_records['chunking']:.2f} sec")

        self.chunk_query_retriever = self.vector_store.as_retriever(search_kwargs={"k": n_retrieved})

    def run(self, query: str) -> None:
        """
        Executes a query against the vector store and retrieves relevant document chunks.

        :param query: The search query to retrieve relevant content.
        """
        start_time = time.time()
        context = helper.retrieve_context_per_question(query, self.chunk_query_retriever)

        self.time_records['retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['retrieval']:.2f} seconds")

        helper.show_context(context)


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Validates command-line arguments to ensure correct input values.

    :param args: Parsed command-line arguments.
    :return: Validated arguments.
    """
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


def parse_args() -> argparse.Namespace:
    """
    Parses and validates command-line arguments.

    :return: Parsed and validated arguments.
    """
    parser = argparse.ArgumentParser(description="Encode a PDF document and test a simple RAG retriever.")
    parser.add_argument("--path", type=str, default="../data/Agents_whitepaper.pdf",
                        help="Path to the PDF file to encode.")
    parser.add_argument("--chunk_size", type=int, default=1000,
                        help="Size of each text chunk (default: 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=200,
                        help="Overlap between consecutive chunks (default: 200).")
    parser.add_argument("--n_retrieved", type=int, default=2,
                        help="Number of chunks to retrieve for each query (default: 2).")
    parser.add_argument("--query", type=str, default="What is an agent?",
                        help="Query to test the retriever (default: 'What is an agent?').")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate the retriever's performance (default: False).")

    return validate_args(parser.parse_args())


def main(args: argparse.Namespace) -> None:
    """
    Main function that initializes the retriever and executes a query.

    :param args: Parsed command-line arguments.
    """
    simple_rag = SimplePdfRag(
        path=args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved
    )

    simple_rag.run(args.query)


if __name__ == '__main__':
    main(parse_args())