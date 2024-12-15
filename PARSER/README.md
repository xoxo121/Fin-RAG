The above file contains the implementation of the docling parser. It is an open source parser by IBM. To run the parser follow the instructions given below.
Parser is implemented as FastAPI endpoint which can be queried to parse the PDFs.

1. Move the 'USE-Parser' folder out of the current 'Parser' folder.
2. Install all the requirements from the 'requirements.txt' folder.
3. Download the 'Ollama app' from the website and run follwing instructions to make sure 'Llama3.2 3B' model.
    Ollama pull llama3.2
    Ollama run llama3.2
4. Put the PDFs to be processed in the sub folder 'to-process' in USE-Folder.
5. Initiate the FastAPI endpoint by using the command 'uvicorn final:app --reload'.
6. Set the chunk size and overlap size in 'instruct.py'.
7. Run the 'instruct.py' file in 'USE-Folder'.
8. A zip folder will be downloaded in the 'USE-Parser' folder with 'chunks', 'tables' in markdown format, extracted figures and the original PDF.
