from fastapi import FastAPI, File, UploadFile, Form
from pathlib import Path
import json
import zipfile
import shutil
from starlette.responses import FileResponse
from fast import converter, save_tables, remove_loc_tags, process_document, get_chunks
from langchain_ollama import ChatOllama


# Initialize FastAPI app
app = FastAPI()

# Constants
BASE_PROCESSED_PATH = Path("processed_data")

LLM = ChatOllama(model='llama3.2')

@app.post("/process-pdf/")
async def process_pdf(
    CHUNK_SIZE: int = Form(512),  # Default value for CHUNK_SIZE
    OVERLAP: int = Form(100),      # Default value for OVERLAP
    files: list[UploadFile] = File(...),
):
    zip_files = []

    # Process each uploaded file
    for file in files:
        # Create a unique directory for each file
        if file.filename is None:
            raise ValueError("Uploaded file must have a filename")
        file_stem = Path(file.filename).stem  # Filename without extension
        processed_folder = BASE_PROCESSED_PATH / file_stem
        processed_folder.mkdir(parents=True, exist_ok=True)

        # Define paths within the unique folder
        doc_path = processed_folder / "uploaded_document.pdf"
        image_path = processed_folder / "images"
        table_path = processed_folder / "tables"
        image_path.mkdir(exist_ok=True)
        table_path.mkdir(exist_ok=True)

        # Save uploaded file temporarily
        with open(doc_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Initialize converter and convert document
        convertor = converter()
        conv_res = convertor.convert(str(doc_path))

        # Save images and tables
        save_tables(conv_res, str(table_path))

        # Process document and clean tags
        doc_format = conv_res.document.export_to_document_tokens()
        cleaned_doc = remove_loc_tags(doc_format)
        json_doc = await process_document(conv_res, cleaned_doc, LLM)

        # Write JSON output to processed folder
        output_json_path = processed_folder / "output.json"
        with open(output_json_path, "w", encoding='utf-8') as f:
            json.dump(json_doc, f, indent=4)

        # Chunk the JSON data
        chunks = get_chunks(json_doc, CHUNK_SIZE, OVERLAP)
        chunks_path = processed_folder / "chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=4)

        # Zip the processed folder for each file
        zip_file_path = BASE_PROCESSED_PATH / f"{file_stem}.zip"
        with zipfile.ZipFile(zip_file_path, "w") as zipf:
            for file_path in processed_folder.rglob("*"):
                zipf.write(file_path, arcname=file_path.relative_to(BASE_PROCESSED_PATH))

        zip_files.append(zip_file_path)

    # Create a final zip with all individual zips
    final_zip_path = BASE_PROCESSED_PATH / "all_processed_data.zip"
    with zipfile.ZipFile(final_zip_path, "w") as final_zip:
        for zip_file in zip_files:
            final_zip.write(zip_file, arcname=zip_file.name)

    # Return the final zip file as response
    return FileResponse(
        final_zip_path,
        media_type="application/zip",
        filename="all_processed_data.zip"
    )

@app.get("/")
async def root():
    return {"message": "FastAPI server is running"}
