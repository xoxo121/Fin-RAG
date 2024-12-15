import logging
import time
from pathlib import Path
import re
import json
from docling_core.types.doc.document import PictureItem
import asyncio
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

CHUNK_SIZE = 256
IMAGE_RESOLUTION_SCALE = 2.0

llm = ChatOllama(model="llama3.2")

logging.basicConfig(level=logging.INFO)

input_doc_path = ""

from docling.datamodel.pipeline_options import EasyOcrOptions

def converter():
    pipeline_options = PdfPipelineOptions(
        # Disable OCR but provide required options
        do_ocr=False,
        ocr_options=EasyOcrOptions(force_full_page_ocr=False),
        
        # Table settings
        do_table_structure=True,
        table_structure_options=TableStructureOptions(),
        
        # # Image settings
        # images_scale=IMAGE_RESOLUTION_SCALE,
        # generate_page_images=False,
        # generate_table_images=False,
        # generate_picture_images=True
    )

    try:
        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        logging.info("Document converter initialized successfully")
        return doc_converter
        
    except Exception as e:
        logging.error(f"Failed to initialize document converter: {str(e)}")
        raise

# def save_images(conv_res, save_path: str | Path) -> None:
#     """Save images from document to specified path.
    
#     Args:
#         conv_res: Conversion result containing document
#         save_path: Directory path to save images
#     """
#     picture_counter = 0
#     output_dir = Path(save_path)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     doc_filename = conv_res.input.file.stem

#     for element, _level in conv_res.document.iterate_items():
#         if not isinstance(element, PictureItem):
#             continue
            
#         try:
#             if element.image is None:
#                 logging.warning(f"Skipping image - element.image is None")
#                 continue
                
#             if element.image.pil_image is None:
#                 logging.warning(f"Skipping image - pil_image is None")
#                 continue
                
#             picture_counter += 1
#             element_image_filename = (
#                 output_dir / f"{doc_filename}-picture-{picture_counter}.png"
#             )
            
#             with element_image_filename.open("wb") as fp:
#                 element.image.pil_image.save(fp, format="PNG")
#                 logging.info(f"Saved image {picture_counter} to {element_image_filename}")
                
#         except AttributeError as e:
#             logging.error(f"Failed to save image {picture_counter}: {str(e)}")
#         except Exception as e:
#             logging.error(f"Unexpected error saving image {picture_counter}: {str(e)}")

#     logging.info(f"Saved {picture_counter} images to {output_dir}")


def save_tables(conv_res, save_path):
    table_output_dir = Path(save_path)
    table_output_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    for iter, table in enumerate(conv_res.document.tables):
        markdown_filename = table_output_dir / f"table{iter}.md"        
        # Save as Markdown
        with open(markdown_filename, 'w', encoding='utf-8') as f_md:
            f_md.write(table.export_to_markdown())
        k += 1
    logging.info(f"{k} Tables identified and saved in multiple formats.")

        

def remove_loc_tags(text):
    # Regular expression to match <loc*> patterns
    text = re.sub(r"<loc[^>]*>", "", text)
    text = re.sub(r"</loc[^>]*>", "", text)
    text = re.sub(r'<page_\d+>|</page_\d+>', '', text)
    return text

async def process_tables(table, llm):
    prompt = """
    You are being given a table in the form of markdown. 
    Only return the summary of the tables in 5 sentences.
    """
    input = prompt + '\n' + table
    messages = [
        ("system", prompt),
        ("human", input)
    ]
    logging.info("Invoking LLM with messages: %s", messages)
    res = llm.invoke(messages)
    logging.info("Received response from LLM: %s", res.content)
    res = re.sub(r"Here is a summary of the table in 5 sentences:", "", res.content)
    res = res.strip()
    logging.info("Processed response: %s", res)
    print("llm_call")
    return res

async def process_figures(figure, llm):
    await asyncio.sleep(1)  # Simulate processing time
    return f"Processed figure for {figure}"

def add_to_json_final(json_final, entry_type, header, cur_text=None, list_items=None):
    if cur_text:
        json_final.append({
            'type': entry_type,
            'title': header,
            'text': header + ': ' + cur_text 
        })
    if list_items:
        json_final.append({
            'type': 'list',
            'title': header,
            'items': list_items
        })

async def process_document(doc, html_content, llm):
    table_iter = 0
    figure_iter = 0
    tables = doc.document.tables
    json_final = []
    soup = BeautifulSoup(html_content, 'html.parser')

    header = ""
    cur_text = ""
    list_items = []
    table_summaries = []
    figure_summaries = []

    # Fix: soup.find() instead of soup.document
    for tag in soup.find_all(['section_header', 'text', 'list_item', 'table', 'figure']):
        try:
            if tag.name == "section_header":
                add_to_json_final(json_final, 'text', header, cur_text)
                cur_text = ""
                add_to_json_final(json_final, 'list', header, list_items=list_items)
                list_items = []
                header = tag.text.strip()
                if header == 'Table of Contents': 
                    header = ""

            elif tag.name == 'text':
                add_to_json_final(json_final, 'list', header, list_items=list_items)
                list_items = []
                cur_text += tag.text.strip() + " "

            elif tag.name == 'list_item':
                list_items.append(tag.text.strip())

            elif tag.name == 'table' and table_iter < len(tables):
                add_to_json_final(json_final, 'text', header, cur_text)
                cur_text = ""
                add_to_json_final(json_final, 'list', header, list_items=list_items)
                list_items = []
                
                table = tables[table_iter].export_to_markdown()
                table_summaries.append({'index': len(json_final), 'table': table})
                json_final.append({
                    'type': 'table',
                    'text': '',
                    'table_iter': table_iter,
                    'title': header,
                    'reference': f"table{table_iter}.md"
                })
                table_iter += 1

            elif tag.name == 'figure':
                add_to_json_final(json_final, 'text', header, cur_text)
                cur_text = ""
                add_to_json_final(json_final, 'list', header, list_items=list_items)
                list_items = []
                
                json_final.append({
                    'type': 'figure',
                    'text': "",
                    'figure_iter': figure_iter,
                    'title': header
                })
                figure_iter += 1

        except Exception as e:
            logging.error(f"Error processing tag {tag.name}: {str(e)}")
            continue

    # Process final content
    add_to_json_final(json_final, 'text', header, cur_text)
    add_to_json_final(json_final, 'list', header, list_items=list_items)

    # Process tables with error handling
    try:
        table_processing_tasks = [process_tables(entry['table'], llm) for entry in table_summaries]
        table_summaries_results = await asyncio.gather(*table_processing_tasks)
        
        for i, summary in enumerate(table_summaries_results):
            if i < len(table_summaries):
                json_final[table_summaries[i]['index']]['text'] = summary
                
    except Exception as e:
        logging.error(f"Error processing tables: {str(e)}")

    return json_final

def get_chunks(elements, chunk_size: int, overlap_size: int) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
        separators=["\n\n", "\n", ". "]
    )
    chunks = []
    
    try:
        for element in elements:
            if element.get('type') == 'text' and element.get('text'):
                temp_chunks = text_splitter.split_text(element['text'])
                chunks.extend([{
                    'type': 'text',
                    'title': element.get('title', ''),
                    'text': element.get('title', '') + chunk
                } for chunk in temp_chunks])
            else:
                chunks.append(element)
    except Exception as e:
        logging.error(f"Error in get_chunks: {str(e)}")
        
    return chunks