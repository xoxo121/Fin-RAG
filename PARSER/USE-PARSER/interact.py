import requests
import os
# Define the endpoint and the PDF file paths
url = "http://127.0.0.1:8000/process-pdf/"
folder_path = 'to-process'

pdf_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
# Prepare the files dictionary for multiple files
files = [("files", open(pdf, "rb")) for pdf in pdf_paths]
params = {
    'CHUNK_SIZE': 512,
    'OVERLAP': 100
}

response = requests.post(url, files=files, data=params)

for file in files:
    file[1].close()

# Save the downloaded zip file if the request is successful
if response.status_code == 200:
    with open("processed_data.zip", "wb") as f:
        f.write(response.content)
    print("Downloaded processed_data.zip successfully!")
else:
    print(f"Error: {response.status_code} - {response.text}")
