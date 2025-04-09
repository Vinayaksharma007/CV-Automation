import os
import io
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_bytes
import pytesseract
from pydantic import ValidationError
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from pydantic import BaseModel, Field
from typing import List, Optional
from dateutil.parser import parse
from dotenv import load_dotenv
from docx2pdf import convert as convert_docx_to_pdf
import tempfile

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_CONCURRENT_REQUESTS = 5  # Limit concurrent API calls to avoid rate limiting

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in .env file")

# Pydantic models remain the same as in your original code
class PersonalInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None

class Experience(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    address: Optional[str] = None
    duration: Optional[str] = None
    responsibilities: Optional[List[str]] = []
    is_part_time: Optional[bool] = False  

class Education(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    duration: Optional[str] = None

class Project(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    technologies_used: Optional[List[str]] = []

class Certification(BaseModel):
    title: Optional[str] = None
    issuing_organization: Optional[str] = None
    date_issued: Optional[str] = None

class Achievement(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None

class Candidate(BaseModel):
    personal_info: Optional[PersonalInfo] = None
    career_objective: Optional[str] = None
    skills: Optional[List[str]] = []
    experience: Optional[List[Experience]] = []
    education: Optional[List[Education]] = []
    projects: Optional[List[Project]] = []
    certifications: Optional[List[Certification]] = []
    achievements: Optional[List[Achievement]] = []
    total_experience: Optional[float] = None
    relevant_experience: Optional[dict] = None

class AggregatedData(BaseModel):
    candidates: List[Candidate] = []

def log_message(message, start_time=None):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if start_time:
        time_taken = time.time() - start_time
        print(f"[{current_time}] {message} (Time taken: {time_taken:.2f} seconds)")
    else:
        print(f"[{current_time}] {message}")

def authenticate_to_drive(credentials_path):
    """Authenticate to Google Drive using service account credentials."""
    try:
        credentials = Credentials.from_service_account_file(credentials_path, scopes=["https://www.googleapis.com/auth/drive"])
        drive_service = build('drive', 'v3', credentials=credentials)
        log_message("Successfully authenticated to Google Drive.")
        return drive_service
    except Exception as e:
        log_message(f"Authentication failed: {e}")
        raise

async def list_files_in_folder(drive_service, folder_id):
    """List all files in a Google Drive folder."""
    log_message(f"📂 Fetching files from Google Drive Folder: {folder_id}")

    try:
        query = f"'{folder_id}' in parents and trashed=false"
        response = drive_service.files().list(q=query, fields="files(id, name, mimeType)").execute()
        files = response.get('files', [])
        log_message(f"🔍 Found {len(files)} files in the folder.")
        for file in files:
            log_message(f"📄 {file['name']} (ID: {file['id']})")  
        return files
    except Exception as e:
        log_message(f"❌ Error listing files: {e}")
        return []

async def download_file_as_bytes(drive_service, file_id):
    """Download a file from Google Drive as bytes."""
    try:
        request = drive_service.files().get_media(fileId=file_id)
        file_stream = io.BytesIO()
        downloader = MediaIoBaseDownload(file_stream, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            log_message(f"Processing: {int(status.progress() * 100)}%")
        file_stream.seek(0)  
        return file_stream
    except Exception as e:
        log_message(f"Error downloading file: {e}")
        return None

async def convert_word_to_pdf(word_bytes, file_name):
    """Convert Word document to PDF using docx2pdf."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_docx:
            temp_docx.write(word_bytes.read())
            temp_docx_path = temp_docx.name
        
        pdf_path = tempfile.mktemp(suffix=".pdf")
        convert_docx_to_pdf(temp_docx_path, pdf_path)
        
        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = io.BytesIO(pdf_file.read())
        
        # Clean up temporary files
        os.unlink(temp_docx_path)
        os.unlink(pdf_path)
        
        return pdf_bytes
    except Exception as e:
        log_message(f"Error converting Word to PDF: {e}")
        return None

async def pdf_bytes_to_images(pdf_bytes):
    """Convert PDF bytes to images using pdf2image with async wrapper."""
    start_time = time.time()
    log_message("Converting PDF bytes to images.", start_time)
    
    def sync_convert():
        try:
            return convert_from_bytes(pdf_bytes.read())
        except Exception as e:
            log_message(f"Error converting PDF bytes to images: {e}", start_time)
            return []
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        images = await loop.run_in_executor(pool, sync_convert)
    
    log_message(f"Conversion completed. Got {len(images)} images.", start_time)
    return images

async def extract_text_from_image(image):
    """Extract text from image using pytesseract with async wrapper."""
    start_time = time.time()
    log_message("Extracting text from image.", start_time)
    
    def sync_extract():
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            error_message = f"Error extracting text: {str(e)}"
            log_message(error_message, start_time)
            return error_message
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        text = await loop.run_in_executor(pool, sync_extract)
    
    log_message("Text extraction completed.", start_time)
    return text

def parse_date(date_str):
    """Parse a date string into a datetime object, handling multiple formats."""
    if not date_str:
        return None

    try:
        return parse(date_str, fuzzy=True)
    except ValueError:
        pass

    try:
        return datetime.strptime(date_str.strip(), "%B %Y")
    except ValueError:
        pass

    try:
        return datetime.strptime(date_str.strip(), "%Y")
    except ValueError:
        pass

    try:
        return datetime.strptime(date_str.strip(), "%m/%Y")
    except ValueError:
        pass

    try:
        return datetime.strptime(date_str.strip(), "%m-%Y")
    except ValueError:
        pass

    return None

def calculate_total_experience(experiences):
    """Calculate total years of experience from a list of experiences."""
    total_days = 0
    date_ranges = []

    for exp in experiences:
        if not exp.duration:
            continue

        parts = exp.duration.split(' - ')
        if len(parts) != 2:
            continue

        start_date_str, end_date_str = parts

        start_date = parse_date(start_date_str.strip())
        end_date = parse_date(end_date_str.strip()) if end_date_str.strip().lower() not in ['present', 'till now', 'current', 'ongoing', 'on-going'] else datetime.now()

        if not start_date or not end_date:
            continue  

        if exp.is_part_time:
            total_days += (end_date - start_date).days * 0.5  
        else:
            date_ranges.append((start_date, end_date))

    date_ranges.sort(key=lambda x: x[0])

    merged_ranges = []
    for start, end in date_ranges:
        if not merged_ranges:
            merged_ranges.append((start, end))
        else:
            last_start, last_end = merged_ranges[-1]
            if start <= last_end:
                merged_ranges[-1] = (min(last_start, start), max(last_end, end))
            else:
                merged_ranges.append((start, end))

    for start, end in merged_ranges:
        total_days += (end - start).days

    total_years = total_days / 365.25
    return round(total_years, 2)

async def process_text_with_openai(session, extracted_text):
    """Process extracted text with OpenAI API using async."""
    start_time = time.time()
    log_message("Processing text with OpenAI API.", start_time)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    schema = {
        "personal_info": {
            "name": "string or null",
            "email": "string or null",
            "phone": "string or null",
            "address": "string or null",
            "linkedin": "string or null",
            "github": "string or null"
        },
        "career_objective": "string or null",
        "skills": ["string"],
        "experience": [
            {
                "job_title": "string or null",
                "company": "string or null",
                "location": "string or null",
                "duration": "string or null",
                "responsibilities": ["string"],
                "is_part_time": "boolean or null"
            }
        ],
        "education": [
            {
                "degree": "string or null",
                "institution": "string or null",
                "duration": "string or null"
            }
        ],
        "projects": [
            {
                "title": "string or null",
                "description": "string or null",
                "technologies_used": ["string"]
            }
        ],
        "certifications": [
            {
                "title": "string or null",
                "issuing_organization": "string or null",
                "date_issued": "string or null"
            }
        ],
        "achievements": [
            {
                "title": "string or null",
                "description": "string or null"
            }
        ],
        "total_experience": "float or null",
        "relevant_experience": "dict or null"
    }

    prompt_content = (
        "You are a structured JSON generator. Convert the provided resume text into a JSON object "
        f"matching the following schema: {json.dumps(schema, indent=2)}. "
        "### Instructions:\n"
        "1. **Strict Schema Adherence**: Ensure all fields are correctly structured. Use `null` for missing values.\n"
        "2. **Education Extraction**: Only include the highest pursued degree with both full and short form (e.g., 'Master of Science (M.Sc)').\n"
        "3. **Experience Handling**:\n"
        "   - Capture all details, ensuring exact company location (if provided).\n"
        "   - Convert all experience durations into a structured format.\n"
        "   - Handle formats like 'Jan 2020 - Present', 'April 2019 - Nov 2021', '5 months'.\n"
        "   - Convert months to years where applicable (e.g., '2 years 3 months' → 2.25 years).\n"
        "   - If the end date is 'present', 'till', 'current', 'now', 'ongoing', 'on-going', 'till now' calculate experience up to today's date.\n"
        "4. **Total Experience Calculation**:\n"
        "   - Ensure no double counting of overlapping job durations.\n"
        "   - Accurately compute total experience as a numeric value.\n"
        "5. **Relevant Experience Calculation**:\n"
        "   - Compute and map total duration per job title into `relevant_experience`.\n"
        "6. **Ensure Data Integrity**:\n"
        "   - Extract all resume details without omitting any relevant information.\n"
        "   - Maintain correct company addresses, ensuring JSON validity.\n"
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": f"Here is the extracted text from the resume:\n\n{extracted_text}"}
        ],
        "max_tokens": 2000
    }

    try:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                log_message(f"Error: {response.status} - {error_text}")
                return None

            response_data = await response.json()
            content = response_data["choices"][0]["message"]["content"]
            cleaned_content = content.strip("```json").strip("```").strip()
            processed_data = json.loads(cleaned_content)

            if processed_data.get("experience"):
                total_experience = calculate_total_experience([Experience(**exp) for exp in processed_data["experience"]])
                processed_data["total_experience"] = total_experience

            try:
                candidate = Candidate(**processed_data)
                return candidate.dict()
            except ValidationError as e:
                log_message(f"Validation error: {str(e)}")
                return None

    except Exception as e:
        log_message(f"Error during OpenAI API call: {str(e)}")
        return None

async def process_single_file(drive_service, file, semaphore, session):
    """Process a single file (PDF or Word) and return candidate data."""
    async with semaphore:
        start_time = time.time()
        log_message(f"Starting processing for file: {file['name']}", start_time)
        
        try:
            file_stream = await download_file_as_bytes(drive_service, file['id'])
            if not file_stream:
                return None

            if file['mimeType'] in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 
                                   'application/msword']:
                log_message(f"Converting Word document to PDF: {file['name']}")
                file_stream = await convert_word_to_pdf(file_stream, file['name'])
                if not file_stream:
                    return None

            images = await pdf_bytes_to_images(file_stream)
            if not images:
                return None

            extraction_tasks = [extract_text_from_image(image) for image in images]
            extracted_texts = await asyncio.gather(*extraction_tasks)
            extracted_text = " ".join([text for text in extracted_texts if text])

            candidate_data = await process_text_with_openai(session, extracted_text)
            
            if candidate_data:
                log_message(f"Successfully processed file: {file['name']}", start_time)
                return candidate_data
            return None

        except Exception as e:
            log_message(f"Error processing file {file['name']}: {str(e)}", start_time)
            return None

async def process_pdfs_to_nested_json(drive_service, folder_id, output_file):
    """Main async function to process all files in a folder."""
    start_time = time.time()
    log_message("Starting processing of files in Google Drive folder.", start_time)
    
    files = await list_files_in_folder(drive_service, folder_id)
    if not files:
        log_message("No files found in the folder.", start_time)
        return

    # Filter only PDF and Word files
    supported_files = [
        file for file in files 
        if file['mimeType'] in [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword'
        ]
    ]
    
    if not supported_files:
        log_message("No supported files (PDF or Word) found in the folder.", start_time)
        return

    # Use semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    aggregated_data = {"candidates": []}
    
    async with aiohttp.ClientSession() as session:
        processing_tasks = [
            process_single_file(drive_service, file, semaphore, session) 
            for file in supported_files
        ]
        
        results = await asyncio.gather(*processing_tasks)
        
        for result in results:
            if result:
                aggregated_data["candidates"].append(result)

    try:
        with open(output_file, 'w') as json_file:
            json.dump(aggregated_data, json_file, indent=4)
        log_message(f"Aggregated data saved to {output_file}. Processed {len(aggregated_data['candidates'])} resumes.", start_time)
    except Exception as e:
        log_message(f"Error saving aggregated data: {str(e)}", start_time)

async def main():
    credentials_path = "/Users/vinayaksharma/Documents/CV-Testing/securitykey.json"
    drive_folder_id = "1PqxUUvPMmdTQqEidcPvBDf9RJvo5caAO"
    output_file_path = "/Users/vinayaksharma/Documents/cv_automation/aggregated_data.json"

    try:
        drive_service = authenticate_to_drive(credentials_path)
        await process_pdfs_to_nested_json(drive_service, drive_folder_id, output_file_path)
    except Exception as e:
        log_message(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
