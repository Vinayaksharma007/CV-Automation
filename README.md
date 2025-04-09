# CV-Automation


Documentation for Resume Processing Workflow
Overview
This script processes resumes in PDF format stored in a Google Drive folder, extracts structured information from them, and saves the output as a nested JSON file. It uses OCR for text extraction, OpenAI API for data structuring, and Pydantic for validation. The structured data includes personal information, career objectives, skills, experience, education, projects, certifications, achievements, and total experience.

Workflow
1. Authentication
Function: authenticate_to_drive
Authenticates to Google Drive using a service account.

Requires a credentials.json file with access to the Drive API.

2. Listing Files in a Folder
Function: list_files_in_folder
Lists all files in the specified Google Drive folder.

Filters PDF files based on their MIME type (application/pdf).

3. Downloading Files
Function: download_file_as_bytes
Downloads the PDF file from Google Drive as a byte stream.

Uses the Google Drive API.

4. PDF to Images Conversion
Function: pdf_bytes_to_images
Converts the PDF byte stream into images using the pdf2image library.

This step prepares the file for OCR text extraction.

5. Text Extraction
Function: extract_text_from_image
Uses Tesseract OCR to extract text from the images generated from the PDF.

The extracted text serves as input for the OpenAI API.

6. Text Processing with OpenAI API
Function: process_text_with_openai
Sends the extracted text to the OpenAI API for conversion into structured JSON.

Includes a schema for data validation.

Handles missing fields by setting them to null.

OpenAI Schema
The JSON schema includes the following:

personal_info

career_objective

skills

experience

education

projects

certifications

achievements

total_experience

7. Experience Calculation
Function: calculate_total_experience
Computes the total work experience from the experience section.

Handles various date formats and normalizes durations like 2 years 3 months into years.

8. Aggregating Data
Function: process_pdfs_to_nested_json
Iterates over all PDF files in the folder.

Extracts and processes text for each file.

Aggregates all processed data into a single JSON file.

9. Saving Output
The final JSON data is saved to a file specified by the user.
File Requirements
Input
Google Drive Folder: Contains resumes in PDF format.

Service Account Credentials: JSON file with permissions for Drive API.

Output
A nested JSON file (aggregated_data.json) containing structured data for all resumes.
Execution
Steps to Run the Script
Update the following configuration parameters in the __main__ block:

credentials_path: Path to the service account credentials.

Credentials_path
This is the file path to the Google service account key file (a JSON file).

Steps to Obtain:
Create a Service Account:

Go to the Google Cloud Console.

Navigate to IAM & Admin > Service Accounts.

Click Create Service Account, and follow the steps to create it.

Download the Key:

After creating the service account, go to the "Keys" section for that service account.

Click Add Key > Create New Key > Select JSON.

Download the JSON file.

Store the File:
Save the downloaded JSON file in a secure location on your system.The credentials_path is the full file path to this JSON file, e.g., /path/to/service_account.json.

Share the Drive Folder:
Share the target Google Drive folder with the service account's email (e.g., my-service-account@my-project.iam.gserviceaccount.com) to grant it access.

drive_folder_id: ID of the Google Drive folder containing PDFs.
Drive_folder_id
drive.google.com/drive/folders/1SiMWJdDJX-xVrd9-TjbUFEllOL0d6sz7

The <folder_id> part of the URL is the drive_folder_id,

output_file_path: Path to save the output JSON file.

openai_api_key: Your OpenAI API key.

Error Handling
Authentication Errors: Ensures the script stops if Drive authentication fails.

Missing Fields: Handles missing data gracefully by setting them to null.

Parsing Errors: Logs and skips malformed data without crashing the process.

Logs
The script logs the following:

Authentication and API calls.
Processing status of each file.
Errors encountered during any step.
requirements
google-api-python-client==2.92.0

google-auth==2.23.4

google-auth-httplib2==0.1.0

google-auth-oauthlib==1.0.0

pytesseract==0.3.10

pdf2image==1.16.3

requests==2.31.0

python-dateutil==2.8.2

pydantic==1.10.12
