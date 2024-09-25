import streamlit as st
import openai
from PIL import Image
from io import BytesIO
import pandas as pd
import base64
from pdf2image import convert_from_bytes
import io
import os
import numpy as np

# Configure page layout
st.set_page_config(page_title="Expense Tracker", page_icon="💼", layout="wide")

openai.api_key = os.getenv("OPENAI_API_KEY")

# Define categories for the dropdown
CATEGORIES = ["Food", "Transportation", "Furniture", "Clothing", "Entertainment", "Healthcare", "Other"]

# Function to export the data as an Excel file
def export_to_excel(dataframe):
    towrite = BytesIO()
    dataframe.to_excel(towrite, index=False, header=True)
    towrite.seek(0)  # reset pointer
    return towrite

# Extract raw text from receipt image using GPT-4 OCR
def extract_raw_text_from_img_openai(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Please extract all the text from this receipt image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }],
        max_tokens=2048
    )
    
    return response['choices'][0]['message']['content']

# Adjusted GPT prompt to handle multiple receipts
def extract_structured_data(content: str):
    template = """
        You are an expert at extracting information from receipts. Please note that the image may contain more than one receipt. 
        If there are multiple receipts, extract and return the details for each receipt individually.
        
        For each receipt, extract the following details:
        - Date of Purchase
        - Merchant
        - Amount
        - Category (choose from the following: "Food", "Transportation", "Furniture", "Clothing", "Entertainment", "Healthcare", "Other")
        - Summary
        
        If any information is not available, return "Not available".
        
        For the **Summary**, ensure that it is in the format: "<Merchant> payment for <item/transaction made> on <date>." 
        The summary should be 10-20 words long, clearly describing the purpose of the transaction.

        Here is the receipt content:

        {content}

        Return the information in this format:
        Receipt 1:
        Date of Purchase: 
        Merchant:
        Amount:
        Category:
        Summary:
        
        Receipt 2 (if applicable):
        Date of Purchase: 
        Merchant:
        Amount:
        Category:
        Summary:
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": template.format(content=content)
        }],
        max_tokens=1000
    )
    
    return response.choices[0].message['content'].strip()

# Parse multiple receipts from the GPT response
def parse_multiple_receipts(gpt_response: str):
    # Split response into sections based on the "Receipt X:" pattern
    receipts = gpt_response.split('Receipt ')
    parsed_receipts = []

    for receipt in receipts[1:]:  # Skip the first element (before 'Receipt 1:')
        lines = receipt.strip().split('\n')
        receipt_data = {
            "Date of Purchase": lines[1].split(":")[1].strip() if len(lines) > 1 else "Not available",
            "Merchant": lines[2].split(":")[1].strip() if len(lines) > 2 else "Not available",
            "Amount": lines[3].split(":")[1].strip() if len(lines) > 3 else "Not available",
            "Category": lines[4].split(":")[1].strip() if len(lines) > 4 else "Not available",
            "Description": lines[5].split(":")[1].strip() if len(lines) > 5 else "Not available"  # Ensure "Description" is set
        }
        parsed_receipts.append(receipt_data)

    return parsed_receipts

# Function to convert PDF to images
def convert_pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)  # Convert PDF bytes to images
    return [img for img in images]

# Main function for the Streamlit app
def main():
    st.title("💼 Expense Tracker")
    st.subheader("Take a photo of the receipt or upload it (Image or PDF)")

    # Layout for camera input, "OR", and file uploader
    col1, col2, col3 = st.columns([1, 1, 2])

    # Use Streamlit camera input to take a photo
    with col1:
        camera_image = st.camera_input("Take a picture of the receipt")

    # Add "OR" in the middle column
    with col2:
        st.markdown("<h3 style='text-align: center;'>OR</h3>", unsafe_allow_html=True)

    # Use file uploader for images and PDFs
    with col3:
        uploaded_files = st.file_uploader("Upload Receipt", accept_multiple_files=True, type=["pdf", "jpg", "jpeg", "png"])

    # Expense Summary Table to display results
    if 'summary_data' not in st.session_state:
        st.session_state['summary_data'] = []

    # Process camera or uploaded images
    if camera_image or uploaded_files:
        images_list = []

        # If camera image is available, add to images_list
        if camera_image:
            images_list.append(camera_image.getvalue())

        # Handle uploaded files (supporting both PDF and image files)
        if uploaded_files:
            for file in uploaded_files:
                if file.type == "application/pdf":  # If PDF, convert to images
                    pdf_bytes = file.getvalue()
                    images = convert_pdf_to_images(pdf_bytes)
                    for img in images:
                        # Convert each page image to byte format
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        images_list.append(buffered.getvalue())  # Save image bytes
                else:
                    images_list.append(file.getbuffer())

        # Extract raw text and structured data from each image
        for idx, image_bytes in enumerate(images_list):
            raw_text = extract_raw_text_from_img_openai(image_bytes)
            structured_data = extract_structured_data(raw_text)

            # Parse multiple receipts (if applicable)
            receipts = parse_multiple_receipts(structured_data)

            # Append each receipt to the summary data
            for receipt in receipts:
                # Ensure all fields are initialized properly
                st.session_state['summary_data'].append({
                    "Date of Purchase": receipt["Date of Purchase"],
                    "Merchant": receipt["Merchant"],
                    "Amount": receipt["Amount"],
                    "Category": receipt["Category"],
                    "Description": receipt["Description"]  # Use the "Description" field
                })

            # Display receipt image and receipt details in columns
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}", caption="Uploaded Receipt", use_column_width=True)

            with col2:
                for idx, receipt in enumerate(receipts):
                    with st.form(f'edit_form_{idx}'):  # Track form by index
                        st.subheader(f"📝 Edit Receipt {idx+1}")
                        date_input = st.text_input("Date of Purchase:", receipt["Date of Purchase"])
                        merchant_input = st.text_input("Merchant:", receipt["Merchant"])
                        amount_input = st.text_input("Amount:", receipt["Amount"])
                        category_input = st.selectbox("Category:", CATEGORIES, index=CATEGORIES.index(receipt["Category"]) if receipt["Category"] in CATEGORIES else 0)
                        description_input = st.text_area("Description:", receipt["Description"], height=100)

                        # Layout for update button and success message
                        button_col, success_col = st.columns([1, 3])

                        with button_col:
                            submitted = st.form_submit_button("Update")

                        if submitted:
                            # Update the row in session_state
                            st.session_state['summary_data'][idx] = {
                                "Date of Purchase": date_input,
                                "Merchant": merchant_input,
                                "Amount": amount_input,
                                "Category": category_input,
                                "Description": description_input
                            }
                            with success_col:
                                st.success("Details updated!", icon="✅")

    # Display Expense Summary Table
    if st.session_state['summary_data']:
        st.subheader("📊 Expense Summary")
        summary_df = pd.DataFrame(st.session_state['summary_data'])
        
        # Ensure the "Description" column is available before displaying the table
        if "Description" in summary_df.columns:
            st.table(summary_df[['Date of Purchase', 'Merchant', 'Amount', 'Category', 'Description']])
        else:
            st.table(summary_df[['Date of Purchase', 'Merchant', 'Amount', 'Category']])  # Fallback if "Description" is missing

        # Add export to Excel button
        excel_data = export_to_excel(summary_df[['Date of Purchase', 'Merchant', 'Amount', 'Category', 'Description']])
        st.download_button(
            label="Download data as Excel",
            data=excel_data,
            file_name='expense_summary.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

# Run the app
if __name__ == "__main__":
    main()
