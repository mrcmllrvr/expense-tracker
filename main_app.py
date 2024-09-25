import streamlit as st
import uuid
import openai
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import base64
import io

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
        model="gpt-4o",
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

# Extract structured receipt data using GPT-4
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
        Date of Purchase: 
        Merchant:
        Amount:
        Category: <choose from the provided list based on the receipt content and context>
        Summary: <Merchant> payment for <item/transaction made> on <date>.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{
            "role": "user",
            "content": template.format(content=content)
        }],
        max_tokens=500
    )
    
    return response.choices[0].message['content'].strip()

# Function to directly update the existing row (replacing the old one)
def update_existing_row(index, date, merchant, amount, category, description):
    # Replace the existing row's data without appending a new one
    st.session_state['summary_data'][index] = {
        "Date of Purchase": date,
        "Merchant": merchant,
        "Amount": amount,
        "Category": category,
        "Description": description
    }

# Main function for the Streamlit app
def main():
    st.title("💼 Expense Tracker")
    st.subheader("Hey there! I’m your new expense tracking sidekick. Just send me a photo or upload your receipts, and I’ll handle the data entry.")

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
        uploaded_files = st.file_uploader("Upload receipt", accept_multiple_files=True, type=["pdf", "jpg", "jpeg", "png"])

    # Expense Summary Table to display results
    if 'summary_data' not in st.session_state:
        st.session_state['summary_data'] = []

    # Display captured or uploaded receipt and extracted details
    if camera_image or uploaded_files:
        images_list = []

        # If camera image is available, use it
        if camera_image:
            images_list.append(camera_image.getvalue())

        # Handle uploaded files
        if uploaded_files:
            for file in uploaded_files:
                images_list.append(file.getbuffer())

        # Extract raw text and structured data from each image
        for idx, image_bytes in enumerate(images_list):
            raw_text = extract_raw_text_from_img_openai(image_bytes)
            structured_data = extract_structured_data(raw_text)

            # Parse structured data returned from GPT-4
            fields = structured_data.split('\n')

            # Extract fields from the response
            date_of_purchase = fields[0].split(":")[1].strip() if len(fields) > 0 else ""
            merchant = fields[1].split(":")[1].strip() if len(fields) > 1 else ""
            amount = fields[2].split(":")[1].strip() if len(fields) > 2 else ""
            category = fields[3].split(":")[1].strip() if len(fields) > 3 else "Not available"
            description = fields[4].split(":")[1].strip() if len(fields) > 4 else ""

            # If this is the first time, add the extracted data to the summary table
            if idx >= len(st.session_state['summary_data']):
                st.session_state['summary_data'].append({
                    "Date of Purchase": date_of_purchase,
                    "Merchant": merchant,
                    "Amount": amount,
                    "Category": category,
                    "Description": description
                })

            # Layout to display receipt and autofill fields side by side
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode('utf-8')}", caption="Uploaded Receipt", use_column_width=True)

            with col2:
                with st.form(f'edit_form_{idx}'):  # Track form by index
                    st.subheader("📝 Edit Receipt")
                    date_input = st.text_input("Date of Purchase:", date_of_purchase)
                    merchant_input = st.text_input("Merchant:", merchant)
                    amount_input = st.text_input("Amount:", amount)
                    category_input = st.selectbox("Category:", CATEGORIES, index=CATEGORIES.index(category) if category in CATEGORIES else 0)
                    description_input = st.text_area("Description:", description, height=100)
                    
                    # Create a layout for the button and success message side by side
                    button_col, success_col = st.columns([1, 10])
                    
                    with button_col:
                        submitted = st.form_submit_button("Update")
                    
                    if submitted:
                        # Update the row directly at index 'idx'
                        update_existing_row(idx, date_input, merchant_input, amount_input, category_input, description_input)
                        with success_col:
                            st.success("Details updated!", icon="✅")

    # Display Expense Summary Table below the form
    if st.session_state['summary_data']:
        st.subheader("📊 Expense Summary")
        
        # Create a DataFrame for the summary
        summary_df = pd.DataFrame(st.session_state['summary_data'])
        
        # Display the table with specific column headers
        st.table(summary_df[['Date of Purchase', 'Merchant', 'Amount', 'Category', 'Description']])

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
