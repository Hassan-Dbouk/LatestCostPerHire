import os
import streamlit as st
import pandas as pd
from google.cloud import bigquery
from datetime import date, datetime
import plotly.express as px
import re
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
import tempfile

# Set page config
st.set_page_config(page_title="Cost Per Hire Analysis", layout="wide")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Setup Google Cloud credentials
def setup_credentials():
    """Setup Google Cloud credentials from Streamlit secrets or uploaded file"""
    try:
        # Try to use Streamlit secrets first (for deployment)
        if "gcp_service_account" in st.secrets:
            credentials_info = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            
            # Set up environment for BigQuery
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(credentials_info, f)
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                
            return credentials
        else:
            st.warning("⚠️ Google Cloud credentials not found in secrets. Please configure them in Streamlit Cloud.")
            return None
    except Exception as e:
        st.error(f"❌ Error setting up credentials: {str(e)[:100]}...")
        return None

# Load Main Data from GCP
@st.cache_data
def load_gcp_data():
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()  # Return empty dataframe if no credentials
        
    client = bigquery.Client(credentials=credentials)
    query = """
        SELECT 
            User_ID,
            `Application Created` AS application_date,
            `Successful_Date` AS successful_date,
            `Location Category Updated` AS location_category,
            `Nationality Category Updated` AS nationality_category,
            `Nationality Updated` AS nationality,
            `Country Updated` AS country
        FROM `data-driven-attributes.AT_marketing_db.ATD_New_Last_Action_by_User_PivotData_View`
        WHERE `Application Created` IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
    df['successful_date'] = pd.to_datetime(df['successful_date'], errors='coerce')
    return df.dropna(subset=['application_date'])

# Load CSV Data from uploaded file
def load_csv_data(uploaded_file):
    if uploaded_file is None:
        return None
        
    encodings = ['utf-16', 'utf-8', 'latin1', 'ISO-8859-1', 'cp1252']   
    
    for encoding in encodings:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
            st.success(f"Successfully read file with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                st.error("Could not read file with any of the attempted encodings")
                return None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Rename columns to match the original BigQuery names
    column_mapping = {
        'Application ID': 'maid_id',
        'Application Creation Date': 'application_date',
        'Landed': 'successful_date',
        'Location Category': 'location_category',
        'Category': 'nationality_category',
        'Nationality': 'nationality',
        'Country': 'country',
        'Freedom Operator': 'freedom_operator',
        'Exit Loan': 'exit_loan',
        'Applicant Name': 'applicant_name'
    }
    
    # Only rename columns that actually exist in the file
    column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=column_mapping)
    
    # Convert values to lowercase
    for col in ['location_category', 'nationality_category', 'nationality', 'country']:
        if col in df.columns:
            df[col] = df[col].str.lower() if pd.api.types.is_object_dtype(df[col]) else df[col]
    
    # Convert dates
    for col in ['application_date', 'successful_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert Exit Loan to numeric
    if 'exit_loan' in df.columns:
        df['exit_loan'] = pd.to_numeric(df['exit_loan'], errors='coerce')
    
    # Process the nationality_category column
    if 'nationality_category' in df.columns:
        df['original_nationality_category'] = df['nationality_category'].copy()
        
        if df['nationality_category'].str.contains(' ').any():
            df['type'] = df['nationality_category'].str.split(' ', n=1).str[1]
            df['nationality_category'] = df['nationality_category'].str.split(' ', n=1).str[0]
        
        df['nationality_category'] = df['nationality_category'].str.rstrip('s')
        
        if 'nationality' in df.columns:
            mask = (df['nationality_category'] == 'african') & (df['nationality'] == 'ethiopian')
            df.loc[mask, 'nationality_category'] = 'ethiopian'
    
    if 'maid_id' in df.columns:
        df["maid_id"] = df["maid_id"].astype(str)
    
    return df

# Load T-Visa data
@st.cache_data(show_spinner=False)
def load_t_visa():
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    client = bigquery.Client(credentials=credentials)
    tvisa_query = """
        SELECT DISTINCT CAST(maid_id AS STRING) AS maid_id
        FROM `data-driven-attributes.AT_marketing_db.maid_tvisa_tracker`
    """
    return client.query(tvisa_query).to_dataframe()

# Load fixed cost data
@st.cache_data(show_spinner=False)
def load_fixed_cost():
    credentials = setup_credentials()
    if not credentials:
        return pd.Series()  # Return empty series if no credentials
        
    client = bigquery.Client(credentials=credentials)
    fixed_query = """
        SELECT *
        FROM `data-driven-attributes.AT_marketing_db.maid_visa_fixed_cost`
    """
    result = client.query(fixed_query).to_dataframe()
    return result.iloc[0] if not result.empty else pd.Series()

# Load lost visas data
def read_lost_visas_sheet():
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    service = build("sheets", "v4", credentials=credentials)
    
    SHEET_ID = "1q0427FhcmmnIpXYrGKvmAszf56Jngwvk3QUTQ_hCztQ"
    SHEET_NAME = "Sheet1"
    
    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID,
        range=SHEET_NAME,
        majorDimension='ROWS'
    ).execute()
    
    rows = result.get("values", [])
    if not rows:
        raise ValueError("No data found in the sheet.")
    
    headers = rows[0]
    data_rows = rows[1:]
    
    normalized_rows = []
    for row in data_rows:
        row += [None] * (len(headers) - len(row))
        normalized_rows.append(row)
    
    df = pd.DataFrame(normalized_rows, columns=headers)
    
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# Load tickets data
def load_tickets_data():
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    service = build('sheets', 'v4', credentials=credentials)
    SPREADSHEET_ID = "1dDKXQKTE-yp4znJI3x-CBDwYgj9EfUDAvyBh6sneUnU"
    RANGE_NAME = "Sheet1"
    
    sheet = service.spreadsheets()
    result = sheet.get(
        spreadsheetId=SPREADSHEET_ID,
        ranges=[RANGE_NAME],
        includeGridData=True
    ).execute()
    
    rows = result['sheets'][0]['data'][0]['rowData']
    
    headers = [cell.get('formattedValue', '') for cell in rows[0]['values']]
    
    # Debug: Print headers to see what columns are available
    st.info(f"Available columns in tickets sheet: {headers}")
    
    try:
        applicant_name_index = headers.index("Applicant Name")
    except ValueError:
        # Try alternative column names
        possible_names = ["Name", "Maid Name", "Applicant", "Worker Name"]
        applicant_name_index = None
        for name in possible_names:
            if name in headers:
                applicant_name_index = headers.index(name)
                break
        
        if applicant_name_index is None:
            st.warning("Could not find applicant name column. Using first column as fallback.")
            applicant_name_index = 0
    
    data = []
    for row in rows:
        row_data = []
        maid_id = None
        values = row.get('values', [])
        
        for cell in values:
            row_data.append(cell.get('formattedValue', ''))
        
        if applicant_name_index < len(values):
            link = values[applicant_name_index].get('hyperlink', '')
            match = re.findall(r'\d{4,}', link)
            maid_id = max(match, key=len) if match else None
        
        if any(row_data):
            row_data.append(maid_id)
            data.append(row_data)
    
    headers.append("maid_id")
    df_tickets = pd.DataFrame(data[1:], columns=headers)
    
    # Handle different possible column names for price
    price_columns = ['Price AED', 'Price', 'Cost AED', 'Cost', 'Amount AED', 'Amount']
    price_col = None
    for col in price_columns:
        if col in df_tickets.columns:
            price_col = col
            break
    
    if price_col:
        df_tickets['Price AED'] = pd.to_numeric(df_tickets[price_col].astype(str).str.replace(',', ''), errors='coerce')
    else:
        st.warning("Could not find price column. Setting Price AED to 0.")
        df_tickets['Price AED'] = 0
    
    # Handle different possible column names for refunds
    refund_columns = ['Amount to be Refunded (AED)', 'Refund AED', 'Refund Amount', 'Refund', 'Amount Refunded']
    refund_col = None
    for col in refund_columns:
        if col in df_tickets.columns:
            refund_col = col
            break
    
    if refund_col:
        df_tickets['Amount to be Refunded (AED)'] = pd.to_numeric(
            df_tickets[refund_col].astype(str).str.replace(',', ''), errors='coerce'
        )
    else:
        st.warning("Could not find refund column. Setting Amount to be Refunded to 0.")
        df_tickets['Amount to be Refunded (AED)'] = 0
    
    # Handle Type column
    type_columns = ['Type', 'Ticket Type', 'Category', 'Status']
    type_col = None
    for col in type_columns:
        if col in df_tickets.columns:
            type_col = col
            break
    
    if type_col and type_col != 'Type':
        df_tickets['Type'] = df_tickets[type_col]
    elif 'Type' not in df_tickets.columns:
        st.warning("Could not find type column. Including all records.")
        df_tickets['Type'] = 'Unknown'
    
    # Filter by type if Type column exists and has valid values
    if 'Type' in df_tickets.columns:
        valid_types = df_tickets['Type'].dropna().unique()
        st.info(f"Available ticket types: {valid_types}")
        
        # Filter for known good types, but be flexible
        filter_types = ['Real', 'Dummy', 'FO Marilyn']
        available_filter_types = [t for t in filter_types if t in valid_types]
        
        if available_filter_types:
            df_tickets = df_tickets[df_tickets['Type'].isin(available_filter_types)]
        else:
            st.warning("No matching ticket types found. Including all records.")
    
    return df_tickets

# Load BAs cost data
def read_cost_sheet(sheet_id, sheet_name):
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    service = build("sheets", "v4", credentials=credentials)
    
    USD_TO_AED = 3.67
    
    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range=sheet_name,
        majorDimension='ROWS'
    ).execute()
    
    rows = result.get("values", [])
    if not rows:
        raise ValueError(f"No data found in sheet: {sheet_name}")
    
    headers = rows[0]
    data_rows = rows[1:]
    
    df = pd.DataFrame(data_rows, columns=headers)
    
    df.columns = [col.strip().lower() for col in df.columns]
    df = df.rename(columns={
        "month": "Month",
        "filipina share usd": "filipina_share_usd",
        "african share usd": "african_share_usd",
        "ethiopian share usd": "ethiopian_share_usd"
    })
    
    for col in ["filipina_share_usd", "african_share_usd", "ethiopian_share_usd"]:
        df[col] = df[col].replace(",", "", regex=True).astype(float)
    
    df["filipina_share_aed"] = df["filipina_share_usd"] * USD_TO_AED
    df["african_share_aed"] = df["african_share_usd"] * USD_TO_AED
    df["ethiopian_share_aed"] = df["ethiopian_share_usd"] * USD_TO_AED
    
    df["Month"] = pd.to_datetime(df["Month"], format="%B %Y").dt.strftime("%b-%y")
    
    return df[["Month", "filipina_share_aed", "african_share_aed", "ethiopian_share_aed"]]

# Load agents data
def read_and_transform_agents_sheet(sheet_id, sheet_name="Agents"):
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    service = build("sheets", "v4", credentials=credentials)
    
    USD_TO_AED = 3.67
    
    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range=sheet_name,
        majorDimension='ROWS'
    ).execute()
    
    rows = result.get("values", [])
    if not rows:
        raise ValueError(f"No data found in sheet: {sheet_name}")
    
    headers = [h.strip() for h in rows[0]]
    data_rows = rows[1:]
    
    normalized_rows = []
    for row in data_rows:
        padded_row = row + [None] * (len(headers) - len(row))
        normalized_rows.append(padded_row[:len(headers)])
    
    df = pd.DataFrame(normalized_rows, columns=headers)
    
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.rename(columns={"salary_usd": "salary_usd", "nationality_category": "nationality_category"}, inplace=True)
    
    df["nationality_category"] = df["nationality_category"].replace('', None).ffill()
    
    df["month"] = pd.to_datetime(df["month"], format="%B %Y", errors="coerce").dt.strftime("%b-%y")
    
    df = df.dropna(subset=["salary_usd", "month"])
    
    df["salary_usd"] = df["salary_usd"].replace(",", "", regex=True).astype(float)
    df["salary_aed"] = df["salary_usd"] * USD_TO_AED
    
    exploded_rows = []
    for _, row in df.iterrows():
        nat = str(row["nationality_category"]).strip().lower()
        
        if "all" in nat:
            categories = ["filipina", "ethiopian", "african"]
        elif "+" in nat:
            categories = [x.strip().lower() for x in nat.split("+")]
        else:
            categories = [nat]
        
        share = row["salary_aed"] / len(categories)
        for cat in categories:
            exploded_rows.append({
                "month": row["month"],
                "nationality_category": cat,
                "salary_aed": round(share, 2)
            })
    
    exploded_df = pd.DataFrame(exploded_rows)
    
    pivot_df = exploded_df.pivot_table(
        index="month",
        columns="nationality_category",
        values="salary_aed",
        aggfunc="sum",
        fill_value=0
    ).reset_index()
    
    pivot_df = pivot_df.rename(columns={
        "filipina": "filipina_cost_aed",
        "ethiopian": "ethiopian_cost_aed",
        "african": "african_cost_aed"
    })
    
    for col in ["filipina_cost_aed", "ethiopian_cost_aed", "african_cost_aed"]:
        if col not in pivot_df.columns:
            pivot_df[col] = 0.0
    
    return pivot_df

# Load LLM costs
def read_llm_costs_sheet_fixed():
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    service = build("sheets", "v4", credentials=credentials)
    
    SHEET_ID = "192G2EAL_D7lKEGAaJ-6BkRacpU0UgS2_akZxqTT5lWo"
    SHEET_NAME = "Sheet1"
    
    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID,
        range=SHEET_NAME,
        majorDimension='ROWS'
    ).execute()
    
    rows = result.get("values", [])
    if not rows:
        raise ValueError("No data found in the sheet.")
    
    headers = rows[0]
    data_rows = rows[1:]
    
    df = pd.DataFrame(data_rows, columns=headers)
    
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Maids At"] = df["Maids At"].str.replace("$", "", regex=False).astype(float)
    
    df = df.dropna(subset=["Date", "Maids At"])
    
    df["Cost AED"] = df["Maids At"] * 3.67
    df["Month"] = df["Date"].dt.strftime('%b-%y')
    
    df_LLM = df.groupby("Month", as_index=False)["Cost AED"].sum()
    df_LLM = df_LLM.rename(columns={"Cost AED": "Cost"})
    
    return df_LLM

# Load referrals data
def read_referrals_sheet():
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    service = build("sheets", "v4", credentials=credentials)
    
    REFERRALS_SHEET_ID = "1FlETNT-_hPcGgN_hzMNJ7UNUMwvZntu0vxNT05NORk8"
    SHEET_NAME = "Sheet1"
    
    result = service.spreadsheets().values().get(
        spreadsheetId=REFERRALS_SHEET_ID,
        range=SHEET_NAME,
        majorDimension='ROWS'
    ).execute()
    
    rows = result.get("values", [])
    headers = rows[0]
    data_rows = rows[1:]
    
    num_cols = len(headers)
    normalized_rows = []
    for row in data_rows:
        row = row + [None] * (num_cols - len(row)) if len(row) < num_cols else row[:num_cols]
        normalized_rows.append(row)
    
    df = pd.DataFrame(normalized_rows, columns=headers)
    
    if "Referred Maid Applicant ID" not in df.columns:
        raise ValueError("Column 'Referred Maid Applicant ID' not found in the sheet.")
    df = df.rename(columns={"Referred Maid Applicant ID": "maid_id"})
    df["maid_id"] = df["maid_id"].astype(str)
    
    if "CC/MV" not in df.columns:
        raise ValueError("Column 'CC/MV' not found in the sheet.")
    df = df[df["CC/MV"].str.strip().str.upper() == "CC"]
    
    return df

# Load broadcasts data
def read_broadcasts_sheet():
    credentials = setup_credentials()
    if not credentials:
        return pd.DataFrame()
        
    service = build("sheets", "v4", credentials=credentials)
    
    SHEET_ID = "1jw2te4TeXRx0GHAubr9ZyXtJF2gNEacJITpmwxUNFS0"
    SHEET_NAME = "Sheet1"
    
    result = service.spreadsheets().values().get(
        spreadsheetId=SHEET_ID,
        range=SHEET_NAME,
        majorDimension='ROWS'
    ).execute()
    
    rows = result.get("values", [])
    if not rows:
        raise ValueError("No data found in the sheet.")
    
    headers = rows[0]
    data_rows = rows[1:]
    
    broadcasts_df = pd.DataFrame(data_rows, columns=headers)
    
    broadcasts_df.columns = broadcasts_df.columns.str.strip().str.lower()
    broadcasts_df = broadcasts_df.rename(columns={
        "month": "Month",
        "filipina": "filipina_broadcast",
        "ethiopian": "ethiopian_broadcast",
        "african": "african_broadcast"
    })
    
    broadcasts_df["Month"] = pd.to_datetime(broadcasts_df["Month"], format="%B %y", errors='coerce').dt.strftime("%b-%y")
    
    for col in ["filipina_broadcast", "ethiopian_broadcast", "african_broadcast"]:
        broadcasts_df[col] = pd.to_numeric(broadcasts_df[col], errors="coerce").fillna(0)
    
    return broadcasts_df

# Modified marketing cost calculation with date range parameters
def calculate_marketing_cost_per_hire(df, num_months=12, behavior_start_date=None, behavior_end_date=None):
    """
    Calculate marketing cost per hire using the weighted approach with date range control for behavior analysis.
    """
    credentials = setup_credentials()
    if not credentials:
        df['marketing_cost_per_hire'] = 0.0
        return df
        
    # Load Spend Data
    client = bigquery.Client(credentials=credentials)
    query = """
        SELECT
            DATE_TRUNC(application_created_date, MONTH) AS spend_month,
            nationality_category,
            location_category,
            SUM(total_spend_aed) AS monthly_spend
        FROM `data-driven-attributes.AT_marketing_db.AT_Country_Daily_Performance_Spend_ERP_Updated`
        GROUP BY spend_month, nationality_category, location_category
    """
    spend_df = client.query(query).to_dataframe()
    spend_df['spend_month'] = pd.to_datetime(spend_df['spend_month']).dt.to_period('M').dt.to_timestamp()
    
    # Modified time-to-hire distribution function with date filtering
    def compute_time_to_hire_distribution(df_filtered, start_date=None, end_date=None):
        df_filtered = df_filtered[df_filtered['successful_date'].notna()].copy()
        
        # Apply date range filter for behavior analysis if provided
        if start_date is not None and end_date is not None:
            df_filtered = df_filtered[
                (df_filtered['application_date'] >= pd.to_datetime(start_date)) & 
                (df_filtered['application_date'] <= pd.to_datetime(end_date))
            ]
        
        df_filtered['application_month'] = df_filtered['application_date'].dt.to_period("M").dt.to_timestamp()
        df_filtered['hire_month'] = df_filtered['successful_date'].dt.to_period("M").dt.to_timestamp()
        df_filtered['month_name'] = df_filtered['application_date'].dt.strftime("%b")
        
        month_wise_brackets = {}
        for month_name, month_data in df_filtered.groupby('month_name'):
            if month_data.empty:
                month_wise_brackets[month_name] = [0.0] * num_months
                continue
                
            total_hires = len(month_data)
            bracket_counts = [0] * num_months
            
            for cohort_month, group in month_data.groupby('application_month'):
                for offset in range(num_months):
                    start = cohort_month + pd.DateOffset(months=offset)
                    end = cohort_month + pd.DateOffset(months=offset + 1)
                    count = group[(group['hire_month'] >= start) & (group['hire_month'] < end)].shape[0]
                    bracket_counts[offset] += count
                    
            month_wise_brackets[month_name] = [(c / total_hires if total_hires > 0 else 0) for c in bracket_counts]
        
        return month_wise_brackets
    
    # Calculate CAC for each combination of nationality and location category
    cac_lookup = {}
    
    nationality_location_pairs = df[df['successful_date'].notna()][['nationality_category', 'location_category']].drop_duplicates()
    
    for _, row in nationality_location_pairs.iterrows():
        nationality = row['nationality_category']
        location = row['location_category']
        
        if pd.isna(nationality) or pd.isna(location):
            continue
        
        filtered_df = df[
            (df['nationality_category'] == nationality) &
            (df['location_category'] == location)
        ].copy()
        
        # Pass date range to behavior analysis
        month_wise_brackets = compute_time_to_hire_distribution(
            filtered_df, 
            start_date=behavior_start_date, 
            end_date=behavior_end_date
        )
        
        filtered_spend = spend_df[
            (spend_df['nationality_category'] == nationality) &
            (spend_df['location_category'] == location)
        ].copy()
        
        monthly_spend = filtered_spend.groupby('spend_month')['monthly_spend'].sum().reset_index()
        
        hire_data = filtered_df[filtered_df['successful_date'].notna()].copy()
        
        hire_data['hire_month'] = hire_data['successful_date'].dt.to_period("M").dt.to_timestamp()
        monthly_hires = hire_data.groupby('hire_month').size().reset_index(name='hires')
        
        for _, hire_row in monthly_hires.iterrows():
            hire_month = hire_row['hire_month']
            hires = hire_row['hires']
            
            if hires == 0:
                continue
                
            weighted_spend = 0
            for i in range(num_months):
                spend_month = (hire_month - pd.DateOffset(months=i)).to_period('M').to_timestamp()
                spend = monthly_spend[monthly_spend['spend_month'] == spend_month]['monthly_spend'].sum()
                
                month_name = spend_month.strftime("%b")
                weight = month_wise_brackets.get(month_name, [0]*num_months)[i] if month_name in month_wise_brackets else 0
                
                weighted_spend += spend * weight
            
            cac = weighted_spend / hires if hires > 0 else 0
            
            month_key = hire_month.strftime('%Y-%m')
            if (nationality, location) not in cac_lookup:
                cac_lookup[(nationality, location)] = {}
            
            cac_lookup[(nationality, location)][month_key] = cac
    
    # Add CAC column to main dataframe
    df['marketing_cost_per_hire'] = 0.0
    
    for idx, row in df[df['successful_date'].notna()].iterrows():
        nat = row['nationality_category']
        loc = row['location_category']
        month_key = pd.to_datetime(row['successful_date']).strftime('%Y-%m')
        
        if pd.isna(nat) or pd.isna(loc) or (nat, loc) not in cac_lookup or month_key not in cac_lookup.get((nat, loc), {}):
            continue
            
        df.at[idx, 'marketing_cost_per_hire'] = cac_lookup[(nat, loc)][month_key]
    
    return df

# Main data processing function
def process_all_data(behavior_start_date=None, behavior_end_date=None, uploaded_csv=None):
    """Process all data and return the final dataframe with all cost components"""
    
    with st.spinner("Loading and processing data..."):
        # Load main data from GCP
        st.info("Loading data from GCP...")
        df_gcp = load_gcp_data()
        
        if df_gcp.empty:
            st.error("Could not load data from GCP. Please check your credentials.")
            return pd.DataFrame()
            
        df_gcp = df_gcp.rename(columns={"User_ID": "maid_id"})
        df_gcp["maid_id"] = df_gcp["maid_id"].astype(str)
        
        # Apply initial nationality_category transformation
        df_gcp['nationality_category'] = df_gcp.apply(
            lambda row: 'ethiopian' if row['nationality_category'] == 'african' and row['nationality'] == 'ethiopian' else row['nationality_category'],
            axis=1)
        
        # Load CSV data if uploaded
        if uploaded_csv is not None:
            try:
                st.info("Loading data from uploaded CSV...")
                df_csv = load_csv_data(uploaded_csv)
                
                if df_csv is not None:
                    # Process CSV integration logic here (shortened for space)
                    csv_maid_ids = set(df_csv['maid_id'])
                    
                    # Remove Ethiopian records from GCP data
                    ethiopian_count = sum(df_gcp['nationality_category'] == 'ethiopian')
                    df_gcp = df_gcp[df_gcp['nationality_category'] != 'ethiopian']
                    
                    # Set all successful_date values to null in GCP data
                    df_gcp['successful_date'] = pd.NaT
                    
                    # Add missing columns
                    for col in ['applicant_name', 'type', 'exit_loan', 'freedom_operator']:
                        if col not in df_gcp.columns:
                            df_gcp[col] = None if col in ['applicant_name', 'type'] else (0 if col == 'exit_loan' else '')
                    
                    # Update GCP data with CSV data
                    update_dict = {}
                    for idx, row in df_csv.iterrows():
                        maid_id = row['maid_id']
                        update_values = {}
                        
                        for field in ['nationality_category', 'nationality', 'location_category', 'successful_date', 'applicant_name', 'type']:
                            if field in df_csv.columns and pd.notna(row[field]):
                                update_values[field] = row[field]
                        
                        update_values['exit_loan'] = row['exit_loan'] if 'exit_loan' in df_csv.columns and pd.notna(row['exit_loan']) else 0
                        update_values['freedom_operator'] = row['freedom_operator'] if 'freedom_operator' in df_csv.columns and pd.notna(row['freedom_operator']) else ''
                        
                        update_dict[maid_id] = update_values
                    
                    # Apply updates
                    for idx, row in df_gcp.iterrows():
                        maid_id = row['maid_id']
                        if maid_id in update_dict:
                            for field, value in update_dict[maid_id].items():
                                df_gcp.at[idx, field] = value
                    
                    # Add CSV-only records
                    gcp_maid_ids = set(df_gcp['maid_id'])
                    csv_only_ids = csv_maid_ids - gcp_maid_ids
                    csv_only_records = df_csv[df_csv['maid_id'].isin(csv_only_ids)].copy()
                    
                    new_records = pd.DataFrame(columns=df_gcp.columns)
                    for idx, row in csv_only_records.iterrows():
                        new_row = pd.Series(index=df_gcp.columns)
                        for col in df_gcp.columns:
                            if col in csv_only_records.columns and pd.notna(row[col]):
                                new_row[col] = row[col]
                        new_records = pd.concat([new_records, pd.DataFrame([new_row])], ignore_index=True)
                    
                    df = pd.concat([df_gcp, new_records], ignore_index=True)
                    st.success(f"Successfully integrated CSV data. Total records: {len(df)}")
                else:
                    df = df_gcp
                    st.warning("Could not load CSV data. Using GCP data only.")
                    
            except Exception as e:
                st.error(f"Error loading CSV data: {e}")
                df = df_gcp
        else:
            df = df_gcp
            st.info("No CSV file uploaded. Using GCP data only.")
        
        # Load visa data
        st.info("Loading visa data...")
        df_t_visa = load_t_visa()
        t_visa_set = set(df_t_visa["maid_id"])
        fixed_cost = load_fixed_cost()
        
        # Compute visa costs
        def compute_actual_cost(row):
            nat = row["nationality_category"]
            loc = row["location_category"]
            
            if nat == "filipina":
                if loc == "inside_uae":
                    return fixed_cost.e_visa_inside
                else:
                    return (
                        fixed_cost.t_visa_outside
                        if row["maid_id"] in t_visa_set
                        else fixed_cost.e_visa_outside
                    )
            
            if nat in ["african", "ethiopian"]:
                if loc == "outside_uae":
                    return fixed_cost.e_visa_outside
                else:
                    return fixed_cost.e_visa_inside
            
            return 0
        
        df["actual_visa_cost"] = df.apply(compute_actual_cost, axis=1)
        
        # Load and process lost visas
        lost_visas_df = read_lost_visas_sheet()
        lost_visas_df.columns = lost_visas_df.columns.str.strip().str.lower()
        
        # Ensure successful_date is datetime before creating month columns
        df['successful_date'] = pd.to_datetime(df['successful_date'], errors='coerce')
        df['successful_month'] = df['successful_date'].dt.strftime('%b-%y')
        df['lost_evisa_share'] = 0.0
        
        for _, row in lost_visas_df.iterrows():
            month = row['month']
            for nationality in ['filipina', 'ethiopian', 'african']:
                lost_total = pd.to_numeric(row[nationality], errors='coerce')
                mask = (df['successful_month'] == month) & (df['nationality_category'].str.lower() == nationality)
                count = mask.sum()
                if count > 0 and pd.notnull(lost_total):
                    share = lost_total / count
                    df.loc[mask, 'lost_evisa_share'] = round(share, 2)
        
        # Load and process tickets
        st.info("Loading tickets data...")
        df_tickets = load_tickets_data()
        
        df['maid_id'] = df['maid_id'].astype(str)
        df_tickets['maid_id'] = df_tickets['maid_id'].astype(str)
        
        df_successful = df[df['successful_date'].notnull()].copy()
        
        ticket_cost = (
            df_tickets.groupby('maid_id', as_index=False)['Price AED']
            .sum()
            .rename(columns={'Price AED': 'total_ticket_cost'})
        )
        
        ticket_refund = (
            df_tickets.groupby('maid_id', as_index=False)['Amount to be Refunded (AED)']
            .sum()
            .rename(columns={'Amount to be Refunded (AED)': 'total_ticket_refund'})
        )
        
        df_successful = df_successful.merge(ticket_cost, on='maid_id', how='left')
        df_successful = df_successful.merge(ticket_refund, on='maid_id', how='left')
        
        df_successful['total_ticket_cost'] = df_successful['total_ticket_cost'].fillna(0)
        df_successful['total_ticket_refund'] = df_successful['total_ticket_refund'].fillna(0)
        
        df = df.merge(
            df_successful[['maid_id', 'total_ticket_cost', 'total_ticket_refund']],
            on='maid_id',
            how='left'
        )
        
        # Process lost tickets
        df_tickets['Travel Date'] = pd.to_datetime(df_tickets['Travel Date'], errors='coerce')
        today = pd.to_datetime(datetime.today().date())
        tickets_past = df_tickets[df_tickets['Travel Date'] < today].copy()
        
        hired_maids = df[df['successful_date'].notna()]['maid_id'].astype(str).unique()
        tickets_past['maid_id'] = tickets_past['maid_id'].astype(str)
        tickets_lost = tickets_past[~tickets_past['maid_id'].isin(hired_maids)].copy()
        
        df_maids_lookup = df[['maid_id', 'nationality_category', 'location_category']].drop_duplicates()
        tickets_lost = tickets_lost.merge(df_maids_lookup, on='maid_id', how='left')
        
        tickets_lost['ticket_month'] = tickets_lost['Travel Date'].dt.strftime('%b-%y')
        tickets_lost['Price AED'] = pd.to_numeric(tickets_lost['Price AED'], errors='coerce').fillna(0)
        tickets_lost['Amount to be Refunded (AED)'] = pd.to_numeric(tickets_lost['Amount to be Refunded (AED)'], errors='coerce').fillna(0)
        tickets_lost['net_lost_cost'] = tickets_lost['Price AED'] - tickets_lost['Amount to be Refunded (AED)']
        
        lost_ticket_grouped = tickets_lost.groupby(['ticket_month', 'nationality_category', 'location_category'])['net_lost_cost'].sum().reset_index()
        
        # Ensure successful_date is datetime and create Month column safely
        df['successful_date'] = pd.to_datetime(df['successful_date'], errors='coerce')
        df['Month'] = df['successful_date'].dt.strftime('%b-%y')
        
        hire_counts = df[df['successful_date'].notna()].groupby(['Month', 'nationality_category', 'location_category'])['maid_id'].count().reset_index()
        hire_counts = hire_counts.rename(columns={'maid_id': 'hire_count'})
        
        merged_costs = lost_ticket_grouped.merge(hire_counts, left_on=['ticket_month', 'nationality_category', 'location_category'],
                                               right_on=['Month', 'nationality_category', 'location_category'], how='left')
        
        merged_costs['lost_ticket_share'] = merged_costs['net_lost_cost'] / merged_costs['hire_count']
        merged_costs = merged_costs[['Month', 'nationality_category', 'location_category', 'lost_ticket_share']]
        
        df = df.merge(merged_costs, on=['Month', 'nationality_category', 'location_category'], how='left')
        
        # Load staff costs
        st.info("Loading staff costs...")
        SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
        bas_cost_df = read_cost_sheet(SHEET_ID, "BAs")
        daspgs_cost_df = read_cost_sheet(SHEET_ID, "Programmers&DAs")
        
        def assign_costs_by_month(df_maids, df_costs, target_col_name):
            df_maids[target_col_name] = 0.0
            for _, row in df_costs.iterrows():
                month = row["Month"]
                for nat in ["filipina", "african", "ethiopian"]:
                    cost_value = row[f"{nat}_share_aed"]
                    mask = (
                        (df_maids["successful_month"] == month) &
                        (df_maids["nationality_category"].str.lower() == nat) &
                        (df_maids["successful_date"].notnull())
                    )
                    count = mask.sum()
                    if count > 0 and cost_value > 0:
                        df_maids.loc[mask, target_col_name] = cost_value / count
            return df_maids
        
        df = assign_costs_by_month(df, bas_cost_df, "bas_cost_share")
        df = assign_costs_by_month(df, daspgs_cost_df, "DataAnalysts_and_Programmers_cost_share")
        
        # Load agents costs
        agents_cost_summary = read_and_transform_agents_sheet(SHEET_ID)
        
        df['Agents_cost_share'] = 0.0
        for _, row in agents_cost_summary.iterrows():
            month = row['month']
            for nat in ['filipina', 'ethiopian', 'african']:
                cost_col = f"{nat}_cost_aed"
                if cost_col not in agents_cost_summary.columns:
                    continue
                total_cost = row[cost_col]
                if pd.isna(total_cost) or total_cost == 0:
                    continue
                
                month_parts = month.split('-')
                if len(month_parts) != 2:
                    continue
                
                month_name, year = month_parts
                year_full = f"20{year}" if len(year) == 2 else year
                
                month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                           'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                
                if month_name not in month_map:
                    continue
                
                month_num = month_map[month_name]
                start_date = f"{year_full}-{month_num:02d}-01"
                
                if month_num in [4, 6, 9, 11]:
                    end_date = f"{year_full}-{month_num:02d}-30"
                elif month_num == 2:
                    is_leap = (int(year_full) % 4 == 0 and int(year_full) % 100 != 0) or (int(year_full) % 400 == 0)
                    end_date = f"{year_full}-{month_num:02d}-29" if is_leap else f"{year_full}-{month_num:02d}-28"
                else:
                    end_date = f"{year_full}-{month_num:02d}-31"
                
                mask = (
                    (df['successful_date'] >= start_date) &
                    (df['successful_date'] <= end_date) &
                    (df['nationality_category'].str.lower() == nat)
                )
                
                count = mask.sum()
                if count == 0:
                    continue
                
                cost_per_maid = total_cost / count
                df.loc[mask, 'Agents_cost_share'] = cost_per_maid
        
        # Load LLM costs
        st.info("Loading LLM and other costs...")
        df_LLM = read_llm_costs_sheet_fixed()
        
        # Add LLM cost share (simplified for space)
        df['llm_cost_share'] = 0.0
        
        # Load referrals
        referrals_df = read_referrals_sheet()
        df['maid_id'] = df['maid_id'].astype(str).str.strip()
        referrals_df['Maid A Applicant ID'] = referrals_df['Maid A Applicant ID'].astype(str).str.strip()
        referred_maid_ids = set(referrals_df['Maid A Applicant ID'])
        df['referral_cost'] = df['maid_id'].apply(lambda x: 1000 if x in referred_maid_ids else 0)
        
        # Load broadcasts
        broadcasts_df = read_broadcasts_sheet()
        df_hires = df[df['successful_date'].notnull() & df['nationality_category'].notnull()].copy()
        
        # Ensure successful_date is datetime before creating Month column
        df_hires['successful_date'] = pd.to_datetime(df_hires['successful_date'], errors='coerce')
        df_hires['Month'] = df_hires['successful_date'].dt.strftime('%b-%y')
        
        broadcast_cost_map = {}
        for _, row in broadcasts_df.iterrows():
            month = row['Month']
            for nat in ['filipina', 'ethiopian', 'african']:
                total_cost = row.get(f"{nat}_broadcast", 0)
                if total_cost == 0:
                    continue
                hires = df_hires[(df_hires['Month'] == month) & (df_hires['nationality_category'].str.lower() == nat)]
                if hires.empty:
                    continue
                cost_per_maid = total_cost / len(hires)
                for maid_id in hires['maid_id']:
                    broadcast_cost_map[maid_id] = cost_per_maid
        
        df['broadcast_cost'] = df['maid_id'].map(broadcast_cost_map).fillna(0)
        
        # Calculate marketing costs with date range
        st.info("Calculating marketing costs...")
        df = calculate_marketing_cost_per_hire(df, num_months=12, behavior_start_date=behavior_start_date, behavior_end_date=behavior_end_date)
        
        # Add operator costs
        def calculate_operator_cost(row):
            USD_TO_AED = 3.67
            cost = 0
            operator = str(row['freedom_operator']).lower() if pd.notna(row['freedom_operator']) else ""
            
            if 'marilyn' in operator:
                return 1966
            
            if row['nationality_category'] == 'ethiopian':
                if 'wa' in operator:
                    return 900 * USD_TO_AED
                elif any(op in operator for op in ['fiseha', 'natnael', 'tadesse']):
                    return 1000 * USD_TO_AED
                elif 'berana' in operator:
                    return 900 * USD_TO_AED
            
            return cost
        
        df['operator_cost'] = df.apply(calculate_operator_cost, axis=1)
        
        # Initialize attestation cost (placeholder)
        df['attestation_cost'] = 0
        
        st.success("Data processing completed successfully!")
        return df

# Main Streamlit App
def main():
    st.title("🏠 Cost Per Hire Analysis Dashboard")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Cost per Hire", "Visas", "Tickets", "BAs, DAs, Programmers", 
        "Agents", "LLMs", "Referrals", "Broadcasts", "Ad Spend", "Operator", "Attestation"
    ])
    
    with tab1:
        st.markdown("<h1 style='text-align: center; font-size: 48px;'>This is to estimate the cost per Hire</h1>", unsafe_allow_html=True)
        
        # Re-read data button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Re-Read Data", type="primary", use_container_width=True):
                st.session_state.data_loaded = False
                st.session_state.df = None
                st.cache_data.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Date range slider for behavior analysis
        st.subheader("Application Period To Consider - For Ad Spend Weights Derivation")
        st.info("Select the date range to use for calculating time-to-hire behavior patterns. This controls which historical applications are used to learn conversion patterns, while spend attribution still uses all available data.")
        
        col1, col2 = st.columns(2)
        with col1:
            behavior_start_date = st.date_input(
                "Start Date for Behavior Analysis",
                value=date(2022, 1, 1),
                key="behavior_start"
            )
        with col2:
            behavior_end_date = st.date_input(
                "End Date for Behavior Analysis",
                value=date(2025, 1, 31),
                key="behavior_end"
            )
        
        # CSV File Upload
        st.subheader("📤 Upload CSV Data (Optional)")
        st.info("Upload your Daily Conversion Report CSV file to integrate with GCP data. If not uploaded, the system will use GCP data only.")
        uploaded_csv = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your Daily Conversion Report CSV file"
        )
        
        # Load and process data
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading data..."):
                st.session_state.df = process_all_data(behavior_start_date, behavior_end_date, uploaded_csv)
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")
        
        # Only show data if it's loaded
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            st.markdown("---")
            st.subheader("📈 Data Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                successful_hires = df['successful_date'].notna().sum()
                st.metric("Successful Hires", f"{successful_hires:,}")
            with col3:
                avg_visa_cost = df[df['successful_date'].notna()]['actual_visa_cost'].mean()
                st.metric("Avg Visa Cost", f"{avg_visa_cost:.2f} AED")
            with col4:
                total_marketing_cost = df[df['successful_date'].notna()]['marketing_cost_per_hire'].sum()
                st.metric("Total Marketing Cost", f"{total_marketing_cost:,.0f} AED")
            
            # Filter for successful hires only
            successful_hires_df = df[df['successful_date'].notna()].copy()
            
            if len(successful_hires_df) > 0:
                # Add total cost per hire calculation
                cost_columns = [
                    'actual_visa_cost', 'lost_evisa_share', 'total_ticket_cost', 
                    'total_ticket_refund', 'lost_ticket_share', 'bas_cost_share',
                    'DataAnalysts_and_Programmers_cost_share', 'Agents_cost_share',
                    'llm_cost_share', 'referral_cost', 'broadcast_cost',
                    'marketing_cost_per_hire', 'operator_cost', 'attestation_cost'
                ]
                
                # Ensure all cost columns exist and fill NaN with 0
                for col in cost_columns:
                    if col not in successful_hires_df.columns:
                        successful_hires_df[col] = 0
                    successful_hires_df[col] = successful_hires_df[col].fillna(0)
                
                # Calculate net ticket cost (cost minus refund)
                successful_hires_df['net_ticket_cost'] = successful_hires_df['total_ticket_cost'] - successful_hires_df['total_ticket_refund']
                
                # Calculate total cost per hire
                successful_hires_df['total_cost_per_hire'] = (
                    successful_hires_df['actual_visa_cost'] +
                    successful_hires_df['lost_evisa_share'] +
                    successful_hires_df['net_ticket_cost'] +
                    successful_hires_df['lost_ticket_share'] +
                    successful_hires_df['bas_cost_share'] +
                    successful_hires_df['DataAnalysts_and_Programmers_cost_share'] +
                    successful_hires_df['Agents_cost_share'] +
                    successful_hires_df['llm_cost_share'] +
                    successful_hires_df['referral_cost'] +
                    successful_hires_df['broadcast_cost'] +
                    successful_hires_df['marketing_cost_per_hire'] +
                    successful_hires_df['operator_cost'] +
                    successful_hires_df['attestation_cost']
                )
                
                st.markdown("---")
                st.subheader("💰 Complete Cost per Hire Analysis")
                
                # Display download section
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.info("💾 **Download Complete Cost Analysis** - Export all successful hires with detailed cost breakdown")
                with col2:
                    # Prepare data for download
                    download_df = successful_hires_df.copy()
                    # Round numerical columns to 2 decimal places
                    numeric_columns = download_df.select_dtypes(include=['float64', 'int64']).columns
                    download_df[numeric_columns] = download_df[numeric_columns].round(2)
                    
                    csv_data = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"cost_per_hire_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                
                # Display summary of cost components
                st.subheader("📊 Cost Components Summary")
                cost_summary = pd.DataFrame({
                    'Cost Component': [
                        'Visa Costs', 'Lost E-Visa Share', 'Net Ticket Costs', 'Lost Ticket Share',
                        'BA Costs', 'DA & Programmer Costs', 'Agent Costs', 'LLM Costs',
                        'Referral Costs', 'Broadcast Costs', 'Marketing Costs', 'Operator Costs', 'Attestation Costs'
                    ],
                    'Total Amount (AED)': [
                        successful_hires_df['actual_visa_cost'].sum(),
                        successful_hires_df['lost_evisa_share'].sum(),
                        successful_hires_df['net_ticket_cost'].sum(),
                        successful_hires_df['lost_ticket_share'].sum(),
                        successful_hires_df['bas_cost_share'].sum(),
                        successful_hires_df['DataAnalysts_and_Programmers_cost_share'].sum(),
                        successful_hires_df['Agents_cost_share'].sum(),
                        successful_hires_df['llm_cost_share'].sum(),
                        successful_hires_df['referral_cost'].sum(),
                        successful_hires_df['broadcast_cost'].sum(),
                        successful_hires_df['marketing_cost_per_hire'].sum(),
                        successful_hires_df['operator_cost'].sum(),
                        successful_hires_df['attestation_cost'].sum()
                    ],
                    'Average per Hire (AED)': [
                        successful_hires_df['actual_visa_cost'].mean(),
                        successful_hires_df['lost_evisa_share'].mean(),
                        successful_hires_df['net_ticket_cost'].mean(),
                        successful_hires_df['lost_ticket_share'].mean(),
                        successful_hires_df['bas_cost_share'].mean(),
                        successful_hires_df['DataAnalysts_and_Programmers_cost_share'].mean(),
                        successful_hires_df['Agents_cost_share'].mean(),
                        successful_hires_df['llm_cost_share'].mean(),
                        successful_hires_df['referral_cost'].mean(),
                        successful_hires_df['broadcast_cost'].mean(),
                        successful_hires_df['marketing_cost_per_hire'].mean(),
                        successful_hires_df['operator_cost'].mean(),
                        successful_hires_df['attestation_cost'].mean()
                    ]
                })
                cost_summary = cost_summary.round(2)
                st.dataframe(cost_summary, use_container_width=True)
                
                # Display total cost summary
                total_cost = successful_hires_df['total_cost_per_hire'].sum()
                avg_cost = successful_hires_df['total_cost_per_hire'].mean()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Total Cost (All Hires)", f"{total_cost:,.2f} AED")
                with col2:
                    st.metric("📊 Average Cost per Hire", f"{avg_cost:,.2f} AED")
                with col3:
                    st.metric("🎯 Total Successful Hires", f"{len(successful_hires_df):,}")
                
                # Display detailed view with cost breakdown
                st.subheader("📋 Detailed Cost per Hire View")
                st.info("👁️ **Preview of downloadable data** - Showing first 50 records with all cost components")
                
                # Select key columns for display
                display_columns = [
                    'maid_id', 'applicant_name', 'successful_date', 'nationality_category', 
                    'location_category', 'actual_visa_cost', 'net_ticket_cost', 
                    'marketing_cost_per_hire', 'operator_cost', 'referral_cost', 
                    'total_cost_per_hire'
                ]
                
                # Filter columns that exist
                available_display_columns = [col for col in display_columns if col in successful_hires_df.columns]
                
                # Sort by successful_date descending and show first 50 records
                preview_df = successful_hires_df[available_display_columns].sort_values('successful_date', ascending=False).head(50)
                st.dataframe(preview_df, use_container_width=True, height=400)
                
                # Show column information
                st.subheader("📝 Dataset Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Total Columns in Dataset:** {len(successful_hires_df.columns)}")
                    st.write(f"**Records Available for Download:** {len(successful_hires_df)}")
                with col2:
                    st.write("**Key Cost Components:**")
                    key_costs = ['actual_visa_cost', 'marketing_cost_per_hire', 'operator_cost', 'referral_cost', 'total_cost_per_hire']
                    for cost in key_costs:
                        if cost in successful_hires_df.columns:
                            st.write(f"• {cost.replace('_', ' ').title()}")
            else:
                st.warning("No successful hires found in the dataset.")
    
    with tab2:
        st.header("🛂 Visa Data")
        
        try:
            st.subheader("T-Visa Data")
            df_t_visa = load_t_visa()
            st.dataframe(df_t_visa.head(), use_container_width=True)
            st.info(f"Total T-Visa records: {len(df_t_visa)}")
            
            st.subheader("Fixed Visa Costs")
            fixed_cost = load_fixed_cost()
            fixed_cost_df = pd.DataFrame([fixed_cost])
            st.dataframe(fixed_cost_df, use_container_width=True)
            
            st.subheader("Lost E-Visas Data")
            lost_visas_df = read_lost_visas_sheet()
            st.dataframe(lost_visas_df.head(), use_container_width=True)
            st.info(f"Total Lost E-Visa records: {len(lost_visas_df)}")
            
        except Exception as e:
            st.error(f"Error loading visa data: {e}")
    
    with tab3:
        st.header("✈️ Tickets Data")
        
        try:
            df_tickets = load_tickets_data()
            st.dataframe(df_tickets.head(), use_container_width=True)
            st.info(f"Total ticket records: {len(df_tickets)}")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_price = df_tickets['Price AED'].mean()
                st.metric("Average Ticket Price", f"{avg_price:.2f} AED")
            with col2:
                total_refunds = df_tickets['Amount to be Refunded (AED)'].sum()
                st.metric("Total Refunds", f"{total_refunds:,.2f} AED")
            with col3:
                ticket_types = df_tickets['Type'].nunique()
                st.metric("Ticket Types", ticket_types)
                
        except Exception as e:
            st.error(f"Error loading tickets data: {e}")
    
    with tab4:
        st.header("👥 BAs, DAs, and Programmers Data")
        
        try:
            SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
            
            st.subheader("Business Analysts Costs")
            bas_cost_df = read_cost_sheet(SHEET_ID, "BAs")
            st.dataframe(bas_cost_df.head(), use_container_width=True)
            
            st.subheader("Programmers & Data Analysts Costs")
            daspgs_cost_df = read_cost_sheet(SHEET_ID, "Programmers&DAs")
            st.dataframe(daspgs_cost_df.head(), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading staff cost data: {e}")
    
    with tab5:
        st.header("🤝 Agents Data")
        
        try:
            SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
            agents_cost_summary = read_and_transform_agents_sheet(SHEET_ID)
            st.dataframe(agents_cost_summary.head(), use_container_width=True)
            st.info(f"Total agent cost records: {len(agents_cost_summary)}")
            
        except Exception as e:
            st.error(f"Error loading agents data: {e}")
    
    with tab6:
        st.header("🤖 LLM Costs Data")
        
        try:
            df_LLM = read_llm_costs_sheet_fixed()
            st.dataframe(df_LLM.head(), use_container_width=True)
            st.info(f"Total LLM cost records: {len(df_LLM)}")
            
            # Summary
            total_llm_cost = df_LLM['Cost'].sum()
            st.metric("Total LLM Costs", f"{total_llm_cost:,.2f} AED")
            
        except Exception as e:
            st.error(f"Error loading LLM data: {e}")
    
    with tab7:
        st.header("🔗 Referrals Data")
        
        try:
            referrals_df = read_referrals_sheet()
            st.dataframe(referrals_df.head(), use_container_width=True)
            st.info(f"Total referral records: {len(referrals_df)}")
            
        except Exception as e:
            st.error(f"Error loading referrals data: {e}")
    
    with tab8:
        st.header("📢 Broadcasts Data")
        
        try:
            broadcasts_df = read_broadcasts_sheet()
            st.dataframe(broadcasts_df.head(), use_container_width=True)
            st.info(f"Total broadcast records: {len(broadcasts_df)}")
            
        except Exception as e:
            st.error(f"Error loading broadcasts data: {e}")
    
    with tab9:
        st.header("💰 Ad Spend Data")
        st.info("Marketing cost calculation uses weighted attribution based on time-to-hire patterns")
        st.markdown("""
        **How it works:**
        1. **Behavior Analysis**: Uses historical data within your selected date range to learn conversion patterns
        2. **Spend Attribution**: Looks back at ALL available spend data when calculating costs
        3. **Weighted Attribution**: Each month's spend is weighted based on historical conversion probability
        """)
        
        if st.session_state.df is not None:
            df = st.session_state.df
            marketing_summary = df[df['successful_date'].notna()].groupby(['nationality_category', 'location_category'])['marketing_cost_per_hire'].agg(['count', 'mean', 'sum']).round(2)
            marketing_summary.columns = ['Hire Count', 'Avg Cost per Hire', 'Total Cost']
            st.dataframe(marketing_summary, use_container_width=True)
    
    with tab10:
        st.header("⚙️ Operator Costs Data")
        st.info("Operator costs are calculated based on freedom operator assignments and nationality")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            operator_summary = df[df['operator_cost'] > 0].groupby(['nationality_category', 'freedom_operator'])['operator_cost'].agg(['count', 'mean']).round(2)
            operator_summary.columns = ['Count', 'Cost per Hire']
            st.dataframe(operator_summary, use_container_width=True)
    
    with tab11:
        st.header("📋 Attestation Data")
        st.info("Attestation costs would be loaded from a separate sheet (placeholder)")
        st.warning("Attestation sheet ID needs to be configured")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            attestation_count = (df['attestation_cost'] > 0).sum()
            st.metric("Records with Attestation Costs", attestation_count)

if __name__ == "__main__":
    main()















# import os
# import streamlit as st
# import pandas as pd
# from google.cloud import bigquery
# from datetime import date, datetime
# import plotly.express as px
# import gspread
# from google.oauth2.service_account import Credentials
# import re
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# import json
# import tempfile

# # Set page config
# st.set_page_config(page_title="Cost Per Hire Analysis", layout="wide")

# # Initialize session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'df' not in st.session_state:
#     st.session_state.df = None

# # Setup Google Cloud credentials
# def setup_credentials():
#     """Setup Google Cloud credentials from Streamlit secrets or uploaded file"""
#     try:
#         # Try to use Streamlit secrets first (for deployment)
#         if "gcp_service_account" in st.secrets:
#             credentials_info = dict(st.secrets["gcp_service_account"])
#             credentials = service_account.Credentials.from_service_account_info(credentials_info)
            
#             # Set up environment for BigQuery
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
#                 json.dump(credentials_info, f)
#                 os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                
#             return credentials
#         else:
#             st.error("Google Cloud credentials not found in secrets. Please configure them in Streamlit Cloud.")
#             return None
#     except Exception as e:
#         st.error(f"Error setting up credentials: {e}")
#         return None

# # Load Main Data from GCP
# @st.cache_data
# def load_gcp_data():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()  # Return empty dataframe if no credentials
        
#     client = bigquery.Client(credentials=credentials)
#     query = """
#         SELECT 
#             User_ID,
#             `Application Created` AS application_date,
#             `Successful_Date` AS successful_date,
#             `Location Category Updated` AS location_category,
#             `Nationality Category Updated` AS nationality_category,
#             `Nationality Updated` AS nationality,
#             `Country Updated` AS country
#         FROM `data-driven-attributes.AT_marketing_db.ATD_New_Last_Action_by_User_PivotData_View`
#         WHERE `Application Created` IS NOT NULL
#     """
#     df = client.query(query).to_dataframe()
#     df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
#     df['successful_date'] = pd.to_datetime(df['successful_date'], errors='coerce')
#     return df.dropna(subset=['application_date'])

# # Load CSV Data from uploaded file
# def load_csv_data(uploaded_file):
#     if uploaded_file is None:
#         return None
        
#     encodings = ['utf-16', 'utf-8', 'latin1', 'ISO-8859-1', 'cp1252']   
    
#     for encoding in encodings:
#         try:
#             # Reset file pointer
#             uploaded_file.seek(0)
#             df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
#             st.success(f"Successfully read file with encoding: {encoding}")
#             break
#         except UnicodeDecodeError:
#             if encoding == encodings[-1]:
#                 st.error("Could not read file with any of the attempted encodings")
#                 return None
#         except Exception as e:
#             st.error(f"Error reading file: {e}")
#             return None
    
#     # Clean column names
#     df.columns = df.columns.str.strip()
    
#     # Rename columns to match the original BigQuery names
#     column_mapping = {
#         'Application ID': 'maid_id',
#         'Application Creation Date': 'application_date',
#         'Landed': 'successful_date',
#         'Location Category': 'location_category',
#         'Category': 'nationality_category',
#         'Nationality': 'nationality',
#         'Country': 'country',
#         'Freedom Operator': 'freedom_operator',
#         'Exit Loan': 'exit_loan',
#         'Applicant Name': 'applicant_name'
#     }
    
#     # Only rename columns that actually exist in the file
#     column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
#     df = df.rename(columns=column_mapping)
    
#     # Convert values to lowercase
#     for col in ['location_category', 'nationality_category', 'nationality', 'country']:
#         if col in df.columns:
#             df[col] = df[col].str.lower() if pd.api.types.is_object_dtype(df[col]) else df[col]
    
#     # Convert dates
#     for col in ['application_date', 'successful_date']:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
    
#     # Convert Exit Loan to numeric
#     if 'exit_loan' in df.columns:
#         df['exit_loan'] = pd.to_numeric(df['exit_loan'], errors='coerce')
    
#     # Process the nationality_category column
#     if 'nationality_category' in df.columns:
#         df['original_nationality_category'] = df['nationality_category'].copy()
        
#         if df['nationality_category'].str.contains(' ').any():
#             df['type'] = df['nationality_category'].str.split(' ', n=1).str[1]
#             df['nationality_category'] = df['nationality_category'].str.split(' ', n=1).str[0]
        
#         df['nationality_category'] = df['nationality_category'].str.rstrip('s')
        
#         if 'nationality' in df.columns:
#             mask = (df['nationality_category'] == 'african') & (df['nationality'] == 'ethiopian')
#             df.loc[mask, 'nationality_category'] = 'ethiopian'
    
#     if 'maid_id' in df.columns:
#         df["maid_id"] = df["maid_id"].astype(str)
    
#     return df

# # Load T-Visa data
# @st.cache_data(show_spinner=False)
# def load_t_visa():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     client = bigquery.Client(credentials=credentials)
#     tvisa_query = """
#         SELECT DISTINCT CAST(maid_id AS STRING) AS maid_id
#         FROM `data-driven-attributes.AT_marketing_db.maid_tvisa_tracker`
#     """
#     return client.query(tvisa_query).to_dataframe()

# # Load fixed cost data
# @st.cache_data(show_spinner=False)
# def load_fixed_cost():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.Series()  # Return empty series if no credentials
        
#     client = bigquery.Client(credentials=credentials)
#     fixed_query = """
#         SELECT *
#         FROM `data-driven-attributes.AT_marketing_db.maid_visa_fixed_cost`
#     """
#     result = client.query(fixed_query).to_dataframe()
#     return result.iloc[0] if not result.empty else pd.Series()

# # Load lost visas data
# def read_lost_visas_sheet():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     SHEET_ID = "1q0427FhcmmnIpXYrGKvmAszf56Jngwvk3QUTQ_hCztQ"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError("No data found in the sheet.")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     normalized_rows = []
#     for row in data_rows:
#         row += [None] * (len(headers) - len(row))
#         normalized_rows.append(row)
    
#     df = pd.DataFrame(normalized_rows, columns=headers)
    
#     for col in df.columns[1:]:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     return df

# # Load tickets data
# def load_tickets_data():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build('sheets', 'v4', credentials=credentials)
#     SPREADSHEET_ID = "1dDKXQKTE-yp4znJI3x-CBDwYgj9EfUDAvyBh6sneUnU"
#     RANGE_NAME = "Sheet1"
    
#     sheet = service.spreadsheets()
#     result = sheet.get(
#         spreadsheetId=SPREADSHEET_ID,
#         ranges=[RANGE_NAME],
#         includeGridData=True
#     ).execute()
    
#     rows = result['sheets'][0]['data'][0]['rowData']
    
#     headers = [cell.get('formattedValue', '') for cell in rows[0]['values']]
    
#     # Debug: Print headers to see what columns are available
#     st.info(f"Available columns in tickets sheet: {headers}")
    
#     try:
#         applicant_name_index = headers.index("Applicant Name")
#     except ValueError:
#         # Try alternative column names
#         possible_names = ["Name", "Maid Name", "Applicant", "Worker Name"]
#         applicant_name_index = None
#         for name in possible_names:
#             if name in headers:
#                 applicant_name_index = headers.index(name)
#                 break
        
#         if applicant_name_index is None:
#             st.warning("Could not find applicant name column. Using first column as fallback.")
#             applicant_name_index = 0
    
#     data = []
#     for row in rows:
#         row_data = []
#         maid_id = None
#         values = row.get('values', [])
        
#         for cell in values:
#             row_data.append(cell.get('formattedValue', ''))
        
#         if applicant_name_index < len(values):
#             link = values[applicant_name_index].get('hyperlink', '')
#             match = re.findall(r'\d{4,}', link)
#             maid_id = max(match, key=len) if match else None
        
#         if any(row_data):
#             row_data.append(maid_id)
#             data.append(row_data)
    
#     headers.append("maid_id")
#     df_tickets = pd.DataFrame(data[1:], columns=headers)
    
#     # Handle different possible column names for price
#     price_columns = ['Price AED', 'Price', 'Cost AED', 'Cost', 'Amount AED', 'Amount']
#     price_col = None
#     for col in price_columns:
#         if col in df_tickets.columns:
#             price_col = col
#             break
    
#     if price_col:
#         df_tickets['Price AED'] = pd.to_numeric(df_tickets[price_col].astype(str).str.replace(',', ''), errors='coerce')
#     else:
#         st.warning("Could not find price column. Setting Price AED to 0.")
#         df_tickets['Price AED'] = 0
    
#     # Handle different possible column names for refunds
#     refund_columns = ['Amount to be Refunded (AED)', 'Refund AED', 'Refund Amount', 'Refund', 'Amount Refunded']
#     refund_col = None
#     for col in refund_columns:
#         if col in df_tickets.columns:
#             refund_col = col
#             break
    
#     if refund_col:
#         df_tickets['Amount to be Refunded (AED)'] = pd.to_numeric(
#             df_tickets[refund_col].astype(str).str.replace(',', ''), errors='coerce'
#         )
#     else:
#         st.warning("Could not find refund column. Setting Amount to be Refunded to 0.")
#         df_tickets['Amount to be Refunded (AED)'] = 0
    
#     # Handle Type column
#     type_columns = ['Type', 'Ticket Type', 'Category', 'Status']
#     type_col = None
#     for col in type_columns:
#         if col in df_tickets.columns:
#             type_col = col
#             break
    
#     if type_col and type_col != 'Type':
#         df_tickets['Type'] = df_tickets[type_col]
#     elif 'Type' not in df_tickets.columns:
#         st.warning("Could not find type column. Including all records.")
#         df_tickets['Type'] = 'Unknown'
    
#     # Filter by type if Type column exists and has valid values
#     if 'Type' in df_tickets.columns:
#         valid_types = df_tickets['Type'].dropna().unique()
#         st.info(f"Available ticket types: {valid_types}")
        
#         # Filter for known good types, but be flexible
#         filter_types = ['Real', 'Dummy', 'FO Marilyn']
#         available_filter_types = [t for t in filter_types if t in valid_types]
        
#         if available_filter_types:
#             df_tickets = df_tickets[df_tickets['Type'].isin(available_filter_types)]
#         else:
#             st.warning("No matching ticket types found. Including all records.")
    
#     return df_tickets

# # Load BAs cost data
# def read_cost_sheet(sheet_id, sheet_name):
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     USD_TO_AED = 3.67
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=sheet_id,
#         range=sheet_name,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError(f"No data found in sheet: {sheet_name}")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     df = pd.DataFrame(data_rows, columns=headers)
    
#     df.columns = [col.strip().lower() for col in df.columns]
#     df = df.rename(columns={
#         "month": "Month",
#         "filipina share usd": "filipina_share_usd",
#         "african share usd": "african_share_usd",
#         "ethiopian share usd": "ethiopian_share_usd"
#     })
    
#     for col in ["filipina_share_usd", "african_share_usd", "ethiopian_share_usd"]:
#         df[col] = df[col].replace(",", "", regex=True).astype(float)
    
#     df["filipina_share_aed"] = df["filipina_share_usd"] * USD_TO_AED
#     df["african_share_aed"] = df["african_share_usd"] * USD_TO_AED
#     df["ethiopian_share_aed"] = df["ethiopian_share_usd"] * USD_TO_AED
    
#     df["Month"] = pd.to_datetime(df["Month"], format="%B %Y").dt.strftime("%b-%y")
    
#     return df[["Month", "filipina_share_aed", "african_share_aed", "ethiopian_share_aed"]]

# # Load agents data
# def read_and_transform_agents_sheet(sheet_id, sheet_name="Agents"):
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     USD_TO_AED = 3.67
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=sheet_id,
#         range=sheet_name,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError(f"No data found in sheet: {sheet_name}")
    
#     headers = [h.strip() for h in rows[0]]
#     data_rows = rows[1:]
    
#     normalized_rows = []
#     for row in data_rows:
#         padded_row = row + [None] * (len(headers) - len(row))
#         normalized_rows.append(padded_row[:len(headers)])
    
#     df = pd.DataFrame(normalized_rows, columns=headers)
    
#     df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
#     df.rename(columns={"salary_usd": "salary_usd", "nationality_category": "nationality_category"}, inplace=True)
    
#     df["nationality_category"] = df["nationality_category"].replace('', None).ffill()
    
#     df["month"] = pd.to_datetime(df["month"], format="%B %Y", errors="coerce").dt.strftime("%b-%y")
    
#     df = df.dropna(subset=["salary_usd", "month"])
    
#     df["salary_usd"] = df["salary_usd"].replace(",", "", regex=True).astype(float)
#     df["salary_aed"] = df["salary_usd"] * USD_TO_AED
    
#     exploded_rows = []
#     for _, row in df.iterrows():
#         nat = str(row["nationality_category"]).strip().lower()
        
#         if "all" in nat:
#             categories = ["filipina", "ethiopian", "african"]
#         elif "+" in nat:
#             categories = [x.strip().lower() for x in nat.split("+")]
#         else:
#             categories = [nat]
        
#         share = row["salary_aed"] / len(categories)
#         for cat in categories:
#             exploded_rows.append({
#                 "month": row["month"],
#                 "nationality_category": cat,
#                 "salary_aed": round(share, 2)
#             })
    
#     exploded_df = pd.DataFrame(exploded_rows)
    
#     pivot_df = exploded_df.pivot_table(
#         index="month",
#         columns="nationality_category",
#         values="salary_aed",
#         aggfunc="sum",
#         fill_value=0
#     ).reset_index()
    
#     pivot_df = pivot_df.rename(columns={
#         "filipina": "filipina_cost_aed",
#         "ethiopian": "ethiopian_cost_aed",
#         "african": "african_cost_aed"
#     })
    
#     for col in ["filipina_cost_aed", "ethiopian_cost_aed", "african_cost_aed"]:
#         if col not in pivot_df.columns:
#             pivot_df[col] = 0.0
    
#     return pivot_df

# # Load LLM costs
# def read_llm_costs_sheet_fixed():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     SHEET_ID = "192G2EAL_D7lKEGAaJ-6BkRacpU0UgS2_akZxqTT5lWo"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError("No data found in the sheet.")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     df = pd.DataFrame(data_rows, columns=headers)
    
#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["Maids At"] = df["Maids At"].str.replace("$", "", regex=False).astype(float)
    
#     df = df.dropna(subset=["Date", "Maids At"])
    
#     df["Cost AED"] = df["Maids At"] * 3.67
#     df["Month"] = df["Date"].dt.strftime('%b-%y')
    
#     df_LLM = df.groupby("Month", as_index=False)["Cost AED"].sum()
#     df_LLM = df_LLM.rename(columns={"Cost AED": "Cost"})
    
#     return df_LLM

# # Load referrals data
# def read_referrals_sheet():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     REFERRALS_SHEET_ID = "1FlETNT-_hPcGgN_hzMNJ7UNUMwvZntu0vxNT05NORk8"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=REFERRALS_SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     num_cols = len(headers)
#     normalized_rows = []
#     for row in data_rows:
#         row = row + [None] * (num_cols - len(row)) if len(row) < num_cols else row[:num_cols]
#         normalized_rows.append(row)
    
#     df = pd.DataFrame(normalized_rows, columns=headers)
    
#     if "Referred Maid Applicant ID" not in df.columns:
#         raise ValueError("Column 'Referred Maid Applicant ID' not found in the sheet.")
#     df = df.rename(columns={"Referred Maid Applicant ID": "maid_id"})
#     df["maid_id"] = df["maid_id"].astype(str)
    
#     if "CC/MV" not in df.columns:
#         raise ValueError("Column 'CC/MV' not found in the sheet.")
#     df = df[df["CC/MV"].str.strip().str.upper() == "CC"]
    
#     return df

# # Load broadcasts data
# def read_broadcasts_sheet():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     SHEET_ID = "1jw2te4TeXRx0GHAubr9ZyXtJF2gNEacJITpmwxUNFS0"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError("No data found in the sheet.")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     broadcasts_df = pd.DataFrame(data_rows, columns=headers)
    
#     broadcasts_df.columns = broadcasts_df.columns.str.strip().str.lower()
#     broadcasts_df = broadcasts_df.rename(columns={
#         "month": "Month",
#         "filipina": "filipina_broadcast",
#         "ethiopian": "ethiopian_broadcast",
#         "african": "african_broadcast"
#     })
    
#     broadcasts_df["Month"] = pd.to_datetime(broadcasts_df["Month"], format="%B %y", errors='coerce').dt.strftime("%b-%y")
    
#     for col in ["filipina_broadcast", "ethiopian_broadcast", "african_broadcast"]:
#         broadcasts_df[col] = pd.to_numeric(broadcasts_df[col], errors="coerce").fillna(0)
    
#     return broadcasts_df

# # Modified marketing cost calculation with date range parameters
# def calculate_marketing_cost_per_hire(df, num_months=12, behavior_start_date=None, behavior_end_date=None):
#     """
#     Calculate marketing cost per hire using the weighted approach with date range control for behavior analysis.
#     """
#     credentials = setup_credentials()
#     if not credentials:
#         df['marketing_cost_per_hire'] = 0.0
#         return df
        
#     # Load Spend Data
#     client = bigquery.Client(credentials=credentials)
#     query = """
#         SELECT
#             DATE_TRUNC(application_created_date, MONTH) AS spend_month,
#             nationality_category,
#             location_category,
#             SUM(total_spend_aed) AS monthly_spend
#         FROM `data-driven-attributes.AT_marketing_db.AT_Country_Daily_Performance_Spend_ERP_Updated`
#         GROUP BY spend_month, nationality_category, location_category
#     """
#     spend_df = client.query(query).to_dataframe()
#     spend_df['spend_month'] = pd.to_datetime(spend_df['spend_month']).dt.to_period('M').dt.to_timestamp()
    
#     # Modified time-to-hire distribution function with date filtering
#     def compute_time_to_hire_distribution(df_filtered, start_date=None, end_date=None):
#         df_filtered = df_filtered[df_filtered['successful_date'].notna()].copy()
        
#         # Apply date range filter for behavior analysis if provided
#         if start_date is not None and end_date is not None:
#             df_filtered = df_filtered[
#                 (df_filtered['application_date'] >= pd.to_datetime(start_date)) & 
#                 (df_filtered['application_date'] <= pd.to_datetime(end_date))
#             ]
        
#         df_filtered['application_month'] = df_filtered['application_date'].dt.to_period("M").dt.to_timestamp()
#         df_filtered['hire_month'] = df_filtered['successful_date'].dt.to_period("M").dt.to_timestamp()
#         df_filtered['month_name'] = df_filtered['application_date'].dt.strftime("%b")
        
#         month_wise_brackets = {}
#         for month_name, month_data in df_filtered.groupby('month_name'):
#             if month_data.empty:
#                 month_wise_brackets[month_name] = [0.0] * num_months
#                 continue
                
#             total_hires = len(month_data)
#             bracket_counts = [0] * num_months
            
#             for cohort_month, group in month_data.groupby('application_month'):
#                 for offset in range(num_months):
#                     start = cohort_month + pd.DateOffset(months=offset)
#                     end = cohort_month + pd.DateOffset(months=offset + 1)
#                     count = group[(group['hire_month'] >= start) & (group['hire_month'] < end)].shape[0]
#                     bracket_counts[offset] += count
                    
#             month_wise_brackets[month_name] = [(c / total_hires if total_hires > 0 else 0) for c in bracket_counts]
        
#         return month_wise_brackets
    
#     # Calculate CAC for each combination of nationality and location category
#     cac_lookup = {}
    
#     nationality_location_pairs = df[df['successful_date'].notna()][['nationality_category', 'location_category']].drop_duplicates()
    
#     for _, row in nationality_location_pairs.iterrows():
#         nationality = row['nationality_category']
#         location = row['location_category']
        
#         if pd.isna(nationality) or pd.isna(location):
#             continue
        
#         filtered_df = df[
#             (df['nationality_category'] == nationality) &
#             (df['location_category'] == location)
#         ].copy()
        
#         # Pass date range to behavior analysis
#         month_wise_brackets = compute_time_to_hire_distribution(
#             filtered_df, 
#             start_date=behavior_start_date, 
#             end_date=behavior_end_date
#         )
        
#         filtered_spend = spend_df[
#             (spend_df['nationality_category'] == nationality) &
#             (spend_df['location_category'] == location)
#         ].copy()
        
#         monthly_spend = filtered_spend.groupby('spend_month')['monthly_spend'].sum().reset_index()
        
#         hire_data = filtered_df[filtered_df['successful_date'].notna()].copy()
        
#         hire_data['hire_month'] = hire_data['successful_date'].dt.to_period("M").dt.to_timestamp()
#         monthly_hires = hire_data.groupby('hire_month').size().reset_index(name='hires')
        
#         for _, hire_row in monthly_hires.iterrows():
#             hire_month = hire_row['hire_month']
#             hires = hire_row['hires']
            
#             if hires == 0:
#                 continue
                
#             weighted_spend = 0
#             for i in range(num_months):
#                 spend_month = (hire_month - pd.DateOffset(months=i)).to_period('M').to_timestamp()
#                 spend = monthly_spend[monthly_spend['spend_month'] == spend_month]['monthly_spend'].sum()
                
#                 month_name = spend_month.strftime("%b")
#                 weight = month_wise_brackets.get(month_name, [0]*num_months)[i] if month_name in month_wise_brackets else 0
                
#                 weighted_spend += spend * weight
            
#             cac = weighted_spend / hires if hires > 0 else 0
            
#             month_key = hire_month.strftime('%Y-%m')
#             if (nationality, location) not in cac_lookup:
#                 cac_lookup[(nationality, location)] = {}
            
#             cac_lookup[(nationality, location)][month_key] = cac
    
#     # Add CAC column to main dataframe
#     df['marketing_cost_per_hire'] = 0.0
    
#     for idx, row in df[df['successful_date'].notna()].iterrows():
#         nat = row['nationality_category']
#         loc = row['location_category']
#         month_key = pd.to_datetime(row['successful_date']).strftime('%Y-%m')
        
#         if pd.isna(nat) or pd.isna(loc) or (nat, loc) not in cac_lookup or month_key not in cac_lookup.get((nat, loc), {}):
#             continue
            
#         df.at[idx, 'marketing_cost_per_hire'] = cac_lookup[(nat, loc)][month_key]
    
#     return df

# # Main data processing function
# def process_all_data(behavior_start_date=None, behavior_end_date=None, uploaded_csv=None):
#     """Process all data and return the final dataframe with all cost components"""
    
#     with st.spinner("Loading and processing data..."):
#         # Load main data from GCP
#         st.info("Loading data from GCP...")
#         df_gcp = load_gcp_data()
        
#         if df_gcp.empty:
#             st.error("Could not load data from GCP. Please check your credentials.")
#             return pd.DataFrame()
            
#         df_gcp = df_gcp.rename(columns={"User_ID": "maid_id"})
#         df_gcp["maid_id"] = df_gcp["maid_id"].astype(str)
        
#         # Apply initial nationality_category transformation
#         df_gcp['nationality_category'] = df_gcp.apply(
#             lambda row: 'ethiopian' if row['nationality_category'] == 'african' and row['nationality'] == 'ethiopian' else row['nationality_category'],
#             axis=1)
        
#         # Load CSV data if uploaded
#         if uploaded_csv is not None:
#             try:
#                 st.info("Loading data from uploaded CSV...")
#                 df_csv = load_csv_data(uploaded_csv)
                
#                 if df_csv is not None:
#                     # Process CSV integration logic here (shortened for space)
#                     csv_maid_ids = set(df_csv['maid_id'])
                    
#                     # Remove Ethiopian records from GCP data
#                     ethiopian_count = sum(df_gcp['nationality_category'] == 'ethiopian')
#                     df_gcp = df_gcp[df_gcp['nationality_category'] != 'ethiopian']
                    
#                     # Set all successful_date values to null in GCP data
#                     df_gcp['successful_date'] = pd.NaT
                    
#                     # Add missing columns
#                     for col in ['applicant_name', 'type', 'exit_loan', 'freedom_operator']:
#                         if col not in df_gcp.columns:
#                             df_gcp[col] = None if col in ['applicant_name', 'type'] else (0 if col == 'exit_loan' else '')
                    
#                     # Update GCP data with CSV data
#                     update_dict = {}
#                     for idx, row in df_csv.iterrows():
#                         maid_id = row['maid_id']
#                         update_values = {}
                        
#                         for field in ['nationality_category', 'nationality', 'location_category', 'successful_date', 'applicant_name', 'type']:
#                             if field in df_csv.columns and pd.notna(row[field]):
#                                 update_values[field] = row[field]
                        
#                         update_values['exit_loan'] = row['exit_loan'] if 'exit_loan' in df_csv.columns and pd.notna(row['exit_loan']) else 0
#                         update_values['freedom_operator'] = row['freedom_operator'] if 'freedom_operator' in df_csv.columns and pd.notna(row['freedom_operator']) else ''
                        
#                         update_dict[maid_id] = update_values
                    
#                     # Apply updates
#                     for idx, row in df_gcp.iterrows():
#                         maid_id = row['maid_id']
#                         if maid_id in update_dict:
#                             for field, value in update_dict[maid_id].items():
#                                 df_gcp.at[idx, field] = value
                    
#                     # Add CSV-only records
#                     gcp_maid_ids = set(df_gcp['maid_id'])
#                     csv_only_ids = csv_maid_ids - gcp_maid_ids
#                     csv_only_records = df_csv[df_csv['maid_id'].isin(csv_only_ids)].copy()
                    
#                     new_records = pd.DataFrame(columns=df_gcp.columns)
#                     for idx, row in csv_only_records.iterrows():
#                         new_row = pd.Series(index=df_gcp.columns)
#                         for col in df_gcp.columns:
#                             if col in csv_only_records.columns and pd.notna(row[col]):
#                                 new_row[col] = row[col]
#                         new_records = pd.concat([new_records, pd.DataFrame([new_row])], ignore_index=True)
                    
#                     df = pd.concat([df_gcp, new_records], ignore_index=True)
#                     st.success(f"Successfully integrated CSV data. Total records: {len(df)}")
#                 else:
#                     df = df_gcp
#                     st.warning("Could not load CSV data. Using GCP data only.")
                    
#             except Exception as e:
#                 st.error(f"Error loading CSV data: {e}")
#                 df = df_gcp
#         else:
#             df = df_gcp
#             st.info("No CSV file uploaded. Using GCP data only.")
        
#         # Load visa data
#         st.info("Loading visa data...")
#         df_t_visa = load_t_visa()
#         t_visa_set = set(df_t_visa["maid_id"])
#         fixed_cost = load_fixed_cost()
        
#         # Compute visa costs
#         def compute_actual_cost(row):
#             nat = row["nationality_category"]
#             loc = row["location_category"]
            
#             if nat == "filipina":
#                 if loc == "inside_uae":
#                     return fixed_cost.e_visa_inside
#                 else:
#                     return (
#                         fixed_cost.t_visa_outside
#                         if row["maid_id"] in t_visa_set
#                         else fixed_cost.e_visa_outside
#                     )
            
#             if nat in ["african", "ethiopian"]:
#                 if loc == "outside_uae":
#                     return fixed_cost.e_visa_outside
#                 else:
#                     return fixed_cost.e_visa_inside
            
#             return 0
        
#         df["actual_visa_cost"] = df.apply(compute_actual_cost, axis=1)
        
#         # Load and process lost visas
#         lost_visas_df = read_lost_visas_sheet()
#         lost_visas_df.columns = lost_visas_df.columns.str.strip().str.lower()
        
#         df['successful_month'] = pd.to_datetime(df['successful_date'], errors='coerce').dt.strftime('%b-%y')
#         df['lost_evisa_share'] = 0.0
        
#         for _, row in lost_visas_df.iterrows():
#             month = row['month']
#             for nationality in ['filipina', 'ethiopian', 'african']:
#                 lost_total = pd.to_numeric(row[nationality], errors='coerce')
#                 mask = (df['successful_month'] == month) & (df['nationality_category'].str.lower() == nationality)
#                 count = mask.sum()
#                 if count > 0 and pd.notnull(lost_total):
#                     share = lost_total / count
#                     df.loc[mask, 'lost_evisa_share'] = round(share, 2)
        
#         # Load and process tickets
#         st.info("Loading tickets data...")
#         df_tickets = load_tickets_data()
        
#         df['maid_id'] = df['maid_id'].astype(str)
#         df_tickets['maid_id'] = df_tickets['maid_id'].astype(str)
        
#         df_successful = df[df['successful_date'].notnull()].copy()
        
#         ticket_cost = (
#             df_tickets.groupby('maid_id', as_index=False)['Price AED']
#             .sum()
#             .rename(columns={'Price AED': 'total_ticket_cost'})
#         )
        
#         ticket_refund = (
#             df_tickets.groupby('maid_id', as_index=False)['Amount to be Refunded (AED)']
#             .sum()
#             .rename(columns={'Amount to be Refunded (AED)': 'total_ticket_refund'})
#         )
        
#         df_successful = df_successful.merge(ticket_cost, on='maid_id', how='left')
#         df_successful = df_successful.merge(ticket_refund, on='maid_id', how='left')
        
#         df_successful['total_ticket_cost'] = df_successful['total_ticket_cost'].fillna(0)
#         df_successful['total_ticket_refund'] = df_successful['total_ticket_refund'].fillna(0)
        
#         df = df.merge(
#             df_successful[['maid_id', 'total_ticket_cost', 'total_ticket_refund']],
#             on='maid_id',
#             how='left'
#         )
        
#         # Process lost tickets
#         df_tickets['Travel Date'] = pd.to_datetime(df_tickets['Travel Date'], errors='coerce')
#         today = pd.to_datetime(datetime.today().date())
#         tickets_past = df_tickets[df_tickets['Travel Date'] < today].copy()
        
#         hired_maids = df[df['successful_date'].notna()]['maid_id'].astype(str).unique()
#         tickets_past['maid_id'] = tickets_past['maid_id'].astype(str)
#         tickets_lost = tickets_past[~tickets_past['maid_id'].isin(hired_maids)].copy()
        
#         df_maids_lookup = df[['maid_id', 'nationality_category', 'location_category']].drop_duplicates()
#         tickets_lost = tickets_lost.merge(df_maids_lookup, on='maid_id', how='left')
        
#         tickets_lost['ticket_month'] = tickets_lost['Travel Date'].dt.strftime('%b-%y')
#         tickets_lost['Price AED'] = pd.to_numeric(tickets_lost['Price AED'], errors='coerce').fillna(0)
#         tickets_lost['Amount to be Refunded (AED)'] = pd.to_numeric(tickets_lost['Amount to be Refunded (AED)'], errors='coerce').fillna(0)
#         tickets_lost['net_lost_cost'] = tickets_lost['Price AED'] - tickets_lost['Amount to be Refunded (AED)']
        
#         lost_ticket_grouped = tickets_lost.groupby(['ticket_month', 'nationality_category', 'location_category'])['net_lost_cost'].sum().reset_index()
        
#         df['Month'] = df['successful_date'].dt.strftime('%b-%y')
#         hire_counts = df[df['successful_date'].notna()].groupby(['Month', 'nationality_category', 'location_category'])['maid_id'].count().reset_index()
#         hire_counts = hire_counts.rename(columns={'maid_id': 'hire_count'})
        
#         merged_costs = lost_ticket_grouped.merge(hire_counts, left_on=['ticket_month', 'nationality_category', 'location_category'],
#                                                right_on=['Month', 'nationality_category', 'location_category'], how='left')
        
#         merged_costs['lost_ticket_share'] = merged_costs['net_lost_cost'] / merged_costs['hire_count']
#         merged_costs = merged_costs[['Month', 'nationality_category', 'location_category', 'lost_ticket_share']]
        
#         df = df.merge(merged_costs, on=['Month', 'nationality_category', 'location_category'], how='left')
        
#         # Load staff costs
#         st.info("Loading staff costs...")
#         SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
#         bas_cost_df = read_cost_sheet(SHEET_ID, "BAs")
#         daspgs_cost_df = read_cost_sheet(SHEET_ID, "Programmers&DAs")
        
#         def assign_costs_by_month(df_maids, df_costs, target_col_name):
#             df_maids[target_col_name] = 0.0
#             for _, row in df_costs.iterrows():
#                 month = row["Month"]
#                 for nat in ["filipina", "african", "ethiopian"]:
#                     cost_value = row[f"{nat}_share_aed"]
#                     mask = (
#                         (df_maids["successful_month"] == month) &
#                         (df_maids["nationality_category"].str.lower() == nat) &
#                         (df_maids["successful_date"].notnull())
#                     )
#                     count = mask.sum()
#                     if count > 0 and cost_value > 0:
#                         df_maids.loc[mask, target_col_name] = cost_value / count
#             return df_maids
        
#         df = assign_costs_by_month(df, bas_cost_df, "bas_cost_share")
#         df = assign_costs_by_month(df, daspgs_cost_df, "DataAnalysts_and_Programmers_cost_share")
        
#         # Load agents costs
#         agents_cost_summary = read_and_transform_agents_sheet(SHEET_ID)
        
#         df['Agents_cost_share'] = 0.0
#         for _, row in agents_cost_summary.iterrows():
#             month = row['month']
#             for nat in ['filipina', 'ethiopian', 'african']:
#                 cost_col = f"{nat}_cost_aed"
#                 if cost_col not in agents_cost_summary.columns:
#                     continue
#                 total_cost = row[cost_col]
#                 if pd.isna(total_cost) or total_cost == 0:
#                     continue
                
#                 month_parts = month.split('-')
#                 if len(month_parts) != 2:
#                     continue
                
#                 month_name, year = month_parts
#                 year_full = f"20{year}" if len(year) == 2 else year
                
#                 month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
#                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                
#                 if month_name not in month_map:
#                     continue
                
#                 month_num = month_map[month_name]
#                 start_date = f"{year_full}-{month_num:02d}-01"
                
#                 if month_num in [4, 6, 9, 11]:
#                     end_date = f"{year_full}-{month_num:02d}-30"
#                 elif month_num == 2:
#                     is_leap = (int(year_full) % 4 == 0 and int(year_full) % 100 != 0) or (int(year_full) % 400 == 0)
#                     end_date = f"{year_full}-{month_num:02d}-29" if is_leap else f"{year_full}-{month_num:02d}-28"
#                 else:
#                     end_date = f"{year_full}-{month_num:02d}-31"
                
#                 mask = (
#                     (df['successful_date'] >= start_date) &
#                     (df['successful_date'] <= end_date) &
#                     (df['nationality_category'].str.lower() == nat)
#                 )
                
#                 count = mask.sum()
#                 if count == 0:
#                     continue
                
#                 cost_per_maid = total_cost / count
#                 df.loc[mask, 'Agents_cost_share'] = cost_per_maid
        
#         # Load LLM costs
#         st.info("Loading LLM and other costs...")
#         df_LLM = read_llm_costs_sheet_fixed()
        
#         # Add LLM cost share (simplified for space)
#         df['llm_cost_share'] = 0.0
        
#         # Load referrals
#         referrals_df = read_referrals_sheet()
#         df['maid_id'] = df['maid_id'].astype(str).str.strip()
#         referrals_df['Maid A Applicant ID'] = referrals_df['Maid A Applicant ID'].astype(str).str.strip()
#         referred_maid_ids = set(referrals_df['Maid A Applicant ID'])
#         df['referral_cost'] = df['maid_id'].apply(lambda x: 1000 if x in referred_maid_ids else 0)
        
#         # Load broadcasts
#         broadcasts_df = read_broadcasts_sheet()
#         df_hires = df[df['successful_date'].notnull() & df['nationality_category'].notnull()].copy()
#         df_hires['Month'] = pd.to_datetime(df_hires['successful_date'], errors='coerce').dt.strftime('%b-%y')
        
#         broadcast_cost_map = {}
#         for _, row in broadcasts_df.iterrows():
#             month = row['Month']
#             for nat in ['filipina', 'ethiopian', 'african']:
#                 total_cost = row.get(f"{nat}_broadcast", 0)
#                 if total_cost == 0:
#                     continue
#                 hires = df_hires[(df_hires['Month'] == month) & (df_hires['nationality_category'].str.lower() == nat)]
#                 if hires.empty:
#                     continue
#                 cost_per_maid = total_cost / len(hires)
#                 for maid_id in hires['maid_id']:
#                     broadcast_cost_map[maid_id] = cost_per_maid
        
#         df['broadcast_cost'] = df['maid_id'].map(broadcast_cost_map).fillna(0)
        
#         # Calculate marketing costs with date range
#         st.info("Calculating marketing costs...")
#         df = calculate_marketing_cost_per_hire(df, num_months=12, behavior_start_date=behavior_start_date, behavior_end_date=behavior_end_date)
        
#         # Add operator costs
#         def calculate_operator_cost(row):
#             USD_TO_AED = 3.67
#             cost = 0
#             operator = str(row['freedom_operator']).lower() if pd.notna(row['freedom_operator']) else ""
            
#             if 'marilyn' in operator:
#                 return 1966
            
#             if row['nationality_category'] == 'ethiopian':
#                 if 'wa' in operator:
#                     return 900 * USD_TO_AED
#                 elif any(op in operator for op in ['fiseha', 'natnael', 'tadesse']):
#                     return 1000 * USD_TO_AED
#                 elif 'berana' in operator:
#                     return 900 * USD_TO_AED
            
#             return cost
        
#         df['operator_cost'] = df.apply(calculate_operator_cost, axis=1)
        
#         # Initialize attestation cost (placeholder)
#         df['attestation_cost'] = 0
        
#         st.success("Data processing completed successfully!")
#         return df

# # Main Streamlit App
# def main():
#     st.title("🏠 Cost Per Hire Analysis Dashboard")
    
#     # Create tabs
#     tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
#         "Cost per Hire", "Visas", "Tickets", "BAs, DAs, Programmers", 
#         "Agents", "LLMs", "Referrals", "Broadcasts", "Ad Spend", "Operator", "Attestation"
#     ])
    
#     with tab1:
#         st.markdown("<h1 style='text-align: center; font-size: 48px;'>This is to estimate the cost per Hire</h1>", unsafe_allow_html=True)
        
#         # Re-read data button
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             if st.button("🔄 Re-Read Data", type="primary", use_container_width=True):
#                 st.session_state.data_loaded = False
#                 st.session_state.df = None
#                 st.cache_data.clear()
#                 st.rerun()
        
#         st.markdown("---")
        
#         # Date range slider for behavior analysis
#         st.subheader("📊 Behavior Analysis Period for Marketing Costs")
#         st.info("Select the date range to use for calculating time-to-hire behavior patterns. This controls which historical applications are used to learn conversion patterns, while spend attribution still uses all available data.")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             behavior_start_date = st.date_input(
#                 "Start Date for Behavior Analysis",
#                 value=date(2022, 1, 1),
#                 key="behavior_start"
#             )
#         with col2:
#             behavior_end_date = st.date_input(
#                 "End Date for Behavior Analysis",
#                 value=date(2025, 1, 31),
#                 key="behavior_end"
#             )
        
#         # CSV File Upload
#         st.subheader("📤 Upload CSV Data (Optional)")
#         st.info("Upload your Daily Conversion Report CSV file to integrate with GCP data. If not uploaded, the system will use GCP data only.")
#         uploaded_csv = st.file_uploader(
#             "Choose CSV file",
#             type=['csv'],
#             help="Upload your Daily Conversion Report CSV file"
#         )
        
#         # Load and process data
#         if not st.session_state.data_loaded or st.session_state.df is None:
#             st.session_state.df = process_all_data(behavior_start_date, behavior_end_date, uploaded_csv)
#             st.session_state.data_loaded = True
        
#         # Display summary statistics
#         if st.session_state.df is not None:
#             df = st.session_state.df
            
#             st.markdown("---")
#             st.subheader("📈 Data Summary")
            
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Total Records", f"{len(df):,}")
#             with col2:
#                 successful_hires = df['successful_date'].notna().sum()
#                 st.metric("Successful Hires", f"{successful_hires:,}")
#             with col3:
#                 avg_visa_cost = df[df['successful_date'].notna()]['actual_visa_cost'].mean()
#                 st.metric("Avg Visa Cost", f"{avg_visa_cost:.2f} AED")
#             with col4:
#                 total_marketing_cost = df[df['successful_date'].notna()]['marketing_cost_per_hire'].sum()
#                 st.metric("Total Marketing Cost", f"{total_marketing_cost:,.0f} AED")
            
#             # Filter for successful hires only
#             successful_hires_df = df[df['successful_date'].notna()].copy()
            
#             if len(successful_hires_df) > 0:
#                 # Add total cost per hire calculation
#                 cost_columns = [
#                     'actual_visa_cost', 'lost_evisa_share', 'total_ticket_cost', 
#                     'total_ticket_refund', 'lost_ticket_share', 'bas_cost_share',
#                     'DataAnalysts_and_Programmers_cost_share', 'Agents_cost_share',
#                     'llm_cost_share', 'referral_cost', 'broadcast_cost',
#                     'marketing_cost_per_hire', 'operator_cost', 'attestation_cost'
#                 ]
                
#                 # Ensure all cost columns exist and fill NaN with 0
#                 for col in cost_columns:
#                     if col not in successful_hires_df.columns:
#                         successful_hires_df[col] = 0
#                     successful_hires_df[col] = successful_hires_df[col].fillna(0)
                
#                 # Calculate net ticket cost (cost minus refund)
#                 successful_hires_df['net_ticket_cost'] = successful_hires_df['total_ticket_cost'] - successful_hires_df['total_ticket_refund']
                
#                 # Calculate total cost per hire
#                 successful_hires_df['total_cost_per_hire'] = (
#                     successful_hires_df['actual_visa_cost'] +
#                     successful_hires_df['lost_evisa_share'] +
#                     successful_hires_df['net_ticket_cost'] +
#                     successful_hires_df['lost_ticket_share'] +
#                     successful_hires_df['bas_cost_share'] +
#                     successful_hires_df['DataAnalysts_and_Programmers_cost_share'] +
#                     successful_hires_df['Agents_cost_share'] +
#                     successful_hires_df['llm_cost_share'] +
#                     successful_hires_df['referral_cost'] +
#                     successful_hires_df['broadcast_cost'] +
#                     successful_hires_df['marketing_cost_per_hire'] +
#                     successful_hires_df['operator_cost'] +
#                     successful_hires_df['attestation_cost']
#                 )
                
#                 st.markdown("---")
#                 st.subheader("💰 Complete Cost per Hire Analysis")
                
#                 # Display download section
#                 col1, col2 = st.columns([2, 1])
#                 with col1:
#                     st.info("💾 **Download Complete Cost Analysis** - Export all successful hires with detailed cost breakdown")
#                 with col2:
#                     # Prepare data for download
#                     download_df = successful_hires_df.copy()
#                     # Round numerical columns to 2 decimal places
#                     numeric_columns = download_df.select_dtypes(include=['float64', 'int64']).columns
#                     download_df[numeric_columns] = download_df[numeric_columns].round(2)
                    
#                     csv_data = download_df.to_csv(index=False)
#                     st.download_button(
#                         label="📥 Download CSV",
#                         data=csv_data,
#                         file_name=f"cost_per_hire_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                         mime="text/csv",
#                         type="primary"
#                     )
                
#                 # Display summary of cost components
#                 st.subheader("📊 Cost Components Summary")
#                 cost_summary = pd.DataFrame({
#                     'Cost Component': [
#                         'Visa Costs', 'Lost E-Visa Share', 'Net Ticket Costs', 'Lost Ticket Share',
#                         'BA Costs', 'DA & Programmer Costs', 'Agent Costs', 'LLM Costs',
#                         'Referral Costs', 'Broadcast Costs', 'Marketing Costs', 'Operator Costs', 'Attestation Costs'
#                     ],
#                     'Total Amount (AED)': [
#                         successful_hires_df['actual_visa_cost'].sum(),
#                         successful_hires_df['lost_evisa_share'].sum(),
#                         successful_hires_df['net_ticket_cost'].sum(),
#                         successful_hires_df['lost_ticket_share'].sum(),
#                         successful_hires_df['bas_cost_share'].sum(),
#                         successful_hires_df['DataAnalysts_and_Programmers_cost_share'].sum(),
#                         successful_hires_df['Agents_cost_share'].sum(),
#                         successful_hires_df['llm_cost_share'].sum(),
#                         successful_hires_df['referral_cost'].sum(),
#                         successful_hires_df['broadcast_cost'].sum(),
#                         successful_hires_df['marketing_cost_per_hire'].sum(),
#                         successful_hires_df['operator_cost'].sum(),
#                         successful_hires_df['attestation_cost'].sum()
#                     ],
#                     'Average per Hire (AED)': [
#                         successful_hires_df['actual_visa_cost'].mean(),
#                         successful_hires_df['lost_evisa_share'].mean(),
#                         successful_hires_df['net_ticket_cost'].mean(),
#                         successful_hires_df['lost_ticket_share'].mean(),
#                         successful_hires_df['bas_cost_share'].mean(),
#                         successful_hires_df['DataAnalysts_and_Programmers_cost_share'].mean(),
#                         successful_hires_df['Agents_cost_share'].mean(),
#                         successful_hires_df['llm_cost_share'].mean(),
#                         successful_hires_df['referral_cost'].mean(),
#                         successful_hires_df['broadcast_cost'].mean(),
#                         successful_hires_df['marketing_cost_per_hire'].mean(),
#                         successful_hires_df['operator_cost'].mean(),
#                         successful_hires_df['attestation_cost'].mean()
#                     ]
#                 })
#                 cost_summary = cost_summary.round(2)
#                 st.dataframe(cost_summary, use_container_width=True)
                
#                 # Display total cost summary
#                 total_cost = successful_hires_df['total_cost_per_hire'].sum()
#                 avg_cost = successful_hires_df['total_cost_per_hire'].mean()
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("💰 Total Cost (All Hires)", f"{total_cost:,.2f} AED")
#                 with col2:
#                     st.metric("📊 Average Cost per Hire", f"{avg_cost:,.2f} AED")
#                 with col3:
#                     st.metric("🎯 Total Successful Hires", f"{len(successful_hires_df):,}")
                
#                 # Display detailed view with cost breakdown
#                 st.subheader("📋 Detailed Cost per Hire View")
#                 st.info("👁️ **Preview of downloadable data** - Showing first 50 records with all cost components")
                
#                 # Select key columns for display
#                 display_columns = [
#                     'maid_id', 'applicant_name', 'successful_date', 'nationality_category', 
#                     'location_category', 'actual_visa_cost', 'net_ticket_cost', 
#                     'marketing_cost_per_hire', 'operator_cost', 'referral_cost', 
#                     'total_cost_per_hire'
#                 ]
                
#                 # Filter columns that exist
#                 available_display_columns = [col for col in display_columns if col in successful_hires_df.columns]
                
#                 # Sort by successful_date descending and show first 50 records
#                 preview_df = successful_hires_df[available_display_columns].sort_values('successful_date', ascending=False).head(50)
#                 st.dataframe(preview_df, use_container_width=True, height=400)
                
#                 # Show column information
#                 st.subheader("📝 Dataset Information")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.write(f"**Total Columns in Dataset:** {len(successful_hires_df.columns)}")
#                     st.write(f"**Records Available for Download:** {len(successful_hires_df)}")
#                 with col2:
#                     st.write("**Key Cost Components:**")
#                     key_costs = ['actual_visa_cost', 'marketing_cost_per_hire', 'operator_cost', 'referral_cost', 'total_cost_per_hire']
#                     for cost in key_costs:
#                         if cost in successful_hires_df.columns:
#                             st.write(f"• {cost.replace('_', ' ').title()}")
#             else:
#                 st.warning("No successful hires found in the dataset.")
    
#     with tab2:
#         st.header("🛂 Visa Data")
        
#         try:
#             st.subheader("T-Visa Data")
#             df_t_visa = load_t_visa()
#             st.dataframe(df_t_visa.head(), use_container_width=True)
#             st.info(f"Total T-Visa records: {len(df_t_visa)}")
            
#             st.subheader("Fixed Visa Costs")
#             fixed_cost = load_fixed_cost()
#             fixed_cost_df = pd.DataFrame([fixed_cost])
#             st.dataframe(fixed_cost_df, use_container_width=True)
            
#             st.subheader("Lost E-Visas Data")
#             lost_visas_df = read_lost_visas_sheet()
#             st.dataframe(lost_visas_df.head(), use_container_width=True)
#             st.info(f"Total Lost E-Visa records: {len(lost_visas_df)}")
            
#         except Exception as e:
#             st.error(f"Error loading visa data: {e}")
    
#     with tab3:
#         st.header("✈️ Tickets Data")
        
#         try:
#             df_tickets = load_tickets_data()
#             st.dataframe(df_tickets.head(), use_container_width=True)
#             st.info(f"Total ticket records: {len(df_tickets)}")
            
#             # Summary statistics
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 avg_price = df_tickets['Price AED'].mean()
#                 st.metric("Average Ticket Price", f"{avg_price:.2f} AED")
#             with col2:
#                 total_refunds = df_tickets['Amount to be Refunded (AED)'].sum()
#                 st.metric("Total Refunds", f"{total_refunds:,.2f} AED")
#             with col3:
#                 ticket_types = df_tickets['Type'].nunique()
#                 st.metric("Ticket Types", ticket_types)
                
#         except Exception as e:
#             st.error(f"Error loading tickets data: {e}")
    
#     with tab4:
#         st.header("👥 BAs, DAs, and Programmers Data")
        
#         try:
#             SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
            
#             st.subheader("Business Analysts Costs")
#             bas_cost_df = read_cost_sheet(SHEET_ID, "BAs")
#             st.dataframe(bas_cost_df.head(), use_container_width=True)
            
#             st.subheader("Programmers & Data Analysts Costs")
#             daspgs_cost_df = read_cost_sheet(SHEET_ID, "Programmers&DAs")
#             st.dataframe(daspgs_cost_df.head(), use_container_width=True)
            
#         except Exception as e:
#             st.error(f"Error loading staff cost data: {e}")
    
#     with tab5:
#         st.header("🤝 Agents Data")
        
#         try:
#             SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
#             agents_cost_summary = read_and_transform_agents_sheet(SHEET_ID)
#             st.dataframe(agents_cost_summary.head(), use_container_width=True)
#             st.info(f"Total agent cost records: {len(agents_cost_summary)}")
            
#         except Exception as e:
#             st.error(f"Error loading agents data: {e}")
    
#     with tab6:
#         st.header("🤖 LLM Costs Data")
        
#         try:
#             df_LLM = read_llm_costs_sheet_fixed()
#             st.dataframe(df_LLM.head(), use_container_width=True)
#             st.info(f"Total LLM cost records: {len(df_LLM)}")
            
#             # Summary
#             total_llm_cost = df_LLM['Cost'].sum()
#             st.metric("Total LLM Costs", f"{total_llm_cost:,.2f} AED")
            
#         except Exception as e:
#             st.error(f"Error loading LLM data: {e}")
    
#     with tab7:
#         st.header("🔗 Referrals Data")
        
#         try:
#             referrals_df = read_referrals_sheet()
#             st.dataframe(referrals_df.head(), use_container_width=True)
#             st.info(f"Total referral records: {len(referrals_df)}")
            
#         except Exception as e:
#             st.error(f"Error loading referrals data: {e}")
    
#     with tab8:
#         st.header("📢 Broadcasts Data")
        
#         try:
#             broadcasts_df = read_broadcasts_sheet()
#             st.dataframe(broadcasts_df.head(), use_container_width=True)
#             st.info(f"Total broadcast records: {len(broadcasts_df)}")
            
#         except Exception as e:
#             st.error(f"Error loading broadcasts data: {e}")
    
#     with tab9:
#         st.header("💰 Ad Spend Data")
#         st.info("Marketing cost calculation uses weighted attribution based on time-to-hire patterns")
#         st.markdown("""
#         **How it works:**
#         1. **Behavior Analysis**: Uses historical data within your selected date range to learn conversion patterns
#         2. **Spend Attribution**: Looks back at ALL available spend data when calculating costs
#         3. **Weighted Attribution**: Each month's spend is weighted based on historical conversion probability
#         """)
        
#         if st.session_state.df is not None:
#             df = st.session_state.df
#             marketing_summary = df[df['successful_date'].notna()].groupby(['nationality_category', 'location_category'])['marketing_cost_per_hire'].agg(['count', 'mean', 'sum']).round(2)
#             marketing_summary.columns = ['Hire Count', 'Avg Cost per Hire', 'Total Cost']
#             st.dataframe(marketing_summary, use_container_width=True)
    
#     with tab10:
#         st.header("⚙️ Operator Costs Data")
#         st.info("Operator costs are calculated based on freedom operator assignments and nationality")
        
#         if st.session_state.df is not None:
#             df = st.session_state.df
#             operator_summary = df[df['operator_cost'] > 0].groupby(['nationality_category', 'freedom_operator'])['operator_cost'].agg(['count', 'mean']).round(2)
#             operator_summary.columns = ['Count', 'Cost per Hire']
#             st.dataframe(operator_summary, use_container_width=True)
    
#     with tab11:
#         st.header("📋 Attestation Data")
#         st.info("Attestation costs would be loaded from a separate sheet (placeholder)")
#         st.warning("Attestation sheet ID needs to be configured")
        
#         if st.session_state.df is not None:
#             df = st.session_state.df
#             attestation_count = (df['attestation_cost'] > 0).sum()
#             st.metric("Records with Attestation Costs", attestation_count)

# if __name__ == "__main__":
#     main()



































# import os
# import streamlit as st
# import pandas as pd
# from google.cloud import bigquery
# from datetime import date, datetime
# import plotly.express as px
# import gspread
# from google.oauth2.service_account import Credentials
# import re
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# import json
# import tempfile

# # Set page config
# st.set_page_config(page_title="Cost Per Hire Analysis", layout="wide")

# # Initialize session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'df' not in st.session_state:
#     st.session_state.df = None

# # Setup Google Cloud credentials
# def setup_credentials():
#     """Setup Google Cloud credentials from Streamlit secrets or uploaded file"""
#     try:
#         # Try to use Streamlit secrets first (for deployment)
#         if "gcp_service_account" in st.secrets:
#             credentials_info = dict(st.secrets["gcp_service_account"])
#             credentials = service_account.Credentials.from_service_account_info(credentials_info)
            
#             # Set up environment for BigQuery
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
#                 json.dump(credentials_info, f)
#                 os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
                
#             return credentials
#         else:
#             st.error("Google Cloud credentials not found in secrets. Please configure them in Streamlit Cloud.")
#             return None
#     except Exception as e:
#         st.error(f"Error setting up credentials: {e}")
#         return None

# # Load Main Data from GCP
# @st.cache_data
# def load_gcp_data():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()  # Return empty dataframe if no credentials
        
#     client = bigquery.Client(credentials=credentials)
#     query = """
#         SELECT 
#             User_ID,
#             `Application Created` AS application_date,
#             `Successful_Date` AS successful_date,
#             `Location Category Updated` AS location_category,
#             `Nationality Category Updated` AS nationality_category,
#             `Nationality Updated` AS nationality,
#             `Country Updated` AS country
#         FROM `data-driven-attributes.AT_marketing_db.ATD_New_Last_Action_by_User_PivotData_View`
#         WHERE `Application Created` IS NOT NULL
#     """
#     df = client.query(query).to_dataframe()
#     df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
#     df['successful_date'] = pd.to_datetime(df['successful_date'], errors='coerce')
#     return df.dropna(subset=['application_date'])

# # Load CSV Data from uploaded file
# def load_csv_data(uploaded_file):
#     if uploaded_file is None:
#         return None
        
#     encodings = ['utf-16', 'utf-8', 'latin1', 'ISO-8859-1', 'cp1252']   
    
#     for encoding in encodings:
#         try:
#             # Reset file pointer
#             uploaded_file.seek(0)
#             df = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
#             st.success(f"Successfully read file with encoding: {encoding}")
#             break
#         except UnicodeDecodeError:
#             if encoding == encodings[-1]:
#                 st.error("Could not read file with any of the attempted encodings")
#                 return None
#         except Exception as e:
#             st.error(f"Error reading file: {e}")
#             return None
    
#     # Clean column names
#     df.columns = df.columns.str.strip()
    
#     # Rename columns to match the original BigQuery names
#     column_mapping = {
#         'Application ID': 'maid_id',
#         'Application Creation Date': 'application_date',
#         'Landed': 'successful_date',
#         'Location Category': 'location_category',
#         'Category': 'nationality_category',
#         'Nationality': 'nationality',
#         'Country': 'country',
#         'Freedom Operator': 'freedom_operator',
#         'Exit Loan': 'exit_loan',
#         'Applicant Name': 'applicant_name'
#     }
    
#     # Only rename columns that actually exist in the file
#     column_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
#     df = df.rename(columns=column_mapping)
    
#     # Convert values to lowercase
#     for col in ['location_category', 'nationality_category', 'nationality', 'country']:
#         if col in df.columns:
#             df[col] = df[col].str.lower() if pd.api.types.is_object_dtype(df[col]) else df[col]
    
#     # Convert dates
#     for col in ['application_date', 'successful_date']:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
    
#     # Convert Exit Loan to numeric
#     if 'exit_loan' in df.columns:
#         df['exit_loan'] = pd.to_numeric(df['exit_loan'], errors='coerce')
    
#     # Process the nationality_category column
#     if 'nationality_category' in df.columns:
#         df['original_nationality_category'] = df['nationality_category'].copy()
        
#         if df['nationality_category'].str.contains(' ').any():
#             df['type'] = df['nationality_category'].str.split(' ', n=1).str[1]
#             df['nationality_category'] = df['nationality_category'].str.split(' ', n=1).str[0]
        
#         df['nationality_category'] = df['nationality_category'].str.rstrip('s')
        
#         if 'nationality' in df.columns:
#             mask = (df['nationality_category'] == 'african') & (df['nationality'] == 'ethiopian')
#             df.loc[mask, 'nationality_category'] = 'ethiopian'
    
#     if 'maid_id' in df.columns:
#         df["maid_id"] = df["maid_id"].astype(str)
    
#     return df

# # Load T-Visa data
# @st.cache_data(show_spinner=False)
# def load_t_visa():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     client = bigquery.Client(credentials=credentials)
#     tvisa_query = """
#         SELECT DISTINCT CAST(maid_id AS STRING) AS maid_id
#         FROM `data-driven-attributes.AT_marketing_db.maid_tvisa_tracker`
#     """
#     return client.query(tvisa_query).to_dataframe()

# # Load fixed cost data
# @st.cache_data(show_spinner=False)
# def load_fixed_cost():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.Series()  # Return empty series if no credentials
        
#     client = bigquery.Client(credentials=credentials)
#     fixed_query = """
#         SELECT *
#         FROM `data-driven-attributes.AT_marketing_db.maid_visa_fixed_cost`
#     """
#     result = client.query(fixed_query).to_dataframe()
#     return result.iloc[0] if not result.empty else pd.Series()

# # Load lost visas data
# def read_lost_visas_sheet():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     SHEET_ID = "1q0427FhcmmnIpXYrGKvmAszf56Jngwvk3QUTQ_hCztQ"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError("No data found in the sheet.")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     normalized_rows = []
#     for row in data_rows:
#         row += [None] * (len(headers) - len(row))
#         normalized_rows.append(row)
    
#     df = pd.DataFrame(normalized_rows, columns=headers)
    
#     for col in df.columns[1:]:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     return df

# # Load tickets data
# def load_tickets_data():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build('sheets', 'v4', credentials=credentials)
#     SPREADSHEET_ID = "1dDKXQKTE-yp4znJI3x-CBDwYgj9EfUDAvyBh6sneUnU"
#     RANGE_NAME = "Sheet1"
    
#     sheet = service.spreadsheets()
#     result = sheet.get(
#         spreadsheetId=SPREADSHEET_ID,
#         ranges=[RANGE_NAME],
#         includeGridData=True
#     ).execute()
    
#     rows = result['sheets'][0]['data'][0]['rowData']
    
#     headers = [cell.get('formattedValue', '') for cell in rows[0]['values']]
    
#     # Debug: Print headers to see what columns are available
#     st.info(f"Available columns in tickets sheet: {headers}")
    
#     try:
#         applicant_name_index = headers.index("Applicant Name")
#     except ValueError:
#         # Try alternative column names
#         possible_names = ["Name", "Maid Name", "Applicant", "Worker Name"]
#         applicant_name_index = None
#         for name in possible_names:
#             if name in headers:
#                 applicant_name_index = headers.index(name)
#                 break
        
#         if applicant_name_index is None:
#             st.warning("Could not find applicant name column. Using first column as fallback.")
#             applicant_name_index = 0
    
#     data = []
#     for row in rows:
#         row_data = []
#         maid_id = None
#         values = row.get('values', [])
        
#         for cell in values:
#             row_data.append(cell.get('formattedValue', ''))
        
#         if applicant_name_index < len(values):
#             link = values[applicant_name_index].get('hyperlink', '')
#             match = re.findall(r'\d{4,}', link)
#             maid_id = max(match, key=len) if match else None
        
#         if any(row_data):
#             row_data.append(maid_id)
#             data.append(row_data)
    
#     headers.append("maid_id")
#     df_tickets = pd.DataFrame(data[1:], columns=headers)
    
#     # Handle different possible column names for price
#     price_columns = ['Price AED', 'Price', 'Cost AED', 'Cost', 'Amount AED', 'Amount']
#     price_col = None
#     for col in price_columns:
#         if col in df_tickets.columns:
#             price_col = col
#             break
    
#     if price_col:
#         df_tickets['Price AED'] = pd.to_numeric(df_tickets[price_col].astype(str).str.replace(',', ''), errors='coerce')
#     else:
#         st.warning("Could not find price column. Setting Price AED to 0.")
#         df_tickets['Price AED'] = 0
    
#     # Handle different possible column names for refunds
#     refund_columns = ['Amount to be Refunded (AED)', 'Refund AED', 'Refund Amount', 'Refund', 'Amount Refunded']
#     refund_col = None
#     for col in refund_columns:
#         if col in df_tickets.columns:
#             refund_col = col
#             break
    
#     if refund_col:
#         df_tickets['Amount to be Refunded (AED)'] = pd.to_numeric(
#             df_tickets[refund_col].astype(str).str.replace(',', ''), errors='coerce'
#         )
#     else:
#         st.warning("Could not find refund column. Setting Amount to be Refunded to 0.")
#         df_tickets['Amount to be Refunded (AED)'] = 0
    
#     # Handle Type column
#     type_columns = ['Type', 'Ticket Type', 'Category', 'Status']
#     type_col = None
#     for col in type_columns:
#         if col in df_tickets.columns:
#             type_col = col
#             break
    
#     if type_col and type_col != 'Type':
#         df_tickets['Type'] = df_tickets[type_col]
#     elif 'Type' not in df_tickets.columns:
#         st.warning("Could not find type column. Including all records.")
#         df_tickets['Type'] = 'Unknown'
    
#     # Filter by type if Type column exists and has valid values
#     if 'Type' in df_tickets.columns:
#         valid_types = df_tickets['Type'].dropna().unique()
#         st.info(f"Available ticket types: {valid_types}")
        
#         # Filter for known good types, but be flexible
#         filter_types = ['Real', 'Dummy', 'FO Marilyn']
#         available_filter_types = [t for t in filter_types if t in valid_types]
        
#         if available_filter_types:
#             df_tickets = df_tickets[df_tickets['Type'].isin(available_filter_types)]
#         else:
#             st.warning("No matching ticket types found. Including all records.")
    
#     return df_tickets

# # Load BAs cost data
# def read_cost_sheet(sheet_id, sheet_name):
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     USD_TO_AED = 3.67
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=sheet_id,
#         range=sheet_name,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError(f"No data found in sheet: {sheet_name}")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     df = pd.DataFrame(data_rows, columns=headers)
    
#     df.columns = [col.strip().lower() for col in df.columns]
#     df = df.rename(columns={
#         "month": "Month",
#         "filipina share usd": "filipina_share_usd",
#         "african share usd": "african_share_usd",
#         "ethiopian share usd": "ethiopian_share_usd"
#     })
    
#     for col in ["filipina_share_usd", "african_share_usd", "ethiopian_share_usd"]:
#         df[col] = df[col].replace(",", "", regex=True).astype(float)
    
#     df["filipina_share_aed"] = df["filipina_share_usd"] * USD_TO_AED
#     df["african_share_aed"] = df["african_share_usd"] * USD_TO_AED
#     df["ethiopian_share_aed"] = df["ethiopian_share_usd"] * USD_TO_AED
    
#     df["Month"] = pd.to_datetime(df["Month"], format="%B %Y").dt.strftime("%b-%y")
    
#     return df[["Month", "filipina_share_aed", "african_share_aed", "ethiopian_share_aed"]]

# # Load agents data
# def read_and_transform_agents_sheet(sheet_id, sheet_name="Agents"):
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     USD_TO_AED = 3.67
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=sheet_id,
#         range=sheet_name,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError(f"No data found in sheet: {sheet_name}")
    
#     headers = [h.strip() for h in rows[0]]
#     data_rows = rows[1:]
    
#     normalized_rows = []
#     for row in data_rows:
#         padded_row = row + [None] * (len(headers) - len(row))
#         normalized_rows.append(padded_row[:len(headers)])
    
#     df = pd.DataFrame(normalized_rows, columns=headers)
    
#     df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
#     df.rename(columns={"salary_usd": "salary_usd", "nationality_category": "nationality_category"}, inplace=True)
    
#     df["nationality_category"] = df["nationality_category"].replace('', None).ffill()
    
#     df["month"] = pd.to_datetime(df["month"], format="%B %Y", errors="coerce").dt.strftime("%b-%y")
    
#     df = df.dropna(subset=["salary_usd", "month"])
    
#     df["salary_usd"] = df["salary_usd"].replace(",", "", regex=True).astype(float)
#     df["salary_aed"] = df["salary_usd"] * USD_TO_AED
    
#     exploded_rows = []
#     for _, row in df.iterrows():
#         nat = str(row["nationality_category"]).strip().lower()
        
#         if "all" in nat:
#             categories = ["filipina", "ethiopian", "african"]
#         elif "+" in nat:
#             categories = [x.strip().lower() for x in nat.split("+")]
#         else:
#             categories = [nat]
        
#         share = row["salary_aed"] / len(categories)
#         for cat in categories:
#             exploded_rows.append({
#                 "month": row["month"],
#                 "nationality_category": cat,
#                 "salary_aed": round(share, 2)
#             })
    
#     exploded_df = pd.DataFrame(exploded_rows)
    
#     pivot_df = exploded_df.pivot_table(
#         index="month",
#         columns="nationality_category",
#         values="salary_aed",
#         aggfunc="sum",
#         fill_value=0
#     ).reset_index()
    
#     pivot_df = pivot_df.rename(columns={
#         "filipina": "filipina_cost_aed",
#         "ethiopian": "ethiopian_cost_aed",
#         "african": "african_cost_aed"
#     })
    
#     for col in ["filipina_cost_aed", "ethiopian_cost_aed", "african_cost_aed"]:
#         if col not in pivot_df.columns:
#             pivot_df[col] = 0.0
    
#     return pivot_df

# # Load LLM costs
# def read_llm_costs_sheet_fixed():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     SHEET_ID = "192G2EAL_D7lKEGAaJ-6BkRacpU0UgS2_akZxqTT5lWo"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError("No data found in the sheet.")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     df = pd.DataFrame(data_rows, columns=headers)
    
#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     df["Maids At"] = df["Maids At"].str.replace("$", "", regex=False).astype(float)
    
#     df = df.dropna(subset=["Date", "Maids At"])
    
#     df["Cost AED"] = df["Maids At"] * 3.67
#     df["Month"] = df["Date"].dt.strftime('%b-%y')
    
#     df_LLM = df.groupby("Month", as_index=False)["Cost AED"].sum()
#     df_LLM = df_LLM.rename(columns={"Cost AED": "Cost"})
    
#     return df_LLM

# # Load referrals data
# def read_referrals_sheet():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     REFERRALS_SHEET_ID = "1FlETNT-_hPcGgN_hzMNJ7UNUMwvZntu0vxNT05NORk8"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=REFERRALS_SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     num_cols = len(headers)
#     normalized_rows = []
#     for row in data_rows:
#         row = row + [None] * (num_cols - len(row)) if len(row) < num_cols else row[:num_cols]
#         normalized_rows.append(row)
    
#     df = pd.DataFrame(normalized_rows, columns=headers)
    
#     if "Referred Maid Applicant ID" not in df.columns:
#         raise ValueError("Column 'Referred Maid Applicant ID' not found in the sheet.")
#     df = df.rename(columns={"Referred Maid Applicant ID": "maid_id"})
#     df["maid_id"] = df["maid_id"].astype(str)
    
#     if "CC/MV" not in df.columns:
#         raise ValueError("Column 'CC/MV' not found in the sheet.")
#     df = df[df["CC/MV"].str.strip().str.upper() == "CC"]
    
#     return df

# # Load broadcasts data
# def read_broadcasts_sheet():
#     credentials = setup_credentials()
#     if not credentials:
#         return pd.DataFrame()
        
#     service = build("sheets", "v4", credentials=credentials)
    
#     SHEET_ID = "1jw2te4TeXRx0GHAubr9ZyXtJF2gNEacJITpmwxUNFS0"
#     SHEET_NAME = "Sheet1"
    
#     result = service.spreadsheets().values().get(
#         spreadsheetId=SHEET_ID,
#         range=SHEET_NAME,
#         majorDimension='ROWS'
#     ).execute()
    
#     rows = result.get("values", [])
#     if not rows:
#         raise ValueError("No data found in the sheet.")
    
#     headers = rows[0]
#     data_rows = rows[1:]
    
#     broadcasts_df = pd.DataFrame(data_rows, columns=headers)
    
#     broadcasts_df.columns = broadcasts_df.columns.str.strip().str.lower()
#     broadcasts_df = broadcasts_df.rename(columns={
#         "month": "Month",
#         "filipina": "filipina_broadcast",
#         "ethiopian": "ethiopian_broadcast",
#         "african": "african_broadcast"
#     })
    
#     broadcasts_df["Month"] = pd.to_datetime(broadcasts_df["Month"], format="%B %y", errors='coerce').dt.strftime("%b-%y")
    
#     for col in ["filipina_broadcast", "ethiopian_broadcast", "african_broadcast"]:
#         broadcasts_df[col] = pd.to_numeric(broadcasts_df[col], errors="coerce").fillna(0)
    
#     return broadcasts_df

# # Modified marketing cost calculation with date range parameters
# def calculate_marketing_cost_per_hire(df, num_months=12, behavior_start_date=None, behavior_end_date=None):
#     """
#     Calculate marketing cost per hire using the weighted approach with date range control for behavior analysis.
#     """
#     credentials = setup_credentials()
#     if not credentials:
#         df['marketing_cost_per_hire'] = 0.0
#         return df
        
#     # Load Spend Data
#     client = bigquery.Client(credentials=credentials)
#     query = """
#         SELECT
#             DATE_TRUNC(application_created_date, MONTH) AS spend_month,
#             nationality_category,
#             location_category,
#             SUM(total_spend_aed) AS monthly_spend
#         FROM `data-driven-attributes.AT_marketing_db.AT_Country_Daily_Performance_Spend_ERP_Updated`
#         GROUP BY spend_month, nationality_category, location_category
#     """
#     spend_df = client.query(query).to_dataframe()
#     spend_df['spend_month'] = pd.to_datetime(spend_df['spend_month']).dt.to_period('M').dt.to_timestamp()
    
#     # Modified time-to-hire distribution function with date filtering
#     def compute_time_to_hire_distribution(df_filtered, start_date=None, end_date=None):
#         df_filtered = df_filtered[df_filtered['successful_date'].notna()].copy()
        
#         # Apply date range filter for behavior analysis if provided
#         if start_date is not None and end_date is not None:
#             df_filtered = df_filtered[
#                 (df_filtered['application_date'] >= pd.to_datetime(start_date)) & 
#                 (df_filtered['application_date'] <= pd.to_datetime(end_date))
#             ]
        
#         df_filtered['application_month'] = df_filtered['application_date'].dt.to_period("M").dt.to_timestamp()
#         df_filtered['hire_month'] = df_filtered['successful_date'].dt.to_period("M").dt.to_timestamp()
#         df_filtered['month_name'] = df_filtered['application_date'].dt.strftime("%b")
        
#         month_wise_brackets = {}
#         for month_name, month_data in df_filtered.groupby('month_name'):
#             if month_data.empty:
#                 month_wise_brackets[month_name] = [0.0] * num_months
#                 continue
                
#             total_hires = len(month_data)
#             bracket_counts = [0] * num_months
            
#             for cohort_month, group in month_data.groupby('application_month'):
#                 for offset in range(num_months):
#                     start = cohort_month + pd.DateOffset(months=offset)
#                     end = cohort_month + pd.DateOffset(months=offset + 1)
#                     count = group[(group['hire_month'] >= start) & (group['hire_month'] < end)].shape[0]
#                     bracket_counts[offset] += count
                    
#             month_wise_brackets[month_name] = [(c / total_hires if total_hires > 0 else 0) for c in bracket_counts]
        
#         return month_wise_brackets
    
#     # Calculate CAC for each combination of nationality and location category
#     cac_lookup = {}
    
#     nationality_location_pairs = df[df['successful_date'].notna()][['nationality_category', 'location_category']].drop_duplicates()
    
#     for _, row in nationality_location_pairs.iterrows():
#         nationality = row['nationality_category']
#         location = row['location_category']
        
#         if pd.isna(nationality) or pd.isna(location):
#             continue
        
#         filtered_df = df[
#             (df['nationality_category'] == nationality) &
#             (df['location_category'] == location)
#         ].copy()
        
#         # Pass date range to behavior analysis
#         month_wise_brackets = compute_time_to_hire_distribution(
#             filtered_df, 
#             start_date=behavior_start_date, 
#             end_date=behavior_end_date
#         )
        
#         filtered_spend = spend_df[
#             (spend_df['nationality_category'] == nationality) &
#             (spend_df['location_category'] == location)
#         ].copy()
        
#         monthly_spend = filtered_spend.groupby('spend_month')['monthly_spend'].sum().reset_index()
        
#         hire_data = filtered_df[filtered_df['successful_date'].notna()].copy()
        
#         hire_data['hire_month'] = hire_data['successful_date'].dt.to_period("M").dt.to_timestamp()
#         monthly_hires = hire_data.groupby('hire_month').size().reset_index(name='hires')
        
#         for _, hire_row in monthly_hires.iterrows():
#             hire_month = hire_row['hire_month']
#             hires = hire_row['hires']
            
#             if hires == 0:
#                 continue
                
#             weighted_spend = 0
#             for i in range(num_months):
#                 spend_month = (hire_month - pd.DateOffset(months=i)).to_period('M').to_timestamp()
#                 spend = monthly_spend[monthly_spend['spend_month'] == spend_month]['monthly_spend'].sum()
                
#                 month_name = spend_month.strftime("%b")
#                 weight = month_wise_brackets.get(month_name, [0]*num_months)[i] if month_name in month_wise_brackets else 0
                
#                 weighted_spend += spend * weight
            
#             cac = weighted_spend / hires if hires > 0 else 0
            
#             month_key = hire_month.strftime('%Y-%m')
#             if (nationality, location) not in cac_lookup:
#                 cac_lookup[(nationality, location)] = {}
            
#             cac_lookup[(nationality, location)][month_key] = cac
    
#     # Add CAC column to main dataframe
#     df['marketing_cost_per_hire'] = 0.0
    
#     for idx, row in df[df['successful_date'].notna()].iterrows():
#         nat = row['nationality_category']
#         loc = row['location_category']
#         month_key = pd.to_datetime(row['successful_date']).strftime('%Y-%m')
        
#         if pd.isna(nat) or pd.isna(loc) or (nat, loc) not in cac_lookup or month_key not in cac_lookup.get((nat, loc), {}):
#             continue
            
#         df.at[idx, 'marketing_cost_per_hire'] = cac_lookup[(nat, loc)][month_key]
    
#     return df

# # Main data processing function
# def process_all_data(behavior_start_date=None, behavior_end_date=None, uploaded_csv=None):
#     """Process all data and return the final dataframe with all cost components"""
    
#     with st.spinner("Loading and processing data..."):
#         # Load main data from GCP
#         st.info("Loading data from GCP...")
#         df_gcp = load_gcp_data()
        
#         if df_gcp.empty:
#             st.error("Could not load data from GCP. Please check your credentials.")
#             return pd.DataFrame()
            
#         df_gcp = df_gcp.rename(columns={"User_ID": "maid_id"})
#         df_gcp["maid_id"] = df_gcp["maid_id"].astype(str)
        
#         # Apply initial nationality_category transformation
#         df_gcp['nationality_category'] = df_gcp.apply(
#             lambda row: 'ethiopian' if row['nationality_category'] == 'african' and row['nationality'] == 'ethiopian' else row['nationality_category'],
#             axis=1)
        
#         # Load CSV data if uploaded
#         if uploaded_csv is not None:
#             try:
#                 st.info("Loading data from uploaded CSV...")
#                 df_csv = load_csv_data(uploaded_csv)
            
#             if df_csv is not None:
#                 # Process CSV integration logic here (shortened for space)
#                 csv_maid_ids = set(df_csv['maid_id'])
                
#                 # Remove Ethiopian records from GCP data
#                 ethiopian_count = sum(df_gcp['nationality_category'] == 'ethiopian')
#                 df_gcp = df_gcp[df_gcp['nationality_category'] != 'ethiopian']
                
#                 # Set all successful_date values to null in GCP data
#                 df_gcp['successful_date'] = pd.NaT
                
#                 # Add missing columns
#                 for col in ['applicant_name', 'type', 'exit_loan', 'freedom_operator']:
#                     if col not in df_gcp.columns:
#                         df_gcp[col] = None if col in ['applicant_name', 'type'] else (0 if col == 'exit_loan' else '')
                
#                 # Update GCP data with CSV data
#                 update_dict = {}
#                 for idx, row in df_csv.iterrows():
#                     maid_id = row['maid_id']
#                     update_values = {}
                    
#                     for field in ['nationality_category', 'nationality', 'location_category', 'successful_date', 'applicant_name', 'type']:
#                         if field in df_csv.columns and pd.notna(row[field]):
#                             update_values[field] = row[field]
                    
#                     update_values['exit_loan'] = row['exit_loan'] if 'exit_loan' in df_csv.columns and pd.notna(row['exit_loan']) else 0
#                     update_values['freedom_operator'] = row['freedom_operator'] if 'freedom_operator' in df_csv.columns and pd.notna(row['freedom_operator']) else ''
                    
#                     update_dict[maid_id] = update_values
                
#                 # Apply updates
#                 for idx, row in df_gcp.iterrows():
#                     maid_id = row['maid_id']
#                     if maid_id in update_dict:
#                         for field, value in update_dict[maid_id].items():
#                             df_gcp.at[idx, field] = value
                
#                 # Add CSV-only records
#                 gcp_maid_ids = set(df_gcp['maid_id'])
#                 csv_only_ids = csv_maid_ids - gcp_maid_ids
#                 csv_only_records = df_csv[df_csv['maid_id'].isin(csv_only_ids)].copy()
                
#                 new_records = pd.DataFrame(columns=df_gcp.columns)
#                 for idx, row in csv_only_records.iterrows():
#                     new_row = pd.Series(index=df_gcp.columns)
#                     for col in df_gcp.columns:
#                         if col in csv_only_records.columns and pd.notna(row[col]):
#                             new_row[col] = row[col]
#                     new_records = pd.concat([new_records, pd.DataFrame([new_row])], ignore_index=True)
                
#                 df = pd.concat([df_gcp, new_records], ignore_index=True)
#                 st.success(f"Successfully integrated CSV data. Total records: {len(df)}")
#             else:
#                 df = df_gcp
#                 st.warning("Could not load CSV data. Continuing with GCP data only.")
                
#             except Exception as e:
#                 st.error(f"Error loading CSV data: {e}")
#                 df = df_gcp
#         else:
#             df = df_gcp
#             st.info("No CSV file uploaded. Using GCP data only.")
        
#         # Load visa data
#         st.info("Loading visa data...")
#         df_t_visa = load_t_visa()
#         t_visa_set = set(df_t_visa["maid_id"])
#         fixed_cost = load_fixed_cost()
        
#         # Compute visa costs
#         def compute_actual_cost(row):
#             nat = row["nationality_category"]
#             loc = row["location_category"]
            
#             if nat == "filipina":
#                 if loc == "inside_uae":
#                     return fixed_cost.e_visa_inside
#                 else:
#                     return (
#                         fixed_cost.t_visa_outside
#                         if row["maid_id"] in t_visa_set
#                         else fixed_cost.e_visa_outside
#                     )
            
#             if nat in ["african", "ethiopian"]:
#                 if loc == "outside_uae":
#                     return fixed_cost.e_visa_outside
#                 else:
#                     return fixed_cost.e_visa_inside
            
#             return 0
        
#         df["actual_visa_cost"] = df.apply(compute_actual_cost, axis=1)
        
#         # Load and process lost visas
#         lost_visas_df = read_lost_visas_sheet()
#         lost_visas_df.columns = lost_visas_df.columns.str.strip().str.lower()
        
#         df['successful_month'] = pd.to_datetime(df['successful_date'], errors='coerce').dt.strftime('%b-%y')
#         df['lost_evisa_share'] = 0.0
        
#         for _, row in lost_visas_df.iterrows():
#             month = row['month']
#             for nationality in ['filipina', 'ethiopian', 'african']:
#                 lost_total = pd.to_numeric(row[nationality], errors='coerce')
#                 mask = (df['successful_month'] == month) & (df['nationality_category'].str.lower() == nationality)
#                 count = mask.sum()
#                 if count > 0 and pd.notnull(lost_total):
#                     share = lost_total / count
#                     df.loc[mask, 'lost_evisa_share'] = round(share, 2)
        
#         # Load and process tickets
#         st.info("Loading tickets data...")
#         df_tickets = load_tickets_data()
        
#         df['maid_id'] = df['maid_id'].astype(str)
#         df_tickets['maid_id'] = df_tickets['maid_id'].astype(str)
        
#         df_successful = df[df['successful_date'].notnull()].copy()
        
#         ticket_cost = (
#             df_tickets.groupby('maid_id', as_index=False)['Price AED']
#             .sum()
#             .rename(columns={'Price AED': 'total_ticket_cost'})
#         )
        
#         ticket_refund = (
#             df_tickets.groupby('maid_id', as_index=False)['Amount to be Refunded (AED)']
#             .sum()
#             .rename(columns={'Amount to be Refunded (AED)': 'total_ticket_refund'})
#         )
        
#         df_successful = df_successful.merge(ticket_cost, on='maid_id', how='left')
#         df_successful = df_successful.merge(ticket_refund, on='maid_id', how='left')
        
#         df_successful['total_ticket_cost'] = df_successful['total_ticket_cost'].fillna(0)
#         df_successful['total_ticket_refund'] = df_successful['total_ticket_refund'].fillna(0)
        
#         df = df.merge(
#             df_successful[['maid_id', 'total_ticket_cost', 'total_ticket_refund']],
#             on='maid_id',
#             how='left'
#         )
        
#         # Process lost tickets
#         df_tickets['Travel Date'] = pd.to_datetime(df_tickets['Travel Date'], errors='coerce')
#         today = pd.to_datetime(datetime.today().date())
#         tickets_past = df_tickets[df_tickets['Travel Date'] < today].copy()
        
#         hired_maids = df[df['successful_date'].notna()]['maid_id'].astype(str).unique()
#         tickets_past['maid_id'] = tickets_past['maid_id'].astype(str)
#         tickets_lost = tickets_past[~tickets_past['maid_id'].isin(hired_maids)].copy()
        
#         df_maids_lookup = df[['maid_id', 'nationality_category', 'location_category']].drop_duplicates()
#         tickets_lost = tickets_lost.merge(df_maids_lookup, on='maid_id', how='left')
        
#         tickets_lost['ticket_month'] = tickets_lost['Travel Date'].dt.strftime('%b-%y')
#         tickets_lost['Price AED'] = pd.to_numeric(tickets_lost['Price AED'], errors='coerce').fillna(0)
#         tickets_lost['Amount to be Refunded (AED)'] = pd.to_numeric(tickets_lost['Amount to be Refunded (AED)'], errors='coerce').fillna(0)
#         tickets_lost['net_lost_cost'] = tickets_lost['Price AED'] - tickets_lost['Amount to be Refunded (AED)']
        
#         lost_ticket_grouped = tickets_lost.groupby(['ticket_month', 'nationality_category', 'location_category'])['net_lost_cost'].sum().reset_index()
        
#         df['Month'] = df['successful_date'].dt.strftime('%b-%y')
#         hire_counts = df[df['successful_date'].notna()].groupby(['Month', 'nationality_category', 'location_category'])['maid_id'].count().reset_index()
#         hire_counts = hire_counts.rename(columns={'maid_id': 'hire_count'})
        
#         merged_costs = lost_ticket_grouped.merge(hire_counts, left_on=['ticket_month', 'nationality_category', 'location_category'],
#                                                right_on=['Month', 'nationality_category', 'location_category'], how='left')
        
#         merged_costs['lost_ticket_share'] = merged_costs['net_lost_cost'] / merged_costs['hire_count']
#         merged_costs = merged_costs[['Month', 'nationality_category', 'location_category', 'lost_ticket_share']]
        
#         df = df.merge(merged_costs, on=['Month', 'nationality_category', 'location_category'], how='left')
        
#         # Load staff costs
#         st.info("Loading staff costs...")
#         SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
#         bas_cost_df = read_cost_sheet(SHEET_ID, "BAs")
#         daspgs_cost_df = read_cost_sheet(SHEET_ID, "Programmers&DAs")
        
#         def assign_costs_by_month(df_maids, df_costs, target_col_name):
#             df_maids[target_col_name] = 0.0
#             for _, row in df_costs.iterrows():
#                 month = row["Month"]
#                 for nat in ["filipina", "african", "ethiopian"]:
#                     cost_value = row[f"{nat}_share_aed"]
#                     mask = (
#                         (df_maids["successful_month"] == month) &
#                         (df_maids["nationality_category"].str.lower() == nat) &
#                         (df_maids["successful_date"].notnull())
#                     )
#                     count = mask.sum()
#                     if count > 0 and cost_value > 0:
#                         df_maids.loc[mask, target_col_name] = cost_value / count
#             return df_maids
        
#         df = assign_costs_by_month(df, bas_cost_df, "bas_cost_share")
#         df = assign_costs_by_month(df, daspgs_cost_df, "DataAnalysts_and_Programmers_cost_share")
        
#         # Load agents costs
#         agents_cost_summary = read_and_transform_agents_sheet(SHEET_ID)
        
#         df['Agents_cost_share'] = 0.0
#         for _, row in agents_cost_summary.iterrows():
#             month = row['month']
#             for nat in ['filipina', 'ethiopian', 'african']:
#                 cost_col = f"{nat}_cost_aed"
#                 if cost_col not in agents_cost_summary.columns:
#                     continue
#                 total_cost = row[cost_col]
#                 if pd.isna(total_cost) or total_cost == 0:
#                     continue
                
#                 month_parts = month.split('-')
#                 if len(month_parts) != 2:
#                     continue
                
#                 month_name, year = month_parts
#                 year_full = f"20{year}" if len(year) == 2 else year
                
#                 month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
#                            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                
#                 if month_name not in month_map:
#                     continue
                
#                 month_num = month_map[month_name]
#                 start_date = f"{year_full}-{month_num:02d}-01"
                
#                 if month_num in [4, 6, 9, 11]:
#                     end_date = f"{year_full}-{month_num:02d}-30"
#                 elif month_num == 2:
#                     is_leap = (int(year_full) % 4 == 0 and int(year_full) % 100 != 0) or (int(year_full) % 400 == 0)
#                     end_date = f"{year_full}-{month_num:02d}-29" if is_leap else f"{year_full}-{month_num:02d}-28"
#                 else:
#                     end_date = f"{year_full}-{month_num:02d}-31"
                
#                 mask = (
#                     (df['successful_date'] >= start_date) &
#                     (df['successful_date'] <= end_date) &
#                     (df['nationality_category'].str.lower() == nat)
#                 )
                
#                 count = mask.sum()
#                 if count == 0:
#                     continue
                
#                 cost_per_maid = total_cost / count
#                 df.loc[mask, 'Agents_cost_share'] = cost_per_maid
        
#         # Load LLM costs
#         st.info("Loading LLM and other costs...")
#         df_LLM = read_llm_costs_sheet_fixed()
        
#         # Add LLM cost share (simplified for space)
#         df['llm_cost_share'] = 0.0
        
#         # Load referrals
#         referrals_df = read_referrals_sheet()
#         df['maid_id'] = df['maid_id'].astype(str).str.strip()
#         referrals_df['Maid A Applicant ID'] = referrals_df['Maid A Applicant ID'].astype(str).str.strip()
#         referred_maid_ids = set(referrals_df['Maid A Applicant ID'])
#         df['referral_cost'] = df['maid_id'].apply(lambda x: 1000 if x in referred_maid_ids else 0)
        
#         # Load broadcasts
#         broadcasts_df = read_broadcasts_sheet()
#         df_hires = df[df['successful_date'].notnull() & df['nationality_category'].notnull()].copy()
#         df_hires['Month'] = pd.to_datetime(df_hires['successful_date'], errors='coerce').dt.strftime('%b-%y')
        
#         broadcast_cost_map = {}
#         for _, row in broadcasts_df.iterrows():
#             month = row['Month']
#             for nat in ['filipina', 'ethiopian', 'african']:
#                 total_cost = row.get(f"{nat}_broadcast", 0)
#                 if total_cost == 0:
#                     continue
#                 hires = df_hires[(df_hires['Month'] == month) & (df_hires['nationality_category'].str.lower() == nat)]
#                 if hires.empty:
#                     continue
#                 cost_per_maid = total_cost / len(hires)
#                 for maid_id in hires['maid_id']:
#                     broadcast_cost_map[maid_id] = cost_per_maid
        
#         df['broadcast_cost'] = df['maid_id'].map(broadcast_cost_map).fillna(0)
        
#         # Calculate marketing costs with date range
#         st.info("Calculating marketing costs...")
#         df = calculate_marketing_cost_per_hire(df, num_months=12, behavior_start_date=behavior_start_date, behavior_end_date=behavior_end_date)
        
#         # Add operator costs
#         def calculate_operator_cost(row):
#             USD_TO_AED = 3.67
#             cost = 0
#             operator = str(row['freedom_operator']).lower() if pd.notna(row['freedom_operator']) else ""
            
#             if 'marilyn' in operator:
#                 return 1966
            
#             if row['nationality_category'] == 'ethiopian':
#                 if 'wa' in operator:
#                     return 900 * USD_TO_AED
#                 elif any(op in operator for op in ['fiseha', 'natnael', 'tadesse']):
#                     return 1000 * USD_TO_AED
#                 elif 'berana' in operator:
#                     return 900 * USD_TO_AED
            
#             return cost
        
#         df['operator_cost'] = df.apply(calculate_operator_cost, axis=1)
        
#         # Initialize attestation cost (placeholder)
#         df['attestation_cost'] = 0
        
#         st.success("Data processing completed successfully!")
#         return df

# # Main Streamlit App
# def main():
#     st.title("🏠 Cost Per Hire Analysis Dashboard")
    
#     # Create tabs
#     tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
#         "Cost per Hire", "Visas", "Tickets", "BAs, DAs, Programmers", 
#         "Agents", "LLMs", "Referrals", "Broadcasts", "Ad Spend", "Operator", "Attestation"
#     ])
    
#     with tab1:
#         st.markdown("<h1 style='text-align: center; font-size: 48px;'>This is to estimate the cost per Hire</h1>", unsafe_allow_html=True)
        
#         # Re-read data button
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             if st.button("🔄 Re-Read Data", type="primary", use_container_width=True):
#                 st.session_state.data_loaded = False
#                 st.session_state.df = None
#                 st.cache_data.clear()
#                 st.rerun()
        
#         st.markdown("---")
        
#         # Date range slider for behavior analysis
#         st.subheader("📊 Behavior Analysis Period for Marketing Costs")
#         st.info("Select the date range to use for calculating time-to-hire behavior patterns. This controls which historical applications are used to learn conversion patterns, while spend attribution still uses all available data.")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             behavior_start_date = st.date_input(
#                 "Start Date for Behavior Analysis",
#                 value=date(2022, 1, 1),
#                 key="behavior_start"
#             )
#         with col2:
#             behavior_end_date = st.date_input(
#                 "End Date for Behavior Analysis",
#                 value=date(2025, 1, 31),
#                 key="behavior_end"
#             )
        
#         # CSV File Upload
#         st.subheader("📤 Upload CSV Data (Optional)")
#         st.info("Upload your Daily Conversion Report CSV file to integrate with GCP data. If not uploaded, the system will use GCP data only.")
#         uploaded_csv = st.file_uploader(
#             "Choose CSV file",
#             type=['csv'],
#             help="Upload your Daily Conversion Report CSV file"
#         )
        
#         # Load and process data
#         if not st.session_state.data_loaded or st.session_state.df is None:
#             st.session_state.df = process_all_data(behavior_start_date, behavior_end_date, uploaded_csv)
#             st.session_state.data_loaded = True
        
#         # Display summary statistics
#         if st.session_state.df is not None:
#             df = st.session_state.df
            
#             st.markdown("---")
#             st.subheader("📈 Data Summary")
            
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Total Records", f"{len(df):,}")
#             with col2:
#                 successful_hires = df['successful_date'].notna().sum()
#                 st.metric("Successful Hires", f"{successful_hires:,}")
#             with col3:
#                 avg_visa_cost = df[df['successful_date'].notna()]['actual_visa_cost'].mean()
#                 st.metric("Avg Visa Cost", f"{avg_visa_cost:.2f} AED")
#             with col4:
#                 total_marketing_cost = df[df['successful_date'].notna()]['marketing_cost_per_hire'].sum()
#                 st.metric("Total Marketing Cost", f"{total_marketing_cost:,.0f} AED")
            
#             # Filter for successful hires only
#             successful_hires_df = df[df['successful_date'].notna()].copy()
            
#             if len(successful_hires_df) > 0:
#                 # Add total cost per hire calculation
#                 cost_columns = [
#                     'actual_visa_cost', 'lost_evisa_share', 'total_ticket_cost', 
#                     'total_ticket_refund', 'lost_ticket_share', 'bas_cost_share',
#                     'DataAnalysts_and_Programmers_cost_share', 'Agents_cost_share',
#                     'llm_cost_share', 'referral_cost', 'broadcast_cost',
#                     'marketing_cost_per_hire', 'operator_cost', 'attestation_cost'
#                 ]
                
#                 # Ensure all cost columns exist and fill NaN with 0
#                 for col in cost_columns:
#                     if col not in successful_hires_df.columns:
#                         successful_hires_df[col] = 0
#                     successful_hires_df[col] = successful_hires_df[col].fillna(0)
                
#                 # Calculate net ticket cost (cost minus refund)
#                 successful_hires_df['net_ticket_cost'] = successful_hires_df['total_ticket_cost'] - successful_hires_df['total_ticket_refund']
                
#                 # Calculate total cost per hire
#                 successful_hires_df['total_cost_per_hire'] = (
#                     successful_hires_df['actual_visa_cost'] +
#                     successful_hires_df['lost_evisa_share'] +
#                     successful_hires_df['net_ticket_cost'] +
#                     successful_hires_df['lost_ticket_share'] +
#                     successful_hires_df['bas_cost_share'] +
#                     successful_hires_df['DataAnalysts_and_Programmers_cost_share'] +
#                     successful_hires_df['Agents_cost_share'] +
#                     successful_hires_df['llm_cost_share'] +
#                     successful_hires_df['referral_cost'] +
#                     successful_hires_df['broadcast_cost'] +
#                     successful_hires_df['marketing_cost_per_hire'] +
#                     successful_hires_df['operator_cost'] +
#                     successful_hires_df['attestation_cost']
#                 )
                
#                 st.markdown("---")
#                 st.subheader("💰 Complete Cost per Hire Analysis")
                
#                 # Display download section
#                 col1, col2 = st.columns([2, 1])
#                 with col1:
#                     st.info("💾 **Download Complete Cost Analysis** - Export all successful hires with detailed cost breakdown")
#                 with col2:
#                     # Prepare data for download
#                     download_df = successful_hires_df.copy()
#                     # Round numerical columns to 2 decimal places
#                     numeric_columns = download_df.select_dtypes(include=['float64', 'int64']).columns
#                     download_df[numeric_columns] = download_df[numeric_columns].round(2)
                    
#                     csv_data = download_df.to_csv(index=False)
#                     st.download_button(
#                         label="📥 Download CSV",
#                         data=csv_data,
#                         file_name=f"cost_per_hire_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                         mime="text/csv",
#                         type="primary"
#                     )
                
#                 # Display summary of cost components
#                 st.subheader("📊 Cost Components Summary")
#                 cost_summary = pd.DataFrame({
#                     'Cost Component': [
#                         'Visa Costs', 'Lost E-Visa Share', 'Net Ticket Costs', 'Lost Ticket Share',
#                         'BA Costs', 'DA & Programmer Costs', 'Agent Costs', 'LLM Costs',
#                         'Referral Costs', 'Broadcast Costs', 'Marketing Costs', 'Operator Costs', 'Attestation Costs'
#                     ],
#                     'Total Amount (AED)': [
#                         successful_hires_df['actual_visa_cost'].sum(),
#                         successful_hires_df['lost_evisa_share'].sum(),
#                         successful_hires_df['net_ticket_cost'].sum(),
#                         successful_hires_df['lost_ticket_share'].sum(),
#                         successful_hires_df['bas_cost_share'].sum(),
#                         successful_hires_df['DataAnalysts_and_Programmers_cost_share'].sum(),
#                         successful_hires_df['Agents_cost_share'].sum(),
#                         successful_hires_df['llm_cost_share'].sum(),
#                         successful_hires_df['referral_cost'].sum(),
#                         successful_hires_df['broadcast_cost'].sum(),
#                         successful_hires_df['marketing_cost_per_hire'].sum(),
#                         successful_hires_df['operator_cost'].sum(),
#                         successful_hires_df['attestation_cost'].sum()
#                     ],
#                     'Average per Hire (AED)': [
#                         successful_hires_df['actual_visa_cost'].mean(),
#                         successful_hires_df['lost_evisa_share'].mean(),
#                         successful_hires_df['net_ticket_cost'].mean(),
#                         successful_hires_df['lost_ticket_share'].mean(),
#                         successful_hires_df['bas_cost_share'].mean(),
#                         successful_hires_df['DataAnalysts_and_Programmers_cost_share'].mean(),
#                         successful_hires_df['Agents_cost_share'].mean(),
#                         successful_hires_df['llm_cost_share'].mean(),
#                         successful_hires_df['referral_cost'].mean(),
#                         successful_hires_df['broadcast_cost'].mean(),
#                         successful_hires_df['marketing_cost_per_hire'].mean(),
#                         successful_hires_df['operator_cost'].mean(),
#                         successful_hires_df['attestation_cost'].mean()
#                     ]
#                 })
#                 cost_summary = cost_summary.round(2)
#                 st.dataframe(cost_summary, use_container_width=True)
                
#                 # Display total cost summary
#                 total_cost = successful_hires_df['total_cost_per_hire'].sum()
#                 avg_cost = successful_hires_df['total_cost_per_hire'].mean()
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("💰 Total Cost (All Hires)", f"{total_cost:,.2f} AED")
#                 with col2:
#                     st.metric("📊 Average Cost per Hire", f"{avg_cost:,.2f} AED")
#                 with col3:
#                     st.metric("🎯 Total Successful Hires", f"{len(successful_hires_df):,}")
                
#                 # Display detailed view with cost breakdown
#                 st.subheader("📋 Detailed Cost per Hire View")
#                 st.info("👁️ **Preview of downloadable data** - Showing first 50 records with all cost components")
                
#                 # Select key columns for display
#                 display_columns = [
#                     'maid_id', 'applicant_name', 'successful_date', 'nationality_category', 
#                     'location_category', 'actual_visa_cost', 'net_ticket_cost', 
#                     'marketing_cost_per_hire', 'operator_cost', 'referral_cost', 
#                     'total_cost_per_hire'
#                 ]
                
#                 # Filter columns that exist
#                 available_display_columns = [col for col in display_columns if col in successful_hires_df.columns]
                
#                 # Sort by successful_date descending and show first 50 records
#                 preview_df = successful_hires_df[available_display_columns].sort_values('successful_date', ascending=False).head(50)
#                 st.dataframe(preview_df, use_container_width=True, height=400)
                
#                 # Show column information
#                 st.subheader("📝 Dataset Information")
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.write(f"**Total Columns in Dataset:** {len(successful_hires_df.columns)}")
#                     st.write(f"**Records Available for Download:** {len(successful_hires_df)}")
#                 with col2:
#                     st.write("**Key Cost Components:**")
#                     key_costs = ['actual_visa_cost', 'marketing_cost_per_hire', 'operator_cost', 'referral_cost', 'total_cost_per_hire']
#                     for cost in key_costs:
#                         if cost in successful_hires_df.columns:
#                             st.write(f"• {cost.replace('_', ' ').title()}")
#             else:
#                 st.warning("No successful hires found in the dataset.")
    
#     with tab2:
#         st.header("🛂 Visa Data")
        
#         try:
#             st.subheader("T-Visa Data")
#             df_t_visa = load_t_visa()
#             st.dataframe(df_t_visa.head(), use_container_width=True)
#             st.info(f"Total T-Visa records: {len(df_t_visa)}")
            
#             st.subheader("Fixed Visa Costs")
#             fixed_cost = load_fixed_cost()
#             fixed_cost_df = pd.DataFrame([fixed_cost])
#             st.dataframe(fixed_cost_df, use_container_width=True)
            
#             st.subheader("Lost E-Visas Data")
#             lost_visas_df = read_lost_visas_sheet()
#             st.dataframe(lost_visas_df.head(), use_container_width=True)
#             st.info(f"Total Lost E-Visa records: {len(lost_visas_df)}")
            
#         except Exception as e:
#             st.error(f"Error loading visa data: {e}")
    
#     with tab3:
#         st.header("✈️ Tickets Data")
        
#         try:
#             df_tickets = load_tickets_data()
#             st.dataframe(df_tickets.head(), use_container_width=True)
#             st.info(f"Total ticket records: {len(df_tickets)}")
            
#             # Summary statistics
#             col1, col2, col3 = st.columns(3)
#             with col1:
#                 avg_price = df_tickets['Price AED'].mean()
#                 st.metric("Average Ticket Price", f"{avg_price:.2f} AED")
#             with col2:
#                 total_refunds = df_tickets['Amount to be Refunded (AED)'].sum()
#                 st.metric("Total Refunds", f"{total_refunds:,.2f} AED")
#             with col3:
#                 ticket_types = df_tickets['Type'].nunique()
#                 st.metric("Ticket Types", ticket_types)
                
#         except Exception as e:
#             st.error(f"Error loading tickets data: {e}")
    
#     with tab4:
#         st.header("👥 BAs, DAs, and Programmers Data")
        
#         try:
#             SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
            
#             st.subheader("Business Analysts Costs")
#             bas_cost_df = read_cost_sheet(SHEET_ID, "BAs")
#             st.dataframe(bas_cost_df.head(), use_container_width=True)
            
#             st.subheader("Programmers & Data Analysts Costs")
#             daspgs_cost_df = read_cost_sheet(SHEET_ID, "Programmers&DAs")
#             st.dataframe(daspgs_cost_df.head(), use_container_width=True)
            
#         except Exception as e:
#             st.error(f"Error loading staff cost data: {e}")
    
#     with tab5:
#         st.header("🤝 Agents Data")
        
#         try:
#             SHEET_ID = "1bbpwM_6C2f4Z0KeOH2CI7NKr0oq-Za6DJsgRZaHd1v8"
#             agents_cost_summary = read_and_transform_agents_sheet(SHEET_ID)
#             st.dataframe(agents_cost_summary.head(), use_container_width=True)
#             st.info(f"Total agent cost records: {len(agents_cost_summary)}")
            
#         except Exception as e:
#             st.error(f"Error loading agents data: {e}")
    
#     with tab6:
#         st.header("🤖 LLM Costs Data")
        
#         try:
#             df_LLM = read_llm_costs_sheet_fixed()
#             st.dataframe(df_LLM.head(), use_container_width=True)
#             st.info(f"Total LLM cost records: {len(df_LLM)}")
            
#             # Summary
#             total_llm_cost = df_LLM['Cost'].sum()
#             st.metric("Total LLM Costs", f"{total_llm_cost:,.2f} AED")
            
#         except Exception as e:
#             st.error(f"Error loading LLM data: {e}")
    
#     with tab7:
#         st.header("🔗 Referrals Data")
        
#         try:
#             referrals_df = read_referrals_sheet()
#             st.dataframe(referrals_df.head(), use_container_width=True)
#             st.info(f"Total referral records: {len(referrals_df)}")
            
#         except Exception as e:
#             st.error(f"Error loading referrals data: {e}")
    
#     with tab8:
#         st.header("📢 Broadcasts Data")
        
#         try:
#             broadcasts_df = read_broadcasts_sheet()
#             st.dataframe(broadcasts_df.head(), use_container_width=True)
#             st.info(f"Total broadcast records: {len(broadcasts_df)}")
            
#         except Exception as e:
#             st.error(f"Error loading broadcasts data: {e}")
    
#     with tab9:
#         st.header("💰 Ad Spend Data")
#         st.info("Marketing cost calculation uses weighted attribution based on time-to-hire patterns")
#         st.markdown("""
#         **How it works:**
#         1. **Behavior Analysis**: Uses historical data within your selected date range to learn conversion patterns
#         2. **Spend Attribution**: Looks back at ALL available spend data when calculating costs
#         3. **Weighted Attribution**: Each month's spend is weighted based on historical conversion probability
#         """)
        
#         if st.session_state.df is not None:
#             df = st.session_state.df
#             marketing_summary = df[df['successful_date'].notna()].groupby(['nationality_category', 'location_category'])['marketing_cost_per_hire'].agg(['count', 'mean', 'sum']).round(2)
#             marketing_summary.columns = ['Hire Count', 'Avg Cost per Hire', 'Total Cost']
#             st.dataframe(marketing_summary, use_container_width=True)
    
#     with tab10:
#         st.header("⚙️ Operator Costs Data")
#         st.info("Operator costs are calculated based on freedom operator assignments and nationality")
        
#         if st.session_state.df is not None:
#             df = st.session_state.df
#             operator_summary = df[df['operator_cost'] > 0].groupby(['nationality_category', 'freedom_operator'])['operator_cost'].agg(['count', 'mean']).round(2)
#             operator_summary.columns = ['Count', 'Cost per Hire']
#             st.dataframe(operator_summary, use_container_width=True)
    
#     with tab11:
#         st.header("📋 Attestation Data")
#         st.info("Attestation costs would be loaded from a separate sheet (placeholder)")
#         st.warning("Attestation sheet ID needs to be configured")
        
#         if st.session_state.df is not None:
#             df = st.session_state.df
#             attestation_count = (df['attestation_cost'] > 0).sum()
#             st.metric("Records with Attestation Costs", attestation_count)

# if __name__ == "__main__":
#     main()
