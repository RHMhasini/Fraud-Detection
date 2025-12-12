import streamlit as st
import requests
import pandas as pd
import hashlib
from datetime import datetime
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Tuple

# -------------------
# Configuration
# -------------------
API_BASE_URL = "http://127.0.0.1:8000"
TRANSACTION_ENDPOINT = f"{API_BASE_URL}/transaction"
TRANSACTIONS_ENDPOINT = f"{API_BASE_URL}/transactions"
ML_SCORES_ENDPOINT = f"{API_BASE_URL}/ml_scores"
VERIFY_LAST_ENDPOINT = f"{API_BASE_URL}/verify_last"
VERIFY_ENDPOINT = f"{API_BASE_URL}/verify"
VERIFICATIONS_ENDPOINT = f"{API_BASE_URL}/verifications"

# Thresholds - Matched with API
# - Pass: < 0.3 (30% fraud probability) - Low risk
# - Flag: 0.3 - 0.6 (30-60% fraud probability) - Medium risk, needs review
# - Deny: >= 0.6 (60%+ fraud probability) - High risk, should be denied
THRESHOLD_PASS = 0.3
THRESHOLD_DENY = 0.6

# -------------------
# Helper Functions
# -------------------
def hash_string(text: str) -> str:
    """Hash a string using SHA256"""
    return hashlib.sha256(text.encode()).hexdigest()

def classify_score(score: float) -> str:
    """Classify transaction based on ensemble score"""
    if score < THRESHOLD_PASS:
        return "pass"
    elif score < THRESHOLD_DENY:
        return "flag"
    else:
        return "deny"

def get_score_color(score: float) -> str:
    """Get color based on score - matches classification thresholds"""
    if score < THRESHOLD_PASS:
        return "green"
    elif score < THRESHOLD_DENY:
        return "yellow"
    else:
        return "red"

def check_api_connection():
    """Check if API is available"""
    try:
        response = requests.get(TRANSACTIONS_ENDPOINT, timeout=2)
        return response.status_code == 200
    except:
        return False

def preprocess_transaction(user_id, signup_time, purchase_time, purchase_value, 
                          device_id, ip_address, browser, source, sex, age):
    """Preprocess raw transaction data into API format (raw transaction format)"""
    # Return raw transaction format - API will handle preprocessing
    return {
        "signup_time": signup_time.isoformat() if isinstance(signup_time, datetime) else signup_time,
        "purchase_time": purchase_time.isoformat() if isinstance(purchase_time, datetime) else purchase_time,
        "purchase_value": float(purchase_value),
        "age": int(age),
        "device_id": device_id,
        "ip_address": ip_address,
        "user_id": user_id,
        "source": source,
        "browser": browser,
        "sex": sex
    }

def create_pdf_report(df: pd.DataFrame, summary: dict) -> BytesIO:
    """Generate a PDF report for download."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Fraud Detection Transaction Report")
    y -= 30

    c.setFont("Helvetica", 12)
    for label, value in summary.items():
        c.drawString(40, y, f"{label}: {value}")
        y -= 18
        if y < 60:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12)

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Transactions")
    y -= 20

    c.setFont("Helvetica", 10)
    for _, row in df.head(100).iterrows():
        line = (
            f"#{row['transaction_id']} | {row['time']} | ${row['amount']:.2f} | "
            f"Score {row['score']:.3f} | {row['status'].upper()}"
        )
        c.drawString(40, y, line)
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 10)

    c.save()
    buffer.seek(0)
    return buffer

def delete_transaction_via_api(transaction_id: int) -> Tuple[bool, str]:
    """Call API to delete a transaction."""
    try:
        response = requests.delete(f"{TRANSACTION_ENDPOINT}/{transaction_id}", timeout=5)
        if response.status_code == 200:
            return True, "Transaction deleted successfully."
        elif response.status_code == 404:
            return False, f"Transaction {transaction_id} not found."
        return False, response.json().get("detail", "Unable to delete transaction.")
    except requests.exceptions.RequestException as exc:
        return False, f"API error: {exc}"

# -------------------
# Page: Transaction Submission
# -------------------
def page_submit_transaction():
    st.header(" Submit Transaction")
    
    if not check_api_connection():
        st.error(" API unavailable. Please ensure the FastAPI backend is running at http://127.0.0.1:8000")
        return
    
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input("User ID", value="user_123")
            st.write("**Signup Time**")
            signup_date = st.date_input("Signup Date", value=datetime.now().date(), key="signup_date")
            signup_time_input = st.time_input("Signup Time", value=datetime.now().time(), key="signup_time")
            signup_time = datetime.combine(signup_date, signup_time_input)
            
            st.write("**Purchase Time**")
            purchase_date = st.date_input("Purchase Date", value=datetime.now().date(), key="purchase_date")
            purchase_time_input = st.time_input("Purchase Time", value=datetime.now().time(), key="purchase_time")
            purchase_time = datetime.combine(purchase_date, purchase_time_input)
            
            purchase_value = st.number_input("Purchase Value ($)", min_value=0.01, value=100.0, step=0.01)
            device_id = st.text_input("Device ID", value="device_123")
            ip_address = st.text_input("IP Address", value="192.168.1.1")
        
        with col2:
            browser = st.selectbox("Browser", ["Chrome", "FireFox", "IE", "Opera", "Safari"])
            source = st.selectbox("Source", ["Ads", "Direct", "SEO"])
            sex = st.selectbox("Sex", ["F", "M"])
            age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        
        submitted = st.form_submit_button("Submit Transaction", use_container_width=True)
    
    if submitted:
        with st.spinner("Processing transaction..."):
            try:
                # Preprocess data
                transaction_data = preprocess_transaction(
                    user_id, signup_time, purchase_time, purchase_value,
                    device_id, ip_address, browser, source, sex, age
                )
                
                # Submit to API
                response = requests.post(TRANSACTION_ENDPOINT, json=transaction_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(" Transaction submitted successfully!")
                        
                        # Extract ML scores from result
                        ml_scores = result.get("ml_scores", {})
                        verification = result.get("verification", {})
                        
                        # Display scores
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Transaction ID", result.get("transaction_id"))
                        with col2:
                            st.metric("RF Score", f"{ml_scores.get('RF_score', 0):.3f}")
                        with col3:
                            st.metric("XGB Score", f"{ml_scores.get('XGB_score', 0):.3f}")
                        with col4:
                            ensemble_score = ml_scores.get("ensemble_score", 0)
                            status = verification.get("status", classify_score(ensemble_score))
                            st.metric("Ensemble Score", f"{ensemble_score:.3f}", 
                                     delta=status.upper())
                        
                        # Display anomaly detection results
                        anomaly = result.get("anomaly_detection", {})
                        if anomaly.get("has_anomalies"):
                            st.warning(f" Anomalies detected: {', '.join(anomaly.get('anomalies', []))}")
                        
                        # Display verification and explanation
                        if verification.get("status") in ["flag", "deny"]:
                            if verification.get("explanation"):
                                st.subheader(" Explanation")
                                st.info(verification["explanation"])
                        
                        # Display alerts if generated
                        alerts = result.get("alerts", {})
                        if alerts and alerts.get("alert_count", 0) > 0:
                            st.error(f" {alerts.get('alert_count', 0)} alert(s) generated!")
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ API Error: {error_detail}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection error: {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# -------------------
# Page: Transaction History Dashboard
# -------------------
def page_transaction_history():
    st.header(" Transaction History Dashboard")
    
    if not check_api_connection():
        st.error(" API unavailable. Please ensure the FastAPI backend is running at http://127.0.0.1:8000")
        return
    
    if st.button(" Refresh Data", use_container_width=True):
        st.rerun()
    
    with st.spinner("Loading transaction data..."):
        try:
            # Fetch transactions, scores, and verifications
            trans_response = requests.get(TRANSACTIONS_ENDPOINT)
            scores_response = requests.get(ML_SCORES_ENDPOINT)
            verifications_response = requests.get(VERIFICATIONS_ENDPOINT)
            
            if trans_response.status_code != 200 or scores_response.status_code != 200:
                st.error("Failed to fetch data from API")
                return
            
            transactions = trans_response.json()
            scores = scores_response.json()
            verifications = verifications_response.json() if verifications_response.status_code == 200 else []
            
            if not transactions or not scores:
                st.info("No transactions found. Please submit a transaction first.")
                return
            
            trans_df = pd.DataFrame(transactions)
            scores_df = pd.DataFrame(scores)
            
            merged_df = trans_df.merge(
                scores_df[["transaction_id", "ensemble_score"]],
                on="transaction_id",
                how="inner"
            )
            
            # Merge verifications to get explanations
            if verifications:
                verifications_df = pd.DataFrame(verifications)
                merged_df = merged_df.merge(
                    verifications_df[["transaction_id", "explanation"]],
                    on="transaction_id",
                    how="left"
                )
            else:
                merged_df["explanation"] = None
            
            merged_df["purchase_time_dt"] = pd.to_datetime(merged_df["purchase_time"], format='ISO8601', errors='coerce')
            merged_df["status"] = merged_df["ensemble_score"].apply(classify_score)
            merged_df["display_time"] = merged_df["purchase_time_dt"].dt.strftime("%Y-%m-%d %H:%M")
            merged_df["user_short"] = merged_df["hashed_user_id"].str[:8] + "..."
            merged_df["device_short"] = merged_df["hashed_device_id"].str[:8] + "..."
            
            merged_df[["hashed_user_id", "hashed_device_id", "hashed_ip_address"]] = (
                merged_df[["hashed_user_id", "hashed_device_id", "hashed_ip_address"]].fillna("")
            )
            min_date = merged_df["purchase_time_dt"].min().date()
            max_date = merged_df["purchase_time_dt"].max().date()

            search_term = ""
            status_filter = []
            score_range = (0.0, 1.0)
            date_range = (min_date, max_date)
            delete_id = None

            with st.expander("Filters & Actions", expanded=False):
                search_term = st.text_input("Search (transaction ID, user hash, device hash, IP)")
                status_filter = st.multiselect("Status filter", ["pass", "flag", "deny"])
                score_range = st.slider(
                    "Score range",
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.01
                )
                date_range = st.date_input(
                    "Purchase date range",
                    value=(min_date, max_date)
                )
                col_select, col_action = st.columns([2, 1])
                delete_id = col_select.selectbox(
                    "Select transaction to delete",
                    merged_df["transaction_id"].tolist(),
                    format_func=lambda x: f"#{x}"
                )
                if col_action.button("Delete Transaction", use_container_width=True):
                    success, message = delete_transaction_via_api(delete_id)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                st.caption("Tip: apply filters before deleting or exporting reports.")

            filtered_df = merged_df.copy()
            if search_term:
                term = search_term.lower()
                mask = (
                    filtered_df["transaction_id"].astype(str).str.contains(term, case=False)
                    | filtered_df["hashed_user_id"].str.lower().str.contains(term)
                    | filtered_df["hashed_device_id"].str.lower().str.contains(term)
                    | filtered_df["hashed_ip_address"].str.lower().str.contains(term)
                )
                filtered_df = filtered_df[mask]
            if status_filter:
                filtered_df = filtered_df[filtered_df["status"].isin(status_filter)]
            filtered_df = filtered_df[
                (filtered_df["ensemble_score"] >= score_range[0]) &
                (filtered_df["ensemble_score"] <= score_range[1])
            ]
            start_date = end_date = None
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
            elif date_range:
                start_date = pd.to_datetime(date_range)
                end_date = start_date + pd.Timedelta(days=1)
            if start_date is not None:
                filtered_df = filtered_df[
                    (filtered_df["purchase_time_dt"] >= start_date) &
                    (filtered_df["purchase_time_dt"] < end_date)
                ]

            if filtered_df.empty:
                st.warning("No transactions match the selected filters.")
                return

            # Format explanations for display
            display_df = pd.DataFrame({
                "transaction_id": filtered_df["transaction_id"],
                "user": filtered_df["user_short"],
                "amount": filtered_df["purchase_value"],
                "time": filtered_df["display_time"],
                "device": filtered_df["device_short"],
                "score": filtered_df["ensemble_score"],
                "status": filtered_df["status"]
            })
            
            # Add explanation column (only for flagged/denied)
            explanations_dict = {}
            for idx, row in filtered_df.iterrows():
                tx_id = row["transaction_id"]
                status = row["status"]
                if status in ["flag", "deny"]:
                    explanation = row.get("explanation")
                    if pd.notna(explanation) and explanation:
                        explanations_dict[tx_id] = explanation
            
            display_df["explanation"] = display_df["transaction_id"].map(explanations_dict)

            st.subheader("All Transactions")

            def color_rows(row):
                score = row["score"]
                if score < THRESHOLD_PASS:
                    return ['background-color: #4FC978'] * len(row)
                elif score < THRESHOLD_DENY:
                    return ['background-color: #EFD033'] * len(row)
                else:
                    return ['background-color: #c30F18'] * len(row)

            styled_df = display_df[["transaction_id", "user", "amount", "time", "device", "score", "status"]].style.apply(color_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Show explanations below table for flagged/denied transactions
            flagged_denied = display_df[display_df["status"].isin(["flag", "deny"])]
            if not flagged_denied.empty:
                st.subheader("Explanations for Flagged/Denied Transactions")
                for idx, row in flagged_denied.iterrows():
                    tx_id = int(row["transaction_id"])
                    status = row["status"]
                    explanation = row.get("explanation")
                    
                    with st.expander(f"Transaction #{tx_id} - {status.upper()}", expanded=False):
                        if pd.notna(explanation) and explanation:
                            st.info(explanation)
                        else:
                            if st.button(f"Generate Explanation", key=f"gen_expl_{tx_id}"):
                                with st.spinner("Generating explanation..."):
                                    try:
                                        verify_response = requests.post(f"{VERIFY_ENDPOINT}/{tx_id}")
                                        if verify_response.status_code == 200:
                                            verify_result = verify_response.json()
                                            explanation = verify_result.get("explanation", "Explanation unavailable.")
                                            st.info(explanation)
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Failed to generate explanation: {str(e)}")

            summary_stats = {
                "Total Transactions": len(display_df),
                "Passed": int((display_df["status"] == "pass").sum()),
                "Flagged": int((display_df["status"] == "flag").sum()),
                "Denied": int((display_df["status"] == "deny").sum())
            }

            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", summary_stats["Total Transactions"])
            with col2:
                st.metric("Passed", summary_stats["Passed"])
            with col3:
                st.metric("Flagged", summary_stats["Flagged"])
            with col4:
                st.metric("Denied", summary_stats["Denied"])

            pdf_buffer = create_pdf_report(display_df, summary_stats)
            download_col, _ = st.columns([1, 3])
            with download_col:
                st.download_button(
                    " Download PDF Report",
                    data=pdf_buffer,
                    file_name="transaction_report.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                    type="secondary"
                )
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# -------------------
# Page: Alerts
# -------------------
def page_alerts():
    st.header(" High-Risk Transaction Alerts")
    
    if not check_api_connection():
        st.error(" API unavailable. Please ensure the FastAPI backend is running at http://127.0.0.1:8000")
        return
    
    # Cache for explanations to avoid repeated API calls
    if "alert_explanations" not in st.session_state:
        st.session_state.alert_explanations = {}
    
    with st.spinner("Loading alerts..."):
        try:
            # Fetch transactions and scores
            trans_response = requests.get(TRANSACTIONS_ENDPOINT)
            scores_response = requests.get(ML_SCORES_ENDPOINT)
            
            if trans_response.status_code != 200 or scores_response.status_code != 200:
                st.error("Failed to fetch data from API")
                return
            
            transactions = trans_response.json()
            scores = scores_response.json()
            
            if not transactions or not scores:
                st.info("No transactions found. Please submit a transaction first.")
                return
            
            # Merge and filter high-risk (score >= THRESHOLD_DENY for deny status)
            trans_df = pd.DataFrame(transactions)
            scores_df = pd.DataFrame(scores)
            
            merged_df = trans_df.merge(
                scores_df[["transaction_id", "ensemble_score"]],
                on="transaction_id",
                how="inner"
            )
            
            high_risk = merged_df[merged_df["ensemble_score"] >= THRESHOLD_DENY].copy()
            
            if high_risk.empty:
                st.success(" No high-risk transactions detected.")
                return
            
            st.warning(f" Found {len(high_risk)} high-risk transaction(s)")
            
            # Display each alert
            for idx, row in high_risk.iterrows():
                transaction_id = int(row["transaction_id"])
                ensemble_score = row["ensemble_score"]
                
                with st.expander(f" Transaction #{transaction_id} - Score: {ensemble_score:.3f}", expanded=True):
                    # Get explanation (cached or fetch)
                    if transaction_id not in st.session_state.alert_explanations:
                        with st.spinner("Generating explanation..."):
                            try:
                                verify_response = requests.post(f"{VERIFY_ENDPOINT}/{transaction_id}")
                                if verify_response.status_code == 200:
                                    verify_result = verify_response.json()
                                    explanation = verify_result.get("explanation", "This transaction requires manual review due to suspicious activity patterns.")
                                    st.session_state.alert_explanations[transaction_id] = explanation
                                else:
                                    explanation = "This transaction requires manual review due to suspicious activity patterns."
                                    st.session_state.alert_explanations[transaction_id] = explanation
                            except:
                                explanation = "This transaction requires manual review due to suspicious activity patterns."
                                st.session_state.alert_explanations[transaction_id] = explanation
                    else:
                        explanation = st.session_state.alert_explanations[transaction_id]
                    
                    st.write(f"**Score:** {ensemble_score:.3f}")
                    st.write(f"**Status:** {classify_score(ensemble_score).upper()}")
                    st.write(f"**Explanation:** {explanation}")
                    
                    # Show transaction details
                    with st.expander("Transaction Details"):
                        detail_cols = ["purchase_value", "age", "purchase_time", "hashed_user_id", "hashed_device_id"]
                        for col in detail_cols:
                            if col in row:
                                st.write(f"**{col.replace('_', ' ').title()}:** {row[col]}")
                
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# -------------------
# Page: ML Insights
# -------------------
def page_ml_insights():
    st.header(" ML Model Insights")
    
    if not check_api_connection():
        st.error(" API unavailable. Please ensure the FastAPI backend is running at http://127.0.0.1:8000")
        return
    
    with st.spinner("Loading ML scores..."):
        try:
            # Fetch scores
            scores_response = requests.get(ML_SCORES_ENDPOINT)
            
            if scores_response.status_code != 200:
                st.error("Failed to fetch scores from API")
                return
            
            scores = scores_response.json()
            
            if not scores:
                st.info("No ML scores found. Please submit a transaction first.")
                return
            
            scores_df = pd.DataFrame(scores)
            
            # Histogram of ensemble scores
            st.subheader(" Score Distribution")
            fig = px.histogram(
                scores_df,
                x="ensemble_score",
                nbins=20,
                title="Distribution of Ensemble Fraud Scores",
                labels={"ensemble_score": "Ensemble Score", "count": "Frequency"}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (hardcoded based on typical RF/XGB outputs)
            st.subheader(" Feature Importance")
            feature_importance = {
                "purchase_value": 0.4,
                "device_id_count": 0.3,
                "age": 0.2,
                "ip_address_count": 0.15,
                "time_to_purchase": 0.12,
                "source_Ads": 0.08,
                "browser_Chrome": 0.05,
                "sex_M": 0.03
            }
            
            importance_df = pd.DataFrame({
                "Feature": list(feature_importance.keys()),
                "Importance": list(feature_importance.values())
            }).sort_values("Importance", ascending=False)
            
            fig_bar = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Features by Importance",
                labels={"Importance": "Importance Score", "Feature": "Feature Name"}
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Threshold slider
            st.subheader(" Threshold Configuration")
            threshold = st.slider(
                "Fraud Detection Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Adjust the threshold to see how it affects classification"
            )
            
            # Dynamic classification table
            st.subheader(" Classification Preview")
            preview_df = scores_df[["transaction_id", "ensemble_score"]].copy()
            preview_df["status"] = preview_df["ensemble_score"].apply(classify_score)
            
            # Color code preview
            def color_preview(row):
                score = row["ensemble_score"]
                if score < threshold:
                    return ['background-color: #4FC978'] * len(row)  # Green for pass
                elif score < threshold + (THRESHOLD_DENY - THRESHOLD_PASS):
                    return ['background-color: #EFD033'] * len(row)  # Yellow for flag
                else:
                    return ['background-color: #c30F18'] * len(row)  # Red for deny
            
            styled_preview = preview_df.head(20).style.apply(color_preview, axis=1)
            st.dataframe(styled_preview, use_container_width=True, hide_index=True)
            
            # Model explanation
            st.subheader(" Model Information")
            st.info("""
            **The system uses an ensemble of Random Forest and XGBoost models for fraud detection.**
            
            - **Random Forest (RF)**: A tree-based ensemble method that aggregates predictions from multiple decision trees
            - **XGBoost (XGB)**: A gradient boosting framework that builds models sequentially to correct errors
            - **Ensemble Score**: The average of RF and XGB scores, providing a more robust fraud prediction
            
            The models analyze transaction patterns including purchase behavior, device usage, IP addresses, 
            and user demographics to identify potentially fraudulent activities.
            """)
            
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection error: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# -------------------
# Main App
# -------------------
def main():
    st.set_page_config(
        page_title="Fraud Detection System",
        layout="wide"
    )
    
    st.title("Fraud Detection System")
    status_text = "API Connected" if check_api_connection() else "API Unavailable"
    st.caption(status_text)
    st.markdown("---")

    pages = [
        "Submit Transaction",
        "Transaction History",
        "Alerts",
        "ML Insights"
    ]

    with st.sidebar:
        st.markdown("### Navigation")
        selected_page = st.radio(
            "Pages",
            pages,
            index=0,
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("**API Status**")
        if check_api_connection():
            st.success("Connected")
        else:
            st.error("Disconnected")

    page_map = {
        "Submit Transaction": page_submit_transaction,
        "Transaction History": page_transaction_history,
        "Alerts": page_alerts,
        "ML Insights": page_ml_insights
    }

    page_map[selected_page]()

if __name__ == "__main__":
    main()

