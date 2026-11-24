import streamlit as st
import pandas as pd
import plotly.express as px
from backend.data_loader import load_file, DataLoaderError, get_last_encoding
from data_quality.profiler import get_dataset_metadata, get_column_summary
import plotly.graph_objects as go
from logger_setup import setup_logger

# Initialize logger
logger = setup_logger("frontend.app")

# Initialize Session State at Startup
if "df" not in st.session_state:
    st.session_state.df = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# Global UI Custom Utilities and Helpers for Enterprise Layout
def get_column_icon_and_title(col: str) -> tuple[str, str]:
    col_lower = col.lower()
    title = col
    if col_lower in ["isactivemember", "active_member", "active"]:
        title = "Membership Status"
        icon = "🟢"
    elif col_lower in ["hascrcard", "has_credit_card", "credit_card"]:
        title = "Credit Card Ownership"
        icon = "💳"
    elif col_lower in ["exited", "churn", "attrition", "default"]:
        title = "Customer Churn Status"
        icon = "📈"
    elif col_lower in ["country", "geography", "region", "city", "state"]:
        title = "Geography"
        icon = "🌍"
    elif col_lower in ["gender", "sex"]:
        title = "Gender"
        icon = "👤"
    elif col_lower in ["age"]:
        title = "Age Profile"
        icon = "👥"
    elif col_lower in ["balance", "amount", "revenue", "sales", "income", "profit"]:
        title = col.replace("_", " ").title()
        icon = "💰"
    else:
        import re
        title = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', col)
        title = re.sub(r'([A-Z])([A-Z][a-z])', r'\1 \2', title)
        title = title.replace("_", " ").replace("-", " ").title()
        icon = "📍"
    return icon, title

def clean_html(html_str: str) -> str:
    return "\n".join(line.lstrip() for line in html_str.split("\n"))

def classify_filter_group(col_name: str) -> str:
    c = col_name.lower().strip()
    demographics_keywords = ["geography", "gender", "age", "country", "city", "region", "education", "marital", "status", "occupation", "job", "demographic"]
    financial_keywords = ["creditscore", "balance", "salary", "income", "assets", "debt", "financial", "revenue", "spending", "amount", "price"]
    target_keywords = ["exited", "churn", "attrition", "default", "left", "target"]
    
    if any(k in c for k in demographics_keywords):
        return "Demographics"
    elif any(k in c for k in financial_keywords):
        return "Financial"
    elif any(k in c for k in target_keywords):
        return "Target"
    else:
        return "Behavior"

def render_empty_state(icon: str, message: str, explanation: str):
    html = f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 40px 20px;
        background: var(--card-bg, #ffffff);
        border: 2px dashed var(--card-border, rgba(0, 0, 0, 0.08));
        border-radius: 12px;
        text-align: center;
        margin: 20px 0;
    ">
        <div style="font-size: 3rem; margin-bottom: 12px;">{icon}</div>
        <div style="font-weight: 700; font-size: 1.1rem; color: var(--text-primary); margin-bottom: 6px; font-family: 'Outfit', sans-serif;">{message}</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); max-width: 400px; margin: 0 auto; font-family: 'Outfit', sans-serif;">{explanation}</div>
    </div>
    """
    st.markdown(clean_html(html), unsafe_allow_html=True)

def get_stepper_html(current_step: int, steps: list) -> str:
    html = """
    <div class="loading-stepper-container">
        <div class="stepper-header">
            <div class="stepper-spinner"></div>
            <span>Executing Analytical Pipeline</span>
        </div>
    """
    for idx, step_name in enumerate(steps):
        if idx < current_step:
            status_class = "completed"
            badge = "✅"
        elif idx == current_step:
            status_class = "active"
            badge = "<div class='stepper-spinner'></div>"
        else:
            status_class = "pending"
            badge = "⚪"
            
        html += f"""
        <div class="stepper-item {status_class}">
            <div class="stepper-badge">{badge}</div>
            <div class="stepper-name">{step_name}</div>
        </div>
        """
    html += "</div>"
    return clean_html(html)

def render_kpi_cards(df: pd.DataFrame, meta: dict, report: dict, target_col: str):
    # 1. Target Rate / Churn Rate / Attrition Rate
    target_rate_val = None
    target_label = "Target Rate"
    target_badge = "Active"
    target_subtext = "Favorable baseline rate"
    
    if target_col and report:
        domain_type = report["domain_analysis"]["domain_type"]
        if domain_type == "churn":
            target_rate_val = report["domain_analysis"]["metrics"].get("churn_rate")
            target_label = "Churn Rate"
        elif domain_type == "hr_attrition":
            target_rate_val = report["domain_analysis"]["metrics"].get("attrition_rate")
            target_label = "Attrition Rate"
        else:
            target_rate_val = report["domain_analysis"]["metrics"].get("target_rate")
            if target_rate_val is None:
                cleaned = df[target_col].dropna()
                mapped = cleaned.map(lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1", "1.0", "y", "t", "exited", "left", "cancelled", "churned"] else 0)
                total_valid = df[target_col].notnull().sum()
                target_rate_val = (mapped.sum() / total_valid * 100) if total_valid > 0 else 0.0
            target_label = "Target Rate"
            
        if target_rate_val is not None:
            if target_rate_val > 15.0:
                target_subtext = "Threshold exceeded"
            else:
                target_subtext = "Favorable rate"
                
    target_val_str = f"{target_rate_val:.2f}%" if target_rate_val is not None else "N/A"
    
    # 2. Data Quality Index (DQI)
    density_val = meta.get('data_density', '100%')
    try:
        density_num = float(density_val.replace('%', ''))
    except:
        density_num = 100.0
    
    dqi_val_str = f"{density_num:.1f}%"
    dqi_badge_class = "success" if density_num >= 98.0 else ("warning" if density_num >= 90.0 else "danger")
    dqi_badge_text = "Optimal" if dqi_badge_class == "success" else ("Warning" if dqi_badge_class == "warning" else "Critical")
    dqi_subtext = "cell coverage"
        
    # 3. Data density, missing, duplicates
    missing_val = meta.get('missing_cells', 0)
    dup_val = meta.get('duplicate_rows', 0)
    
    # Status badges map
    target_badge_class = "danger" if target_rate_val is not None and target_rate_val > 15.0 else "success"
    target_badge_text = "High Risk" if target_badge_class == "danger" else "Favorable"
    
    density_badge_class = "success"
    if density_val != '100%':
        try:
            val_num = float(density_val.replace('%',''))
            if val_num < 95.0:
                density_badge_class = "warning"
        except:
            pass
    density_badge_text = "Optimal" if density_badge_class == "success" else "Warning"
    
    missing_badge_class = "success" if missing_val == 0 else "danger"
    missing_badge_text = "Perfect" if missing_val == 0 else "Action Required"
    missing_subtext = "No null values" if missing_val == 0 else f"{missing_val:,} cells to verify"
    
    dup_badge_class = "success" if dup_val == 0 else "warning"
    dup_badge_text = "Clean" if dup_val == 0 else "Warning"
    dup_subtext = "No duplicates" if dup_val == 0 else f"{dup_val:,} rows detected"

    # Render HTML grid
    kpis_html = f"""
    <div class="enterprise-kpi-grid">
        <div class="enterprise-kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Total Records</span>
                <span class="kpi-icon">📊</span>
            </div>
            <div class="kpi-value">{len(df):,}</div>
            <div class="kpi-footer">
                <span class="kpi-trend info">Active</span>
                <span class="kpi-subtitle">filtered scope</span>
            </div>
        </div>
        <div class="enterprise-kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">{target_label}</span>
                <span class="kpi-icon">📈</span>
            </div>
            <div class="kpi-value">{target_val_str}</div>
            <div class="kpi-footer">
                <span class="kpi-trend {target_badge_class}">{target_badge_text}</span>
                <span class="kpi-subtitle">{target_subtext}</span>
            </div>
        </div>
        <div class="enterprise-kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Data Quality Index</span>
                <span class="kpi-icon">🎯</span>
            </div>
            <div class="kpi-value">{dqi_val_str}</div>
            <div class="kpi-footer">
                <span class="kpi-trend {dqi_badge_class}">{dqi_badge_text}</span>
                <span class="kpi-subtitle">{dqi_subtext}</span>
            </div>
        </div>
        <div class="enterprise-kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Total Features</span>
                <span class="kpi-icon">📐</span>
            </div>
            <div class="kpi-value">{meta['columns']:,}</div>
            <div class="kpi-footer">
                <span class="kpi-trend info">Baseline</span>
                <span class="kpi-subtitle">dataset attributes</span>
            </div>
        </div>
        <div class="enterprise-kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Data Density</span>
                <span class="kpi-icon">🟢</span>
            </div>
            <div class="kpi-value">{density_val}</div>
            <div class="kpi-footer">
                <span class="kpi-trend {density_badge_class}">{density_badge_text}</span>
                <span class="kpi-subtitle">completeness</span>
            </div>
        </div>
        <div class="enterprise-kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Missing Cells</span>
                <span class="kpi-icon">⚠️</span>
            </div>
            <div class="kpi-value">{missing_val:,}</div>
            <div class="kpi-footer">
                <span class="kpi-trend {missing_badge_class}">{missing_badge_text}</span>
                <span class="kpi-subtitle">{missing_subtext}</span>
            </div>
        </div>
        <div class="enterprise-kpi-card">
            <div class="kpi-top">
                <span class="kpi-title">Duplicate Rows</span>
                <span class="kpi-icon">👥</span>
            </div>
            <div class="kpi-value">{dup_val:,}</div>
            <div class="kpi-footer">
                <span class="kpi-trend {dup_badge_class}">{dup_badge_text}</span>
                <span class="kpi-subtitle">{dup_subtext}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(kpis_html, unsafe_allow_html=True)

def render_recommendation_cards(recs_list: list):
    cards_html = "<div style='display:grid;grid-template-columns:repeat(auto-fit, minmax(320px, 1fr));gap:16px;width:100%;margin-top:14px;'>"
    for idx, rec in enumerate(recs_list[:6]):
        if isinstance(rec, dict):
            # Dynamic structured recommendation
            p_lvl = str(rec.get("implementation_priority", "MEDIUM")).upper()
            if p_lvl == "HIGH":
                p_class = "danger"
                p_text = "● High Priority"
            elif p_lvl == "LOW":
                p_class = "info"
                p_text = "● Low Priority"
            else:
                p_class = "warning"
                p_text = "● Medium Priority"
                
            problem = rec.get("problem", "")
            action = rec.get("recommendation", "")
            benefit = rec.get("expected_benefit", "")
            roi_label = "Expected ROI"
            roi = benefit
            impact = problem
            improvement = benefit
        else:
            rec_lower = rec.lower()
            if "germany" in rec_lower:
                p_class = "danger"
                p_text = "● High Priority"
                impact = "Reduce Germany churn rate and prevent loss"
                roi = "+12%"
                roi_label = "Expected ROI"
                improvement = "Lower Germany Churn by 12%"
                action = rec
            elif "member" in rec_lower or "membership" in rec_lower:
                p_class = "danger"
                p_text = "● High Priority"
                impact = "Optimize active membership conversion rates"
                roi = "+8%"
                roi_label = "Expected Improvement"
                improvement = "Increase Active Membership Share by 8%"
                action = rec
            elif "credit card" in rec_lower or "crcard" in rec_lower:
                p_class = "warning"
                p_text = "● Medium Priority"
                impact = "Boost credit card ownership and client lifetime value"
                roi = "+$65K"
                roi_label = "Expected ROI"
                improvement = "Optimize CrCard ownership value gap"
                action = rec
            elif "balance" in rec_lower or "amount" in rec_lower or "revenue" in rec_lower:
                p_class = "danger"
                p_text = "● High Priority"
                impact = "Retain high-asset customer balance and prevent wealth outflow"
                roi = "+15%"
                roi_label = "Expected Impact"
                improvement = "15% Improvement in Balance Retention"
                action = rec
            elif "duplicate" in rec_lower or "missing" in rec_lower or "quality" in rec_lower:
                p_class = "warning"
                p_text = "● Medium Priority"
                impact = "Fix missing cells and resolve duplicate rows"
                roi = "100%"
                roi_label = "Expected Completeness"
                improvement = "100% Data Quality Ingestion Match"
                action = rec
            else:
                if idx <= 1:
                    p_class = "danger"
                    p_text = "● High Priority"
                    impact = "Optimize demographic segment strategies"
                    roi = "+10%"
                    roi_label = "Expected ROI"
                    improvement = "10% Higher Customer Engagement"
                elif idx <= 3:
                    p_class = "warning"
                    p_text = "● Medium Priority"
                    impact = "Address high churn risks across low engagement categories"
                    roi = "+5%"
                    roi_label = "Expected ROI"
                    improvement = "5% Risk Reduction"
                else:
                    p_class = "info"
                    p_text = "● Low Priority"
                    impact = "Refine overall process efficiency and performance tracking"
                    roi = "+3%"
                    roi_label = "Expected ROI"
                    improvement = "3% Operating Efficiency Uplift"
                action = rec

        cards_html += f"""
        <div class="mckinsey-card">
            <div class="kpi-top" style="margin-bottom:10px;">
                <span class="kpi-title" style="font-size:0.75rem;">Action Brief {idx+1}</span>
                <span class="kpi-badge {p_class}">{p_text}</span>
            </div>
            <div style="font-size:0.92rem;font-weight:700;color:var(--text-primary);margin-bottom:6px;font-family:'Outfit';">{impact}</div>
            <div style="font-size:0.82rem;color:var(--text-secondary);margin-bottom:12px;line-height:1.4;font-family:'Inter';">
                <strong>{roi_label}:</strong> <span style="color:var(--success-color);font-weight:600;">{roi}</span><br/>
                <strong>Focus Area:</strong> {improvement}
            </div>
            <div style="background:var(--workspace-bg);border:1px solid var(--card-border);border-radius:8px;padding:12px;font-size:0.82rem;color:var(--text-primary);line-height:1.5;font-family:'Inter';">
                <strong>Suggested Action:</strong><br/>
                {action}
            </div>
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

def render_timeline_recommendations(recs: list):
    short_term = []
    med_term = []
    long_term = []
    for idx, r in enumerate(recs):
        # Handle dict or string recommendations
        rec_str = r.get("recommendation", "") if isinstance(r, dict) else str(r)
        if idx % 3 == 0:
            short_term.append(rec_str)
        elif idx % 3 == 1:
            med_term.append(rec_str)
        else:
            long_term.append(rec_str)
            
    html = """
    <div class="consulting-timeline">
        <div class="timeline-lane">
            <div class="lane-header short-term">
                <span class="lane-badge">0-3 Months</span>
                <span class="lane-title">Horizon 1: Quick Wins & Operational Fixes</span>
            </div>
            <div class="lane-body">
    """
    for item in short_term:
        html += f'<div class="timeline-item"><span class="bullet"></span><span class="item-text">{item}</span></div>'
    html += """
            </div>
        </div>
        <div class="timeline-lane">
            <div class="lane-header med-term">
                <span class="lane-badge">3-6 Months</span>
                <span class="lane-title">Horizon 2: Tactical Campaigns & Program Launch</span>
            </div>
            <div class="lane-body">
    """
    for item in med_term:
        html += f'<div class="timeline-item"><span class="bullet"></span><span class="item-text">{item}</span></div>'
    html += """
            </div>
        </div>
        <div class="timeline-lane">
            <div class="lane-header long-term">
                <span class="lane-badge">6+ Months</span>
                <span class="lane-title">Horizon 3: Strategic Structural Upgrades</span>
            </div>
            <div class="lane-body">
    """
    for item in long_term:
        html += f'<div class="timeline-item"><span class="bullet"></span><span class="item-text">{item}</span></div>'
    html += """
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# Page Configuration
st.set_page_config(
    page_title="Senior Data Analyst Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to generate mock sample data
def get_sample_dataframe() -> pd.DataFrame:
    """Generates a representative sample dataframe with diverse dtypes, missing values, and duplicates."""
    data = {
        "Transaction ID": [1001, 1002, 1003, 1004, 1005, 1005, 1007, 1008, 1009, 1010],
        "Customer Name": ["Alice Jones", "Bob Smith", "Charlie Brown", None, "Eva Davis", "Eva Davis", "Grace Lee", "Henry Wilson", "Ivy Taylor", "Jack Miller"],
        "Purchase Date": [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-02"), pd.Timestamp("2026-06-03"), pd.Timestamp("2026-06-04"), pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-05"), pd.Timestamp("2026-06-07"), pd.Timestamp("2026-06-08"), pd.Timestamp("2026-06-09"), pd.Timestamp("2026-06-10")],
        "Product Category": ["Electronics", "Clothing", "Home", "Electronics", "Clothing", "Clothing", None, "Home", "Electronics", "Home"],
        "Amount (USD)": [120.50, 45.00, None, 250.00, 89.99, 89.99, 15.75, 110.00, 310.25, None],
        "Is Member": [True, False, True, True, False, False, True, None, True, False]
    }
    return pd.DataFrame(data)

# Premium Enterprise SaaS CSS styling
st.markdown(
    clean_html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #3B82F6; /* Premium Blue Accent */
        --primary-light: rgba(59, 130, 246, 0.1);
        --primary-hover: #2563EB;
        --success-color: #10B981;
        --success-light: rgba(16, 185, 129, 0.1);
        --warning-color: #F59E0B;
        --warning-light: rgba(245, 158, 11, 0.1);
        --danger-color: #EF4444;
        --danger-light: rgba(239, 68, 68, 0.1);
        --info-color: #7C3AED;
        --card-bg: #161B22; /* Dark Card Background */
        --card-border: rgba(255, 255, 255, 0.08);
        --card-border-light: rgba(255, 255, 255, 0.04);
        --card-radius: 18px; /* 18px rounded corners */
        --card-padding: 20px;
        --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        --card-shadow-hover: 0 10px 30px rgba(59, 130, 246, 0.15);
        --text-primary: #F3F4F6;
        --text-secondary: #9CA3AF;
        --sidebar-bg: #111827; /* Dark Sidebar */
        --workspace-bg: #0B0F19; /* Main background */
    }

    /* Hide default Streamlit elements for pure software look */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    footer {
        visibility: hidden;
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    button[data-testid="stSidebarCollapse"] {
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }

    /* Global typography overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--workspace-bg);
    }
    h1, h2, h3, h4, h5, h6, .outfit-font {
        font-family: 'Outfit', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
    }
    
    /* Sticky Header parent styling */
    div[data-testid="stVerticalBlock"] > div:has(> div > .sticky-header-marker) {
        position: sticky !important;
        top: 0px !important;
        z-index: 1000 !important;
        background: var(--card-bg) !important;
        border-bottom: 1px solid var(--card-border) !important;
        padding: 10px 24px !important;
        margin-left: -2.0rem !important;
        margin-right: -2.0rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04) !important;
    }

    /* Main container settings */
    .block-container {
        padding-top: 0.0rem !important; /* Top padding is handled by header */
        padding-bottom: 3.0rem !important;
        max-width: 1440px !important;
        background-color: var(--workspace-bg);
    }

    /* Sidebar modifications to white theme */
    [data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--card-border) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.0rem !important;
    }

    /* Sidebar premium navigation buttons */
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 12px !important;
        padding: 10px 16px !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-left: 3px solid transparent !important;
        border-radius: 8px !important;
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        transition: all 0.22s ease !important;
        box-shadow: none !important;
        margin-bottom: 4px !important;
        font-family: 'Outfit', sans-serif !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: var(--primary-light) !important;
        color: var(--primary-color) !important;
        transform: translateX(4px) !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:has(div:contains("▸")) {
        background: var(--primary-light) !important;
        color: var(--primary-color) !important;
        border-left: 3px solid var(--primary-color) !important;
        font-weight: 600 !important;
    }

    /* Sidebar brand / logo styling */
    .sidebar-brand-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 15px 0;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--card-border);
    }
    .sidebar-logo {
        width: 44px;
        height: 44px;
        margin-bottom: 8px;
        filter: drop-shadow(0 0 12px rgba(59, 130, 246, 0.5));
    }
    .sidebar-brand-name {
        font-family: 'Outfit', sans-serif;
        font-size: 1.25rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: 2px;
    }
    .sidebar-brand-sub {
        font-family: 'Outfit', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Uploader panel and sidebar block border styling */
    [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--card-border) !important;
        border-radius: 10px !important;
        background: var(--workspace-bg) !important;
        padding: 6px !important;
    }

    /* Native containers stylized as rounded SaaS cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: var(--card-radius) !important;
        padding: var(--card-padding) !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 16px !important;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: var(--card-shadow-hover) !important;
        border-color: rgba(15, 82, 186, 0.15) !important;
    }

    /* Equal height KPI layout */
    .enterprise-kpi-grid {
        display: grid !important;
        grid-template-columns: repeat(7, 1fr) !important;
        gap: 16px !important;
        margin-bottom: 24px !important;
        width: 100% !important;
    }
    @media (max-width: 1200px) {
        .enterprise-kpi-grid { grid-template-columns: repeat(4, 1fr) !important; }
    }
    @media (max-width: 768px) {
        .enterprise-kpi-grid { grid-template-columns: repeat(2, 1fr) !important; }
    }
    @media (max-width: 480px) {
        .enterprise-kpi-grid { grid-template-columns: 1fr !important; }
    }

    .enterprise-kpi-card {
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: var(--card-shadow) !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: space-between !important;
        height: 135px !important;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
    }
    .enterprise-kpi-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: var(--card-shadow-hover) !important;
        border-color: rgba(15, 82, 186, 0.25) !important;
    }
    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    .kpi-title {
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-family: 'Outfit', sans-serif;
    }
    .kpi-icon {
        font-size: 1.15rem;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--text-primary);
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.5px;
        line-height: 1.1;
        margin: 4px 0;
    }
    .kpi-footer {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
    }
    .kpi-trend {
        display: inline-flex;
        align-items: center;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .kpi-trend.success { background: rgba(16, 185, 129, 0.08); color: var(--success-color); border: 1px solid rgba(16, 185, 129, 0.15); }
    .kpi-trend.warning { background: rgba(245, 158, 11, 0.08); color: var(--warning-color); border: 1px solid rgba(245, 158, 11, 0.15); }
    .kpi-trend.danger { background: rgba(239, 68, 68, 0.08); color: var(--danger-color); border: 1px solid rgba(239, 68, 68, 0.15); }
    .kpi-trend.info { background: rgba(59, 130, 246, 0.08); color: var(--info-color); border: 1px solid rgba(59, 130, 246, 0.15); }
    .kpi-subtitle {
        color: var(--text-secondary);
        font-weight: 400;
    }

    /* Executive report timelines and badging */
    .consulting-timeline {
        display: flex;
        flex-direction: column;
        gap: 20px;
        width: 100%;
        margin: 20px 0;
    }
    .timeline-lane {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        box-shadow: var(--card-shadow);
        overflow: hidden;
    }
    .lane-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 12px 20px;
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    .lane-header.short-term { background: linear-gradient(135deg, #0F52BA, #3B82F6); }
    .lane-header.med-term { background: linear-gradient(135deg, #8B5CF6, #EC4899); }
    .lane-header.long-term { background: linear-gradient(135deg, #10B981, #059669); }
    .lane-badge {
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .lane-title { font-size: 0.95rem; }
    .lane-body {
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .timeline-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    .timeline-item .bullet {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary-color);
        margin-top: 6px;
        flex-shrink: 0;
    }
    .timeline-item .item-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        color: var(--text-primary);
        line-height: 1.4;
    }

    /* Executive sticky top header layout elements */
    .sticky-header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        font-family: 'Outfit', sans-serif;
    }
    .hdr-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: 1.5px;
    }
    .hdr-sub {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--primary-color);
        background: var(--primary-light);
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 6px;
    }
    .hdr-dataset {
        font-size: 0.84rem;
        color: var(--text-secondary);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .hdr-status-pill {
        font-size: 0.65rem;
        font-weight: 700;
        padding: 1px 6px;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .hdr-status-pill.success { background: rgba(16, 185, 129, 0.1); color: var(--success-color); border: 1px solid rgba(16, 185, 129, 0.2); }
    .hdr-status-pill.idle { background: rgba(100, 116, 139, 0.1); color: var(--text-secondary); border: 1px solid rgba(100, 116, 139, 0.2); }
    .hdr-time {
        font-size: 0.82rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-align: right;
    }

    /* Chat bubble layout stylings */
    .chat-bubble-container {
        display: flex;
        gap: 14px;
        margin-bottom: 18px;
        width: 100%;
        align-items: flex-start;
    }
    .chat-bubble-container.user {
        flex-direction: row-reverse;
    }
    .chat-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: var(--primary-light);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        border: 1px solid var(--card-border);
    }
    .chat-bubble-container.user .chat-avatar {
        background: var(--primary-color);
        color: #ffffff;
    }
    .chat-bubble {
        max-width: 75%;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: var(--card-shadow);
    }
    .chat-bubble-container.user .chat-bubble {
        background: var(--primary-light);
        border-color: rgba(15, 82, 186, 0.15);
    }
    .chat-role {
        font-family: 'Outfit', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-secondary);
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .chat-text {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: var(--text-primary);
        line-height: 1.5;
    }
    .chat-actions {
        display: flex;
        gap: 8px;
        margin-top: 10px;
        padding-top: 8px;
        border-top: 1px solid var(--card-border);
    }
    .chat-action-btn {
        background: transparent;
        border: 1px solid var(--card-border);
        border-radius: 6px;
        padding: 4px 10px;
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--text-secondary);
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .chat-action-btn:hover {
        border-color: var(--primary-color);
        color: var(--primary-color);
        background: var(--primary-light);
    }

    /* Typing indicators */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 5px;
        height: 17px;
    }
    .typing-indicator span {
        width: 6px;
        height: 6px;
        background: var(--primary-color);
        border-radius: 50%;
        animation: typing-pulse 1.4s infinite ease-in-out both;
    }
    .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
    .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes typing-pulse {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1.0); }
    }

    /* Business insight grids */
    .insights-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 20px;
        margin-top: 16px;
        width: 100%;
    }
    .insight-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        box-shadow: var(--card-shadow);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .insight-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--card-shadow-hover);
        border-color: rgba(15, 82, 186, 0.15) !important;
    }
    .insight-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }
    .insight-icon { font-size: 1.4rem; }
    .insight-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    .insight-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.84rem;
        color: var(--text-secondary);
        margin-bottom: 12px;
    }
    .insight-impact-container {
        background: var(--workspace-bg);
        border-radius: 8px;
        padding: 10px 14px;
        border-left: 3px solid var(--primary-color);
        margin-bottom: 16px;
    }
    .insight-impact-label {
        font-family: 'Outfit', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: var(--primary-color);
        display: block;
        margin-bottom: 4px;
    }
    .insight-impact {
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        color: var(--text-primary);
        line-height: 1.4;
        margin: 0;
    }
    .insight-footer {
        display: flex;
        gap: 8px;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .badge-danger { background: rgba(239, 68, 68, 0.06); color: var(--danger-color); border: 1px solid rgba(239, 68, 68, 0.12); }
    .badge-warning { background: rgba(245, 158, 11, 0.06); color: var(--warning-color); border: 1px solid rgba(245, 158, 11, 0.12); }
    .badge-success { background: rgba(16, 185, 129, 0.06); color: var(--success-color); border: 1px solid rgba(16, 185, 129, 0.12); }
    .badge-info { background: rgba(59, 130, 246, 0.06); color: var(--info-color); border: 1px solid rgba(59, 130, 246, 0.12); }

    /* McKinsey cards styling for other metrics */
    .mckinsey-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 20px;
        box-shadow: var(--card-shadow);
        transition: transform 0.22s ease, box-shadow 0.22s ease;
    }
    .mckinsey-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--card-shadow-hover);
        border-color: rgba(15, 82, 186, 0.15) !important;
    }

    /* Stepper Loading Experience */
    .loading-stepper-container {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 28px;
        max-width: 480px;
        margin: 60px auto;
        box-shadow: var(--card-shadow);
        font-family: 'Inter', sans-serif;
    }
    .stepper-header {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .stepper-item {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 14px;
        opacity: 0.4;
        transition: all 0.3s ease;
    }
    .stepper-item:last-child {
        margin-bottom: 0;
    }
    .stepper-item.active {
        opacity: 1;
        font-weight: 600;
    }
    .stepper-item.completed {
        opacity: 0.85;
    }
    .stepper-badge {
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    .stepper-name {
        font-size: 0.9rem;
        color: var(--text-primary);
    }
    .stepper-spinner {
        width: 14px;
        height: 14px;
        border: 2px solid var(--primary-color);
        border-top: 2px solid transparent;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* PowerPoint preview elements */
    .deck-preview-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        width: 100%;
        margin-top: 20px;
    }
    @media (max-width: 900px) {
        .deck-preview-grid { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 600px) {
        .deck-preview-grid { grid-template-columns: 1fr; }
    }
    .slide-thumbnail {
        background: #f1f5f9;
        border: 1px solid var(--card-border);
        border-radius: 8px;
        aspect-ratio: 16 / 9;
        padding: 12px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .slide-thumbnail::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 4px;
        background: var(--primary-color);
    }
    .slide-title {
        font-family: 'Outfit', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .slide-body-mock {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        gap: 4px;
        margin-top: 8px;
    }
    .slide-line-mock {
        height: 4px;
        background: #cbd5e1;
        border-radius: 2px;
    }
    .slide-line-mock.short { width: 40%; }
    .slide-line-mock.med { width: 70%; }
    .slide-line-mock.long { width: 90%; }
    .slide-footer-mock {
        font-size: 0.6rem;
        color: #94a3b8;
        text-align: right;
    }

    /* ── Divider ─────────────────────────────────────────────── */
    hr {
        border-color: var(--card-border) !important;
        margin: 24px 0 !important;
    }

    /* ── Scrollbar ───────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(100,116,139,0.2);
        border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary-color) !important; }
    
    /* Style chatbot suggested question buttons */
    div.stButton > button:has(div:contains("❓")) {
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        color: var(--text-secondary) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        box-shadow: var(--card-shadow) !important;
        transition: all 0.2s ease !important;
        height: 52px !important;
    }
    div.stButton > button:has(div:contains("❓")):hover {
        background-color: var(--primary-light) !important;
        border-color: rgba(15, 82, 186, 0.2) !important;
        color: var(--primary-color) !important;
        transform: translateY(-2px) !important;
        box-shadow: var(--card-shadow-hover) !important;
    }
    
    /* Native spinner color override */
    div[data-testid="stSpinner"] > div {
        border-top-color: var(--primary-color) !important;
    }
    </style>
    """),
    unsafe_allow_html=True
)

# Global Sticky Top Bar Header in Main Workspace
with st.container():
    st.markdown('<div class="sticky-header-marker"></div>', unsafe_allow_html=True)
    import datetime
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    col_hdr_l, col_hdr_m, col_hdr_r = st.columns([5, 4, 3])
    with col_hdr_l:
        st.markdown('<div class="hdr-title">🏢 VALKYRIE <span class="hdr-sub">INTELLIGENCE</span></div>', unsafe_allow_html=True)
    with col_hdr_m:
        if st.session_state.filename:
            st.markdown(f'<div class="hdr-dataset">📂 {st.session_state.filename} <span class="hdr-status-pill success">Connected</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="hdr-dataset">📂 No Active Dataset <span class="hdr-status-pill idle">Offline</span></div>', unsafe_allow_html=True)
    with col_hdr_r:
        st.markdown(f'<div class="hdr-time">🕒 {now_str}</div>', unsafe_allow_html=True)

def st_plotly_chart(fig, key=None):
    """
    Renders a Plotly figure with high-resolution export configuration,
    disabling the logo and removing unused tools.
    """
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        config={
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'valkyrie_analytics_export',
                'height': 600,
                'width': 900,
                'scale': 2
            }
        }
    )

def render_chart_with_fallback(chart_func, *args, key=None, **kwargs):
    """
    Renders any Plotly fig using chart_func, validating that df is non-empty 
    and required columns exist. Displays clean fallbacks on validation failure.
    """
    try:
        df_arg = args[0] if len(args) > 0 else None
        
        # 1. Check if df is empty
        if df_arg is None or (isinstance(df_arg, pd.DataFrame) and df_arg.empty):
            render_empty_state("🔍", "No data after filters", "The selected filters returned 0 records. Please adjust your criteria.")
            return
            
        # 2. Extract column names from arguments to check presence
        cols_to_check = []
        for arg in args[1:]:
            if isinstance(arg, str):
                cols_to_check.append(arg)
        for val in kwargs.values():
            if isinstance(val, str) and isinstance(df_arg, pd.DataFrame) and val in df_arg.columns:
                cols_to_check.append(val)
                
        # Validate columns exist in df
        if isinstance(df_arg, pd.DataFrame):
            for col in cols_to_check:
                if col not in df_arg.columns:
                    render_empty_state("📊", "No visualization available", f"The required column '{col}' is not present in the active dataset.")
                    return
                    
        # 3. Call the visualization engine function
        fig = chart_func(*args, **kwargs)
        
        # 4. Check if figure is valid and non-empty
        if fig is None or not hasattr(fig, "data") or not fig.data:
            render_empty_state("📊", "No visualization available", "The analytical data could not be visualized.")
        else:
            st_plotly_chart(fig, key=key)
            
    except Exception as e:
        logger.error(f"Error rendering chart: {str(e)}", exc_info=True)
        render_empty_state("⚠️", "Visualization Error", f"An unexpected error occurred: {str(e)}")

def render_time_trend_with_fallback(time_res, metric_col, date_col, key=None):
    """
    Renders time trend intelligence chart safely with fallback message.
    """
    try:
        if not time_res or not isinstance(time_res, dict) or "time_series" not in time_res or time_res["time_series"].empty:
            render_empty_state("🕒", "No date column available", "A date column is required to render temporal analysis projections.")
            return
        fig = create_time_trend_chart(time_res, metric_col, date_col)
        if fig is None or not hasattr(fig, "data") or not fig.data:
            render_empty_state("📊", "No visualization available", "Time Intelligence trend could not be rendered.")
        else:
            st_plotly_chart(fig, key=key)
    except Exception as e:
        logger.error(f"Error rendering time trend chart: {str(e)}", exc_info=True)
        render_empty_state("⚠️", "Visualization Error", f"An unexpected error occurred: {str(e)}")

# App Title & Description — Premium Hero Banner
st.markdown("""
<div class="dashboard-hero">
    <div style="font-size:0.75rem;font-weight:700;opacity:0.72;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;font-family:'Outfit',sans-serif;">🏢 Enterprise Intelligence Platform</div>
    <h1>📊 Senior Data Analyst Platform</h1>
    <p>Upload any CSV or Excel dataset to unlock automated analytics, predictive modelling, executive insights, and AI-powered recommendations — powered by your local analytical engines.</p>
</div>
""", unsafe_allow_html=True)

# Session state initialized at startup

# Sidebar Brand Header
st.sidebar.markdown("""
<div class="sidebar-brand-container">
    <svg class="sidebar-logo" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="url(#grad1)" />
        <path d="M2 17L12 22L22 17" stroke="url(#grad1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        <path d="M2 12L12 17L22 12" stroke="url(#grad1)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#3B82F6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#0F52BA;stop-opacity:1" />
            </linearGradient>
        </defs>
    </svg>
    <div class="sidebar-brand-name">VALKYRIE</div>
    <div class="sidebar-brand-sub">Enterprise Data Intelligence</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='font-size:0.8rem;font-weight:700;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;font-family:Outfit,sans-serif;'>📁 Data Source</div>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Upload your dataset here. Maximum file size is 200MB."
)

# Sample Data trigger button
load_sample = st.sidebar.button(
    "💡 Load Sample Dataset",
    help="Test the platform features immediately with a pre-configured sample dataset.",
    use_container_width=True
)

local_path = st.sidebar.text_input(
    "💡 Local Path (Dev Upload)",
    "",
    help="Enter local workspace file path to load directly, e.g., 'data/samples/sales_dataset.csv'"
)

if local_path:
    if st.session_state.filename != local_path:
        import os
        if os.path.exists(local_path):
            try:
                logger.info(f"Loading local file: {local_path}")
                with st.spinner(f"Ingesting {local_path}..."):
                    df = load_file(local_path, local_path)
                    st.session_state.df = df
                    st.session_state.filename = local_path
                    st.session_state.loaded_encoding = get_last_encoding()
                    st.sidebar.success("Loaded successfully!")
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to load: {str(e)}")
        else:
            st.sidebar.error("File does not exist.")
elif uploaded_file is not None:
    # Check if a new file is uploaded
    if st.session_state.filename != uploaded_file.name:
        try:
            logger.info(f"New file uploaded by user: {uploaded_file.name}")
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                # Load the file
                df = load_file(uploaded_file, uploaded_file.name)
                # Store in session state
                st.session_state.df = df
                st.session_state.filename = uploaded_file.name
                st.session_state.loaded_encoding = get_last_encoding()
                st.success("File uploaded and ingested successfully!")
        except DataLoaderError as e:
            st.sidebar.error(f"Ingestion failed: {str(e)}")
            logger.error(f"Failed to ingest file {uploaded_file.name}: {str(e)}")
            st.session_state.df = None
            st.session_state.filename = None
            st.session_state.loaded_encoding = None
        except Exception as e:
            st.sidebar.error(f"An unexpected error occurred during ingestion.")
            logger.error(f"Unexpected error loading file {uploaded_file.name}: {str(e)}", exc_info=True)
            st.session_state.df = None
            st.session_state.filename = None
            st.session_state.loaded_encoding = None

elif load_sample:
    logger.info("User requested sample dataset loading.")
    df_sample = get_sample_dataframe()
    st.session_state.df = df_sample
    st.session_state.filename = "sample_dataset.csv"
    st.session_state.loaded_encoding = None
    st.sidebar.success("Sample dataset loaded successfully!")

# Reset Workspace action (clear dataset)
st.sidebar.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
if st.sidebar.button("🗑️ Clear Dataset & Upload New", help="Remove current dataset and return to the upload screen", use_container_width=True):
    st.session_state.df = None
    st.session_state.filename = None
    st.session_state.loaded_encoding = None
    st.rerun()

# Check if dataset is loaded
df = st.session_state.df

if df is not None:
    from analytics import (
        get_numeric_columns,
        get_categorical_columns,
        get_numeric_summary,
        get_categorical_distribution,
        generate_numeric_insights,
        generate_categorical_insights,
        generate_cross_insights,
        get_semantic_roles,
        perform_rca,
        perform_time_analysis,
        generate_executive_report,
        get_analyst_response,
        generate_pdf,
        extract_risk_level,
        generate_presentation
    )
    from analytics.entity_investigation import run_entity_investigation
    from visualization import (
        create_distribution_plot,
        create_categorical_bar_chart,
        create_categorical_pie_chart,
        create_cross_variable_bar_chart,
        create_rca_contribution_chart,
        create_time_trend_chart,
        create_correlation_heatmap,
        create_box_plot_by_target,
        create_scatter_plot,
        create_horizontal_bar_chart,
        apply_enterprise_layout
    )
    
    if st.session_state.get("loaded_encoding"):
        st.markdown(f"""
<div style="background:var(--primary-light);border:1px solid rgba(37,99,235,0.2);border-radius:10px;padding:10px 16px;display:flex;align-items:center;gap:10px;margin-bottom:16px;font-family:'Outfit',sans-serif;">
    <span style="font-size:1.2rem;">📂</span>
    <div>
        <div style="font-size:0.82rem;font-weight:700;color:var(--text-primary);">{st.session_state.filename}</div>
        <div style="font-size:0.74rem;color:var(--text-secondary);">Encoding: {st.session_state.loaded_encoding}</div>
    </div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div style="background:var(--primary-light);border:1px solid rgba(37,99,235,0.2);border-radius:10px;padding:10px 16px;display:flex;align-items:center;gap:10px;margin-bottom:16px;font-family:'Outfit',sans-serif;">
    <span style="font-size:1.2rem;">📂</span>
    <div>
        <div style="font-size:0.82rem;font-weight:700;color:var(--text-primary);">{st.session_state.filename}</div>
        <div style="font-size:0.74rem;color:var(--text-secondary);">Active Dataset</div>
    </div>
</div>""", unsafe_allow_html=True)
    
    # 1. Classify semantic roles
    try:
        semantic_roles = get_semantic_roles(df)
    except Exception as e:
        logger.error(f"Error classifying semantic roles: {str(e)}", exc_info=True)
        semantic_roles = {col: "Metric" if pd.api.types.is_numeric_dtype(df[col].dtype) else "Category" for col in df.columns}

    # 2. Sidebar Filters with Professional Redesign
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
<div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;'>
    <div style='width:30px;height:30px;border-radius:8px;background:var(--primary-light);display:flex;align-items:center;justify-content:center;font-size:1.0rem;color:var(--primary-color);'>🔎</div>
    <div style='font-size:1.05rem;font-weight:800;color:var(--text-primary);font-family:Outfit,sans-serif;letter-spacing:-0.2px;'>Dashboard Filters</div>
</div>""", unsafe_allow_html=True)

    filterable_cols = [
        col for col in get_categorical_columns(df)
        if semantic_roles.get(col) in ["Category", "Segment", "Boolean"]
        and 2 <= df[col].dropna().nunique() <= 25
    ]

    active_filters = {}

    def reset_all_filters_state():
        for c in filterable_cols:
            st.session_state[f"filter_{c}"] = []
            if c in df.columns:
                for val in df[c].dropna().unique():
                    st.session_state.pop(f"chk_{c}_{val}", None)
                    st.session_state.pop(f"multi_{c}", None)

    col_act1, col_act2 = st.sidebar.columns(2)
    with col_act1:
        if st.button("🧹 Clear Filters", key="btn_clear_all_filters", use_container_width=True):
            reset_all_filters_state()
            st.rerun()
    with col_act2:
        if st.button("🔄 Reset Filters", key="btn_reset_filters", use_container_width=True):
            reset_all_filters_state()
            st.rerun()

    def clear_filter_val(col_name, val_to_remove):
        key = f"filter_{col_name}"
        if key in st.session_state:
            curr = list(st.session_state[key])
            if val_to_remove in curr:
                curr.remove(val_to_remove)
                st.session_state[key] = curr
                st.session_state.pop(f"chk_{col_name}_{val_to_remove}", None)
                st.rerun()

    # Group filters dynamically
    groups = {"Demographics": [], "Financial": [], "Behavior": [], "Target": []}
    for col in filterable_cols:
        grp = classify_filter_group(col)
        groups[grp].append(col)

    for grp_name, cols in groups.items():
        if not cols:
            continue
        with st.sidebar.expander(f"📁 {grp_name} Filters", expanded=True):
            for col in cols:
                unique_vals = sorted(df[col].dropna().unique().tolist())
                counts_dict = df[col].value_counts().to_dict()
                icon, display_title = get_column_icon_and_title(col)
                
                selected_key = f"filter_{col}"
                if selected_key not in st.session_state:
                    st.session_state[selected_key] = []
                curr_selection = st.session_state[selected_key]
                
                val_map = {}
                for val in unique_vals:
                    lbl = str(val)
                    if col.lower() in ["isactivemember", "active_member", "active"]:
                        lbl = "Active" if val in [1, 1.0, True, "1", "1.0", "active"] else "Inactive"
                    elif col.lower() in ["hascrcard", "has_credit_card", "credit_card"]:
                        lbl = "Yes" if val in [1, 1.0, True, "1", "1.0", "yes"] else "No"
                    elif col.lower() in ["exited", "churn"]:
                        lbl = "Churned" if val in [1, 1.0, True, "1", "1.0", "yes"] else "Stayed"
                    elif isinstance(val, bool) or val in [True, False]:
                        lbl = "Yes" if val else "No"
                    elif str(val) == "1" and len(unique_vals) <= 2:
                        lbl = "Yes"
                    elif str(val) == "0" and len(unique_vals) <= 2:
                        lbl = "No"
                    val_map[val] = lbl
                    
                def get_display_label(val_item):
                    label_text = val_map.get(val_item, str(val_item))
                    frequency = counts_dict.get(val_item, 0)
                    return f"{label_text} ({frequency:,})"
                
                st.markdown(f"**{icon} {display_title}**")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("Select All", key=f"sel_all_{col}", use_container_width=True):
                        st.session_state[selected_key] = unique_vals
                        st.rerun()
                with col_btn2:
                    if st.button("Clear All", key=f"clr_all_{col}", use_container_width=True):
                        st.session_state[selected_key] = []
                        st.rerun()
                
                selected_vals = st.multiselect(
                    label=f"Filter {col}",
                    options=unique_vals,
                    default=curr_selection,
                    key=f"multi_{col}",
                    format_func=get_display_label,
                    label_visibility="collapsed"
                )
                
                if sorted(selected_vals) != sorted(curr_selection):
                    st.session_state[selected_key] = selected_vals
                    st.rerun()
                    
                if selected_vals:
                    active_filters[col] = selected_vals

    active_filters_count = sum(len(vals) for vals in active_filters.values())
    if active_filters_count > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"<div style='font-size:0.84rem;font-weight:800;color:var(--text-primary);margin-bottom:8px;font-family:Outfit,sans-serif;'>Applied Filters</div>",
            unsafe_allow_html=True
        )
        for col, vals in active_filters.items():
            icon, display_title = get_column_icon_and_title(col)
            for val in vals:
                friendly_val = val
                if col.lower() in ["isactivemember", "active_member", "active"]:
                    friendly_val = "Active" if val in [1, 1.0, True, "1", "1.0"] else "Inactive"
                elif col.lower() in ["hascrcard", "has_credit_card", "credit_card"]:
                    friendly_val = "Yes" if val in [1, 1.0, True, "1", "1.0"] else "No"
                elif col.lower() in ["exited", "churn"]:
                    friendly_val = "Churned" if val in [1, 1.0, True, "1", "1.0"] else "Stayed"
                elif isinstance(val, bool) or val in [True, False]:
                    friendly_val = "Yes" if val else "No"
                
                st.sidebar.button(
                    f"  {icon} {friendly_val}",
                    key=f"clear_chip_{col}_{val}",
                    on_click=clear_filter_val,
                    args=(col, val),
                    help=f"Remove filter on {display_title}",
                    use_container_width=True
                )
        if st.sidebar.button("🧹 Clear All Filters", key="btn_clear_all_chips_panel", use_container_width=True):
            reset_all_filters_state()
            st.rerun()

    df_filtered = df.copy()
    for col, selected_vals in active_filters.items():
        df_filtered = df_filtered[df_filtered[col].isin(selected_vals)]

    if df_filtered.empty:
        st.warning("⚠️ The selected filters returned 0 records. Please adjust the interactive filters in the sidebar.")
        st.stop()
        
    total_records = len(st.session_state.df)
    filtered_records = len(df_filtered)
    coverage_pct = (filtered_records / total_records * 100) if total_records > 0 else 0.0
    
    # Filter Summary card at bottom of sidebar
    st.sidebar.markdown("---")
    with st.sidebar.container(border=True):
        st.markdown("**Current Dataset**")
        st.caption(st.session_state.filename)
        st.markdown(f"**Records:** `{filtered_records:,} / {total_records:,}`")
        st.markdown(f"**Filters Applied:** `{active_filters_count}`")
        st.markdown(f"**Coverage:** `{coverage_pct:.1f}%`")
        st.progress(min(coverage_pct / 100.0, 1.0))
        
    df = df_filtered
    
    # Calculate metadata metrics
    try:
        meta = get_dataset_metadata(df)
    except Exception as e:
        st.error("Failed to compute basic dataset profiling metadata.")
        logger.error(f"Error calculating metadata: {str(e)}", exc_info=True)
        meta = {"rows": 0, "columns": 0, "missing_cells": 0, "duplicate_rows": 0, "data_density": "0.0%"}
        
    density_val = meta.get('data_density', '100%')
    missing_val = meta.get('missing_cells', 0)
    dup_val = meta.get('duplicate_rows', 0)
    
    # Precompute executive report (Cached for performance)
    if "report" not in st.session_state or st.session_state.get("report_filename") != st.session_state.filename:
        steps = [
            "Uploading Dataset",
            "Cleaning & Profiling Data",
            "Executing Statistical Analytics",
            "Synthesising Segment Distributions",
            "Generating Domain Insights",
            "Building Executive Dashboard",
            "Completed"
        ]
        
        stepper_placeholder = st.empty()
        
        # Step 0: Uploading
        stepper_placeholder.markdown(get_stepper_html(0, steps), unsafe_allow_html=True)
        import time
        time.sleep(0.5)
        
        # Step 1: Cleaning
        stepper_placeholder.markdown(get_stepper_html(1, steps), unsafe_allow_html=True)
        time.sleep(0.5)
        
        # Step 2: Analyzing
        stepper_placeholder.markdown(get_stepper_html(2, steps), unsafe_allow_html=True)
        time.sleep(0.5)
        
        # Step 3: AutoML modeling & report generator
        stepper_placeholder.markdown(get_stepper_html(3, steps), unsafe_allow_html=True)
        try:
            st.session_state.report = generate_executive_report(df)
            st.session_state.report_filename = st.session_state.filename
        except Exception as e:
            st.error(f"Error compiling report: {str(e)}")
            logger.error(f"Error compiling report: {str(e)}", exc_info=True)
            st.session_state.report = None
            st.session_state.report_filename = None
            st.stop()
            
        # Step 4: Domain Insights
        stepper_placeholder.markdown(get_stepper_html(4, steps), unsafe_allow_html=True)
        time.sleep(0.5)
        
        # Step 5: Dashboard
        stepper_placeholder.markdown(get_stepper_html(5, steps), unsafe_allow_html=True)
        time.sleep(0.5)
        
        # Step 6: Completed
        stepper_placeholder.markdown(get_stepper_html(6, steps), unsafe_allow_html=True)
        time.sleep(0.5)
        
        stepper_placeholder.empty()
        st.rerun()
        
    report = st.session_state.report
    target_col = report["domain_analysis"]["target_column"] if report and "domain_analysis" in report else None
    automl_res = report.get("automl", {}).get("results") if report else None

    # Global column variables for page routing scope
    primary_date = report["analyzed_columns"]["date"] if report and "analyzed_columns" in report else None
    primary_metric = report["analyzed_columns"]["metrics"][0] if report and "analyzed_columns" in report and report["analyzed_columns"]["metrics"] else None
    primary_category = report["analyzed_columns"]["categories"][0] if report and "analyzed_columns" in report and report["analyzed_columns"]["categories"] else None

    # Precompute variables for global exports
    findings = []
    high_segment = "None"
    high_explanation = "No critical risk identified."
    med_segment = "None"
    med_explanation = "No moderate risk identified."
    low_segment = "Database Metrics"
    low_explanation = "Baseline parameters are stable."
    
    if report:
        meta = report["metadata"]
        quality_alerts = report.get("quality_alerts", [])
        
        # Populate findings directly from the central BI engine output
        findings = report.get("findings", [])
        
        # Populate risks from structured risks
        risks_struct = report.get("risks_structured", {})
        if risks_struct:
            high_segment = risks_struct.get("high", {}).get("segment", "None")
            high_explanation = risks_struct.get("high", {}).get("explanation", "Stable.")
            med_segment = risks_struct.get("medium", {}).get("segment", "None")
            med_explanation = risks_struct.get("medium", {}).get("explanation", "Stable.")
            low_segment = risks_struct.get("low", {}).get("segment", "None")
            low_explanation = risks_struct.get("low", {}).get("explanation", "Stable.")

    # Compute target labels and density values globally for KPIs
    target_rate_val = None
    target_label = "Target Rate"
    target_badge = ":blue[● Baseline]"
    target_subtext = "Favorable baseline rate"
    if target_col and report:
        domain_type = report["domain_analysis"]["domain_type"]
        if domain_type == "churn":
            target_rate_val = report["domain_analysis"]["metrics"].get("churn_rate")
            target_label = "Churn Rate"
        elif domain_type == "hr_attrition":
            target_rate_val = report["domain_analysis"]["metrics"].get("attrition_rate")
            target_label = "Attrition Rate"
        else:
            target_rate_val = report["domain_analysis"]["metrics"].get("target_rate")
            if target_rate_val is None:
                cleaned = df[target_col].dropna()
                mapped = cleaned.map(lambda x: 1 if str(x).strip().lower() in ["yes", "true", "1", "1.0", "y", "t", "exited", "left", "cancelled", "churned"] else 0)
                total_valid = df[target_col].notnull().sum()
                target_rate_val = (mapped.sum() / total_valid * 100) if total_valid > 0 else 0.0
            target_label = "Target Rate"
        if target_rate_val is not None:
            if target_rate_val > 15.0:
                target_badge = ":red[● High Risk]"
                target_subtext = "Threshold exceeded"
            else:
                target_badge = ":green[● Favorable]"
                target_subtext = "Favorable rate"
    target_val_str = f"{target_rate_val:.2f}%" if target_rate_val is not None else "N/A"

    density_val = meta.get('data_density', '100%')
    density_badge = ":green[● Optimal]"
    if density_val != '100%':
        try:
            val_num = float(density_val.replace('%',''))
            if val_num < 95.0:
                density_badge = ":orange[● Warning]"
        except:
            pass

    # Sidebar Navigation Menu Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='display:flex;align-items:center;gap:8px;margin-bottom:14px;'>
        <div style='width:30px;height:30px;border-radius:8px;background:var(--primary-light);display:flex;align-items:center;justify-content:center;font-size:1.0rem;color:var(--primary-color);'>🎯</div>
        <div style='font-size:1.05rem;font-weight:800;color:var(--text-primary);font-family:Outfit,sans-serif;letter-spacing:-0.2px;'>Navigation</div>
    </div>""", unsafe_allow_html=True)
    
    pages = {
        "Dashboard": "📊 Dashboard",
        "Business Insights": "💡 Business Insights",
        "Root Cause Analysis": "🎯 Root Cause Analysis",
        "Trend Analysis": "📈 Trend Analysis",
        "Executive Report": "📝 Executive Report",
        "Presentation": "💼 Presentation",
        "Export": "📥 Export"
    }
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
        
    for page_id, page_lbl in pages.items():
        is_active = st.session_state.current_page == page_id
        indicator_prefix = "▸ " if is_active else "  "
        if st.sidebar.button(f"{indicator_prefix}{page_lbl}", key=f"btn_nav_{page_id}", use_container_width=True):
            st.session_state.current_page = page_id
            st.rerun()
            
    # Header is rendered globally at the top of the app

    # 1. DASHBOARD PAGE
    if st.session_state.current_page == "Dashboard":
        if report:
            st.markdown("### 🏢 Executive BI Overview")
            
            # Executive KPI Cards
            render_kpi_cards(df, meta, report, target_col)

            # Row 2: Trend & Distribution
            col_r2_left, col_r2_right = st.columns(2)
            with col_r2_left:
                with st.container(border=True):
                    st.markdown("### 🕒 Temporal Trend Analysis")
                    if primary_date:
                        render_time_trend_with_fallback(report.get("time_findings"), primary_metric or df.select_dtypes(include='number').columns[0], primary_date, key="db_trend_chart")
                    else:
                        render_empty_state("🕒", "Temporal Trend Unavailable", "No date dimension was identified to map trends.")
            
            with col_r2_right:
                with st.container(border=True):
                    st.markdown("### 📊 Distribution Analysis")
                    if primary_metric:
                        fig_dist = create_distribution_plot(df, primary_metric)
                        st_plotly_chart(fig_dist, key="db_dist_chart")
                    else:
                        render_empty_state("📊", "Distribution Plot Unavailable", "Requires numeric metrics column.")

            st.divider()

            # Row 3: Top Categories & Segment Breakdown
            col_r3_left, col_r3_right = st.columns(2)
            with col_r3_left:
                with st.container(border=True):
                    st.markdown("### 📈 Top Categories & Contribution Average")
                    if primary_metric and primary_category:
                        fig_cross = create_cross_variable_bar_chart(df, primary_metric, primary_category)
                        st_plotly_chart(fig_cross, key="db_cross_chart")
                    else:
                        render_empty_state("📊", "Category Analysis Unavailable", "Requires numeric metrics and categories.")
            
            with col_r3_right:
                with st.container(border=True):
                    st.markdown("### 🍩 Proportional Segment Breakdown")
                    if primary_category:
                        fig_pie = create_categorical_pie_chart(df, primary_category)
                        st_plotly_chart(fig_pie, key="db_pie_chart")
                    else:
                        render_empty_state("🍩", "Segment Breakdown Unavailable", "Requires a categorical segment column.")

    # 2. VISUAL ANALYTICS PAGE
    elif st.session_state.current_page == "Visual Analytics":
        if report:
            st.markdown("### 📈 Visual Analytics & Deep Exploration")
            
            sub_vis1, sub_vis2, sub_vis3, sub_vis4 = st.tabs([
                "📊 Attribute Distributions",
                "🌡️ Correlation Matrix",
                "🔍 Segment & Category Exploration",
                "🏆 Top N Summaries"
            ])

            with sub_vis1:
                with st.container(border=True):
                    st.markdown("### Distribution Analysis")
                    num_cols = get_numeric_columns(df)
                    if num_cols:
                        selected_dist_col = st.selectbox("Select metric to view distribution:", num_cols, key="vis_dist_col")
                        render_chart_with_fallback(create_distribution_plot, df, selected_dist_col, key="vis_dist_plot")
                    else:
                        render_empty_state("📊", "No metrics found", "Value distribution requires at least one numeric metric column.")

            with sub_vis2:
                with st.container(border=True):
                    st.markdown("### Correlation Matrix Heatmap")
                    render_chart_with_fallback(create_correlation_heatmap, df, key="vis_corr_heatmap")

            with sub_vis3:
                # Segment Analysis and Category Comparisons
                st.markdown("### Category Segment Exploration")
                col_sub_v1, col_sub_v2 = st.columns(2)
                with col_sub_v1:
                    with st.container(border=True):
                        st.markdown("#### Category Comparison")
                        cat_cols = [col for col in get_categorical_columns(df) if df[col].dropna().nunique() <= 25]
                        if cat_cols:
                            selected_cat_col = st.selectbox("Select category to analyze:", cat_cols, key="vis_cat_col")
                            render_chart_with_fallback(create_categorical_bar_chart, df, selected_cat_col, key="vis_cat_chart")
                        else:
                            st.info("No categorical fields available for category comparison.")
                with col_sub_v2:
                    with st.container(border=True):
                        st.markdown("#### Segment Investigation Profiling")
                        drill_cols = [col for col in get_categorical_columns(df) if semantic_roles.get(col) in ["Category", "Segment", "Boolean"]]
                        drill_cols = [col for col in drill_cols if df[col].dropna().nunique() <= 100]
                        options = []
                        for col in drill_cols:
                            unique_vals = sorted(df[col].dropna().unique().tolist())
                            for val in unique_vals:
                                options.append({"display": f"{col} = {val}", "column": col, "value": val})
                        
                        if options:
                            selected_option = st.selectbox("Select segment value to focus investigation:", options=options, format_func=lambda x: x["display"], key="vis_drill_down_select")
                            if selected_option:
                                col_name = selected_option["column"]
                                val_name = selected_option["value"]
                                seg_count = len(df[df[col_name] == val_name])
                                seg_pct = (seg_count / len(df) * 100)
                                st.markdown(f"**Focused Segment:** `{col_name} = {val_name}`  \n**Segment Size:** `{seg_count:,} records` (`{seg_pct:.1f}%` of active dataset)")
                                if st.button("🔬 Analyze Segment Drivers & Gaps", key="vis_drill_btn", use_container_width=True):
                                    st.session_state.focus_column = col_name
                                    st.session_state.focus_value = val_name
                                    st.rerun()
                        else:
                            st.info("No categorical features available to run segment profiling.")

                # Detail rendering for Segment investigation
                focus_col = st.session_state.get("focus_column")
                focus_val = st.session_state.get("focus_value")
                if focus_col and focus_val:
                    with st.spinner(f"Running Segment Investigation for {focus_col} = {focus_val}..."):
                        try:
                            entity_res = run_entity_investigation(df, focus_val, focus_col, roles=semantic_roles)
                        except Exception as e:
                            st.error(f"Error running segment investigation: {str(e)}")
                            entity_res = None
                            
                    if entity_res and entity_res.get("success"):
                        st.markdown(f"#### Segment Investigation Detail — `{focus_col} = {focus_val}`")
                        col_seg_kpi1, col_seg_kpi2, col_seg_kpi3 = st.columns(3)
                        seg_count = len(df[df[focus_col] == focus_val])
                        seg_pct = (seg_count / len(df) * 100)
                        with col_seg_kpi1:
                            with st.container(border=True):
                                st.markdown("##### Segment Summary")
                                st.markdown(f"### {seg_count:,} Records")
                                st.caption(f":blue[● Info] · {seg_pct:.1f}% of active dataset")
                        with col_seg_kpi2:
                            with st.container(border=True):
                                if entity_res.get("has_binary_target"):
                                    delta = entity_res['target_rate_entity'] - entity_res['target_rate_others']
                                    delta_badge = ":red[▲ Higher]" if delta > 0 else ":green[▼ Lower]"
                                    st.markdown("##### Segment Target Rate")
                                    st.markdown(f"### {entity_res['target_rate_entity']:.2f}%")
                                    st.caption(f"{delta_badge} · {delta:+.2f}% vs others")
                                else:
                                    seg_share = (len(df[df[focus_col] == focus_val]) / len(df) * 100)
                                    st.markdown("##### Segment Share")
                                    st.markdown(f"### {seg_share:.1f}%")
                                    st.caption(":blue[● Info] · Proportional share")
                        with col_seg_kpi3:
                            with st.container(border=True):
                                score_class = ":green[✓ High]" if entity_res['confidence_level'].lower() == "high" else ":orange[● Medium]"
                                st.markdown("##### Driver Score Confidence")
                                st.markdown(f"### {entity_res['confidence_score']}%")
                                st.caption(f"{score_class} · {entity_res['confidence_level']} confidence")
                                
                        rankings = entity_res.get("rankings", [])
                        interp_text = f"The selected segment **{focus_col} = {focus_val}** represents **{seg_pct:.1f}%** of the active dataset."
                        if entity_res.get("has_binary_target"):
                            delta = entity_res['target_rate_entity'] - entity_res['target_rate_others']
                            dir_word = "higher" if delta > 0 else "lower"
                            interp_text += f" Quantitative profiling shows a target conversion rate of **{entity_res['target_rate_entity']:.2f}%**, which is **{abs(delta):.2f}% {dir_word}** than the remaining records. "
                        if rankings:
                            interp_text += f"Drivers analysis isolates **{rankings[0]['feature']}** as the primary segment driver, showing a driver relevance coefficient of **{rankings[0]['driver_score']:.3f}**. "
                        
                        with st.container(border=True):
                            st.markdown("**Business Interpretation**")
                            st.markdown(interp_text)
                            
                        col_seg_left, col_seg_right = st.columns([3, 2])
                        with col_seg_left:
                            with st.container(border=True):
                                st.markdown("##### Quantitative Evidence Table")
                                evidence_data = entity_res.get("metrics_evidence", [])
                                if evidence_data:
                                    rows = []
                                    for item in evidence_data:
                                        ent_val_str = f"{item['entity_val']:.2f}{item['unit']}" if not item['is_percentage'] else f"{item['entity_val']:.2f}%"
                                        oth_val_str = f"{item['others_val']:.2f}{item['unit']}" if not item['is_percentage'] else f"{item['others_val']:.2f}%"
                                        diff_val_str = f"{item['difference']:+,.2f}{item['unit']}" if not item['is_percentage'] else f"{item['difference']:+,.2f}%"
                                        pct_diff_str = f"{item['pct_difference']:+,.1f}%" if item['pct_difference'] is not None else "N/A"
                                        rows.append({
                                            "Metric": item["metric_name"],
                                            "Segment Value": ent_val_str,
                                            "Other Records": oth_val_str,
                                            "Difference": diff_val_str,
                                            "% Diff": pct_diff_str
                                        })
                                    st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)
                                else:
                                    render_empty_state("📊", "No comparison data", "Comparative statistical metrics are unavailable for this segment.")
                                    
                        with col_seg_right:
                            with st.container(border=True):
                                st.markdown("##### Key Segment Business Drivers")
                                if rankings:
                                    top_drivers = rankings[:4]
                                    for idx, drv in enumerate(top_drivers, 1):
                                        st.markdown(f"**{idx}. {drv['feature']}** (Driver Score: `{drv['driver_score']:.3f}`)")
                                        st.caption(drv['evidence'])
                                else:
                                    render_empty_state("🎯", "No drivers identified", "Driver rankings could not be computed for this specific segment.")

            with sub_vis4:
                with st.container(border=True):
                    st.markdown("### Top N Performance Summaries")
                    if num_cols and cat_cols:
                        col_n1, col_n2, col_n3 = st.columns(3)
                        with col_n1:
                            n_cat = st.selectbox("Group By:", cat_cols, key="vis_n_cat")
                        with col_n2:
                            n_metric = st.selectbox("Metric Summary:", num_cols, key="vis_n_metric")
                        with col_n3:
                            n_val = st.slider("Top N Counts:", min_value=3, max_value=20, value=10, key="vis_n_val")
                        
                        top_n_df = df.groupby(n_cat)[n_metric].mean().reset_index().sort_values(by=n_metric, ascending=False).head(n_val)
                        fig_top_n = px.bar(top_n_df, x=n_cat, y=n_metric, title=f"Top {n_val} Average {n_metric} by {n_cat}")
                        st_plotly_chart(fig_top_n, key="vis_top_n_chart")
                    else:
                        st.info("Insufficient metrics or categorical attributes to display Top N summary.")



    # 2. BUSINESS INSIGHTS PAGE
    elif st.session_state.current_page == "Business Insights":
        if report:
            st.markdown("### 💡 Strategic Business Insights")
            st.markdown("Automated quantitative findings with priority classifications and actionable next steps.")
            
            for idx, f in enumerate(findings, 1):
                with st.container(border=True):
                    # Finding Header
                    col_icon, col_title = st.columns([1, 15])
                    with col_icon:
                        st.markdown(f"<div style='font-size:2rem; padding-top:4px;'>{f['icon']}</div>", unsafe_allow_html=True)
                    with col_title:
                        st.markdown(f"#### Finding {idx}: {f['obs']}")
                    
                    priority = "High" if idx <= 2 else ("Medium" if idx <= 4 else "Low")
                    p_color = "red" if priority == "High" else ("orange" if priority == "Medium" else "green")
                    confidence = "High" if idx % 2 == 0 or "significant" in f['impact'].lower() else "Medium"
                    c_color = "green" if confidence == "High" else "orange"
                    
                    st.markdown(f"**Priority:** :{p_color}[● {priority} Priority] | **Confidence:** :{c_color}[✓ {confidence} Confidence]")
                    
                    # Expandable details
                    with st.expander("🔍 View Segment Evidence & Recommendations"):
                        st.markdown(f"**Supporting Evidence:** `{f['metric']}`")
                        st.markdown(f"**Business Impact:** {f['impact']}")
                        
                        recs_list = report.get("recommendations", [])
                        rec_val = recs_list[idx - 1] if idx - 1 < len(recs_list) else "Establish ongoing reporting metrics to monitor segment deviations."
                        st.markdown(f"**Actionable Recommendation:** {rec_val}")

    # 3. ROOT CAUSE ANALYSIS PAGE
    elif st.session_state.current_page == "Root Cause Analysis":
        if report:
            st.markdown("### 🎯 Root Cause Analysis (RCA)")
            st.markdown("Diagnose why performance KPIs shifted and trace segment contribution drivers.")
            
            # Row 1: Segment Contribution Analysis
            col_rca_left, col_rca_right = st.columns([3, 2])
            with col_rca_left:
                st.markdown("#### 📊 Segment Contribution Share")
                if report.get("rca_findings") and primary_metric and primary_category:
                    fig_rca = create_rca_contribution_chart(report["rca_findings"], primary_metric, primary_category)
                    st_plotly_chart(fig_rca, key="rca_contribution_chart")
                else:
                    render_empty_state("📊", "RCA Unavailable", "Root cause segment details require metric and category columns.")
            
            with col_rca_right:
                with st.container(border=True):
                    st.markdown("#### 🎯 Driver Hierarchy & Synthesis")
                    if report.get("rca_findings"):
                        st.markdown("**Segment Impact Concentration Description:**")
                        st.write(report["rca_findings"].get("concentration", {}).get("description", "No concentration alerts identified."))
                        
                        top_contrib = report["rca_findings"]["contribution"]["top"]
                        st.markdown(f"""
                        <div style='background:rgba(59,130,246,0.1); border-left:4px solid #3B82F6; padding:12px; border-radius:4px; margin-top:14px;'>
                            <strong>Top Contributing Driver:</strong><br/>
                            Segment Category: <code>{primary_category}</code><br/>
                            Driver Segment: <code>{top_contrib['dimension_value']}</code><br/>
                            Total Aggregation: <code>{top_contrib['sum']:,.0f}</code><br/>
                            Contribution Ratio: <code>{top_contrib['share_percentage']}%</code>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write("Baseline distribution remains stable across segments.")
            
            st.divider()
            
            # Row 2: Evidence Table & Business Explanation
            st.markdown("#### 🔍 Segment Evidence Table")
            if report.get("rca_findings") and "contribution" in report["rca_findings"]:
                breakdown = report["rca_findings"]["contribution"]["full_breakdown"]
                ev_df = pd.DataFrame(breakdown)
                ev_df.columns = ["Segment Value", "Aggregated Sum", "Share of Total (%)"]
                st.dataframe(ev_df.round(2), use_container_width=True, hide_index=True)
                
                # Waterfall/Step calculation
                st.markdown("#### ⚙️ Contribution Step Analysis")
                fig_wf = go.Figure(go.Waterfall(
                    name="RCA Waterfall",
                    orientation="v",
                    measure=["relative"] * len(breakdown) + ["total"],
                    x=[item["dimension_value"] for item in breakdown] + ["Total"],
                    textposition="outside",
                    text=[f"{item['share_percentage']}%" for item in breakdown] + ["100%"],
                    y=[item["share_percentage"] for item in breakdown] + [100.0],
                    connector={"line": {"color": "rgba(255,255,255,0.2)"}},
                    increasing={"marker": {"color": "#3B82F6"}},
                    totals={"marker": {"color": "#7C3AED"}}
                ))
                apply_enterprise_layout(fig_wf, "Segment Relative Contribution Waterfall")
                st_plotly_chart(fig_wf, key="rca_waterfall")
            else:
                st.info("No detailed segment breakdown available.")

    # 4. TREND ANALYSIS PAGE
    elif st.session_state.current_page == "Trend Analysis":
        if report:
            st.markdown("### 📈 Time Intelligence & Trend Analysis")
            st.markdown("Trace historical variance, rolling averages, and statistical regression projections.")
            
            if primary_date:
                df_time = df.copy()
                df_time[primary_date] = pd.to_datetime(df_time[primary_date])
                
                tab_trend_m, tab_trend_w, tab_trend_d = st.tabs([
                    "📅 Monthly Trend Analysis",
                    "🗓️ Weekly Trend Analysis",
                    "⏱️ Daily Trend Analysis"
                ])
                
                with tab_trend_m:
                    with st.container(border=True):
                        df_m = df_time.groupby(df_time[primary_date].dt.to_period("M"))[primary_metric].sum().reset_index()
                        df_m[primary_date] = df_m[primary_date].dt.to_timestamp()
                        
                        fig_trend_m = go.Figure()
                        fig_trend_m.add_trace(go.Scatter(
                            x=df_m[primary_date], y=df_m[primary_metric],
                            mode="lines+markers", name="Monthly Aggregated Sum",
                            line=dict(color="#3B82F6", width=3)
                        ))
                        import numpy as np
                        from sklearn.linear_model import LinearRegression
                        x_num = np.arange(len(df_m)).reshape(-1, 1)
                        y_val = df_m[primary_metric].values.reshape(-1, 1)
                        lr = LinearRegression().fit(x_num, y_val)
                        trend_y = lr.predict(x_num)
                        fig_trend_m.add_trace(go.Scatter(
                            x=df_m[primary_date], y=trend_y.flatten(),
                            mode="lines", name="Regression Trend",
                            line=dict(color="#10B981", width=2, dash="dot")
                        ))
                        apply_enterprise_layout(fig_trend_m, f"Monthly Aggregated Sum: {primary_metric}")
                        st_plotly_chart(fig_trend_m, key="trend_monthly_chart")
                        
                with tab_trend_w:
                    with st.container(border=True):
                        df_w = df_time.groupby(df_time[primary_date].dt.to_period("W"))[primary_metric].sum().reset_index()
                        df_w[primary_date] = df_w[primary_date].dt.to_timestamp()
                        
                        fig_trend_w = go.Figure()
                        fig_trend_w.add_trace(go.Scatter(
                            x=df_w[primary_date], y=df_w[primary_metric],
                            mode="lines+markers", name="Weekly Aggregated Sum",
                            line=dict(color="#7C3AED", width=2.5)
                        ))
                        apply_enterprise_layout(fig_trend_w, f"Weekly Aggregated Sum: {primary_metric}")
                        st_plotly_chart(fig_trend_w, key="trend_weekly_chart")
                        
                with tab_trend_d:
                    with st.container(border=True):
                        df_d = df_time.groupby(df_time[primary_date].dt.date)[primary_metric].sum().reset_index()
                        
                        fig_trend_d = go.Figure()
                        fig_trend_d.add_trace(go.Scatter(
                            x=df_d[primary_date], y=df_d[primary_metric],
                            mode="lines", name="Daily Aggregated Sum",
                            line=dict(color="#06B6D4", width=1.5)
                        ))
                        apply_enterprise_layout(fig_trend_d, f"Daily Aggregated Sum: {primary_metric}")
                        st_plotly_chart(fig_trend_d, key="trend_daily_chart")
                        
                st.markdown("#### 🔮 Forecast Projections")
                time_res = perform_time_analysis(df, primary_metric, primary_date)
                if time_res and "forecast" in time_res:
                    with st.container(border=True):
                        fig_fc = create_time_trend_chart(time_res, primary_metric, primary_date)
                        st_plotly_chart(fig_fc, key="trend_forecast_chart")
            else:
                st.info("No time column available.")

    # 6. EXECUTIVE REPORT PAGE
    elif st.session_state.current_page == "Executive Report":
        if report:
            st.markdown("### 📋 Management-Ready Executive Report")
            
            # Simple KPIs cards
            risk_val = extract_risk_level(report.get('executive_summary', '')) or 'Medium'
            risk_class = "danger" if risk_val == "High" else ("warning" if risk_val == "Medium" else "success")
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:16px;margin-bottom:20px;width:100%;">
                <div class="enterprise-kpi-card" style="height: 100px;">
                    <div class="kpi-top">
                        <span class="kpi-title">Aggregated Records</span>
                        <span class="kpi-icon">📊</span>
                    </div>
                    <div class="kpi-value" style="font-size:1.55rem; margin:2px 0;">{len(df):,}</div>
                </div>
                <div class="enterprise-kpi-card" style="height: 100px;">
                    <div class="kpi-top">
                        <span class="kpi-title">{target_label}</span>
                        <span class="kpi-icon">📈</span>
                    </div>
                    <div class="kpi-value" style="font-size:1.55rem; margin:2px 0;">{target_val_str}</div>
                </div>
                <div class="enterprise-kpi-card" style="height: 100px;">
                    <div class="kpi-top">
                        <span class="kpi-title">Assessed Risk Level</span>
                        <span class="kpi-icon">⚠️</span>
                    </div>
                    <div class="kpi-value" style="font-size:1.55rem; margin:2px 0; font-weight:800; color: var(--{risk_class}-color);">{risk_val}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()

            # Executive Summary text
            st.markdown("#### 📝 Executive Summary")
            with st.container(border=True):
                st.markdown(report.get("executive_summary", "No executive summary available."))

            st.divider()

            # Findings
            st.markdown("#### 🔑 Key Business Findings")
            for idx, f in enumerate(findings, 1):
                with st.container(border=True):
                    st.markdown(f"**Finding {idx}:** {f['obs']}")
                    st.markdown(f"*Business Impact:* {f['impact']}")
                    st.markdown(f"*Supporting Metric:* {f['metric']}")
            
            st.divider()

            # Risks
            st.markdown("#### ⚠️ Business Risks Assessment")
            col_risk_l, col_risk_r = st.columns(2)
            with col_risk_l:
                with st.container(border=True):
                    st.markdown("##### Primary Severity Risk")
                    st.markdown(f"**Segment Dimension:** `{high_segment}`")
                    st.write(high_explanation)
            with col_risk_r:
                with st.container(border=True):
                    st.markdown("##### Secondary Severity Risk")
                    st.markdown(f"**Segment Dimension:** `{med_segment}`")
                    st.write(med_explanation)

            st.divider()

            # Strategic Business Recommendations Roadmap (Horizon Timeline)
            st.markdown("#### 💡 Strategic Business Recommendations Roadmap")
            render_timeline_recommendations(report.get("recommendations", []))

            st.divider()

            # Expected Business Impact
            st.markdown("#### 📈 Expected Business Impact Summary")
            with st.container(border=True):
                impacts = report.get("expected_business_impact_structured", [])
                if impacts:
                    for imp in impacts:
                        st.markdown(f"- **Outcome Benefit:** {imp}")
                else:
                    st.markdown(
                        "- **Optimized Retention Operations**: Implementing localized campaigns for primary drivers stabilizes base outflow.\n"
                        "- **Targeted Marketing Allocation**: Marketing campaigns directed at highly relevant demographic segments improve conversions.\n"
                        "- **Data-Driven Capital Allocation**: Insights derived from temporal growth trends minimize seasonal performance waste."
                    )

            st.divider()

            # Recommended Next Steps
            st.markdown("#### ⏭️ Recommended Next Steps")
            with st.container(border=True):
                steps = report.get("next_steps_structured", [])
                if steps:
                    for step in steps:
                        st.markdown(f"- {step}")
                else:
                    st.markdown(
                        "- Review local system service or product friction indicators.\n"
                        "- Design target campaign materials for re-engagement segments.\n"
                        "- Schedule workflow optimization review to correct data gaps."
                    )

            st.divider()

            # Concluding Remarks
            st.markdown("#### 🏁 Concluding Remarks")
            with st.container(border=True):
                st.markdown(
                    "This management briefing summarizes the operational and financial profile of the target segment base. "
                    "All data metrics are dynamically compiled from active query boundaries. "
                    "For offline management briefings, please navigate to the **Presentation** section or use the central **Export** hub."
                )

    # 6. PRESENTATION PAGE
    elif st.session_state.current_page == "Presentation":
        if report:
            st.markdown("### 💼 PowerPoint Presentation Deck")
            st.markdown("Compile all diagnostic observations and strategic roadmaps into a widescreen presentation deck.")
            
            st.markdown("#### 📺 Presentation Slides Preview")
            
            # Slides mockup preview grid
            st.markdown("""
            <div class="deck-preview-grid">
                <div class="slide-thumbnail">
                    <div class="slide-title">1. Executive Overview Cover</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock med"></div>
                        <div class="slide-line-mock short"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Cover Slide</div>
                </div>
                <div class="slide-thumbnail">
                    <div class="slide-title">2. Executive Summary Context</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock short"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Slide 2</div>
                </div>
                <div class="slide-thumbnail">
                    <div class="slide-title">3. Key Performance Indicators</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock med"></div>
                        <div class="slide-line-mock med"></div>
                        <div class="slide-line-mock long"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Slide 3</div>
                </div>
                <div class="slide-thumbnail">
                    <div class="slide-title">4. Visual Performance Highlights</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock short"></div>
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock med"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Slide 4</div>
                </div>
                <div class="slide-thumbnail">
                    <div class="slide-title">5. Core Business Findings</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock short"></div>
                        <div class="slide-line-mock med"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Slide 5</div>
                </div>
                <div class="slide-thumbnail">
                    <div class="slide-title">6. Strategic Risks Assessment</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock med"></div>
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock short"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Slide 6</div>
                </div>
                <div class="slide-thumbnail">
                    <div class="slide-title">7. Action Roadmaps Timeline</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock short"></div>
                        <div class="slide-line-mock med"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Slide 7</div>
                </div>
                <div class="slide-thumbnail">
                    <div class="slide-title">8. Executive Conclusion Plan</div>
                    <div class="slide-body-mock">
                        <div class="slide-line-mock med"></div>
                        <div class="slide-line-mock long"></div>
                        <div class="slide-line-mock short"></div>
                    </div>
                    <div class="slide-footer-mock">VALKYRIE • Slide 8</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            st.markdown("#### ⚙️ Deck Generation & Export Options")
            
            with st.container(border=True):
                if not st.session_state.get("pptx_created"):
                    st.markdown("""
                    <div style="padding: 10px 0 20px 0;">
                        <h5>Ready to generate presentation?</h5>
                        <p style="font-size: 0.85rem; color: var(--text-secondary);">
                            Compile all business parameters, root cause findings, and timeline recommendations into a widescreen 16:9 widescreen presentation deck.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("📊 Create Executive Presentation", key="btn_create_pptx_page", use_container_width=True):
                        progress_placeholder = st.empty()
                        steps = [
                            "Initializing PPTX Template (16:9 Widescreen)",
                            "Generating Slide 1: Cover Page Title & Metadata",
                            "Generating Slide 2: Executive Summary Overview",
                            "Generating Slide 3: KPI Metrics Cards Grid",
                            "Generating Slide 4: Visual Dashboard Highlights",
                            "Generating Slide 5: Core Business Findings Log",
                            "Generating Slide 6: Strategic Risks Severity Matrix",
                            "Generating Slide 7: Action Recommendation Roadmaps",
                            "Generating Slide 8: Executive Conclusion Plan",
                            "Completed"
                        ]
                        
                        import time
                        for i in range(len(steps)):
                            progress_placeholder.markdown(get_stepper_html(i, steps), unsafe_allow_html=True)
                            time.sleep(0.4)
                        
                        try:
                            pptx_bytes = generate_presentation(
                                df=df,
                                report=report,
                                findings=findings,
                                risks={"high": {"segment": high_segment, "explanation": high_explanation},
                                       "medium": {"segment": med_segment, "explanation": med_explanation},
                                       "low": {"segment": low_segment, "explanation": low_explanation}},
                                recommendations=report.get("recommendations", []),
                                filename=st.session_state.filename
                            )
                            st.session_state.pptx_bytes = pptx_bytes
                            st.session_state.pptx_created = True
                        except Exception as pptx_err:
                            logger.error(f"Failed to generate PowerPoint: {str(pptx_err)}", exc_info=True)
                            st.error(f"PowerPoint generation failed: {str(pptx_err)}")
                            st.stop()
                            
                        st.rerun()
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 20px 0;">
                        <span style="font-size: 3rem; color: var(--success-color);">✅</span>
                        <h4 style="margin-top: 10px; font-family: Outfit; font-weight: 700; color: var(--text-primary);">PowerPoint Presentation Generated</h4>
                        <p style="font-size: 0.88rem; color: var(--text-secondary); max-width: 450px; margin: 6px auto 20px auto;">
                            Widescreen (16:9) corporate presentation is ready for delivery. Contains 8 structured consulting slides.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.download_button(
                        label="⬇ Download Executive Presentation (.pptx)",
                        data=st.session_state.pptx_bytes,
                        file_name="executive_briefing.pptx",
                        mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        use_container_width=True,
                        key="report_pptx_download_btn"
                    )
                    
                    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
                    if st.button("🔄 Create New Presentation", key="btn_reset_pptx_page", use_container_width=True):
                        st.session_state.pptx_created = False
                        st.session_state.pptx_bytes = None
                        st.rerun()

    # 7. EXPORT PAGE
    elif st.session_state.current_page == "Export":
        if report:
            st.markdown("### 📥 Central Export Hub")
            st.markdown("Download management reports, executive slide decks, tabular summaries, and high-resolution chart assets.")
            
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                with st.container(border=True):
                    st.markdown("#### 📄 Executive Briefing Report (PDF)")
                    st.markdown("Full text report containing summary analysis, segment breakdowns, and next step lists.")
                    try:
                        pdf_lines = [
                            "# EXECUTIVE BRIEFING REPORT",
                            f"**Source Dataset:** {st.session_state.filename}",
                            f"**Total Records:** {len(df):,}",
                            "",
                            "## Executive Summary",
                            report.get("executive_summary", "N/A"),
                            "",
                            "## Business Findings",
                        ]
                        for idx, f in enumerate(findings, 1):
                            pdf_lines.append(f"Finding {idx}: {f['obs']}")
                            pdf_lines.append(f"- Business Impact: {f['impact']}")
                            pdf_lines.append(f"- Supporting Metric: {f['metric']}")
                            pdf_lines.append("")
                            
                        pdf_lines.extend([
                            "## Risk Assessment",
                            f"- High Risk: {high_segment} - {high_explanation}",
                            f"- Medium Risk: {med_segment} - {med_explanation}",
                            f"- Low Risk: {low_segment} - {low_explanation}",
                            "",
                            "## Recommendations",
                        ])
                        for r in report.get("recommendations", []):
                            pdf_lines.append(f"- {r}")
                            
                        pdf_text = "\n".join(pdf_lines)
                        pdf_bytes = generate_pdf(pdf_text, title="Executive Briefing Report")
                        st.download_button(
                            label="📥 Download PDF Briefing",
                            data=pdf_bytes,
                            file_name="executive_briefing.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            key="export_page_pdf"
                        )
                    except Exception as e:
                        st.error(f"Failed to compile PDF Briefing: {str(e)}")
            
            with col_ex2:
                with st.container(border=True):
                    st.markdown("#### 📊 Tabular Data Summary (CSV/Excel)")
                    st.markdown("Row summary metadata, computed statistics, segment risk profiles, and recommendation logs.")
                    try:
                        import csv
                        import io
                        csv_io = io.StringIO()
                        writer = csv.writer(csv_io)
                        writer.writerow(["Section", "Metric/Item", "Detail"])
                        writer.writerow(["Metadata", "Filename", st.session_state.filename])
                        writer.writerow(["Metadata", "Total Records", len(df)])
                        writer.writerow(["Executive Summary", "Brief", report.get("executive_summary", "N/A").replace("\n", " ")])
                        for idx, f in enumerate(findings, 1):
                            writer.writerow([f"Finding {idx}", f["obs"], f["metric"]])
                        writer.writerow(["Risk", "High Severity", f"{high_segment}: {high_explanation}"])
                        writer.writerow(["Risk", "Medium Severity", f"{med_segment}: {med_explanation}"])
                        writer.writerow(["Risk", "Low Severity", f"{low_segment}: {low_explanation}"])
                        for idx, r in enumerate(report.get("recommendations", []), 1):
                            writer.writerow([f"Recommendation {idx}", r, "N/A"])
                            
                        st.download_button(
                            label="📊 Export Summary (CSV)",
                            data=csv_io.getvalue(),
                            file_name="executive_summary.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="export_page_csv"
                        )
                    except Exception as e:
                        st.error("Failed to compile CSV Export.")
                        
            col_ex3, col_ex4 = st.columns(2)
            with col_ex3:
                with st.container(border=True):
                    st.markdown("#### 💼 Executive Presentation Deck (PPTX)")
                    st.markdown("Consulting slide deck containing widescreen layouts. Compile it in the **Presentation** menu first.")
                    if st.session_state.get("pptx_created"):
                        st.download_button(
                            label="⬇ Download Slide Deck (.pptx)",
                            data=st.session_state.pptx_bytes,
                            file_name="executive_briefing.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            use_container_width=True,
                            key="export_page_pptx"
                        )
                    else:
                        st.info("⚠️ Presentation deck has not been compiled yet. Please navigate to the **Presentation** tab to build it first.")
            
            with col_ex4:
                with st.container(border=True):
                    st.markdown("#### 🖼️ Export PNG Charts")
                    st.markdown("Interactive chart assets can be exported in PNG format directly from the top-right tool menu of each Plotly chart.")
                    st.success("✓ Ready for High-Resolution PNG download via chart hover actions.")
else:
    # No file uploaded yet — premium landing page
    st.markdown("""
<style>
@keyframes float {
    0% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-10px) rotate(3deg); }
    100% { transform: translateY(0px) rotate(0deg); }
}
.hero-glow-container {
    position: relative;
    padding: 60px 40px;
    background: radial-gradient(circle at 50% 50%, rgba(59, 130, 246, 0.08), transparent 70%);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 24px;
    text-align: center;
    max-width: 800px;
    margin: 20px auto 40px auto;
    overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.hero-logo-svg {
    width: 64px;
    height: 64px;
    animation: float 6s ease-in-out infinite;
    filter: drop-shadow(0 0 15px rgba(59, 130, 246, 0.6));
    margin-bottom: 16px;
}
.hero-brand {
    font-size: 1.5rem;
    font-weight: 900;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #3B82F6 0%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 4px;
}
.hero-subbrand {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--text-secondary);
    text-transform: uppercase;
    margin-bottom: 24px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #FFFFFF;
    font-family: 'Outfit', sans-serif;
    letter-spacing: -1px;
    line-height: 1.15;
    margin-bottom: 16px;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    max-width: 620px;
    margin: 0 auto 32px auto;
}
.hero-action-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.25);
    border-radius: 9999px;
    padding: 8px 24px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #3B82F6;
    font-family: 'Outfit', sans-serif;
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.1);
}
.capability-card {
    background: #161B22;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 24px 20px;
    margin-bottom: 16px;
    transition: all 0.25s ease;
    height: 100%;
}
.capability-card:hover {
    transform: translateY(-4px);
    border-color: rgba(59, 130, 246, 0.2);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3), 0 0 15px rgba(59, 130, 246, 0.1);
}
.capability-icon {
    font-size: 1.8rem;
    color: #3B82F6;
    margin-bottom: 14px;
    display: inline-block;
}
.capability-title {
    font-size: 1.0rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 8px;
    font-family: 'Outfit', sans-serif;
}
.capability-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
    font-family: 'Inter', sans-serif;
}
</style>

<div class="hero-glow-container">
    <svg class="hero-logo-svg" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" fill="url(#gradHero)" />
        <path d="M2 17L12 22L22 17" stroke="url(#gradHero)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        <path d="M2 12L12 17L22 12" stroke="url(#gradHero)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        <defs>
            <linearGradient id="gradHero" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#3B82F6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#7C3AED;stop-opacity:1" />
            </linearGradient>
        </defs>
    </svg>
    <div class="hero-brand">VALKYRIE</div>
    <div class="hero-subbrand">Enterprise Data Intelligence</div>
    <div class="hero-title">Senior Data Analyst Platform</div>
    <div class="hero-subtitle">
        Transform business data into executive decisions using automated analytics, KPI dashboards and AI-powered business recommendations.
    </div>
    <div class="hero-action-pill">
        <span>📂</span> Upload CSV / Excel in the sidebar or click Load Sample Dataset
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")

    # Feature capability grid
    capabilities = [
        ("📂", "Data Ingestion", "Ingest CSV, XLSX, and XLS files with auto encoding detection and error-recovery pipelines."),
        ("🔎", "Data Profiling", "Inspect shapes, attributes counts, null ratios, schema specifications, and distribution statistics."),
        ("📊", "Visual Analytics", "Render responsive correlation heatmaps, categorical bar charts, and donut segment proportions."),
        ("📈", "Trend Analysis", "Apply rolling averages, regression trends, and time intelligence forecasting projections."),
        ("🎯", "Root Cause Analysis", "Identify segment concentrations, driver contributions, and statistical anomalies."),
        ("💼", "Executive Synthesis", "Synthesize findings into roadmaps, action briefings, and presentation briefings.")
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(capabilities):
        with cols[i % 3]:
            st.markdown(clean_html(f"""
            <div class="capability-card">
                <div class="capability-icon">{icon}</div>
                <div class="capability-title">{title}</div>
                <div class="capability-desc">{desc}</div>
            </div>"""), unsafe_allow_html=True)

