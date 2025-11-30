# modern_faculty_dashboard_enhanced_fixed.py
"""
Modern Faculty Research Dashboard (single file) — Fixed & cleaned
- Fixed syntax errors
- Auto-detect & clean messy CSV headers
- No accidental DataFrame prints (no st.write(df)/bare df)
- Defensive guards for plotting / empty data
- Keeps features: KPIs, profile cards, dept comparison, leaderboard,
  faculty detail + radar, patents/gender insights, totals table,
  clustering (optional), word cloud
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64

# -----------------------
# Page config + CSS
# -----------------------
st.set_page_config(page_title="Modern Faculty Research Dashboard", layout="wide", page_icon="📚")

CSS = """
<style>
/* background */
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg,#0f1720,#071022); }

/* card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  padding:14px;
  border-radius:14px;
  box-shadow: 0 8px 30px rgba(2,6,23,0.6);
  transition: transform .18s ease, box-shadow .18s ease;
  margin-bottom:12px;
}
.card:hover { transform: translateY(-6px); box-shadow: 0 18px 50px rgba(2,6,23,0.75); }

/* grid */
.card-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap:14px; }

/* KPI */
.kpi { font-size:28px; font-weight:800; color:white; }
.kpi-label { color:#bfc7d6; margin-top:4px; }

/* small */
.small-muted { color:#9aa5b1; font-size:13px; }

/* leaderboard */
.lb-row { display:flex; align-items:center; gap:10px; padding:8px 6px; border-bottom:1px solid rgba(255,255,255,0.03); }
.lb-rank { width:36px; text-align:center; font-weight:700; color:#ffd36e; }

/* responsive */
@media (max-width:600px){
  .card-grid { grid-template-columns: 1fr; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------
# Helpers
# -----------------------
def read_csv_with_header_detection(uploaded_file, header_keywords=("Department", "Faculty", "Faculty Name", "Name")):
    """
    Try to detect a correct header row in messy CSVs by scanning the first ~100 rows.
    If a header-like row is found (contains one of header_keywords), skip rows above it.
    Returns a pandas DataFrame or None on failure.
    """
    try:
        # read a sample without header to inspect rows
        uploaded_file.seek(0)
        sample = pd.read_csv(uploaded_file, header=None, nrows=200, encoding='utf-8', dtype=str)
    except Exception:
        # fallback: read directly with pandas default
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            return None

    header_row = None
    for i, row in sample.iterrows():
        vals = [str(v).strip() for v in row.values]
        # treat the row as header if it contains any header keyword (case-insensitive)
        if any(any(kw.lower() in (v or "").lower() for v in vals) for kw in header_keywords):
            header_row = i
            break

    try:
        uploaded_file.seek(0)
        if header_row is not None and header_row > 0:
            # skip the noisy lines before header_row
            df = pd.read_csv(uploaded_file, skiprows=range(header_row), header=0)
        else:
            df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Failed to parse CSV after header detection: {e}")
        return None

def avatar_data_uri(initials, size=120, bg=(75,207,180), fg=(6,17,26)):
    """Return data URI PNG for avatar (rounded square)."""
    if isinstance(bg, (list, tuple)):
        bg = tuple(int(x) for x in bg)
    if isinstance(fg, (list, tuple)):
        fg = tuple(int(x) for x in fg)
    img = Image.new("RGBA", (size, size), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([(0,0),(size,size)], radius=max(8, size//6), fill=bg + (255,))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size/3))
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), initials, font=font)
    w = bbox[2] - bbox[0]; h = bbox[3] - bbox[1]
    x = (size - w)/2; y = (size - h)/2
    draw.text((x,y), initials, font=font, fill=fg + (255,))
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/png;base64," + b64

def initials(name):
    parts = str(name).strip().split()
    if len(parts) == 0:
        return "?"
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# -----------------------
# Sidebar: upload + options
# -----------------------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload Faculty CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (demo)", value=False)

# Column mapping to normalize variants
AUTO_MAP_COLS = True
column_map = {
    "Dept": "Department",
    "Department Name": "Department",
    "Project Domain": "Domain",
    "Research Domain": "Domain",
    "Amount": "Amount_Lakhs",
    "Funding Amount": "Amount_Lakhs",
    "Funding (Lakhs)": "Amount_Lakhs",
    "Status of Project": "Status",
    "Project Status": "Status",
    "Project Title": "Title",
    "Name of Project": "Title",
    "Journal/Conference Publications": "Journal / Conference Publications",
    "Journal/Conference": "Journal / Conference Publications",
    "Phd Supervised": "No of Phd Supervised",
    "PhD Supervised": "No of Phd Supervised",
    "Awards & Honours": "Award and Honours",
    "Events Organized": "Events Organized( National / International)"
}

if uploaded is None and not use_sample:
    st.markdown(
        "<div class='card'><h2 style='color:white'>Modern Faculty Research Dashboard</h2>"
        "<p class='small-muted'>Upload a CSV to start. Toggle 'Use sample dataset' to preview the layout.</p></div>",
        unsafe_allow_html=True
    )
    st.stop()

# -----------------------
# Load data (uploaded or sample)
# -----------------------
if uploaded is not None:
    df = read_csv_with_header_detection(uploaded)
    if df is None:
        st.stop()
    # normalize headers
    df.columns = df.columns.astype(str).str.strip()
    if AUTO_MAP_COLS:
        rename_map = {col: column_map[col] for col in df.columns if col in column_map}
        if rename_map:
            df = df.rename(columns=rename_map)
else:
    # sample dataset
    sample = {
        "Faculty Name": ["Dr A L Sangal","Dr Urvashi","Dr Lalatendu Behera","Dr Nagendra Pratap Singh","Dr Kunwar Pal","Dr X Y","Dr Z Q","Dr P R"],
        "Department": ["CSE","CSE","CSE","CSE","CSE","ECE","ME","EEE"],
        "Gender": ["M","F","M","M","M","F","M","F"],
        "Designation": ["Professor (HAG) & Head","Asst Professor","Asst Professor","Asst Professor","Asst Professor","Associate Prof","Professor","Assistant Prof"],
        "Patents": [2,21,1,0,1,3,5,0],
        "Research Projects/Collabs": [0,2,3,2,4,1,5,2],
        "Books Published": [3,8,1,8,1,0,2,0],
        "Journal / Conference Publications": [48,12,8,51,18,10,30,5],
        "No of Phd Supervised": [18,2,0,8,6,1,9,0],
        "Professional Affiliations": [7,4,1,0,2,2,5,1],
        "Events Organized( National / International)": [13,9,4,3,6,1,7,2],
        "Award and Honours": [0,0,0,1,0,0,2,0]
    }
    df = pd.DataFrame(sample)
    df.columns = df.columns.str.strip()

# -----------------------
# Optional project-columns check (non-blocking)
# -----------------------
required_project_cols = ["Department", "Domain", "Amount_Lakhs", "Status", "Title"]
missing_project_cols = [c for c in required_project_cols if c not in df.columns]
if uploaded is not None and missing_project_cols:
    st.warning(
        f"Project-portfolio warning — missing columns (optional for faculty dashboard): "
        f"{', '.join(missing_project_cols)}. The dashboard will still run."
    )

# -----------------------
# Ensure numeric columns exist and coerce safely
# -----------------------
numeric_cols = [
    "Patents", "Research Projects/Collabs", "Books Published", "Journal / Conference Publications",
    "No of Phd Supervised", "Professional Affiliations", "Events Organized( National / International)", "Award and Honours"
]
for c in numeric_cols:
    if c in df.columns:
        # assign explicitly (no chained inplace)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    else:
        df[c] = 0.0

# -----------------------
# Filters
# -----------------------
st.sidebar.header("Filters")
dept_vals = sorted(df["Department"].dropna().unique().tolist()) if "Department" in df.columns else []
dept_options = ["All"] + dept_vals
selected_depts = st.sidebar.multiselect("Departments", options=dept_options, default=["All"])
if ("All" in selected_depts) or (not selected_depts):
    df_f = df.copy()
else:
    df_f = df[df["Department"].isin(selected_depts)].copy()

# publications slider - guard max
if "Journal / Conference Publications" in df_f.columns and not df_f["Journal / Conference Publications"].isna().all():
    max_pub = int(df_f["Journal / Conference Publications"].max())
else:
    max_pub = 0
min_pub = st.sidebar.slider("Min publications", 0, max(1000, max_pub), 0)
if "Journal / Conference Publications" in df_f.columns:
    df_f = df_f[df_f["Journal / Conference Publications"] >= min_pub]

# download filtered CSV if there is data
if not df_f.empty:
    st.sidebar.download_button("Download filtered CSV", data=df_to_csv_bytes(df_f), file_name="filtered_faculty.csv", mime="text/csv")

# -----------------------
# Header + KPIs
# -----------------------
st.markdown(
    "<div style='display:flex; justify-content:space-between; align-items:center; gap:20px'>"
    "<div><h1 style='color:white;margin:0'>📚 Faculty Research Dashboard</h1>"
    "<div class='small-muted'>Modern, interactive, and production-ready layout.</div></div>"
    f"<div style='text-align:right'><span class='small-muted'>Records: </span><b style='color:white'>{len(df_f)}</b></div></div>",
    unsafe_allow_html=True
)

# safe KPI computations
if df_f.empty:
    total_faculty = 0
    total_pubs = 0
    total_patents = 0
    avg_pubs = 0.0
else:
    total_faculty = df_f["Faculty Name"].nunique()
    total_pubs = int(df_f["Journal / Conference Publications"].sum())
    total_patents = int(df_f["Patents"].sum())
    avg_pubs = round(float(df_f["Journal / Conference Publications"].mean()), 2)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"<div class='card'><div class='kpi'>{total_faculty}</div><div class='kpi-label'>Total Faculty</div></div>", unsafe_allow_html=True)
with c2:
    if not df_f.empty:
        pub_series = df_f["Journal / Conference Publications"].sort_values(ascending=False).values
        spark = px.line(y=pub_series, height=64)
        spark.update_traces(line=dict(width=2))
        spark.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.markdown(f"<div class='card'><div class='kpi'>{total_pubs}</div><div class='kpi-label'>Total Publications</div>", unsafe_allow_html=True)
        st.plotly_chart(spark, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='card'><div class='kpi'>0</div><div class='kpi-label'>Total Publications</div></div>", unsafe_allow_html=True)
with c3:
    if not df_f.empty:
        pat_series = df_f["Patents"].sort_values(ascending=False).values
        spark2 = px.line(y=pat_series, height=64)
        spark2.update_traces(line=dict(width=2))
        spark2.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.markdown(f"<div class='card'><div class='kpi'>{total_patents}</div><div class='kpi-label'>Total Patents</div>", unsafe_allow_html=True)
        st.plotly_chart(spark2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='card'><div class='kpi'>0</div><div class='kpi-label'>Total Patents</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='card'><div class='kpi'>{avg_pubs}</div><div class='kpi-label'>Avg Publications</div></div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Profile cards grid
# -----------------------
st.markdown("### Faculty Profiles")
if df_f.empty:
    st.info("No records to display. Try uploading a CSV or changing filters.")
else:
    st.markdown("<div class='card-grid'>", unsafe_allow_html=True)
    for _, r in df_f.iterrows():
        name = r.get("Faculty Name", "Unknown")
        dept = r.get("Department", "-")
        desig = r.get("Designation", "-")
        pubs = int(r.get("Journal / Conference Publications", 0))
        patents = int(r.get("Patents", 0))
        books = int(r.get("Books Published", 0))
        awards = int(r.get("Award and Honours", 0))
        avatar = avatar_data_uri(initials(name), size=120)
        html = f"""
        <div class='card' style='padding:12px;'>
          <div style='display:flex; gap:12px; align-items:center;'>
            <img src="{avatar}" style="width:64px; height:64px; border-radius:10px;" />
            <div>
              <div style='font-weight:700; color:white'>{name}</div>
              <div class='small-muted'>{desig} • {dept}</div>
            </div>
          </div>
          <div style='display:flex; gap:8px; margin-top:10px; align-items:center;'>
            <div style='flex:1'>
              <div style='color:#a8d0e6; font-weight:700'>{pubs}</div><div class='small-muted'>Publications</div>
            </div>
            <div style='flex:1'>
              <div style='color:#ffd36e; font-weight:700'>{patents}</div><div class='small-muted'>Patents</div>
            </div>
            <div style='flex:1'>
              <div style='color:#b2f5ea; font-weight:700'>{books}</div><div class='small-muted'>Books</div>
            </div>
            <div style='flex:1'>
              <div style='color:#f6b0c1; font-weight:700'>{awards}</div><div class='small-muted'>Awards</div>
            </div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Department comparison (modern)
# -----------------------
st.markdown("## 📊 Department Comparison")
if df_f.empty:
    st.info("No department data available for current filters.")
    dept_group = pd.DataFrame()
else:
    dept_group = df_f.groupby("Department").agg({
        "Faculty Name": "nunique",
        "Journal / Conference Publications": "sum",
        "Patents": "sum",
        "Books Published": "sum",
        "Award and Honours": "sum",
        "No of Phd Supervised": "sum"
    }).rename(columns={"Faculty Name": "Faculty Count", "Journal / Conference Publications": "Publications"}).reset_index()

colA, colB = st.columns([2.2, 1])
with colA:
    if not dept_group.empty:
        fig_dept = px.bar(
            dept_group,
            x="Department",
            y=["Publications", "Patents", "Books Published"],
            barmode="group",
            title="Dept: Publications vs Patents vs Books",
            template="plotly_dark"
        )
        fig_dept.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_dept, use_container_width=True)
    else:
        st.info("No department data.")
with colB:
    if not dept_group.empty:
        nums = dept_group.select_dtypes(include=["number"]).columns
        fmt = {c: "{:.0f}" for c in nums}
        st.dataframe(dept_group.style.format(fmt), height=340)
    else:
        st.write("—")

# -----------------------
# Additional Insights
# -----------------------
st.markdown("---")
st.markdown("## 🔎 Additional Insights")
insA, insB = st.columns(2)
with insA:
    if not dept_group.empty and dept_group["Patents"].sum() > 0:
        pie = px.pie(dept_group, values='Patents', names='Department', title='Patents by Department', template='plotly_dark')
        st.plotly_chart(pie, use_container_width=True)
    else:
        st.info("No patent data.")
with insB:
    if 'Gender' in df_f.columns and not df_f['Gender'].isna().all():
        gender_group = df_f.groupby('Gender').agg({'Faculty Name': 'nunique', 'Journal / Conference Publications': 'sum'}).reset_index().rename(columns={'Faculty Name': 'Count', 'Journal / Conference Publications': 'Publications'})
        fig_g = px.bar(gender_group, x='Gender', y=['Count', 'Publications'], barmode='group', title='Gender: Count and Publications', template='plotly_dark')
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info('No gender data available.')

# Totals table
if not df_f.empty:
    totals_series = df_f[numeric_cols].sum().astype(int)
    totals_df = pd.DataFrame({'Metric': totals_series.index, 'Total': totals_series.values})
    st.markdown("### Totals across filtered records")
    st.dataframe(totals_df.style.format({'Total': '{:.0f}'}), height=260)
else:
    st.write("No totals to show for empty dataset.")

# Dept with max PhD supervised
if not dept_group.empty and 'No of Phd Supervised' in dept_group.columns:
    idx = dept_group['No of Phd Supervised'].idxmax()
    if pd.notna(idx):
        dept_max_phd = dept_group.loc[idx, 'Department']
        max_phd_val = int(dept_group.loc[idx, 'No of Phd Supervised'])
        st.markdown(f"### 🎓 Department with highest PhD supervised: **{dept_max_phd}** ({max_phd_val} total)")

st.markdown("---")

# -----------------------
# Leaderboard
# -----------------------
st.markdown("## 🏆 Leaderboard — Top Researchers")
if df_f.empty:
    st.info("No records for leaderboard.")
else:
    lb = df_f.sort_values("Journal / Conference Publications", ascending=False).head(10).reset_index(drop=True)
    for i, r in lb.iterrows():
        rank = i + 1
        st.markdown(f"""
        <div class='lb-row card'>
          <div class='lb-rank'>{rank}</div>
          <div style='flex:1'>
            <div style='font-weight:700; color:white'>{r.get('Faculty Name','')}</div>
            <div class='small-muted'>{r.get('Department','')} • {r.get('Designation','')}</div>
          </div>
          <div style='width:140px; text-align:right'>
            <div style='font-weight:800; color:#cdeffd'>{int(r.get('Journal / Conference Publications',0))}</div>
            <div class='small-muted'>Publications</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# Faculty detail + radar
# -----------------------
st.markdown("## Faculty Detail")
if df_f.empty:
    st.info("No faculty to choose from.")
else:
    names = df_f["Faculty Name"].tolist()
    names = list(dict.fromkeys(names))  # preserve order, unique
    selected = st.selectbox("Select faculty", options=names)
    if selected:
        rec = df_f[df_f["Faculty Name"] == selected].iloc[0]
        left, right = st.columns([1, 2])
        with left:
            st.image(avatar_data_uri(initials(rec["Faculty Name"]), size=200), width=160)
        with right:
            st.markdown(f"### {rec['Faculty Name']}")
            st.markdown(f"**{rec.get('Designation','')}**, {rec.get('Department','')}")
            st.metric("Publications", int(rec.get("Journal / Conference Publications", 0)))
            st.metric("Patents", int(rec.get("Patents", 0)))
            st.metric("Books", int(rec.get("Books Published", 0)))
            st.write("")
            df_det = pd.DataFrame({
                "Metric": ["PhD Supervised", "Projects/Collabs", "Affiliations", "Events Organized", "Awards"],
                "Value": [
                    int(rec.get("No of Phd Supervised", 0)),
                    int(rec.get("Research Projects/Collabs", 0)),
                    int(rec.get("Professional Affiliations", 0)),
                    int(rec.get("Events Organized( National / International)", 0)),
                    int(rec.get("Award and Honours", 0))
                ]
            })
            st.table(df_det)

        # Radar chart: selected vs department avg
        metrics = ["Patents", "Books Published", "Journal / Conference Publications", "Research Projects/Collabs", "No of Phd Supervised"]
        faculty_vals = [float(rec.get(m, 0)) for m in metrics]
        dept = rec.get("Department")
        if dept in df_f["Department"].values:
            dept_avg = df_f[df_f["Department"] == dept][metrics].mean()
        else:
            dept_avg = df_f[metrics].mean()
        dept_vals = [float(dept_avg.get(m, 0)) for m in metrics]

        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(r=faculty_vals, theta=metrics, fill='toself', name=selected))
        radar.add_trace(go.Scatterpolar(r=dept_vals, theta=metrics, fill='toself', name=f"{dept} avg"))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", height=420)
        st.plotly_chart(radar, use_container_width=True)

st.markdown("---")

# -----------------------
# Globe placeholder (collaborations)
# -----------------------
st.markdown("## 🌐 International Collaborations (Preview)")
if df_f.empty:
    st.markdown(
        "<div class='card' style='display:flex; align-items:center; gap:18px;'>"
        "<div style='flex:1'><h3 style='margin:0;color:white'>0 Collaborations</h3>"
        "<div class='small-muted'>Faculty with at least one project/collab</div></div>"
        "<div style='width:160px; text-align:center; font-size:64px;'>🌐</div></div>",
        unsafe_allow_html=True
    )
else:
    num_collabs = int((df_f["Research Projects/Collabs"] > 0).sum()) if "Research Projects/Collabs" in df_f.columns else 0
    st.markdown(
        f"<div class='card' style='display:flex; align-items:center; gap:18px;'>"
        f"<div style='flex:1'><h3 style='margin:0;color:white'>{num_collabs} Collaborations</h3>"
        f"<div class='small-muted'>Faculty with at least one project/collab</div></div>"
        "<div style='width:160px; text-align:center; font-size:64px;'>🌐</div></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------
# Clustering: KMeans + PCA
# -----------------------
st.markdown("## 🔬 Cluster Analysis — researcher archetypes")
cluster_metrics = ["Patents", "Journal / Conference Publications", "Books Published", "Research Projects/Collabs", "No of Phd Supervised"]
if df_f.shape[0] >= 2:
    X = df_f[cluster_metrics].fillna(0).values
    # limit K to reasonable bounds
    max_k = min(6, max(2, int(len(df_f))))
    k = st.sidebar.slider("K clusters (KMeans)", 2, max_k, 3)
    if X.shape[0] >= k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        labels = kmeans.labels_
        pca = PCA(n_components=2, random_state=42)
        pcs = pca.fit_transform(X)
        cd = pd.DataFrame({"pc1": pcs[:, 0], "pc2": pcs[:, 1], "cluster": labels, "name": df_f["Faculty Name"].values})
        clust_fig = px.scatter(cd, x="pc1", y="pc2", color="cluster", hover_name="name", template="plotly_dark", height=480)
        st.plotly_chart(clust_fig, use_container_width=True)
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_metrics)
        st.markdown("### Cluster centers (averages)")
        st.dataframe(centers.round(2))
    else:
        st.info("Not enough rows for chosen K.")
else:
    st.info("Add more rows to enable clustering.")

st.markdown("---")

# -----------------------
# Word cloud (safe)
# -----------------------
st.markdown("## ☁ Word Cloud (Designation & Department)")
parts = []
if "Designation" in df_f.columns:
    parts += df_f["Designation"].dropna().astype(str).tolist()
if "Department" in df_f.columns:
    parts += df_f["Department"].dropna().astype(str).tolist()
txt = " ".join(parts).strip()
if len(txt.split()) == 0:
    st.warning("No textual data for word cloud.")
else:
    try:
        wc = WordCloud(width=900, height=360, background_color=None, mode="RGBA", colormap="tab10").generate(txt)
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        st.image(buf)
    except Exception as e:
        st.warning(f"Could not generate word cloud: {e}")

st.markdown("---")
st.markdown("<div class='small-muted'>Pro tips: try department filters, increase min publications, or upload richer CSV (country of collaborators for real maps).</div>", unsafe_allow_html=True)