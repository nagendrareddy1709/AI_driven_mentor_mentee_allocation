import streamlit as st
import pandas as pd
import os
import smtplib
from email.mime.text import MIMEText
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ======================================================
# CONFIG
# ======================================================

LOAD_PENALTY = 0.05
SKILL_WEIGHT = 0.6
RATING_WEIGHT = 0.3
LOAD_WEIGHT = 0.1
DEFAULT_RATING = 3.5

MENTORS_FILE = "mentors.csv"
MENTEES_FILE = "mentees.csv"
SPECIALIZATION_FILE = "master_specializations.csv"
COURSES_FILE = "master_courses.csv"

# ======================================================
# NORMALIZATION FUNCTION
# ======================================================

def normalize(text):
    if pd.isna(text):
        return ""
    return str(text).strip().lower()


# ======================================================
# DATAFRAME STRUCTURE VALIDATION
# ======================================================

def ensure_columns(df, columns_defaults):
    for col, default in columns_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

# ======================================================
# INITIALIZE FILES
# ======================================================

def initialize_files():

    if not os.path.exists(MENTORS_FILE):
        pd.DataFrame(columns=[
            "mentor_name","email","specializations",
            "assigned_count","avg_rating","total_ratings"
        ]).to_csv(MENTORS_FILE,index=False)

    if not os.path.exists(MENTEES_FILE):
        pd.DataFrame(columns=[
            "mentee_name","email","course",
            "assigned_mentor","status","rating"
        ]).to_csv(MENTEES_FILE,index=False)

    if not os.path.exists(SPECIALIZATION_FILE):
        pd.DataFrame(columns=["specialization"]).to_csv(SPECIALIZATION_FILE,index=False)

    if not os.path.exists(COURSES_FILE):
        pd.DataFrame(columns=["course"]).to_csv(COURSES_FILE,index=False)

# ======================================================
# LOAD DATA
# ======================================================

def load_data():

    mentors_df = pd.read_csv(MENTORS_FILE) 

    # ensure mentor columns exist
    mentors_df = ensure_columns(mentors_df,{
        "mentor_name":"",
        "email":"",
        "specializations":"",
        "assigned_count":0,
        "avg_rating":DEFAULT_RATING,
        "total_ratings":0
    })
            
    mentees_df = pd.read_csv(MENTEES_FILE)

    required_cols = [
        "mentee_name",
        "email",
        "course",
        "assigned_mentor",
        "status",
        "rating"
    ]

    for col in required_cols:
        if col not in mentees_df.columns:
            if col == "status":
                mentees_df[col] = "Not Assigned"
            elif col == "assigned_mentor":
                mentees_df[col] = "Unassigned"
            else:
                mentees_df[col] = ""

    mentees_df["assigned_mentor"] = mentees_df["assigned_mentor"].fillna("Unassigned")

    specialization_df = pd.read_csv(SPECIALIZATION_FILE)
    courses_df = pd.read_csv(COURSES_FILE)

    # clean specialization duplicates
    if not specialization_df.empty:
        specialization_df["specialization"] = specialization_df["specialization"].str.strip().str.title()
        specialization_df = specialization_df.drop_duplicates(subset=["specialization"])
        specialization_df.to_csv(SPECIALIZATION_FILE,index=False)

    # clean course duplicates
    if not courses_df.empty:
        courses_df["course"] = courses_df["course"].str.strip().str.title()
        courses_df = courses_df.drop_duplicates(subset=["course"])
        courses_df.to_csv(COURSES_FILE,index=False)

    return mentors_df, mentees_df, specialization_df, courses_df

# ======================================================
# EMAIL FUNCTION
# ======================================================

def send_email(to_email, subject, body):

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")

    if not sender or not password:
        st.warning("Email credentials not configured.")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com",465)
        server.login(sender,password)
        server.sendmail(sender,to_email,msg.as_string())
        server.quit()
    except Exception as e:
        st.error(f"Email failed: {e}")

# ======================================================
# AI MATCHING ENGINE
# ======================================================

def compute_best_mentor(mentee_course, mentors_df):

    if mentors_df.empty:
        return None

    mentor_texts = mentors_df["specializations"].fillna("").tolist()
    mentor_texts = [normalize(t) for t in mentor_texts]

    documents = mentor_texts + [normalize(mentee_course)]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])[0]

    scores = []

    for idx, sim in enumerate(similarity):

        assigned = mentors_df.iloc[idx].get("assigned_count",0)
        rating = mentors_df.iloc[idx].get("avg_rating",DEFAULT_RATING)

        if pd.isna(rating):
            rating = DEFAULT_RATING

        normalized_rating = rating / 5

        final_score = (
            (sim * SKILL_WEIGHT)
            + (normalized_rating * RATING_WEIGHT)
            - (LOAD_PENALTY * assigned * LOAD_WEIGHT)
        )

        scores.append(final_score)

    mentors_copy = mentors_df.copy()
    mentors_copy["temp_score"] = scores
    best_idx = mentors_copy["temp_score"].idxmax()
    return mentors_copy.loc[best_idx]

# ======================================================
# START APP
# ======================================================

initialize_files()

mentors_df, mentees_df, specialization_df, courses_df = load_data()

# dropdown lists
mentor_names = sorted(mentors_df["mentor_name"].dropna().unique().tolist())
mentee_names = sorted(mentees_df["mentee_name"].dropna().unique().tolist())

specializations_list = sorted(
    mentors_df["specializations"]
    .dropna()
    .str.split(r"\s*\|\s*")
    .explode()
    .str.strip()
    .unique()
    .tolist()
)

courses_list = sorted(
    mentees_df["course"]
    .dropna()
    .unique()
    .tolist()
)

# ======================================================
# STREAMLIT UI
# ======================================================

st.title("AI Mentor Allocation Platform")

# ======================================================
# DATA UPLOAD
# ======================================================

st.header("Upload Existing CSV Data")

mentor_file = st.file_uploader("Upload mentors.csv", type="csv")
mentee_file = st.file_uploader("Upload mentees.csv", type="csv")

if st.button("Process Uploaded Files"):

    if mentor_file:

        uploaded_mentors = pd.read_csv(mentor_file)
        uploaded_mentors.columns = uploaded_mentors.columns.str.strip().str.lower()

        # group mentors and combine specializations

        mentor_dict = {}

        for _, row in uploaded_mentors.iterrows():

            name = str(row["mentor_name"]).strip()
            spec = str(row["specializations"]).strip()

            if name not in mentor_dict:
                mentor_dict[name] = []

            mentor_dict[name].append(spec)

        processed_data = []

        for mentor, specs in mentor_dict.items():
            processed_data.append({
                "mentor_name": mentor,
                "email": "",
                "specializations": " | ".join(list(set(specs))),
                "assigned_count": 0,
                "avg_rating": DEFAULT_RATING,
                "total_ratings": 0
            })

        processed_df = pd.DataFrame(processed_data)

        processed_df.to_csv(MENTORS_FILE, index=False)

        all_specs = []

        for specs in processed_df["specializations"]:
            all_specs.extend(specs.split(" | "))

        spec_df = pd.DataFrame({"specialization": list(set(all_specs))})
        spec_df.to_csv(SPECIALIZATION_FILE, index=False)

        st.success("Mentor file processed")

    if mentee_file:

        uploaded_mentees = pd.read_csv(mentee_file)
        uploaded_mentees.columns = uploaded_mentees.columns.str.strip().str.lower()

        production_cols = [
            "mentee_name","email","course",
            "assigned_mentor","status","rating"
        ]

        for col in production_cols:

            if col not in uploaded_mentees.columns:

                if col == "status":
                    uploaded_mentees[col] = "Not Assigned"
                elif col == "rating":
                    uploaded_mentees[col] = 0
                else:
                    uploaded_mentees[col] = ""

        uploaded_mentees = uploaded_mentees[production_cols]
        uploaded_mentees = uploaded_mentees.drop_duplicates(subset=["mentee_name"])

        uploaded_mentees.to_csv(MENTEES_FILE,index=False)

        st.success("Mentee file processed")

    st.rerun()

# ======================================================
# ADD MENTOR
# ======================================================

st.header("Add Mentor")

mentor_name = st.selectbox("Mentor Name", [""] + mentor_names)
mentor_email = st.text_input("Mentor Email")

selected_specs = st.multiselect("Specializations", specializations_list)
new_spec = st.text_input("Add New Specialization")

if st.button("Add / Update Mentor"):

    if new_spec:

        if new_spec.lower() not in specialization_df["specialization"].str.lower().tolist():

            specialization_df.loc[len(specialization_df)] = new_spec.strip()
            specialization_df.to_csv(SPECIALIZATION_FILE,index=False)

            selected_specs.append(new_spec)

    if mentor_name and mentor_email:

        existing = mentors_df[
            (mentors_df["mentor_name"]==mentor_name)
            &
            (mentors_df["email"]==mentor_email)
        ]

        if not existing.empty:

            idx = existing.index[0]
            old_specs = mentors_df.loc[idx,"specializations"].split(" | ")

            merged = list(set(old_specs + selected_specs))
            mentors_df.loc[idx,"specializations"] = " | ".join(merged)

        else:

            mentors_df.loc[len(mentors_df)] = {
                "mentor_name":mentor_name,
                "email":mentor_email,
                "specializations":" | ".join(selected_specs),
                "assigned_count":0,
                "avg_rating":DEFAULT_RATING,
                "total_ratings":0
            }

        mentors_df.to_csv(MENTORS_FILE,index=False)
        st.success("Mentor Saved")
        st.rerun()

# ======================================================
# ADD MENTEE
# ======================================================

st.header("Add Mentee")

mentee_name = st.selectbox("Mentee Name", [""] + mentee_names)
mentee_email = st.text_input("Mentee Email")

selected_course = ""

if mentee_name:

    row = mentees_df[mentees_df["mentee_name"]==mentee_name]

    if not row.empty:
        selected_course = row.iloc[0]["course"]
        st.info(f"Detected Course: {selected_course}")

if not selected_course:
    selected_course = st.selectbox("Course", [""] + courses_list)

new_course = st.text_input("Add New Course")

if st.button("Allocate Mentor"):

    if new_course:

        if new_course.lower() not in courses_df["course"].str.lower().tolist():

            courses_df.loc[len(courses_df)] = new_course
            courses_df.to_csv(COURSES_FILE,index=False)

            selected_course = new_course

    if mentee_name and selected_course:

        best = compute_best_mentor(selected_course, mentors_df)

        if best is not None:

            mentor_assigned = best["mentor_name"]

            mentors_df.loc[
                mentors_df["mentor_name"]==mentor_assigned,
                "assigned_count"
            ] += 1

            mentors_df.to_csv(MENTORS_FILE,index=False)

            existing = mentees_df[
                mentees_df["mentee_name"]==mentee_name
            ]

            if not existing.empty:

                idx = existing.index[0]
                mentees_df.loc[idx,"assigned_mentor"] = mentor_assigned
                mentees_df.loc[idx,"status"] = "Assigned"

            else:

                mentees_df.loc[len(mentees_df)] = {
                    "mentee_name":mentee_name,
                    "email":mentee_email,
                    "course":selected_course,
                    "assigned_mentor":mentor_assigned,
                    "status":"Assigned",
                    "rating":""
                }

            mentees_df.to_csv(MENTEES_FILE,index=False)

            st.success(f"Assigned Mentor: {mentor_assigned}")
            st.rerun()

# ======================================================
# DASHBOARD
# ======================================================

st.header("Dashboard")

unique_mentors = mentors_df[["mentor_name","email"]].drop_duplicates()

st.write("Total Unique Mentors:", len(unique_mentors))
st.write("Total Mentees:", len(mentees_df))

st.subheader("Mentor – Mentee Allocation Overview")

if not mentees_df.empty and "assigned_mentor" in mentees_df.columns:

    # only show mentees who actually have a mentor
    assigned_df = mentees_df[
        (mentees_df["assigned_mentor"].notna()) &
        (mentees_df["assigned_mentor"] != "Unassigned")
    ].copy()

    if not assigned_df.empty:

        # calculate mentee count per mentor
        mentor_counts = assigned_df.groupby("assigned_mentor")["mentee_name"].count()

        # build table rows
        rows = []

        for mentor, group in assigned_df.groupby("assigned_mentor"):

            mentee_count = mentor_counts[mentor]

            for i, row in enumerate(group.itertuples()):

                rows.append({
                    "Mentor Name": mentor if i == 0 else "",
                    "Mentees Assigned": mentee_count if i == 0 else "",
                    "Mentee Name": row.mentee_name,
                    "Course": row.course
                })

        overview_df = pd.DataFrame(rows)

        st.dataframe(
            overview_df.style.set_properties(**{
                "white-space": "normal"
            }),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No mentor allocations yet.")

else:
    st.info("No mentor allocations yet.")

st.subheader("Mentor - Mentee Allocation Distribution")

if not mentors_df.empty:

    # count mentees per mentor
    mentor_counts = mentees_df.groupby("assigned_mentor")["mentee_name"].count()

    # get mentee names per mentor
    mentor_mentees = mentees_df.groupby("assigned_mentor")["mentee_name"].apply(list)

    # ensure all mentors appear
    chart_df = mentors_df.copy()

    chart_df["assigned_count"] = chart_df["mentor_name"].map(mentor_counts).fillna(0)
    chart_df["mentees"] = chart_df["mentor_name"].map(mentor_mentees).apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )

    chart_df = chart_df.sort_values(by="assigned_count", ascending=False)

    num_mentors = len(chart_df)
    fig_height = max(1.5, num_mentors * 0.2)

    fig, ax = plt.subplots(figsize=(12, fig_height), constrained_layout=True)

    bars = ax.barh(
        chart_df["mentor_name"],
        chart_df["assigned_count"],
        height=0.25
    )
    ax.invert_yaxis()
    ax.margins(y=0.005)
    max_count = int(chart_df["assigned_count"].max())

    ax.set_xlim(0, max_count + 1)
    ax.set_xticks(range(0, max_count + 5, 1))
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    # add padding for axis labels
    ax.set_xlabel("Number of Mentees Assigned", fontsize=18, labelpad=15)  # ↑ space from x-axis ticks
    ax.set_ylabel("Mentor Name", fontsize=18, labelpad=15)  # ← space from y-axis ticks

    # ax.set_title("Mentor, Mentee Allocation Distribution", fontsize=16, pad=20)

    # annotate bars
    for i, bar in enumerate(bars):
        count = int(bar.get_width())
        mentee_list = chart_df.iloc[i]["mentees"]
        label = f"{count}    ({mentee_list})"
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=10
        )

    # increase spacing between tick labels and axis labels
    ax.tick_params(axis='y', labelsize=14, pad=5)  # y-axis tick labels padding
    ax.tick_params(axis='x', labelsize=14, pad=5)  # x-axis tick labels padding

    st.pyplot(fig)


# ======================================================
# SYSTEM RESET
# ======================================================

if st.button("Reset Entire System (Demo Only)"):

    files = [MENTORS_FILE,MENTEES_FILE,SPECIALIZATION_FILE,COURSES_FILE]

    for f in files:
        if os.path.exists(f):
            os.remove(f)

    st.success("System Reset Completed")

    st.rerun()


