import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats

from database.database import connect_database, init_database

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "local-student-management-secret")
init_database()

SUBJECTS = ["math_marks", "python_marks", "cn_marks", "wad_marks"]
FEATURES = SUBJECTS + ["attendance"]

def load_data():
    with connect_database() as connection:
        return pd.read_sql_query("SELECT * FROM students ORDER BY student_id DESC", connection)

def validate_form(form):
    roll_no = form.get("roll_no", "").strip()
    name = form.get("name", "").strip()
    branch = form.get("branch", "").strip()
    email = form.get("email", "").strip()
    mobile = form.get("mobile", "").strip()
    if not roll_no or not name or not branch:
        raise ValueError("Roll number, name and branch are required.")
    semester = int(form.get("semester", ""))
    if not 1 <= semester <= 8:
        raise ValueError("Semester must be between 1 and 8.")
    marks = [float(form.get(field, "")) for field in SUBJECTS]
    attendance = float(form.get("attendance", ""))
    if any(mark < 0 or mark > 100 for mark in marks):
        raise ValueError("Subject marks must be between 0 and 100.")
    if attendance < 0 or attendance > 100:
        raise ValueError("Attendance must be between 0 and 100.")
    return (roll_no, name, branch, semester, email, mobile, *marks,
            float(np.mean(marks)), attendance)

@app.route("/")
def index():
    df = load_data()
    return render_template("index.html", students=df.to_dict("records"))

@app.route("/add", methods=["POST"])
def add_student():
    try:
        data = validate_form(request.form)
        with connect_database() as connection:
            connection.execute("""
                INSERT INTO students
                (roll_no,name,branch,semester,email,mobile,math_marks,python_marks,
                 cn_marks,wad_marks,average_marks,attendance)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, data)
        flash("Student added successfully.", "success")
    except ValueError as exc:
        flash(str(exc), "error")
    except Exception as exc:
        message = "That roll number already exists." if "UNIQUE constraint failed" in str(exc) else str(exc)
        flash(f"Could not add student: {message}", "error")
    return redirect(url_for("index"))

@app.route("/update/<int:student_id>", methods=["POST"])
def update_student(student_id):
    try:
        data = validate_form(request.form)
        with connect_database() as connection:
            cursor = connection.execute("""
                UPDATE students SET roll_no=?,name=?,branch=?,semester=?,email=?,mobile=?,
                math_marks=?,python_marks=?,cn_marks=?,wad_marks=?,average_marks=?,attendance=?
                WHERE student_id=?
            """, data + (student_id,))
            changed = cursor.rowcount
        flash("Student updated successfully." if changed else "Student not found.", "success")
    except ValueError as exc:
        flash(str(exc), "error")
    except Exception as exc:
        message = "That roll number already exists." if "UNIQUE constraint failed" in str(exc) else str(exc)
        flash(f"Could not update student: {message}", "error")
    return redirect(url_for("index"))

@app.route("/delete/<int:student_id>", methods=["POST"])
def delete_student(student_id):
    try:
        with connect_database() as connection:
            cursor = connection.execute("DELETE FROM students WHERE student_id=?", (student_id,))
            changed = cursor.rowcount
        flash("Student deleted." if changed else "Student record not found.", "success")
    except Exception as exc:
        flash(f"Could not delete student: {exc}", "error")
    return redirect(url_for("index"))

@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    pattern = f"%{query}%"
    with connect_database() as connection:
        rows = connection.execute("""
            SELECT * FROM students WHERE roll_no LIKE ? OR name LIKE ?
            ORDER BY student_id DESC
        """, (pattern, pattern)).fetchall()
    return render_template("index.html", students=[dict(row) for row in rows], search_query=query)

@app.route("/analytics")
def analytics():
    df = load_data()
    if df.empty:
        flash("Add student records before opening analytics.", "error")
        return redirect(url_for("index"))

    import plotly.graph_objects as go
    subject_names = ["Mathematics", "Python", "Computer Networks", "WAD"]
    subject_averages = [round(float(df[col].mean()), 2) for col in SUBJECTS]
    student_labels = df["roll_no"].astype(str).tolist()
    student_averages = df["average_marks"].round(2).tolist()
    branch_stats = df.groupby("branch")["average_marks"].mean().round(2)
    risk_counts = pd.Series(
        np.where((df["average_marks"] >= 40) & (df["attendance"] >= 75),
                 "Pass criteria met", "Needs attention")
    ).value_counts()

    figures = [
        go.Figure(data=[go.Bar(x=student_labels, y=student_averages)],
                  layout=go.Layout(title="Average Marks by Student",
                                   xaxis_title="Roll Number", yaxis_title="Average Marks (0–100)")),
        go.Figure(data=[go.Bar(x=subject_names, y=subject_averages)],
                  layout=go.Layout(title="Average Marks by Subject",
                                   xaxis_title="Subject", yaxis_title="Average Marks (0–100)")),
        go.Figure(data=[go.Scatter(x=df["attendance"].tolist(), y=df["average_marks"].tolist(),
                                   mode="markers", text=df["roll_no"].astype(str).tolist())],
                  layout=go.Layout(title="Attendance vs Average Marks",
                                   xaxis_title="Attendance (%)", yaxis_title="Average Marks (0–100)")),
        go.Figure(data=[go.Bar(x=branch_stats.index.tolist(), y=branch_stats.values.tolist())],
                  layout=go.Layout(title="Average Marks by Branch",
                                   xaxis_title="Branch", yaxis_title="Average Marks (0–100)")),
        go.Figure(data=[go.Pie(labels=risk_counts.index.tolist(),
                               values=risk_counts.values.tolist(), hole=.35)],
                  layout=go.Layout(title="Student Performance Support Flags"))
    ]
    return render_template("analytics.html",
        graphs=[fig.to_json() for fig in figures],
        total=len(df),
        mean_score=round(float(df["average_marks"].mean()), 2),
        mean_attendance=round(float(df["attendance"].mean()), 2))

@app.route("/machine-learning", methods=["GET", "POST"])
def machine_learning():
    result = None
    report = None
    try:
        df = load_data()
        if len(df) < 6:
            flash("Add at least 6 student records with varied pass/fail outcomes to train the model.", "error")
        else:
            df["result"] = np.where((df["average_marks"] >= 40) & (df["attendance"] >= 75), 1, 0)
            X = df[FEATURES]
            y = df["result"]
            if y.nunique() < 2 or y.value_counts().min() < 2:
                flash("The current records need examples of both outcome classes (at least 2 in each class).", "error")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.30, random_state=42, stratify=y)
                model = DecisionTreeClassifier(random_state=42, max_depth=4)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                result = round(float(accuracy_score(y_test, predictions)) * 100, 2)
                report = classification_report(y_test, predictions, zero_division=0)
    except Exception as exc:
        flash(f"Machine learning error: {exc}", "error")
    return render_template("machine_learning.html", accuracy=result, report=report)

@app.route("/data-analysis")
def data_analysis():
    df = load_data()
    cols = SUBJECTS + ["average_marks", "attendance"]
    summary = df[cols].describe().round(2).to_html(classes="data-table", border=0) if not df.empty else None
    return render_template("data_analysis.html", summary=summary, count=len(df))

if __name__ == "__main__":
    app.run(debug=True)
