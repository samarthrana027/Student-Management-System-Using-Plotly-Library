# Student Management System — Flask + SQLite + Plotly + Machine Learning

## Features
- Add, search, update, and delete student records.
- SQLite database is created automatically at `database/students.db`.
- NumPy average calculation and Pandas/SciPy descriptive analysis.
- Plotly interactive charts: student average marks, subject averages, attendance vs marks, branch averages, and performance support flags.
- Scikit-learn Decision Tree classification demonstration.

## Run in VS Code (Windows)
1. Open this folder in VS Code.
2. Optional: create a virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install libraries:
   ```powershell
   pip install -r requirements.txt
   ```
4. Start the website:
   ```powershell
   python app.py
   ```
5. Open `http://127.0.0.1:5000`.

## Database
No MySQL installation or database setup is required. The application creates `database/students.db` and the `students` table automatically on first launch.

## Pages
- `/` — Student CRUD
- `/data-analysis` — Descriptive statistics
- `/analytics` — Interactive Plotly charts
- `/machine-learning` — Decision Tree classifier demo

## Notes
- Plotly is loaded from a CDN, so an internet connection is required to render the charts.
- The ML target is derived from average marks >= 40 and attendance >= 75%. This is a classroom demonstration, not a validated forecast of future academic performance.
