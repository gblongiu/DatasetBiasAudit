# Student Performance Dashboard

This project is an interactive Dash-based web application for exploring student performance data. The app merges two datasets—one for a Portuguese course and one for a Math course—to provide a comprehensive view of student grades and demographics. Users can filter the data by subject, gender, school, study time, and age range, and view dynamic visualizations including summary statistics, histograms, scatter plots, box plots, and detailed data tables.

## Features

- **Interactive Filters:**  
  Apply filters for subject (Math, Portuguese, or Average), gender, school, study time, and age range to update visualizations in real time.

- **Multiple Visualizations:**  
  - **Overview:** Cards displaying average final grade, average first period grade, and average age.
  - **Histogram:** Final grade distribution by gender.
  - **Scatter Plot:** Relationship between first period and final grades.
  - **Box Plot:** Comparison of final grade distributions by gender.
  - **Data Table:** A detailed, filterable data table.

- **Dark Mode Toggle:**  
  Easily switch between light and dark mode. The dark mode styling applies to the entire app, including the offcanvas filter menu and dropdown options.

- **Responsive Design:**  
  Built using Dash and Bootstrap, ensuring a professional and responsive layout.

## Technologies

- **Python 3.12**
- **Dash & Plotly Express:** For interactive web visualizations.
- **Dash Bootstrap Components:** For layout and styling.
- **Pandas:** For data manipulation and analysis.
- **Gunicorn:** For deployment on cloud platforms.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gblongiu/DatasetBiasAudit.git
   cd DatasetBiasAudit
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *(If you haven’t generated requirements.txt yet, run `pip freeze > requirements.txt`.)*

## Running the App Locally

To start the app locally, run:

```bash
python biasAudit.py
```

Then, open your browser and navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Deployment

This project can be deployed to cloud platforms such as Railway, Render, or Fly.io. For Railway:

1. **Create a GitHub Repository:**  
   Push your code (including `biasAudit.py`, the `assets` folder with custom CSS, `requirements.txt`, and a `Procfile`) to GitHub.

2. **Procfile:**  
   Create a file named `Procfile` in your repository root with the following content:

   ```
   web: gunicorn biasAudit:server
   ```

3. **Deploy to Railway:**
   - Sign up at [Railway](https://railway.app/) and create a new project.
   - Connect your GitHub repository.
   - Railway will automatically build and deploy your app.
   - Access your live app via the provided URL.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements

- [Dash by Plotly](https://plotly.com/dash/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Pandas](https://pandas.pydata.org/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)

Feel free to update this README with additional details, screenshots, or instructions as needed.