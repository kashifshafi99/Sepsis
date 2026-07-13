const { useState, useEffect } = React;

function App() {
  const [page, setPage] = useState("prediction");

  return (
    <div>
      <header className="header-wrapper">
        <img src="logo.png" alt="Clinical Intelligence System Logo" className="nav-logo-outside" />
        <nav className="nav">
          <h1>Clinical Intelligence System</h1>
          <div className="nav-links">
            <a href="#" onClick={(e) => { e.preventDefault(); setPage("overview"); }} style={{ background: page === "overview" ? "rgba(255,255,255,0.1)" : "transparent" }}>
              Overview
            </a>
            <a href="#" onClick={(e) => { e.preventDefault(); setPage("prediction"); }} style={{ background: page === "prediction" ? "rgba(255,255,255,0.1)" : "transparent" }}>
              Prediction Dashboard
            </a>
            <a href="#" onClick={(e) => { e.preventDefault(); setPage("about"); }} style={{ background: page === "about" ? "rgba(255,255,255,0.1)" : "transparent" }}>
              About Developer
            </a>
          </div>
        </nav>
      </header>
      <div className="container">
        {page === "overview" && <Overview />}
        {page === "prediction" && <Prediction />}
        {page === "about" && <About />}
      </div>
    </div>
  );
}

function Overview() {
  return (
    <div className="card">
      <div className="card-header">Project Overview</div>
      <div style={{ lineHeight: "1.6" }}>
        <h2>Sepsis Prediction Model</h2>
        <p>Sepsis is a life-threatening condition that arises when the body's response to an infection causes injury to its tissues and organs. Early detection and intervention are crucial for improving patient outcomes and reducing mortality rates associated with sepsis.</p>
        <p>This project utilizes a <strong>Machine Learning Predictive Model (Logistic Regression)</strong> capable of accurately identifying patients at risk of developing sepsis based on vital clinical variables.</p>
        
        <h3>Dataset Details</h3>
        <ul>
          <li><strong>Source:</strong> Kaggle (PhysioNet Challenge 2019)</li>
          <li><strong>Scope:</strong> Over 1.5 million clinical records encompassing vital signs, laboratory results, and demographic information.</li>
          <li><strong>Features Utilized:</strong> Heart Rate, O2 Saturation, Temperature, Blood Pressure (SBP, DBP, MAP), Respiration Rate, and Age.</li>
        </ul>
        
        <h3>Methodology</h3>
        <p>Our methodology involved robust data processing (imputing missing variables and standard scaling) and predictive modeling to forecast sepsis onset. Logistic Regression was selected for its exceptional interpretability and high efficiency in binary classification tasks within clinical settings.</p>
      </div>
    </div>
  );
}

function Prediction() {
  const [formData, setFormData] = useState({
    HR: "",
    O2Sat: "",
    Temp: "",
    SBP: "",
    MAP: "",
    DBP: "",
    Resp: "",
    Age: "",
    Gender: "1",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
    setError("");
  };

  const requiredFields = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "Age"];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);

    const emptyFields = requiredFields.filter(f => formData[f] === "");
    if (emptyFields.length > 0) {
      setError("Please fill in all fields before analyzing: " + emptyFields.join(", "));
      return;
    }

    setLoading(true);
    setError("");
    
    const numericData = {};
    for (let key in formData) {
      if (formData[key] !== "") {
        numericData[key] = parseFloat(formData[key]);
      }
    }

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: numericData }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      setError("Error connecting to prediction server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">Sepsis Risk Prediction Model</div>
      <form onSubmit={handleSubmit}>
        <div className="grid">
          <div className="input-group">
            <label>Heart Rate (HR)</label>
            <input type="number" step="any" name="HR" value={formData.HR} onChange={handleChange} placeholder="e.g. 80" />
          </div>
          <div className="input-group">
            <label>O2 Saturation (%)</label>
            <input type="number" step="any" name="O2Sat" value={formData.O2Sat} onChange={handleChange} placeholder="e.g. 98" />
          </div>
          <div className="input-group">
            <label>Temperature (C)</label>
            <input type="number" step="any" name="Temp" value={formData.Temp} onChange={handleChange} placeholder="e.g. 37.5" />
          </div>
          <div className="input-group">
            <label>Systolic BP (SBP)</label>
            <input type="number" step="any" name="SBP" value={formData.SBP} onChange={handleChange} placeholder="e.g. 120" />
          </div>
          <div className="input-group">
            <label>Mean Arterial Pressure (MAP)</label>
            <input type="number" step="any" name="MAP" value={formData.MAP} onChange={handleChange} placeholder="e.g. 90" />
          </div>
          <div className="input-group">
            <label>Diastolic BP (DBP)</label>
            <input type="number" step="any" name="DBP" value={formData.DBP} onChange={handleChange} placeholder="e.g. 80" />
          </div>
          <div className="input-group">
            <label>Respiration Rate</label>
            <input type="number" step="any" name="Resp" value={formData.Resp} onChange={handleChange} placeholder="e.g. 16" />
          </div>
          <div className="input-group">
            <label>Age</label>
            <input type="number" step="any" name="Age" value={formData.Age} onChange={handleChange} placeholder="e.g. 45" />
          </div>
        </div>
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Risk"}
        </button>
      </form>

      {error && (
        <div className="result high-risk" style={{ marginTop: "16px" }}>
          <p style={{ margin: 0 }}>{error}</p>
        </div>
      )}

      {result && (
        <div className={`result-card ${result.prediction === 1 ? "result-high" : "result-low"}`}>
          <div className="result-icon-row">
            <div className={`result-icon ${result.prediction === 1 ? "icon-high" : "icon-low"}`}>
              {result.prediction === 1 ? "⚠" : "✓"}
            </div>
            <div>
              <h3 className="result-title">Analysis Complete</h3>
              <p className="result-subtitle">{result.prediction === 1 ? "Immediate clinical attention recommended" : "No immediate sepsis indicators detected"}</p>
            </div>
          </div>
          <div className="result-details">
            <div className="result-detail-item">
              <span className="result-label">Risk Prediction</span>
              <span className={`result-badge ${result.prediction === 1 ? "badge-high" : "badge-low"}`}>
                {result.prediction === 1 ? "Sepsis Alert (High Risk)" : "Stable (Low Risk)"}
              </span>
            </div>
            <div className="result-detail-item">
              <span className="result-label">Confidence Score</span>
              <span className="result-value">{(result.probability * 100).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function About() {
  return (
    <div className="about-container">
      <div className="animated-card profile-card">
        <img src="Kashif.jpg" alt="Kashif Shafi" className="profile-img-large" />
        <div className="profile-info">
          <h2>Kashif Shafi</h2>
          <p className="subtitle">Data Analyst & Project Manager</p>
          <div className="contact-info">
            <span><strong>Phone:</strong> 03471202463</span>
            <span><strong>Email:</strong> k.shafi_18883@khi.iba.edu.pk</span>
          </div>
          <p className="education">BSc. Accounting and Finance (IBA - Institute of Business Administration)</p>
        </div>
      </div>

      <h3 className="section-title">Professional Experience</h3>
      <div className="experience-grid">
        <div className="animated-card exp-card">
          <h4>Finance Executive</h4>
          <span className="company">Herbion International Inc</span>
          <p>Developed complex and visually appealing dashboards in Power BI, utilizing DAX queries for advanced calculations and measures, enhancing financial analysis and reporting.</p>
        </div>
        <div className="animated-card exp-card">
          <h4>Project Manager (Contract)</h4>
          <span className="company">Tapal Tea LLC</span>
          <p>Successfully managed the development and implementation of mobile applications catering to Territory, Zonal, and Regional Sales Managers to facilitate efficient shop capturing and trade visits.</p>
        </div>
      </div>

      <h3 className="section-title">Core Skills</h3>
      <div className="skills-container">
        {['Data Analysis (Python, Power BI, SQL)', 'Leadership & Team Management', 'Financial Management', 'Research and Analytical Skills', 'Effective Communication', 'Customer Relationship Management'].map(skill => (
          <div key={skill} className="skill-chip animated-card">
            {skill}
          </div>
        ))}
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
