import React, { useState, useRef } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import "./App.css";
import logo from "./LOGO.png";
import About from "./About";

const API_URL = "http://localhost:8000";

function Home() {
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [model, setModel] = useState("random_forest");
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  // Handle text predictions
  const handlePredictText = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model }),
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Handle file upload
  const handleFileUpload = async (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", model);

    try {
      const res = await fetch(`${API_URL}/predict-file`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="hero-section">
      <h1 className="title">DETECT MALICIOUS EMAILS</h1>

      {/* Model Dropdown */}
      <div className="model-selector">
        <label htmlFor="model-dropdown">Choose AI Model:</label>
        <div className="dropdown-wrapper">
          <select
            id="model-dropdown"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </div>
      </div>

      {/* Text Input */}
      <div className="input-box">
        <textarea
          placeholder="Enter text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        ></textarea>

        <div className="button-area">
          <button
            onClick={handlePredictText}
            disabled={loading || !text.trim()}
          >
            {loading ? "..." : <i className="fa fa-play"></i>}
          </button>
        </div>
      </div>

      {/* File Upload with Drag and Drop */}
      <div
        className={`file-drop-zone ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: "none" }}
          onChange={(e) => handleFileUpload(e.target.files[0])}
        />
        <div onClick={() => fileInputRef.current.click()}>
          <p>
            <i className="fa fa-cloud-arrow-up"></i>
          </p>
          <p>Drag and drop your file here or click to browse</p>
        </div>
      </div>

      {file && (
        <p className="file-name">
          <i className="fa fa-file"></i> {file.name}
        </p>
      )}

      {/* Result Box with spam color and confidence bar */}
      {result && (
        <div
          className={`output ${
            result.prediction.toLowerCase() === "spam" ? "spam" : "ham"
          }`}
        >
          <p>
            <strong>Prediction:</strong> {result.prediction} (
            {(result.confidence * 100).toFixed(1)}%)
          </p>

          {/* Confidence Bar */}
          <div className="confidence-bar-container">
            <div
              className={`confidence-bar ${
                result.prediction.toLowerCase() === "spam" ? "spam" : "ham"
              }`}
              style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
            ></div>
          </div>
        </div>
      )}

      <p className="disclaimer">
        We are not liable for any losses or damages from use of this analysis.
      </p>
      <p className="footer">2025 ©</p>
    </div>
  );
}

function App() {
  return (
    <Router>
      <div className="main-container">
        <nav className="navbar">
          <div className="nav-left">
            <img src={logo} alt="logo" className="nav-logo" />
            <h1 className="nav-title">MatrixShield</h1>
          </div>
          <div className="nav-right">
            <Link to="/">HOME</Link>
            <Link to="/about">ABOUT</Link>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
