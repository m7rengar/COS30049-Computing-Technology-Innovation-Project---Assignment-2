import React from "react";
import "./App.css";

function About() {
  return (
    <div className="about-section">
      <h1 className="about-title">About This Project</h1>
      <p className="about-text">
        This AI-powered web application detects malicious or spam-like content
        in messages and uploaded files using trained machine learning models such as
        <strong> Random Forest </strong> and <strong> XGBoost</strong>.
        <br /><br />
        Built with <strong>React.js</strong> for the frontend and
        <strong> FastAPI</strong> for the backend, this project demonstrates
        the integration of modern web technologies with cybersecurity analysis.
      </p>

      <div className="about-card">
        <h2>Key Features</h2>
        <ul>
          <li> Real-time text classification</li>
          <li> File upload with instant analysis</li>
          <li> Responsive, cyber-themed interface</li>
          <li> 
            Machine learning model selection</li>
        </ul>
      </div>

      <p className="about-footer">
        Developed by <strong>MatrixShield</strong> © 2025
      </p>
    </div>
  );
}

export default About;
