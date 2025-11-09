/**
 * MatrixShield - Enhanced Spam Detection Web Application
 * COS30049 - Computing Technology Innovation Project - Assignment 3
 * * This React application provides:
 * - Single text prediction with real-time results
 * - Batch prediction with CSV file upload
 * - Prediction history tracking and management (using Local Storage)
 * - 4 interactive data visualizations (Chart.js & Recharts)
 * - Export functionality (CSV/JSON)
 * - Real-time statistics and monitoring (derived from local history)
 * - Comprehensive error handling
 * - Responsive design with smooth animations
 * * Version: 2.1.0 (Aligned with stateless backend)
 * Date: 2025-11-09
 */

import React, { useState, useRef, useEffect, useMemo } from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import axios from "axios";
import Papa from "papaparse";
import "./App.css";
import logo from "./LOGO.png";
import About from "./About";

// Chart.js imports for interactive charts
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar, Pie } from 'react-chartjs-2';


// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// API Configuration
const API_URL = "http://localhost:8000";
const HISTORY_STORAGE_KEY = "predictionHistory";

// ============================================================================
// MAIN HOME COMPONENT
// ============================================================================
function Home() {
  // ========== STATE MANAGEMENT ==========
  const [text, setText] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [model, setModel] = useState("random_forest");
  const [dragActive, setDragActive] = useState(false);
  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  
  const fileInputRef = useRef(null);

  // ========== LOAD HISTORY ON MOUNT ==========
  useEffect(() => {
    loadHistoryFromStorage();
  }, []);

  // ========== DERIVED STATE FOR STATS ==========
  // Calculate stats from history state instead of fetching
  const stats = useMemo(() => {
    if (history.length === 0) {
      return null;
    }
    
    const spam_predictions = history.filter(h => h.is_spam).length;
    const ham_predictions = history.filter(h => !h.is_spam).length;
    const total_predictions = history.length;
    const spam_percentage = total_predictions > 0 
      ? ((spam_predictions / total_predictions) * 100).toFixed(0) 
      : 0;
      
    return {
      prediction_statistics: {
        total_predictions,
        spam_predictions,
        ham_predictions,
        spam_percentage
      }
    };
  }, [history]);

  // ========== LOCAL HISTORY FUNCTIONS ==========

  /**
   * Load history from Local Storage
   */
  const loadHistoryFromStorage = () => {
    try {
      const savedHistory = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (savedHistory) {
        setHistory(JSON.parse(savedHistory));
      }
    } catch (err) {
      console.error("Failed to load history from storage:", err);
      localStorage.removeItem(HISTORY_STORAGE_KEY);
    }
  };

  /**
   * Add a new prediction (or array of) to history and save
   */
  const updateHistory = (newItems) => {
    const itemsArray = Array.isArray(newItems) ? newItems : [newItems];
    setHistory(prevHistory => {
      const updatedHistory = [...itemsArray, ...prevHistory];
      try {
        localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(updatedHistory));
      } catch (err) {
        console.error("Failed to save history to storage:", err);
        setError("Failed to save history. Storage might be full.");
      }
      return updatedHistory;
    });
  };

  // ========== API FUNCTIONS ==========

  /**
   * Handle single text prediction
   */
  const handlePredictText = async () => {
    if (!text.trim()) {
      setError("Please enter some text to analyze");
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const res = await axios.post(`${API_URL}/predict`, {
        text: text.trim(),
        model
      });
      
      const data = res.data;
      
      // Adapt backend response to frontend format
      const resultData = {
        ...data,
        prediction_id: data.timestamp, // Use timestamp as a unique ID
        confidence_percentage: data.confidence * 100
      };
      
      setResult(resultData);
      setSuccessMessage("Prediction completed successfully!");
      
      // Create and save new history item
      const newHistoryItem = {
        ...resultData,
        text: text.trim(),
        text_preview: text.trim().substring(0, 50) + "..."
      };
      updateHistory(newHistoryItem);
      
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);
      
    } catch (err) {
      console.error("Prediction error:", err);
      setError(
        err.response?.data?.detail || 
        "Prediction failed. Please check your connection and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle file selection for parsing and batch prediction
   */
  const handleFileUpload = (selectedFile) => {
    if (!selectedFile) return;

    // Validate file type
    const validTypes = ['.txt', '.csv'];
    const fileExt = selectedFile.name.substring(selectedFile.name.lastIndexOf('.')).toLowerCase();
    
    if (!validTypes.includes(fileExt)) {
      setError("Invalid file type. Please upload a .txt or .csv file");
      return;
    }

    setFile(selectedFile);
    setLoading(true);
    setResult(null);
    setError(null);

    // Read and parse file on client-side
    const fileReader = new FileReader();
    fileReader.onload = (e) => {
      const content = e.target.result;
      let texts = [];
      
      try {
        if (selectedFile.name.endsWith('.csv')) {
          Papa.parse(content, {
            header: false,
            skipEmptyLines: true,
            complete: (results) => {
              texts = results.data.map(row => row[0]); // Assume text is in the first column
              sendBatchPrediction(texts);
            },
            error: (err) => {
              throw new Error(err.message);
            }
          });
        } else if (selectedFile.name.endsWith('.txt')) {
          texts = content.split('\n').filter(line => line.trim() !== '');
          sendBatchPrediction(texts);
        }
      } catch (parseErr) {
        console.error("File parse error:", parseErr);
        setError(`Failed to parse file: ${parseErr.message}`);
        setLoading(false);
      }
    };
    fileReader.onerror = () => {
      setError("Failed to read the file.");
      setLoading(false);
    };
    fileReader.readAsText(selectedFile);
  };
  
  /**
   * Helper function to send parsed texts to /predict-batch
   */
  const sendBatchPrediction = async (texts) => {
    if (texts.length === 0) {
      setError("File is empty or could not be parsed.");
      setLoading(false);
      return;
    }

    try {
      const res = await axios.post(
        `${API_URL}/predict-batch`,
        { texts, model }
      );

      const data = res.data; // This is BatchPredictionResponse

      // Process results to match frontend expectations
      let spam_count = 0;
      let ham_count = 0;
      let confidence_sum = 0;

      const newHistoryItems = data.results.map(item => {
        if (item.is_spam) spam_count++;
        else ham_count++;
        confidence_sum += item.confidence;
        
        // Reconstruct full history item
        return {
          prediction_id: `${data.timestamp}-${item.index}`,
          timestamp: data.timestamp,
          text: texts[item.index], // Get original text
          text_preview: item.text_preview,
          model: data.model_used,
          is_spam: item.is_spam,
          prediction: item.prediction,
          confidence: item.confidence,
          confidence_percentage: item.confidence * 100
        };
      });
      
      const average_confidence = texts.length > 0 ? (confidence_sum / texts.length) : 0;

      // Set the result box
      setResult({
        total_predictions: data.total_predictions,
        spam_count,
        ham_count,
        average_confidence
      });

      setSuccessMessage(
        `Batch prediction completed! Analyzed ${data.total_predictions} messages.`
      );

      // Update history (add in reverse to keep correct order at top)
      updateHistory(newHistoryItems.reverse());

      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);

    } catch (err) {
      console.error("Batch prediction error:", err);
      setError(
        err.response?.data?.detail || 
        "Batch prediction failed. Please check the file format and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  /**
   * Export prediction history (client-side)
   */
  const handleExport = (format = 'csv') => {
    if (history.length === 0) {
      setError("No history to export.");
      return;
    }
    
    try {
      let data, blob, filename;
      
      if (format === 'json') {
        data = JSON.stringify(history, null, 2);
        blob = new Blob([data], { type: "application/json" });
        filename = `predictions_${Date.now()}.json`;
      } else { // csv
        const headers = "timestamp,model,prediction,is_spam,confidence,text_preview,text\n";
        const rows = history.map(h => {
          const textPreview = (h.text_preview || "").replace(/"/g, '""');
          const fullText = (h.text || "").replace(/"/g, '""').replace(/\n/g, '\\n');
          return `"${h.timestamp}","${h.model}","${h.prediction}",${h.is_spam},${h.confidence},"${textPreview}","${fullText}"`;
        }).join('\n');
        data = headers + rows;
        blob = new Blob([data], { type: "text/csv;charset=utf-8," });
        filename = `predictions_${Date.now()}.csv`;
      }

      // Create download link
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url); // Clean up

      setSuccessMessage(`Exported ${history.length} predictions as ${format.toUpperCase()}`);
      setTimeout(() => setSuccessMessage(null), 3000);

    } catch (err) {
      console.error("Export error:", err);
      setError("Failed to export predictions");
    }
  };

  /**
   * Clear prediction history (client-side)
   */
  const handleClearHistory = () => {
    if (!window.confirm("Are you sure you want to clear all prediction history?")) {
      return;
    }

    try {
      setHistory([]);
      setResult(null); // Clear result box
      localStorage.removeItem(HISTORY_STORAGE_KEY);
      setSuccessMessage("History cleared successfully");
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      console.error("Clear history error:", err);
      setError("Failed to clear history");
    }
  };

  // ========== DRAG AND DROP HANDLERS ==========
  
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

  // ========== CHART DATA PREPARATION ==========

  /**
   * Prepare data for confidence trend line chart
   */
  const getConfidenceTrendData = () => {
    const recentHistory = history.slice(0, 20).reverse(); // First 20 (most recent)
    
    return {
      labels: recentHistory.map((_, idx) => `#${history.length - idx}`),
      datasets: [
        {
          label: 'Confidence Score',
          data: recentHistory.map(h => (h.confidence * 100).toFixed(2)),
          borderColor: 'rgb(0, 188, 212)',
          backgroundColor: 'rgba(0, 188, 212, 0.1)',
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointHoverRadius: 6
        }
      ]
    };
  };

  /**
   * Prepare data for spam vs ham comparison bar chart
   */
  const getSpamHamData = () => {
    const spamCount = stats?.prediction_statistics?.spam_predictions || 0;
    const hamCount = stats?.prediction_statistics?.ham_predictions || 0;

    return {
      labels: ['Spam', 'Ham (Not Spam)'],
      datasets: [
        {
          label: 'Message Count',
          data: [spamCount, hamCount],
          backgroundColor: [
            'rgba(255, 99, 132, 0.7)',
            'rgba(75, 192, 192, 0.7)'
          ],
          borderColor: [
            'rgba(255, 99, 132, 1)',
            'rgba(75, 192, 192, 1)'
          ],
          borderWidth: 2
        }
      ]
    };
  };

  /**
   * Prepare data for model usage pie chart
   */
  const getModelUsageData = () => {
    const modelCounts = {};
    history.forEach(h => {
      const modelName = h.model || 'unknown';
      modelCounts[modelName] = (modelCounts[modelName] || 0) + 1;
    });

    return {
      labels: Object.keys(modelCounts).map(m => 
        m.replace('_', ' ').toUpperCase()
      ),
      datasets: [
        {
          label: 'Predictions by Model',
          data: Object.values(modelCounts),
          backgroundColor: [
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(153, 102, 255, 0.7)'
          ],
          borderColor: [
            'rgba(54, 162, 235, 1)',
            'rgba(255, 206, 86, 1)',
            'rgba(153, 102, 255, 1)'
          ],
          borderWidth: 2
        }
      ]
    };
  };

  // ========== CHART OPTIONS ==========
  
  const lineChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'top',
        labels: { color: '#fff', font: { size: 12 } }
      },
      title: {
        display: true,
        text: 'Prediction Confidence Trend (Last 20)',
        color: '#00bcd4',
        font: { size: 16, weight: 'bold' }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `Confidence: ${context.parsed.y}%`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: { color: '#fff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      x: {
        ticks: { color: '#fff' },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      }
    },
    interaction: {
      mode: 'index',
      intersect: false
    }
  };

  const barChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      title: {
        display: true,
        text: 'Spam vs Ham Distribution',
        color: '#00bcd4',
        font: { size: 16, weight: 'bold' }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const total = history.length;
            const percentage = total > 0 ? ((context.parsed.y / total) * 100).toFixed(1) : 0;
            return `Count: ${context.parsed.y} (${percentage}%)`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: { color: '#fff', precision: 0 },
        grid: { color: 'rgba(255, 255, 255, 0.1)' }
      },
      x: {
        ticks: { color: '#fff' },
        grid: { display: false }
      }
    }
  };

  const pieChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: true,
        position: 'bottom',
        labels: { color: '#fff', padding: 15, font: { size: 12 } }
      },
      title: {
        display: true,
        text: 'Model Usage Statistics',
        color: '#00bcd4',
        font: { size: 16, weight: 'bold' }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((context.parsed / total) * 100).toFixed(1);
            return `${context.label}: ${context.parsed} (${percentage}%)`;
          }
        }
      }
    }
  };

  // ========== RENDER ==========
  
  return (
    <div className="hero-section">
      {/* Error Alert */}
      {error && (
        <div className="alert alert-error">
          <i className="fa fa-exclamation-triangle"></i>
          {error}
          <button onClick={() => setError(null)} className="alert-close">×</button>
        </div>
      )}

      {/* Success Alert */}
      {successMessage && (
        <div className="alert alert-success">
          <i className="fa fa-check-circle"></i>
          {successMessage}
        </div>
      )}

      <h1 className="title">DETECT MALICIOUS EMAILS</h1>

      {/* Model Selector */}
      <div className="model-selector">
        <label htmlFor="model-dropdown">Choose AI Model:</label>
        <div className="dropdown-wrapper">
          <select
            id="model-dropdown"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={loading}
          >
            <option value="random_forest">Random Forest</option>
            <option value="xgboost">XGBoost</option>
          </select>
        </div>
      </div>

      {/* Text Input Section */}
      <div className="input-section">
        <div className="input-box">
          <textarea
            placeholder="Enter text here to check if it's spam or malicious..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={loading}
          ></textarea>

          <div className="button-area">
            <button
              onClick={handlePredictText}
              disabled={loading || !text.trim()}
              className="predict-button"
            >
              {loading ? (
                <i className="fa fa-spinner fa-spin"></i>
              ) : (
                <i className="fa fa-play"></i>
              )}
            </button>
          </div>
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
          onChange={(e) => {
            handleFileUpload(e.target.files[0]);
            e.target.value = null; // Reset file input
          }}
          accept=".txt,.csv"
          disabled={loading}
        />
        <div onClick={() => !loading && fileInputRef.current.click()}>
          <p>
            <i className="fa fa-cloud-arrow-up fa-2x"></i>
          </p>
          <p className="upload-text">
            Drag and drop your file here or click to browse
          </p>
          <p className="file-info">Supports: .txt, .csv files</p>
        </div>
      </div>

      {file && !loading && (
        <p className="file-name">
          <i className="fa fa-file"></i> {file.name}
        </p>
      )}

      {/* Result Box */}
      {result && (
        <div className={`output ${result.is_spam || (result.spam_count > result.ham_count) ? 'spam-result' : 'ham-result'}`}>
          {result.total_predictions ? (
            // Batch prediction result
            <div className="batch-result">
              <h3>
                <i className="fa fa-chart-bar"></i> Batch Analysis Complete
              </h3>
              <div className="result-grid">
                <div className="result-stat">
                  <div className="stat-label">Total Analyzed</div>
                  <div className="stat-value">{result.total_predictions}</div>
                </div>
                <div className="result-stat spam">
                  <div className="stat-label">Spam Detected</div>
                  <div className="stat-value">{result.spam_count}</div>
                </div>
                <div className="result-stat ham">
                  <div className="stat-label">Legitimate</div>
                  <div className="stat-value">{result.ham_count}</div>
                </div>
                <div className="result-stat">
                  <div className="stat-label">Avg Confidence</div>
                  <div className="stat-value">
                    {(result.average_confidence * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <button 
                onClick={() => setShowHistory(true)} 
                className="view-details-btn"
              >
                <i className="fa fa-list"></i> View Detailed Results
              </button>
            </div>
          ) : (
            // Single prediction result
            <div className="single-result">
              <div className="result-icon">
                {result.is_spam ? (
                  <i className="fa fa-exclamation-triangle"></i>
                ) : (
                  <i className="fa fa-check-circle"></i>
                )}
              </div>
              <h3 className="result-title">
                {result.is_spam ? "⚠️ SPAM DETECTED" : "✓ LEGITIMATE MESSAGE"}
              </h3>
              <div className="confidence-bar-container">
                <div 
                  className="confidence-bar" 
                  style={{ width: `${result.confidence_percentage}%` }}
                >
                  {result.confidence_percentage.toFixed(1)}% Confident
                </div>
              </div>
              <p className="result-detail">
                <strong>Prediction ID:</strong> {result.prediction_id}
              </p>
              <p className="result-detail">
                <strong>Model Used:</strong> {result.model_used.replace('_', ' ').toUpperCase()}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Statistics Dashboard - Only show if we have history */}
      {history.length > 0 && (
        <div className="stats-dashboard">
          <div className="dashboard-header">
            <h2>
              <i className="fa fa-chart-line"></i> Analytics Dashboard
            </h2>
            <div className="dashboard-controls">
              <button 
                onClick={() => setShowHistory(!showHistory)} 
                className="control-btn"
              >
                <i className={`fa ${showHistory ? 'fa-chart-bar' : 'fa-history'}`}></i>
                {showHistory ? 'Show Charts' : 'Show History'}
              </button>
              <button onClick={() => handleExport('csv')} className="control-btn">
                <i className="fa fa-download"></i> Export CSV
              </button>
              <button onClick={() => handleExport('json')} className="control-btn">
                <i className="fa fa-file-code"></i> Export JSON
              </button>
              <button onClick={handleClearHistory} className="control-btn danger">
                <i className="fa fa-trash"></i> Clear
              </button>
            </div>
          </div>

          {!showHistory ? (
            // Charts View
            <div className="charts-container">
              {/* Chart 1: Confidence Trend (Line Chart) */}
              <div className="chart-card">
                <div className="chart-wrapper">
                  <Line data={getConfidenceTrendData()} options={lineChartOptions} />
                </div>
                <p className="chart-description">
                  Track prediction confidence levels over time. Higher values indicate stronger model certainty.
                </p>
              </div>

              {/* Chart 2: Spam vs Ham Distribution (Bar Chart) */}
              <div className="chart-card">
                <div className="chart-wrapper">
                  <Bar data={getSpamHamData()} options={barChartOptions} />
                </div>
                <p className="chart-description">
                  Compare the distribution of spam vs legitimate messages in your prediction history.
                </p>
              </div>

              {/* Chart 3: Model Usage (Pie Chart) */}
              <div className="chart-card">
                <div className="chart-wrapper">
                  <Pie data={getModelUsageData()} options={pieChartOptions} />
                </div>
                <p className="chart-description">
                  Breakdown of which AI models were used for predictions.
                </p>
              </div>

              {/* Summary Statistics Cards */}
              {stats && stats.prediction_statistics && (
                <div className="stats-cards">
                  <div className="stat-card">
                    <div className="stat-icon">
                      <i className="fa fa-list-check"></i>
                    </div>
                    <div className="stat-content">
                      <div className="stat-number">
                        {stats.prediction_statistics.total_predictions}
                      </div>
                      <div className="stat-label">Total Predictions</div>
                    </div>
                  </div>

                  <div className="stat-card spam">
                    <div className="stat-icon">
                      <i className="fa fa-exclamation-triangle"></i>
                    </div>
                    <div className="stat-content">
                      <div className="stat-number">
                        {stats.prediction_statistics.spam_predictions}
                      </div>
                      <div className="stat-label">Spam Detected</div>
                    </div>
                  </div>

                  <div className="stat-card ham">
                    <div className="stat-icon">
                      <i className="fa fa-shield-check"></i>
                    </div>
                    <div className="stat-content">
                      <div className="stat-number">
                        {stats.prediction_statistics.ham_predictions}
                      </div>
                      <div className="stat-label">Legitimate</div>
                    </div>
                  </div>

                  <div className="stat-card">
                    <div className="stat-icon">
                      <i className="fa fa-percentage"></i>
                    </div>
                    <div className="stat-content">
                      <div className="stat-number">
                        {stats.prediction_statistics.spam_percentage}%
                      </div>
                      <div className="stat-label">Spam Rate</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ) : (
            // History Table View
            <div className="history-container">
              <h3>
                <i className="fa fa-history"></i> Prediction History ({history.length} records)
              </h3>
              <div className="history-table-wrapper">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Time</th>
                      <th>Text Preview</th>
                      <th>Model</th>
                      <th>Result</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.slice(0, 50).map((item, idx) => (
                      <tr key={item.prediction_id || idx} className={item.is_spam ? 'spam-row' : 'ham-row'}>
                        <td>{history.length - idx}</td>
                        <td className="time-cell">
                          {new Date(item.timestamp).toLocaleString()}
                        </td>
                        <td className="text-cell" title={item.text}>
                          {item.text_preview}
                        </td>
                        <td className="model-cell">
                          {item.model?.replace('_', ' ').toUpperCase()}
                        </td>
                        <td className="result-cell">
                          <span className={`badge ${item.is_spam ? 'badge-spam' : 'badge-ham'}`}>
                            {item.is_spam ? '⚠️ SPAM' : '✓ HAM'}
                          </span>
                        </td>
                        <td className="confidence-cell">
                          <div className="confidence-mini-bar">
                            <div 
                              className="confidence-fill" 
                              style={{ width: `${(item.confidence * 100)}%` }}
                            ></div>
                            <span className="confidence-text">
                              {(item.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Disclaimer and Footer */}
      <p className="disclaimer">
        <i className="fa fa-info-circle"></i> We are not liable for any losses or damages from use of this analysis. 
        This tool is for educational and informational purposes only.
      </p>
      <p className="footer">
        MatrixShield © 2025 | COS30049 Innovation Project
      </p>
    </div>
  );
}

// ============================================================================
// MAIN APP COMPONENT WITH ROUTING
// ============================================================================
function App() {
  return (
    <Router>
      <div className="main-container">
        {/* Navigation Bar */}
        <nav className="navbar">
          <div className="nav-left">
            <img src={logo} alt="MatrixShield Logo" className="nav-logo" />
            <h1 className="nav-title">MatrixShield</h1>
          </div>
          <div className="nav-right">
            <Link to="/">
              <i className="fa fa-home"></i> HOME
            </Link>
            <Link to="/about">
              <i className="fa fa-info-circle"></i> ABOUT
            </Link>
          </div>
        </nav>

        {/* Routes */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;