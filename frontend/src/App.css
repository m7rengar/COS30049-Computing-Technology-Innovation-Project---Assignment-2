import React, { useState } from 'react';

const API_URL = 'http://localhost:8000';

function App() {
  const [text, setText] = useState('');
  const [batchText, setBatchText] = useState('');
  const [model, setModel] = useState('random_forest');
  const [result, setResult] = useState(null);
  const [batchResult, setBatchResult] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPredict = async () => {
    const texts = batchText.split('\n').map(t => t.trim()).filter(Boolean);
    if (!texts.length) return;

    setLoading(true);
    setError('');
    setBatchResult([]);

    try {
      const res = await fetch(`${API_URL}/predict-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts, model }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Batch prediction failed');
      setBatchResult(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: 600, margin: '0 auto' }}>
      <h1>Spam Detector</h1>

      {/* Model selection */}
      <div>
        <label>Model:</label>
        <select value={model} onChange={e => setModel(e.target.value)}>
          <option value="random_forest">Random Forest</option>
          <option value="xgboost">XGBoost</option>
        </select>
      </div>

      {/* Single message */}
      <div style={{ marginTop: '1rem' }}>
        <textarea
          rows="4"
          style={{ width: '100%' }}
          placeholder="Enter a message..."
          value={text}
          onChange={e => setText(e.target.value)}
        />
        <button onClick={handlePredict} disabled={loading || !text.trim()}>
          {loading ? 'Analyzing...' : 'Predict Single'}
        </button>
      </div>

      {result && (
        <div style={{ marginTop: '1rem', padding: '1rem', border: '1px solid #ccc' }}>
          <strong>Prediction:</strong> {result.prediction} <br />
          <strong>Confidence:</strong> {(result.confidence * 100).toFixed(1)}%
        </div>
      )}

      {/* Batch messages */}
      <div style={{ marginTop: '2rem' }}>
        <textarea
          rows="6"
          style={{ width: '100%' }}
          placeholder="Enter messages, one per line..."
          value={batchText}
          onChange={e => setBatchText(e.target.value)}
        />
        <button onClick={handleBatchPredict} disabled={loading || !batchText.trim()}>
          {loading ? 'Analyzing...' : 'Predict Batch'}
        </button>
      </div>

      {batchResult.length > 0 && (
        <div style={{ marginTop: '1rem' }}>
          {batchResult.map((r, i) => (
            <div key={i} style={{ padding: '0.5rem', borderBottom: '1px solid #eee' }}>
              <strong>Message:</strong> {r.text_preview} <br />
              <strong>Prediction:</strong> {r.prediction} | <strong>Confidence:</strong> {(r.confidence * 100).toFixed(1)}%
            </div>
          ))}
        </div>
      )}

      {error && <div style={{ color: 'red', marginTop: '1rem' }}>{error}</div>}
    </div>
  );
}

export default App;
