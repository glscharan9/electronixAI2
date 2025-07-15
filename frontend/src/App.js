import React, { useState } from 'react';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    setError('');

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`An error occurred: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      // Since we are now making a cross-origin request, we need to handle potential CORS errors.
      // However, FastAPI by default allows all origins with simple requests, so this should work.
      // A more robust solution in a real-world app would be to configure CORS on the backend.
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const pageStyle = {
      background: '#282c34', minHeight: '100vh', display: 'flex',
      flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      fontSize: 'calc(10px + 2vmin)', color: 'white', textAlign: 'center'
  };
  const textareaStyle = { width: '50%', padding: '10px', margin: '20px', borderRadius: '5px' };
  const buttonStyle = { padding: '10px 20px', cursor: 'pointer', borderRadius: '5px' };
  const resultStyle = { marginTop: '20px', background: '#3a3f4a', padding: '15px', borderRadius: '8px' };
  const errorStyle = { color: '#F44336', marginTop: '10px' };

  return (
    <div style={pageStyle}>
      <h1>Sentiment Analysis</h1>
      <textarea
        style={textareaStyle}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text for analysis..."
        rows="5"
      />
      <button style={buttonStyle} onClick={handlePredict} disabled={loading}>
        {loading ? 'Analyzing...' : 'Predict'}
      </button>

      {result && (
        <div style={resultStyle}>
          <p><strong>Label:</strong> <span style={{color: result.label === 'positive' ? '#4CAF50' : '#F44336'}}>{result.label}</span></p>
          <p><strong>Score:</strong> {result.score}</p>
        </div>
      )}

      {error && <p style={errorStyle}>{error}</p>}
    </div>
  );
}
export default App;