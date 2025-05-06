import React, { useState } from 'react';
import ReportViewer from './components/ReportViewer';

const App: React.FC = () => {
  const [smiles, setSmiles] = useState('');
  const [report, setReport] = useState('');
  const [predictions, setPredictions] = useState<{ logP: number, logD: number, logS: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    if (!smiles) return;
    setLoading(true);
    setError('');
    try {
      const response = await fetch('https://drugdiscoveryusingdl.onrender.com/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      setPredictions(data.predictions);
      setReport(data.report);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 via-purple-100 to-pink-100 p-8 flex items-center justify-center">
      <div className="max-w-2xl w-full bg-white shadow-xl rounded-xl p-8 space-y-6">
        <h1 className="text-3xl font-bold text-center text-indigo-700">Drug Discovery Assistant</h1>
        <input
          type="text"
          placeholder="Enter SMILES (e.g., CCOOH)"
          value={smiles}
          onChange={(e) => setSmiles(e.target.value)}
          className="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
        />
        <button
          onClick={handleSubmit}
          className="w-full bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:scale-105 transition-transform duration-200 disabled:opacity-50"
          disabled={loading}
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <svg className="animate-spin h-5 w-5 mr-2 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
              </svg>
              Generating Report...
            </div>
          ) : (
            'Generate Report'
          )}
        </button>

        {error && <div className="text-red-600 font-medium text-center">{error}</div>}

        {predictions && report && (
          <ReportViewer predictions={predictions} report={report} />
        )}
      </div>
    </div>
  );
};

export default App;
