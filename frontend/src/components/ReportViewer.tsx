import React from 'react';
import ReactMarkdown from 'react-markdown';

interface Props {
  predictions: {
    logP: number;
    logD: number;
    logS: number;
  };
  report: string;
}

const ReportViewer: React.FC<Props> = ({ predictions, report }) => {
  return (
    <div className="mt-6 space-y-6">
      <div className="bg-blue-50 p-4 rounded-lg shadow">
        <h2 className="font-semibold text-blue-800">Predicted Properties</h2>
        <p><strong>logP:</strong> {predictions.logP}</p>
        <p><strong>logD:</strong> {predictions.logD}</p>
        <p><strong>logS:</strong> {predictions.logS}</p>
      </div>
      <div className="bg-purple-50 p-4 rounded-lg shadow">
        <h2 className="font-semibold text-purple-800 mb-2">Scientific Report</h2>
        <div className="prose prose-sm text-gray-700">
        <ReactMarkdown>{report}</ReactMarkdown>
        </div>

      </div>
    </div>
  );
};

export default ReportViewer;
