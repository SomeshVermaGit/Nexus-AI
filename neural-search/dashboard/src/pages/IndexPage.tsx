import React, { useState } from 'react';
import { PlusIcon, TrashIcon } from '@heroicons/react/24/outline';
import axios from 'axios';

const IndexPage: React.FC = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleIndex = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setSuccess(false);

    try {
      // Generate dummy embedding
      const dummyVector = Array(768).fill(0).map(() => Math.random());

      await axios.post('http://localhost:8000/api/index', {
        documents: [{
          text: text,
          vector: dummyVector,
          metadata: {
            indexed_at: new Date().toISOString()
          }
        }]
      });

      setSuccess(true);
      setText('');
      setTimeout(() => setSuccess(false), 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Indexing failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900">Index Management</h2>
        <p className="text-gray-600 mt-1">Add or remove documents from your search index</p>
      </div>

      {/* Success Message */}
      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <p className="text-green-800">✓ Document indexed successfully!</p>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Add Document Form */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Add Document</h3>

        <div className="space-y-4">
          <div>
            <label htmlFor="document-text" className="block text-sm font-medium text-gray-700 mb-2">
              Document Text
            </label>
            <textarea
              id="document-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={8}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
              placeholder="Enter the document text you want to index..."
            />
          </div>

          <button
            onClick={handleIndex}
            disabled={loading || !text.trim()}
            className="w-full px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 flex items-center justify-center gap-2"
          >
            <PlusIcon className="h-5 w-5" />
            {loading ? 'Indexing...' : 'Index Document'}
          </button>
        </div>
      </div>

      {/* Batch Upload */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Batch Upload</h3>
        <p className="text-sm text-gray-600 mb-4">
          Upload multiple documents from a file (JSON, CSV, or TXT)
        </p>

        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
          <input
            type="file"
            id="file-upload"
            className="hidden"
            accept=".json,.csv,.txt"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer inline-flex items-center gap-2 px-4 py-2 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            <PlusIcon className="h-5 w-5 text-gray-400" />
            <span className="text-sm text-gray-600">Choose file</span>
          </label>
          <p className="mt-2 text-xs text-gray-500">
            Supported formats: JSON, CSV, TXT
          </p>
        </div>
      </div>

      {/* Delete Documents */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Delete Documents</h3>

        <div className="space-y-4">
          <div>
            <label htmlFor="doc-ids" className="block text-sm font-medium text-gray-700 mb-2">
              Document IDs (comma-separated)
            </label>
            <input
              id="doc-ids"
              type="text"
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
              placeholder="1, 2, 3, ..."
            />
          </div>

          <button
            className="w-full px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 flex items-center justify-center gap-2"
          >
            <TrashIcon className="h-5 w-5" />
            Delete Documents
          </button>
        </div>
      </div>
    </div>
  );
};

export default IndexPage;