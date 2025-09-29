import React, { useState } from 'react';
import { MagnifyingGlassIcon, ClockIcon } from '@heroicons/react/24/outline';
import axios from 'axios';

interface SearchResult {
  doc_id: number;
  score: number;
  text?: string;
  metadata?: Record<string, any>;
}

interface SearchResponse {
  results: SearchResult[];
  query_time_ms: number;
  total_results: number;
}

const SearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [k, setK] = useState(10);
  const [useHybrid, setUseHybrid] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      // Generate dummy embedding for demo
      const dummyVector = Array(768).fill(0).map(() => Math.random());

      const response = await axios.post<SearchResponse>('http://localhost:8000/api/search', {
        query_text: query,
        query_vector: dummyVector,
        k: k,
        hybrid: useHybrid
      });

      setResults(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Search failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Search</h2>

        {/* Search Input */}
        <div className="flex gap-4">
          <div className="flex-1">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Enter your search query..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-400 flex items-center gap-2"
          >
            <MagnifyingGlassIcon className="h-5 w-5" />
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Search Options */}
        <div className="mt-4 flex gap-6">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-600">Results:</label>
            <input
              type="number"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              min={1}
              max={100}
              className="w-20 px-2 py-1 border border-gray-300 rounded"
            />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="hybrid"
              checked={useHybrid}
              onChange={(e) => setUseHybrid(e.target.checked)}
              className="rounded text-indigo-600"
            />
            <label htmlFor="hybrid" className="text-sm text-gray-600">
              Use Hybrid Search (Dense + BM25)
            </label>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* Search Results */}
      {results && (
        <div className="space-y-4">
          {/* Results Header */}
          <div className="bg-white shadow rounded-lg p-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900">
                {results.total_results} Results
              </h3>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <ClockIcon className="h-4 w-4" />
                <span>{results.query_time_ms.toFixed(2)}ms</span>
              </div>
            </div>
          </div>

          {/* Result Cards */}
          {results.results.map((result, index) => (
            <div key={result.doc_id} className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-500">#{index + 1}</span>
                  <span className="text-xs text-gray-400">ID: {result.doc_id}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-xs text-gray-500">Score:</span>
                  <span className="text-sm font-semibold text-indigo-600">
                    {result.score.toFixed(4)}
                  </span>
                </div>
              </div>

              {result.text && (
                <p className="text-gray-700 mb-3">{result.text}</p>
              )}

              {result.metadata && Object.keys(result.metadata).length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-200">
                  <details className="text-sm">
                    <summary className="cursor-pointer text-gray-600 hover:text-gray-900">
                      Metadata
                    </summary>
                    <pre className="mt-2 p-2 bg-gray-50 rounded text-xs overflow-x-auto">
                      {JSON.stringify(result.metadata, null, 2)}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!results && !loading && (
        <div className="bg-white shadow rounded-lg p-12 text-center">
          <MagnifyingGlassIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No search yet</h3>
          <p className="text-gray-600">Enter a query above to start searching</p>
        </div>
      )}
    </div>
  );
};

export default SearchPage;