import React, { useState } from 'react';
import { Cog6ToothIcon } from '@heroicons/react/24/outline';

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState({
    indexType: 'hnsw',
    embeddingModel: 'all-MiniLM-L6-v2',
    topK: 10,
    efSearch: 50,
    hybridAlpha: 0.5,
    cacheEnabled: true,
  });

  const handleSave = () => {
    // Save settings logic
    console.log('Saving settings:', settings);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <div className="flex items-center gap-3">
          <Cog6ToothIcon className="h-8 w-8 text-gray-700" />
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Settings</h2>
            <p className="text-gray-600 mt-1">Configure your search engine</p>
          </div>
        </div>
      </div>

      {/* Index Settings */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Index Configuration</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Index Type
            </label>
            <select
              value={settings.indexType}
              onChange={(e) => setSettings({ ...settings, indexType: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
            >
              <option value="hnsw">HNSW (Hierarchical Navigable Small World)</option>
              <option value="ivf">IVF (Inverted File Index)</option>
              <option value="flat">Flat (Brute Force)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Default Top K
            </label>
            <input
              type="number"
              value={settings.topK}
              onChange={(e) => setSettings({ ...settings, topK: parseInt(e.target.value) })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
              min={1}
              max={1000}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              HNSW ef_search
            </label>
            <input
              type="number"
              value={settings.efSearch}
              onChange={(e) => setSettings({ ...settings, efSearch: parseInt(e.target.value) })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
              min={10}
              max={500}
            />
            <p className="mt-1 text-xs text-gray-500">
              Higher values improve recall but increase latency
            </p>
          </div>
        </div>
      </div>

      {/* Embedding Settings */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Embedding Configuration</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Embedding Model
            </label>
            <select
              value={settings.embeddingModel}
              onChange={(e) => setSettings({ ...settings, embeddingModel: e.target.value })}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
            >
              <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (Fast, 384 dim)</option>
              <option value="all-mpnet-base-v2">all-mpnet-base-v2 (Balanced, 768 dim)</option>
              <option value="paraphrase-multilingual-mpnet-base-v2">Multilingual (768 dim)</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="cache"
              checked={settings.cacheEnabled}
              onChange={(e) => setSettings({ ...settings, cacheEnabled: e.target.checked })}
              className="rounded text-indigo-600"
            />
            <label htmlFor="cache" className="text-sm text-gray-700">
              Enable embedding cache
            </label>
          </div>
        </div>
      </div>

      {/* Hybrid Search Settings */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Hybrid Search</h3>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dense/Sparse Weight (Alpha)
            </label>
            <input
              type="range"
              value={settings.hybridAlpha}
              onChange={(e) => setSettings({ ...settings, hybridAlpha: parseFloat(e.target.value) })}
              min={0}
              max={1}
              step={0.1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Sparse (BM25)</span>
              <span>{settings.hybridAlpha}</span>
              <span>Dense (Vector)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="flex justify-end">
        <button
          onClick={handleSave}
          className="px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
        >
          Save Settings
        </button>
      </div>
    </div>
  );
};

export default SettingsPage;