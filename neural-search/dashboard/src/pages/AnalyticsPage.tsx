import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ChartBarIcon, ClockIcon, DocumentTextIcon, ServerIcon } from '@heroicons/react/24/outline';
import axios from 'axios';

interface Stats {
  total_documents: number;
  index_type: string;
  memory_usage_mb: number;
  queries_per_second: number;
  avg_query_latency_ms: number;
}

const AnalyticsPage: React.FC = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  // Dummy data for charts
  const queryLatencyData = [
    { time: '00:00', latency: 45 },
    { time: '04:00', latency: 52 },
    { time: '08:00', latency: 48 },
    { time: '12:00', latency: 65 },
    { time: '16:00', latency: 58 },
    { time: '20:00', latency: 43 },
  ];

  const queryVolumeData = [
    { hour: '00:00', queries: 120 },
    { hour: '04:00', queries: 80 },
    { hour: '08:00', queries: 240 },
    { hour: '12:00', queries: 350 },
    { hour: '16:00', queries: 280 },
    { hour: '20:00', queries: 190 },
  ];

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get<Stats>('http://localhost:8000/stats');
      setStats(response.data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="text-center py-12">Loading analytics...</div>;
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h2>
        <p className="text-gray-600 mt-1">Monitor your search engine performance</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={<DocumentTextIcon className="h-8 w-8" />}
          title="Total Documents"
          value={stats?.total_documents.toLocaleString() || '0'}
          color="bg-blue-500"
        />
        <StatCard
          icon={<ChartBarIcon className="h-8 w-8" />}
          title="Queries/Second"
          value={stats?.queries_per_second.toFixed(2) || '0'}
          color="bg-green-500"
        />
        <StatCard
          icon={<ClockIcon className="h-8 w-8" />}
          title="Avg Latency"
          value={`${stats?.avg_query_latency_ms.toFixed(2) || '0'}ms`}
          color="bg-yellow-500"
        />
        <StatCard
          icon={<ServerIcon className="h-8 w-8" />}
          title="Index Type"
          value={stats?.index_type || 'N/A'}
          color="bg-purple-500"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Query Latency Chart */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Query Latency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={queryLatencyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="latency" stroke="#4F46E5" strokeWidth={2} name="Latency (ms)" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Query Volume Chart */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Query Volume</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={queryVolumeData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="queries" fill="#10B981" name="Queries" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* System Info */}
      <div className="bg-white shadow rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <dt className="text-sm font-medium text-gray-500">Index Type</dt>
            <dd className="mt-1 text-sm text-gray-900">{stats?.index_type || 'N/A'}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Memory Usage</dt>
            <dd className="mt-1 text-sm text-gray-900">{stats?.memory_usage_mb.toFixed(2) || '0'} MB</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Total Documents</dt>
            <dd className="mt-1 text-sm text-gray-900">{stats?.total_documents.toLocaleString() || '0'}</dd>
          </div>
          <div>
            <dt className="text-sm font-medium text-gray-500">Average Latency</dt>
            <dd className="mt-1 text-sm text-gray-900">{stats?.avg_query_latency_ms.toFixed(2) || '0'} ms</dd>
          </div>
        </dl>
      </div>
    </div>
  );
};

interface StatCardProps {
  icon: React.ReactNode;
  title: string;
  value: string;
  color: string;
}

const StatCard: React.FC<StatCardProps> = ({ icon, title, value, color }) => {
  return (
    <div className="bg-white shadow rounded-lg p-6">
      <div className="flex items-center">
        <div className={`${color} rounded-lg p-3 text-white`}>
          {icon}
        </div>
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-semibold text-gray-900">{value}</p>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsPage;