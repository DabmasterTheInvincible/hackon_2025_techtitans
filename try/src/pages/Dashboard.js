// src/pages/ReturnAnomalyPage.jsx (Dashboard component only)
import React from 'react';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const Dashboard = ({ returnAnomalies = [] }) => {
  const categoryCounts = {};
  let totalRisk = 0;

  returnAnomalies.forEach(a => {
    categoryCounts[a.category] = (categoryCounts[a.category] || 0) + 1;
    totalRisk += a.riskScore;
  });

  const totalAnomalies = returnAnomalies.length;
  const avgRisk = totalAnomalies > 0 ? (totalRisk / totalAnomalies).toFixed(1) : 0;

  const chartData = {
    labels: Object.keys(categoryCounts),
    datasets: [
      {
        label: 'Cases by Type',
        data: Object.values(categoryCounts),
        backgroundColor: '#7c3aed',
        borderRadius: 6,
      },
    ],
  };

  return (
    <div className="space-y-6">
      <div className="text-white">
        <h2 className="text-2xl font-bold mb-2">Return Anomaly Overview</h2>
        <p className="text-neutral-400">Summary of detected cases by category</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-neutral-800 p-4 rounded-lg border border-neutral-700">
          <p className="text-sm text-neutral-400">Total Cases</p>
          <p className="text-2xl font-bold text-white">{totalAnomalies}</p>
        </div>
        <div className="bg-neutral-800 p-4 rounded-lg border border-neutral-700">
          <p className="text-sm text-neutral-400">Avg Risk Score</p>
          <p className="text-2xl font-bold text-red-400">{avgRisk}%</p>
        </div>
        <div className="bg-neutral-800 p-4 rounded-lg border border-neutral-700">
          <p className="text-sm text-neutral-400">Types Detected</p>
          <p className="text-2xl font-bold text-white">{Object.keys(categoryCounts).length}</p>
        </div>
      </div>

      <div className="bg-neutral-800 p-4 rounded-lg border border-neutral-700">
        <Bar data={chartData} />
      </div>
    </div>
  );
};

export default Dashboard;
