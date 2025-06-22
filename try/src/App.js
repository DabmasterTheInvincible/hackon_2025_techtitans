import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import FraudPage from './pages/FraudPage';
import CounterfeitPage from './pages/CounterfeitPage';
import ReviewSpamPage from './pages/ReviewSpamPage';
import ReturnAnomalyPage from './pages/ReturnAnomalyPage';
import Dashboard from './pages/Dashboard';
import { Eye, Shield, AlertTriangle, Search, Bell, User, Package, House, AlertCircle } from 'lucide-react';

const sampleAlert = {
  id: 'a1',
  title: 'Suspicious Transaction Pattern',
  riskLevel: 'High',
  description: 'Detected anomaly in refund pattern with multiple high-value returns.',
  investigationBrief: 'Customer has initiated 8 refunds totaling $2,347 within 24 hours across different product categories. Pattern suggests potential fraud activity.',
  riskIndicators: ['multi_refund', 'abnormal_timing', 'high_value', 'cross_category'],
  subAgentResults: {
    PatternAnalyzer: { score: 85, confidence: 'High' },
    BehaviorProfiler: { score: 70, confidence: 'Medium' },
    RiskScorer: { score: 92, confidence: 'High' },
  },
};

const Header = () => (
  <header className="bg-gradient-to-r from-blue-900 to-indigo-900 border-b border-neutral-700 shadow-lg">
    <div className="px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Shield className="h-8 w-8 text-blue-400" />
          <div>
            <h1 className="text-2xl font-bold text-white">Trust and Safety Control Center</h1>
            <p className="text-sm text-blue-200">Amazon Security Operations</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 bg-neutral-800 rounded-lg px-3 py-2">
            <Search className="h-4 w-4 text-neutral-400" />
            <input
              type="text"
              placeholder="Search alerts..."
              className="bg-transparent text-white placeholder-neutral-400 text-sm focus:outline-none w-48"
            />
          </div>
          <div className="relative">
            <Bell className="h-5 w-5 text-neutral-300 hover:text-white cursor-pointer" />
            <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">3</span>
          </div>
          <div className="flex items-center space-x-2 text-sm text-neutral-300">
            <User className="h-4 w-4" />
            <span>Security Analyst</span>
          </div>
        </div>
      </div>
    </div>
  </header>
);

const Sidebar = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: House },
    { id: 'fraud', label: 'Fraud Detection', icon: AlertTriangle },
    { id: 'counterfeit', label: 'Counterfeit Products', icon: Shield },
    { id: 'review', label: 'Review Spam', icon: Search },
    { id: 'return', label: 'Return Anomalies', icon: Package },
  ];

  return (
    <aside className="w-72 bg-neutral-800 border-r border-neutral-700">
      <nav className="p-4 space-y-2">
        {navItems.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`w-full flex items-center space-x-3 py-3 px-4 rounded-lg transition-all duration-200 ${
              activeTab === id
                ? 'bg-blue-600 text-white shadow-lg'
                : 'text-neutral-300 hover:bg-neutral-700 hover:text-white'
            }`}
          >
            <Icon className="h-5 w-5" />
            <span className="font-medium">{label}</span>
          </button>
        ))}
      </nav>
      
      <div className="p-4 mt-8">
        <div className="bg-neutral-700 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-white mb-3">System Status</h3>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-neutral-300">ML Models</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-green-400">Online</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-neutral-300">Data Pipeline</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-green-400">Healthy</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-neutral-300">Alert Queue</span>
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                <span className="text-yellow-400">23 Pending</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};

const App = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const selectedAlert = sampleAlert;

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard': return <Dashboard />;
      case 'fraud': return <FraudPage />;
      case 'counterfeit': return <CounterfeitPage />;
      case 'review': return <ReviewSpamPage />;
      case 'return': return <ReturnAnomalyPage />;
      default: return <FraudPage />;
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-neutral-900">
      <Header />
      <div className="flex flex-1">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="flex-1 p-6 space-y-6 overflow-y-auto">
          <div className="bg-neutral-800 p-6 rounded-xl shadow-xl border border-neutral-700">
            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  );
};

export default App;