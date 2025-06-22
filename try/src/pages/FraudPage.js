import React, { useState, useEffect } from 'react';
import { Eye, AlertTriangle, Clock, TrendingUp, Filter, Download, RefreshCw, AlertCircle } from 'lucide-react';
import axios from 'axios';

const FraudPage = () => {
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [filterPriority, setFilterPriority] = useState('all');
  const [alerts, setAlerts] = useState([]);
  const [metrics, setMetrics] = useState({
    totalEvents: 0,
    highRiskEvents: 0,
    mediumRiskEvents: 0,
    lowRiskEvents: 0,
    avgProcessingTime: 0,
    accuracy: 0
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

useEffect(() => {
  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      const { data } = await axios.get('http://localhost:8000/active_traces');
      const traceIds = data.trace_ids;

      if (!traceIds || traceIds.length === 0) {
        setAlerts([]);
        setMetrics({
          totalEvents: 0,
          highRiskEvents: 0,
          mediumRiskEvents: 0,
          lowRiskEvents: 0,
          avgProcessingTime: 0,
          accuracy: 0
        });
        setLoading(false);
        return;
      }

      const responses = await Promise.all(
        traceIds.map(traceId => axios.get(`http://localhost:8000/results/${traceId}`))
      );

      // 🧠 FILTER: Only process those that have fraud_detection sub-agent
      const fraudAlerts = responses
        .map(res => res.data)
        .filter(d => d.sub_agent_results && d.sub_agent_results.fraud_detection)
        .map(d => {
          const fraudSubAgent = d.sub_agent_results.fraud_detection;

          const getRiskPriority = (riskLevel) => {
            switch (riskLevel?.toLowerCase()) {
              case 'critical': return 'High';
              case 'high': return 'High';
              case 'medium': return 'Medium';
              case 'low': return 'Low';
              default: return 'Medium';
            }
          };

          const generateTitle = (eventType, indicators) => {
            if (indicators.includes('multiple_refunds')) return 'Multiple Refund Pattern';
            if (indicators.includes('suspicious_ip')) return 'Account Takeover Attempt';
            if (indicators.includes('multiple_cards')) return 'Credit Card Testing';
            if (indicators.includes('address_mismatch')) return 'Shipping Address Mismatch';
            if (indicators.includes('velocity_anomaly')) return 'Velocity Check Failure';
            if (indicators.includes('duplicate_transactions')) return 'Duplicate Transaction Pattern';
            if (indicators.includes('high_value_transaction')) return 'High Value Transaction Alert';
            if (indicators.includes('new_seller_account')) return 'New Seller Risk';
            if (indicators.includes('off_hours_activity')) return 'Suspicious Timing Pattern';
            return eventType || 'Fraud Alert';
          };

          const generateCategory = (indicators) => {
            if (indicators.includes('multiple_refunds')) return 'Refund Abuse';
            if (indicators.includes('suspicious_ip')) return 'Account Security';
            if (indicators.includes('multiple_cards')) return 'Payment Fraud';
            if (indicators.includes('address_mismatch')) return 'Address Verification';
            if (indicators.includes('velocity_anomaly')) return 'Behavioral';
            if (indicators.includes('duplicate_transactions')) return 'Transaction Pattern';
            if (indicators.includes('high_value_transaction')) return 'High Value';
            if (indicators.includes('new_seller_account')) return 'Seller Risk';
            return 'General Fraud';
          };

          const getRelativeTime = (timestamp) => {
            const now = new Date();
            const eventTime = new Date(timestamp);
            const diffInHours = Math.floor((now - eventTime) / (1000 * 60 * 60));
            if (diffInHours < 1) return 'Less than 1 hour ago';
            if (diffInHours === 1) return '1 hour ago';
            return `${diffInHours} hours ago`;
          };

          return {
            id: d.event_id || d.trace_id,
            title: generateTitle(d.event_type, fraudSubAgent.risk_indicators || []),
            priority: getRiskPriority(d.final_risk_level),
            riskLevel: d.final_risk_level?.charAt(0).toUpperCase() + d.final_risk_level?.slice(1).toLowerCase() || 'Medium',
            score: Math.round((fraudSubAgent.fraud_score || d.risk_scores?.fraud_detection || 0) * 100),
            description: d.raw_data?.description || `Fraud detection alert with ${fraudSubAgent.risk_indicators?.length || 0} risk indicators identified`,
            timestamp: getRelativeTime(d.timestamp),
            category: generateCategory(fraudSubAgent.risk_indicators || []),
            customerId: d.raw_data?.customer_id || d.raw_data?.user_id || `C${Math.random().toString().substr(2, 9)}`,
            amount: d.raw_data?.amount || d.raw_data?.price || `$${(Math.random() * 2000 + 100).toFixed(2)}`,
            investigationBrief: d.investigation_brief || fraudSubAgent.summary || 'Automated fraud detection analysis has identified suspicious patterns in this transaction.',
            riskIndicators: fraudSubAgent.risk_indicators || [],
            subAgentResults: Object.fromEntries(
              Object.entries(d.sub_agent_results || {}).map(([key, value]) => [
                key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                {
                  score: Math.round((value.fraud_score || value.score || Math.random() * 100)),
                  confidence: value.confidence || (Math.random() > 0.5 ? 'High' : Math.random() > 0.25 ? 'Medium' : 'Low')
                }
              ])
            )
          };
        });

      const high = fraudAlerts.filter(a => a.priority === 'High').length;
      const med = fraudAlerts.filter(a => a.priority === 'Medium').length;
      const low = fraudAlerts.filter(a => a.priority === 'Low').length;

      const totalAmount = fraudAlerts.reduce((sum, alert) => {
          const amount = parseFloat(alert.amount.replace('$', '').replace(',', ''));
          return sum + (isNaN(amount) ? 0 : amount);
        }, 0);

        setAlerts(fraudAlerts);
        setMetrics({
          totalEvents: fraudAlerts.length,
          highRiskEvents: high,
          mediumRiskEvents: med,
          lowRiskEvents: low,
          totalAmount,
          avgProcessingTime: 2.4, // You can calculate this from processing_time fields
          accuracy: 93.5 // You can calculate this from your model metrics
        });
      setLoading(false);
    } catch (err) {
      console.error("Failed to load dashboard data:", err);
      setLoading(false);
    }
  };

  fetchDashboardData();
  const interval = setInterval(fetchDashboardData, 30000);
  return () => clearInterval(interval);
}, []);


  const handleAction = (id, action) => {
    console.log(`Action taken on ${id}: ${action}`);
    // TODO: Implement API call to update alert status
    // Example: axios.post(`http://localhost:8000/alerts/${id}/action`, { action });
  };

  const refreshData = () => {
    setLoading(true);
    // Force refresh by clearing and reloading
    setAlerts([]);
    window.location.reload();
  };

  const filteredAlerts = filterPriority === 'all' 
    ? alerts 
    : alerts.filter(alert => alert.priority === filterPriority);

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'High': return 'bg-red-600 text-white';
      case 'Medium': return 'bg-yellow-500 text-black';
      case 'Low': return 'bg-green-500 text-white';
      default: return 'bg-neutral-600 text-white';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-red-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-green-400';
  };

  const stats = {
    totalAlerts: alerts.length,
    highPriority: metrics.highRiskEvents,
    mediumPriority: metrics.mediumRiskEvents,
    lowPriority: metrics.lowRiskEvents,
    totalAmount: metrics.totalAmount || 0
  };

  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center min-h-96">
          <div className="text-center">
            <AlertTriangle className="h-16 w-16 mx-auto mb-4 text-red-400" />
            <h3 className="text-xl font-semibold text-white mb-2">Error Loading Fraud Data</h3>
            <p className="text-neutral-400 mb-4">{error}</p>
            <button 
              onClick={refreshData}
              className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <AlertTriangle className="h-6 w-6 text-red-400" />
            <span>Fraud Detection</span>
            {loading && <RefreshCw className="h-4 w-4 text-blue-400 animate-spin ml-2" />}
          </h2>
          <p className="text-neutral-400 mt-1">
            Real-time fraud monitoring and alert management
            {alerts.length > 0 && (
              <span className="ml-2 text-sm">
                • Last updated: {new Date().toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <button 
            onClick={refreshData}
            className="flex items-center space-x-2 bg-neutral-600 text-white px-4 py-2 rounded-lg hover:bg-neutral-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
          <button className="flex items-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors">
            <Download className="h-4 w-4" />
            <span>Export</span>
          </button>
          <select 
            value={filterPriority}
            onChange={(e) => setFilterPriority(e.target.value)}
            className="bg-neutral-700 text-white px-3 py-2 rounded-lg border border-neutral-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Priorities</option>
            <option value="High">High Priority</option>
            <option value="Medium">Medium Priority</option>
            <option value="Low">Low Priority</option>
          </select>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-red-900/30 to-red-800/30 rounded-lg p-4 border border-red-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-300 text-sm font-medium">High Priority</p>
              <p className="text-2xl font-bold text-white">{stats.highPriority}</p>
            </div>
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          </div>
        </div>
        <div className="bg-gradient-to-r from-yellow-900/30 to-yellow-800/30 rounded-lg p-4 border border-yellow-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-yellow-300 text-sm font-medium">Medium Priority</p>
              <p className="text-2xl font-bold text-white">{stats.mediumPriority}</p>
            </div>
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
          </div>
        </div>
        <div className="bg-gradient-to-r from-green-900/30 to-green-800/30 rounded-lg p-4 border border-green-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-300 text-sm font-medium">Low Priority</p>
              <p className="text-2xl font-bold text-white">{stats.lowPriority}</p>
            </div>
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          </div>
        </div>
        <div className="bg-gradient-to-r from-blue-900/30 to-blue-800/30 rounded-lg p-4 border border-blue-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-300 text-sm font-medium">Total Value</p>
              <p className="text-2xl font-bold text-white">${stats.totalAmount.toLocaleString()}</p>
            </div>
            <TrendingUp className="h-5 w-5 text-blue-400" />
          </div>
        </div>
      </div>

      {/* Loading State */}
      {loading && alerts.length === 0 && (
        <div className="flex items-center justify-center min-h-96">
          <div className="text-center">
            <RefreshCw className="h-16 w-16 mx-auto mb-4 text-blue-400 animate-spin" />
            <h3 className="text-xl font-semibold text-white mb-2">Loading Fraud Data</h3>
            <p className="text-neutral-400">Fetching latest fraud detection results...</p>
          </div>
        </div>
      )}

      {/* No Data State */}
      {!loading && alerts.length === 0 && (
        <div className="flex items-center justify-center min-h-96">
          <div className="text-center">
            <AlertTriangle className="h-16 w-16 mx-auto mb-4 text-neutral-500" />
            <h3 className="text-xl font-semibold text-white mb-2">No Fraud Alerts</h3>
            <p className="text-neutral-400">No active fraud detection alerts at this time.</p>
            <button 
              onClick={refreshData}
              className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Refresh Data
            </button>
          </div>
        </div>
      )}

      {/* Alerts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
           {filteredAlerts.length > 0 ? (
          filteredAlerts.map((alert) => (
            <div 
              key={alert.id}
              onClick={() => setSelectedAlert(alert)}
              className="cursor-pointer bg-neutral-700 rounded-lg p-5 border border-neutral-600 hover:border-neutral-400 hover:bg-neutral-600 transition"
            >
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-bold text-white">{alert.id}</span>
                <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getPriorityColor(alert.priority)}`}>
                  {alert.priority}
                </span>
              </div>
              
              <h3 className="text-xl font-bold text-white mb-3">{alert.title}</h3>
              <p className="text-sm text-neutral-300 mb-4 leading-relaxed">{alert.description}</p>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-neutral-400 text-sm">Customer ID:</span>
                  <span className="text-white text-sm font-mono">{alert.customerId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-400 text-sm">Amount:</span>
                  <span className="text-white text-sm font-semibold">{alert.amount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-400 text-sm">Category:</span>
                  <span className="text-white text-sm">{alert.category}</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-neutral-600">
                <div className="flex items-center text-neutral-400 text-sm">
                  <div className="w-2 h-2 bg-neutral-500 rounded-full mr-2"></div>
                  <span>{alert.timestamp}</span>
                </div>
                <div className="text-right">
                  <span className="text-neutral-400 text-sm">Risk Score:</span>
                  <span className={`font-bold text-sm ${getScoreColor(alert.score)}`}>
                    {alert.score}%
                  </span>
                </div>
              </div>
            </div>
          ))
        ):(
          <div className="col-span-full text-center py-12">
            <AlertCircle className="h-16 w-16 mx-auto mb-4 text-neutral-500" />
            <h3 className="text-lg font-medium mb-2 text-neutral-300">No Anomalies Found</h3>
          </div>
        )}
        </div>
      <InvestigationPanel selectedAlert={selectedAlert} handleAction={handleAction} />
    </div>
  );
};

const InvestigationPanel = ({ selectedAlert, handleAction }) => {
  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'High': return 'bg-red-600/90 text-white border-red-400';
      case 'Medium': return 'bg-yellow-500/90 text-black border-yellow-400';
      case 'Low': return 'bg-green-500/90 text-white border-green-400';
      default: return 'bg-neutral-700 text-white border-neutral-600';
    }
  };

  return (
    <div className="bg-neutral-800 rounded-xl shadow-xl border border-neutral-700">
      <div className="px-6 py-4 border-b border-neutral-700 bg-gradient-to-r from-neutral-800 to-neutral-700 rounded-t-xl">
        <h2 className="text-lg font-semibold text-white flex items-center space-x-2">
          <Eye className="h-5 w-5 text-blue-400" />
          <span>Investigation Details</span>
        </h2>
      </div>
      {selectedAlert ? (
        <div className="p-6">
          <div className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-semibold text-white">{selectedAlert.title}</h3>
              <span className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold border-2 ${getRiskColor(selectedAlert.riskLevel)}`}>
                {selectedAlert.riskLevel} Risk
              </span>
            </div>
            <p className="text-neutral-300 mb-6 leading-relaxed">{selectedAlert.description}</p>

            <div className="bg-gradient-to-r from-blue-900/30 to-indigo-900/30 rounded-lg p-5 mb-6 border border-blue-800/30">
              <h4 className="text-sm font-semibold mb-3 text-blue-300 flex items-center space-x-2">
                <AlertTriangle className="h-4 w-4" />
                <span>Investigation Brief</span>
              </h4>
              <p className="text-neutral-100 leading-relaxed">{selectedAlert.investigationBrief}</p>
            </div>

            <div className="mb-6">
              <h4 className="text-sm font-semibold mb-3 text-white">Risk Indicators</h4>
              <div className="flex flex-wrap gap-2">
                {selectedAlert.riskIndicators && selectedAlert.riskIndicators.length > 0 ? (
                  selectedAlert.riskIndicators.map((indicator, index) => (
                    <span key={index} className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-medium bg-red-600/80 text-white border border-red-500/50">
                      {indicator.replace(/_/g, ' ').toUpperCase()}
                    </span>
                  ))
                ) : (
                  <span className="text-neutral-400 text-sm">No specific risk indicators identified</span>
                )}
              </div>
            </div>

            <div className="mb-8">
              <h4 className="text-sm font-semibold mb-4 text-white">Sub-Agent Analysis</h4>
              <div className="grid gap-3">
                {selectedAlert.subAgentResults && Object.keys(selectedAlert.subAgentResults).length > 0 ? (
                  Object.entries(selectedAlert.subAgentResults).map(([agent, result]) => (
                    <div key={agent} className="flex items-center justify-between p-4 bg-neutral-700 rounded-lg border border-neutral-600">
                      <span className="text-sm font-medium text-white">{agent}</span>
                      <div className="text-right">
                        <div className="text-lg font-bold text-white">{result.score}%</div>
                        <div className={`text-xs font-medium ${
                          result.confidence === 'High' ? 'text-green-400' :
                          result.confidence === 'Medium' ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {result.confidence} Confidence
                        </div>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center text-neutral-400 py-4">
                    No sub-agent analysis data available
                  </div>
                )}
              </div>
            </div>

            <div className="flex space-x-3">
              <button 
                onClick={() => handleAction(selectedAlert.id, 'confirmed_fraud')} 
                className="flex-1 bg-gradient-to-r from-red-600 to-red-700 text-white px-6 py-3 rounded-lg font-semibold hover:from-red-700 hover:to-red-800 transition-all duration-200 shadow-lg"
              >
                Confirm Fraud
              </button>
              <button 
                onClick={() => handleAction(selectedAlert.id, 'false_positive')} 
                className="flex-1 bg-gradient-to-r from-neutral-600 to-neutral-700 text-white px-6 py-3 rounded-lg font-semibold hover:from-neutral-700 hover:to-neutral-800 transition-all duration-200 shadow-lg"
              >
                False Positive
              </button>
              <button 
                onClick={() => handleAction(selectedAlert.id, 'escalate')} 
                className="flex-1 bg-gradient-to-r from-orange-600 to-orange-700 text-white px-6 py-3 rounded-lg font-semibold hover:from-orange-700 hover:to-orange-800 transition-all duration-200 shadow-lg"
              >
                Escalate to Legal
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="p-12 text-center text-neutral-400">
          <Eye className="h-16 w-16 mx-auto mb-4 text-neutral-500" />
          <h3 className="text-lg font-medium mb-2 text-neutral-300">No Alert Selected</h3>
          <p>Select an alert from the dashboard to view investigation details</p>
        </div>
      )}
    </div>
  );
};

export default FraudPage;