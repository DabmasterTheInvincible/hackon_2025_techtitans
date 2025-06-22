import React, { useState, useEffect } from 'react';
import { Shield, Search, Package, AlertCircle, Eye, AlertTriangle, Download, RefreshCw } from 'lucide-react';
import axios from 'axios';

const ReturnAnomalyPage = () => {
  const [selectedAlert, setSelectedAlert] = useState(null);
const [filterPriority, setFilterPriority] = useState('all');
const [returnAlerts, setReturnAlerts] = useState([]);
const [metrics, setMetrics] = useState({
  totalEvents: 0,
  highRiskEvents: 0,
  mediumRiskEvents: 0,
  lowRiskEvents: 0,
});
const [loading, setLoading] = useState(true);
const [error, setError] = useState(null);
const [riskLevelFilter, setRiskLevelFilter] = useState('all');
const filteredAnomalies = riskLevelFilter === 'all'
  ? returnAlerts
  : returnAlerts.filter(alert => alert.priority === riskLevelFilter);
const filteredAlerts = filterPriority === 'all' 
  ? returnAlerts 
  : returnAlerts.filter(alert => alert.priority === filterPriority);

const handleAction = (id, action) => {
  console.log(`Action taken on ${id}: ${action}`);
  // Add actual implementation here
};
const refreshData = () => {
    setLoading(true);
    // Force refresh by clearing and reloading
    setReturnAlerts([]);
    window.location.reload();
  };
useEffect(() => {
  const generateTitle = (eventType, indicators) => {
    if (indicators.includes('multiple_returns')) return 'Excessive Return Volume';
    if (indicators.includes('inconsistent_reasons')) return 'Inconsistent Return Reasons';
    if (indicators.includes('burst_return_activity')) return 'Return Burst Detected';
    if (indicators.includes('category_mismatch')) return 'Return Category Mismatch';
    if (indicators.includes('policy_violation')) return 'Policy Violation Detected';
    return 'Return Anomaly Detected';
  };

  const generateCategory = (indicators) => {
    if (indicators.includes('multiple_returns')) return 'Return Abuse';
    if (indicators.includes('inconsistent_reasons')) return 'Reason Inconsistency';
    if (indicators.includes('burst_return_activity')) return 'Behavioral Anomaly';
    if (indicators.includes('category_mismatch')) return 'Category Mismatch';
    if (indicators.includes('policy_violation')) return 'Policy Violation';
    return 'General Return Risk';
  };

  const getRelativeTime = (timestamp) => {
    const now = new Date();
    const eventTime = new Date(timestamp);
    const diffInHours = Math.floor((now - eventTime) / (1000 * 60 * 60));
    if (diffInHours < 1) return 'Less than 1 hour ago';
    if (diffInHours === 1) return '1 hour ago';
    return `${diffInHours} hours ago`;
  };

  const fetchReturnAlerts = async () => {
    try {
      setLoading(true);
      setError(null);

      const { data } = await axios.get('http://localhost:8000/active_traces');
      const traceIds = data.trace_ids;

      if (!traceIds?.length) {
  setReturnAlerts([]); // ✅ This is good
  setMetrics({ totalEvents: 0, highRiskEvents: 0, mediumRiskEvents: 0, lowRiskEvents: 0 });
  setLoading(false);
  return;
}


      const responses = await Promise.all(
        traceIds.map(traceId => axios.get(`http://localhost:8000/results/${traceId}`))
      );

      const alerts = responses
        .map(res => res.data)
        .filter(d => d.sub_agent_results && d.sub_agent_results.return_anomaly)
        .map(d => {
          const agentData = d.sub_agent_results.return_anomaly;
          const score = agentData.anomaly_score ?? 0;
          const riskLevel = score > 0.8 ? 'High' : score > 0.5 ? 'Medium' : 'Low';
          const indicators = agentData.anomaly_signals || [];

          return {
            id: d.event_id || d.trace_id,
            title: generateTitle(d.event_type, indicators),
            description: d.raw_data?.reason || 'No return reason provided.',
            customerId: d.raw_data?.customer_id || 'Unknown',
            category: generateCategory(indicators),
            timeframe: getRelativeTime(d.timestamp),
            riskScore: Math.round(score * 100),
            priority: riskLevel.toLowerCase(),
            riskLevel,
            investigationBrief: d.investigation_brief || 'Return flagged for anomalous activity.',
            riskIndicators: indicators,
            subAgentResults: {
              'Return Anomaly': {
                score: Math.round(score * 100),
                confidence:
                  agentData.confidence >= 0.8
                    ? 'High'
                    : agentData.confidence >= 0.5
                    ? 'Medium'
                    : 'Low'
              }
            }
          };
        });

      const high = alerts.filter(a => a.riskLevel === 'High').length;
      const med = alerts.filter(a => a.riskLevel === 'Medium').length;
      const low = alerts.filter(a => a.riskLevel === 'Low').length;

      setReturnAlerts(alerts);
      setMetrics({
        totalEvents: alerts.length,
        highRiskEvents: high,
        mediumRiskEvents: med,
        lowRiskEvents: low,
      });
      setLoading(false);
    } catch (err) {
      console.error('Error fetching return anomaly alerts:', err);
      setError(err.message || 'Failed to fetch return anomaly alerts.');
      setLoading(false);
    }
  };

  fetchReturnAlerts();
  const interval = setInterval(fetchReturnAlerts, 30000);
  return () => clearInterval(interval);
}, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Package className="h-6 w-6 text-purple-400" />
            <span>Return Anomaly Detection</span>
            {loading && <RefreshCw className="h-4 w-4 text-blue-400 animate-spin ml-2" />}
          </h2>
          <p className="text-neutral-400 mt-1">Monitoring unusual return patterns and potential abuse</p>
          {returnAlerts.length > 0 && (
              <span className="ml-2 text-sm">
                • Last updated: {new Date().toLocaleTimeString()}
              </span>
            )}
        </div>
        
        {/* Risk Level Filter */}
        <div className="flex items-center space-x-2">
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
            value={riskLevelFilter}
            onChange={(e) => setRiskLevelFilter(e.target.value)}
            className="bg-neutral-700 text-white px-3 py-2 rounded-lg border border-neutral-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Priorities</option>
            <option value="High">High Priority</option>
            <option value="Medium">Medium Priority</option>
            <option value="Low">Low Priority</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-red-900/30 to-red-800/30 rounded-lg p-4 border border-red-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-300 text-sm font-medium">High Risk</p>
              <p className="text-2xl font-bold text-white">18</p>
            </div>
            <AlertCircle className="h-5 w-5 text-red-400" />
          </div>
        </div>
        <div className="bg-gradient-to-r from-yellow-900/30 to-yellow-800/30 rounded-lg p-4 border border-yellow-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-yellow-300 text-sm font-medium">Investigating</p>
              <p className="text-2xl font-bold text-white">7</p>
            </div>
            <Search className="h-5 w-5 text-yellow-400" />
          </div>
        </div>
        <div className="bg-gradient-to-r from-blue-900/30 to-blue-800/30 rounded-lg p-4 border border-blue-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-300 text-sm font-medium">Return Rate</p>
              <p className="text-2xl font-bold text-white">23%</p>
            </div>
            <Package className="h-5 w-5 text-blue-400" />
          </div>
        </div>
        <div className="bg-gradient-to-r from-purple-900/30 to-purple-800/30 rounded-lg p-4 border border-purple-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-300 text-sm font-medium">Blocked Users</p>
              <p className="text-2xl font-bold text-white">5</p>
            </div>
            <Shield className="h-5 w-5 text-purple-400" />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredAnomalies.length > 0 ? (
          filteredAnomalies.map((anomaly) => (
            <div key={anomaly.id} onClick={() => setSelectedAlert(anomaly)} className="cursor-pointer bg-neutral-700 rounded-lg p-5 border border-neutral-600 hover:border-neutral-400 hover:bg-neutral-600 transition">
              <div className="flex items-start justify-between mb-4">
                <h3 className="text-lg font-semibold text-white">{anomaly.id}</h3>
                <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                  anomaly.riskLevel === 'High' ? 'bg-red-600 text-white' :
                  anomaly.riskLevel === 'Medium' ? 'bg-yellow-600 text-white' :
                  'bg-green-600 text-white'
                }`}>
                  {anomaly.riskLevel}
                </span>
              </div>
              
              <h4 className="text-xl font-bold text-white mb-3">{anomaly.pattern}</h4>
              
              <p className="text-neutral-300 text-sm mb-4 leading-relaxed">
                {anomaly.description}
              </p>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-neutral-400 text-sm">Customer ID:</span>
                  <span className="text-white text-sm font-mono">{anomaly.customerId}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-400 text-sm">Amount:</span>
                  <span className="text-white text-sm font-semibold">{anomaly.totalValue}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neutral-400 text-sm">Category:</span>
                  <span className="text-white text-sm">{anomaly.category}</span>
                </div>
              </div>
              
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-neutral-600">
                <div className="flex items-center text-neutral-400 text-sm">
                  <div className="w-2 h-2 bg-neutral-500 rounded-full mr-2"></div>
                  {anomaly.timeframe} ago
                </div>
                <div className="text-right">
                  <span className="text-neutral-400 text-sm">Risk Score: </span>
                  <span className={`font-bold text-sm ${
                    anomaly.riskScore >= 90 ? 'text-red-400' :
                    anomaly.riskScore >= 70 ? 'text-yellow-400' :
                    'text-green-400'
                  }`}>
                    {anomaly.riskScore}%
                  </span>
                </div>
              </div>
            </div>
          ))
        ) : (
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
                {selectedAlert.riskIndicators.map((indicator, index) => (
                  <span key={index} className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-medium bg-red-600/80 text-white border border-red-500/50">
                    {indicator.replace('_', ' ').toUpperCase()}
                  </span>
                ))}
              </div>
            </div>

            <div className="mb-8">
              <h4 className="text-sm font-semibold mb-4 text-white">Sub-Agent Analysis</h4>
              <div className="grid gap-3">
                {Object.entries(selectedAlert.subAgentResults).map(([agent, result]) => (
                  <div key={agent} className="flex items-center justify-between p-4 bg-neutral-700 rounded-lg border border-neutral-600">
                    <span className="text-sm font-medium text-white">{agent.replace(/([A-Z])/g, ' $1').trim()}</span>
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
                ))}
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

export default ReturnAnomalyPage;