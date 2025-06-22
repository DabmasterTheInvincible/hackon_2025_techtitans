import React, { useState, useEffect } from 'react';
import { AlertTriangle, Shield, Eye, TrendingUp, Clock, CheckCircle, XCircle } from 'lucide-react';
import axios from 'axios';

const TSCC_Dashboard = () => {
  const [alerts, setAlerts] = useState([]);
  const [metrics, setMetrics] = useState({
    totalEvents: 0,
    highRiskEvents: 0,
    avgProcessingTime: 0,
    accuracy: 0
  });
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [loading, setLoading] = useState(true);

  // Mock data for demonstration
  /*useEffect(() => {
    const mockAlerts = [
      {
        id: 'alert-001',
        traceId: 'trace-12345',
        timestamp: new Date().toISOString(),
        riskLevel: 'HIGH',
        eventType: 'listing',
        fraudScore: 0.85,
        title: 'Suspicious High-Value Listing',
        description: 'New seller listing premium electronics at 40% below market rate',
        riskIndicators: ['new_seller_account', 'price_anomaly', 'high_value_transaction'],
        investigationBrief: 'High-risk listing detected from 3-day old seller account. Product priced significantly below market average. Recommend immediate review and potential listing suspension.',
        subAgentResults: {
          fraudDetection: { score: 0.85, confidence: 0.92 },
          counterfeitDetection: { score: 0.78, confidence: 0.88 }
        }
      },
      {
        id: 'alert-002', 
        traceId: 'trace-12346',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        riskLevel: 'CRITICAL',
        eventType: 'review',
        fraudScore: 0.95,
        title: 'Review Manipulation Cluster',
        description: 'Multiple 5-star reviews from accounts with similar patterns',
        riskIndicators: ['review_cluster', 'fake_accounts', 'timing_pattern'],
        investigationBrief: 'Critical review manipulation detected. 15 reviews posted within 2-hour window from accounts with minimal purchase history. Recommend immediate investigation and review removal.',
        subAgentResults: {
          reviewSpam: { score: 0.95, confidence: 0.96 },
          fraudDetection: { score: 0.72, confidence: 0.85 }
        }
      },
      {
        id: 'alert-003',
        traceId: 'trace-12347', 
        timestamp: new Date(Date.now() - 600000).toISOString(),
        riskLevel: 'MEDIUM',
        eventType: 'return',
        fraudScore: 0.65,
        title: 'Unusual Return Pattern',
        description: 'Customer returning 5 high-value items in 24 hours',
        riskIndicators: ['high_return_frequency', 'value_pattern'],
        investigationBrief: 'Customer showing unusual return behavior. 5 returns totaling $3,200 in 24 hours. Previous returns were processed normally. Recommend review of return reasons and item conditions.',
        subAgentResults: {
          returnAnomaly: { score: 0.65, confidence: 0.78 },
          fraudDetection: { score: 0.45, confidence: 0.82 }
        }
      }
    ];

    setAlerts(mockAlerts);
    setMetrics({
      totalEvents: 1247,
      highRiskEvents: 23,
      avgProcessingTime: 2.3,
      accuracy: 94.2
    });
    setLoading(false);
  }, []);*/

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        // Step 1: Get active trace IDs
        const { data } = await axios.get('http://localhost:8000/active_traces');
        const traceIds = data.trace_ids;

        // Step 2: Fetch results for each trace
        const alertPromises = traceIds.map(traceId =>
          axios.get(`http://localhost:8000/results/${traceId}`)
        );

        const responses = await Promise.all(alertPromises);
        const alerts = responses.map(res => {
          const d = res.data;
          return {
            id: d.event_id,
            traceId: d.trace_id,
            timestamp: d.timestamp,
            riskLevel: d.final_risk_level.toUpperCase(),
            eventType: d.event_type,
            fraudScore: d.risk_scores?.fraud_detection ?? 0,
            title: d.raw_data?.title || 'Event',
            description: d.raw_data?.description || '',
            riskIndicators: Object.values(d.sub_agent_results || {}).flatMap(r => r.indicators || []),
            investigationBrief: d.investigation_brief,
            subAgentResults: d.sub_agent_results
          };
        });

        const totalEvents = alerts.length;
        const highRiskEvents = alerts.filter(a => ["HIGH", "CRITICAL"].includes(a.riskLevel)).length;

        setAlerts(alerts);
        setMetrics({
          totalEvents,
          highRiskEvents,
          avgProcessingTime: 2.4, // replace with dynamic value if needed
          accuracy: 93.5 // replace with Prometheus scrape if needed
        });
        setLoading(false);
      } catch (err) {
        console.error("Failed to load dashboard data:", err);
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, []);


  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'CRITICAL': return 'text-red-600 bg-red-50 border-red-200';
      case 'HIGH': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'MEDIUM': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const handleAction = (alertId, action) => {
    console.log(`Action ${action} taken on alert ${alertId}`);
    // Update alert status
    setAlerts(alerts.map(alert => 
      alert.id === alertId 
        ? { ...alert, status: action, actionedAt: new Date().toISOString() }
        : alert
    ));
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading TSCC Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">Trust & Safety Command Center</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-sm text-gray-600">
                <Clock className="h-4 w-4 mr-1" />
                Last updated: {formatTimestamp(new Date().toISOString())}
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <TrendingUp className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Total Events</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.totalEvents}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <AlertTriangle className="h-8 w-8 text-orange-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">High Risk Events</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.highRiskEvents}</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <Clock className="h-8 w-8 text-green-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Avg Processing</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.avgProcessingTime}s</p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center">
              <CheckCircle className="h-8 w-8 text-blue-600" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">Accuracy</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.accuracy}%</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Alerts List */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">High Priority Alerts</h2>
            </div>
            <div className="divide-y divide-gray-200">
              {alerts.map((alert) => (
                <div 
                  key={alert.id}
                  className={`p-6 cursor-pointer hover:bg-gray-50 ${selectedAlert?.id === alert.id ? 'bg-blue-50' : ''}`}
                  onClick={() => setSelectedAlert(alert)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getRiskColor(alert.riskLevel)}`}>
                          {alert.riskLevel}
                        </span>
                        <span className="ml-2 text-sm text-gray-600">{alert.eventType}</span>
                      </div>
                      <h3 className="mt-2 text-sm font-medium text-gray-900">{alert.title}</h3>
                      <p className="mt-1 text-sm text-gray-600">{alert.description}</p>
                      <div className="mt-2 flex items-center text-xs text-gray-500">
                        <span>Score: {alert.fraudScore}</span>
                        <span className="mx-2">•</span>
                        <span>{formatTimestamp(alert.timestamp)}</span>
                      </div>
                    </div>
                    <Eye className="h-5 w-5 text-gray-400 ml-4" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Alert Details */}
          <div className="bg-white rounded-lg shadow">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Investigation Details</h2>
            </div>
            {selectedAlert ? (
              <div className="p-6">
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-medium text-gray-900">{selectedAlert.title}</h3>
                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getRiskColor(selectedAlert.riskLevel)}`}>
                      {selectedAlert.riskLevel}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-4">{selectedAlert.description}</p>
                  
                  {/* Investigation Brief */}
                  <div className="bg-gray-50 rounded-lg p-4 mb-4">
                    <h4 className="text-sm font-medium text-gray-900 mb-2">Investigation Brief</h4>
                    <p className="text-sm text-gray-700">{selectedAlert.investigationBrief}</p>
                  </div>

                  {/* Risk Indicators */}
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-gray-900 mb-2">Risk Indicators</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedAlert.riskIndicators.map((indicator, index) => (
                        <span key={index} className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                          {indicator.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Sub-Agent Results */}
                  <div className="mb-6">
                    <h4 className="text-sm font-medium text-gray-900 mb-2">Sub-Agent Analysis</h4>
                    <div className="space-y-2">
                      {Object.entries(selectedAlert.subAgentResults).map(([agent, result]) => (
                        <div key={agent} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                          <span className="text-sm font-medium text-gray-700">{agent.replace(/([A-Z])/g, ' $1')}</span>
                          <div className="text-right">
                            <div className="text-sm font-medium text-gray-900">Score: {result.score}</div>
                            <div className="text-xs text-gray-600">Confidence: {result.confidence}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex space-x-3">
                    <button 
                      onClick={() => handleAction(selectedAlert.id, 'confirmed_fraud')}
                      className="flex-1 bg-red-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-red-700 transition-colors"
                    >
                      Confirm Fraud
                    </button>
                    <button 
                      onClick={() => handleAction(selectedAlert.id, 'false_positive')}
                      className="flex-1 bg-gray-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-700 transition-colors"
                    >
                      False Positive
                    </button>
                    <button 
                      onClick={() => handleAction(selectedAlert.id, 'escalate')}
                      className="flex-1 bg-orange-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-orange-700 transition-colors"
                    >
                      Escalate to Legal
                    </button>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-6 text-center text-gray-500">
                <Eye className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                <p>Select an alert to view investigation details</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TSCC_Dashboard;