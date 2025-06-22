import React, { useState } from 'react';
import { Shield, Search, Package, AlertCircle, Eye, AlertTriangle, Download } from 'lucide-react';

const CounterfeitPage = () => {
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [filterPriority, setFilterPriority] = useState('all');
  
  const counterfeitAlerts = [
    {
      id: 'CF001',
      title: 'Suspected Counterfeit iPhone',
      productName: 'Apple iPhone 15 Pro',
      brand: 'Apple',
      sellerId: 'S12345',
      customerId: 'C001234',
      confidence: 94,
      reason: 'Image similarity analysis flagged product images',
      price: '$899.99',
      amount: '$899.99', // Added missing amount property
      status: 'Under Review',
      pattern: 'Image Anomaly Detection',
      description: 'Product images show inconsistencies with authentic Apple iPhone 15 Pro packaging and product details. Seller has limited history and suspicious pricing.',
      totalValue: '$899.99',
      category: 'Electronics',
      timeframe: '2 hours',
      riskScore: 94,
      riskLevel: 'High',
      priority: 'High', // Added missing priority property
      investigationBrief: 'Computer vision analysis detected 89% similarity to known counterfeit product images. Seller account created 3 days ago with no verification documents. Product listed at 15% below market price with generic product photos.',
      riskIndicators: ['image_similarity_match', 'new_seller_account', 'below_market_pricing', 'generic_product_photos'],
      subAgentResults: {
        'Image Analysis': { score: 89, confidence: 'High' },
        'Seller Verification': { score: 92, confidence: 'High' },
        'Price Analysis': { score: 78, confidence: 'Medium' },
        'Product Authentication': { score: 94, confidence: 'High' }
      }
    },
    {
      id: 'CF002', // Fixed duplicate ID
      title: 'Suspected Counterfeit iPhone - Different Listing',
      productName: 'Apple iPhone 15 Pro',
      brand: 'Apple',
      sellerId: 'S12346', // Different seller ID
      customerId: 'C001235', // Different customer ID
      confidence: 94,
      reason: 'Image similarity analysis flagged product images',
      price: '$899.99',
      amount: '$899.99', // Added missing amount property
      status: 'Under Review',
      pattern: 'Image Anomaly Detection',
      description: 'Product images show inconsistencies with authentic Apple iPhone 15 Pro packaging and product details. Seller has limited history and suspicious pricing.',
      totalValue: '$899.99',
      category: 'Electronics',
      timeframe: '2 hours',
      riskScore: 94,
      riskLevel: 'Medium',
      priority: 'Medium', // Added missing priority property
      investigationBrief: 'Computer vision analysis detected 89% similarity to known counterfeit product images. Seller account created 3 days ago with no verification documents. Product listed at 15% below market price with generic product photos.',
      riskIndicators: ['image_similarity_match', 'new_seller_account', 'below_market_pricing', 'generic_product_photos'],
      subAgentResults: {
        'Image Analysis': { score: 89, confidence: 'High' },
        'Seller Verification': { score: 92, confidence: 'High' },
        'Price Analysis': { score: 78, confidence: 'Medium' },
        'Product Authentication': { score: 94, confidence: 'High' }
      }
    },
    {
      id: 'CF003', // Fixed ID to match the original pattern
      title: 'Unauthorized Nike Retailer',
      productName: 'Nike Air Jordan Retro',
      brand: 'Nike',
      sellerId: 'S67890',
      customerId: 'C005678',
      confidence: 87,
      reason: 'Unauthorized seller detected',
      price: '$150.00',
      amount: '$450.00', // Added missing amount property
      status: 'Suspended',
      pattern: 'Unauthorized Distribution',
      description: 'Seller is not an authorized Nike retailer but is selling premium Nike products. Brand verification failed and seller location inconsistent with authorized distributors.',
      totalValue: '$450.00',
      category: 'Apparel',
      timeframe: '6 hours',
      riskScore: 87,
      riskLevel: 'High',
      priority: 'High', // Added missing priority property
      investigationBrief: 'Seller account not found in Nike\'s authorized retailer database. Multiple listings for limited edition products that should only be available through official channels. Geographic location does not match any authorized distribution centers.',
      riskIndicators: ['unauthorized_seller', 'limited_edition_availability', 'geographic_mismatch', 'brand_verification_failed'],
      subAgentResults: {
        'Brand Authorization': { score: 95, confidence: 'High' },
        'Distribution Verification': { score: 92, confidence: 'High' },
        'Geographic Analysis': { score: 73, confidence: 'Medium' },
        'Product Availability': { score: 88, confidence: 'High' }
      }
    },
    {
      id: 'CF004', // Fixed ID to be unique
      title: 'Luxury Watch Price Anomaly',
      productName: 'Rolex Submariner Watch',
      brand: 'Rolex',
      sellerId: 'S11111',
      customerId: 'C009999',
      confidence: 99,
      reason: 'Price anomaly and seller verification failed',
      price: '$299.99',
      amount: '$299.99', // Added missing amount property
      status: 'Removed',
      pattern: 'Extreme Price Anomaly',
      description: 'Authentic Rolex Submariner listed at $299.99 when market value is $8,000+. Seller has no luxury goods history and product images show quality inconsistencies.',
      totalValue: '$299.99',
      category: 'Luxury Goods',
      timeframe: '1 hour',
      riskScore: 99,
      riskLevel: 'High',
      priority: 'High', // Added missing priority property
      investigationBrief: 'Product listed at 96% below market value for authentic Rolex Submariner. Seller account has no previous luxury goods sales and no authentication certificates provided. Product serial number does not match Rolex database records.',
      riskIndicators: ['extreme_price_anomaly', 'no_authentication_docs', 'invalid_serial_number', 'seller_category_mismatch'],
      subAgentResults: {
        'Price Anomaly Detection': { score: 99, confidence: 'High' },
        'Serial Number Verification': { score: 94, confidence: 'High' },
        'Authentication Analysis': { score: 88, confidence: 'High' },
        'Seller Category History': { score: 96, confidence: 'High' }
      }
    }
  ];

  const filteredAlerts = filterPriority === 'all' 
    ? counterfeitAlerts
    : counterfeitAlerts.filter(alert => alert.priority === filterPriority);

  const handleAction = (id, action) => {
    console.log(`Action taken on ${id}: ${action}`);
    // Add actual implementation here
  };

  // Fixed stats calculation - now uses correct properties and handles parsing properly
  const stats = {
    totalAlerts: counterfeitAlerts.length,
    highPriority: counterfeitAlerts.filter(a => a.priority === 'High').length,
    mediumPriority: counterfeitAlerts.filter(a => a.priority === 'Medium').length,
    lowPriority: counterfeitAlerts.filter(a => a.priority === 'Low').length,
    totalAmount: counterfeitAlerts.reduce((sum, alert) => {
      const amount = parseFloat(alert.amount.replace('$', '').replace(',', ''));
      return sum + (isNaN(amount) ? 0 : amount);
    }, 0)
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center space-x-2">
            <Shield className="h-6 w-6 text-purple-400" />
            <span>Counterfeit Detection</span>
          </h2>
          <p className="text-neutral-400 mt-1">AI-powered detection of counterfeit products and unauthorized sellers</p>
        </div>
        <div className="flex items-center space-x-3">
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

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-red-900/30 to-red-800/30 rounded-lg p-4 border border-red-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-300 text-sm font-medium">Suspected Fakes</p>
              <p className="text-2xl font-bold text-white">18</p>
            </div>
            <AlertCircle className="h-5 w-5 text-red-400" />
          </div>
        </div>
        <div className="bg-gradient-to-r from-yellow-900/30 to-yellow-800/30 rounded-lg p-4 border border-yellow-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-yellow-300 text-sm font-medium">Under Review</p>
              <p className="text-2xl font-bold text-white">7</p>
            </div>
            <Search className="h-5 w-5 text-yellow-400" />
          </div>
        </div>
        <div className="bg-gradient-to-r from-blue-900/30 to-blue-800/30 rounded-lg p-4 border border-blue-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-300 text-sm font-medium">Detection Rate</p>
              <p className="text-2xl font-bold text-white">96%</p>
            </div>
            <Package className="h-5 w-5 text-blue-400" />
          </div>
        </div>
        <div className="bg-gradient-to-r from-purple-900/30 to-purple-800/30 rounded-lg p-4 border border-purple-800/50">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-300 text-sm font-medium">Blocked Sellers</p>
              <p className="text-2xl font-bold text-white">5</p>
            </div>
            <Shield className="h-5 w-5 text-purple-400" />
          </div>
        </div>
      </div>

      {/* Fixed: Now uses filteredAlerts instead of counterfeitAlerts */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredAlerts.map((anomaly) => (
          <div 
            key={anomaly.id} 
            onClick={() => setSelectedAlert(anomaly)} 
            className="cursor-pointer bg-neutral-700 rounded-lg p-5 border border-neutral-600 hover:border-neutral-400 hover:bg-neutral-600 transition"
          >
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
                <span className="text-neutral-400 text-sm">Seller ID:</span>
                <span className="text-white text-sm font-mono">{anomaly.sellerId}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neutral-400 text-sm">Price:</span>
                <span className="text-white text-sm font-semibold">{anomaly.price}</span>
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
        ))}
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
                    {indicator.replace(/_/g, ' ').toUpperCase()}
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
                onClick={() => handleAction(selectedAlert.id, 'confirmed_counterfeit')} 
                className="flex-1 bg-gradient-to-r from-red-600 to-red-700 text-white px-6 py-3 rounded-lg font-semibold hover:from-red-700 hover:to-red-800 transition-all duration-200 shadow-lg"
              >
                Confirm Counterfeit
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
                Escalate to Brand
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

export default CounterfeitPage;