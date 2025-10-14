import React, { useState, useEffect } from 'react';
import { AlertTriangle, Shield, Activity, Lock, Database, Download, RefreshCw, Globe, TrendingUp, Bell } from 'lucide-react';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';

const mockData = {
  overview: {
    active_threats: 287,
    ai_content_24h: 19482,
    critical_incidents: 8,
    cross_border_alerts: 12
  },
  platforms: ["X", "Reddit", "Telegram", "TikTok", "News Blogs"],
  threat_feed: [
    {time: "13:42", event: "Synthetic Disinformation Cluster Detected", source: "X", region: "India", severity: "critical"},
    {time: "13:39", event: "Coordinated AI Narrative", source: "Reddit", region: "Global", severity: "high"},
    {time: "13:34", event: "Fake Video Transcript", source: "Telegram", region: "Russia", severity: "medium"},
    {time: "13:30", event: "Benign Academic Text", source: "News Blogs", region: "UK", severity: "low"}
  ],
  gnn_clusters: [
    {id: "492A-FX", confidence: 0.91, model: "GPT-4", platforms: ["X","Reddit"], speed: "1.8x/min"},
    {id: "213B-ZT", confidence: 0.87, model: "Claude 3", platforms: ["Telegram"], speed: "1.2x/min"}
  ],
  models: {
    "GPT-Series": 41, "Claude": 29, "Gemini": 19, "Perplexity": 11
  },
  topics: [
    {topic: "Election Manipulation", severity: "High", trend: "Rising"},
    {topic: "Vaccine Misinformation", severity: "Medium", trend: "Stable"},
    {topic: "Economic Discontent", severity: "Low", trend: "Rising"}
  ],
  blockchain_status: {
    hash: "0xa5f9b23de79c4e",
    verified: true,
    timestamp: "13:45:23"
  },
  risk_level: {
    score: 92,
    region: "South Asia",
    platform: "X"
  }
};

const trendData = [
  {day: 'Mon', value: 10500},
  {day: 'Tue', value: 12800},
  {day: 'Wed', value: 17600},
  {day: 'Thu', value: 19800},
  {day: 'Fri', value: 22300},
  {day: 'Sat', value: 24400},
  {day: 'Sun', value: 19482}
];

const Dashboard = () => {
  const [time, setTime] = useState(new Date());
  const [secureMode, setSecureMode] = useState(true);
  const [selectedPlatform, setSelectedPlatform] = useState("All Platforms");
  const [timeRange, setTimeRange] = useState("24h");
  const [blockchainHash, setBlockchainHash] = useState(null);
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const modelData = Object.entries(mockData.models).map(([name, value]) => ({
    name, value
  }));

  const COLORS = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B'];

  const handleBlockchainEncode = () => {
    setBlockchainHash({
      hash: `0x${Math.random().toString(16).substr(2, 14)}`,
      timestamp: new Date().toLocaleTimeString('en-IN'),
      verified: true
    });
  };

  const handleRefresh = () => {
    setIsRefreshing(true);
    setTimeout(() => setIsRefreshing(false), 1500);
  };

  const getSeverityColor = (severity) => {
    const colors = {
      critical: 'bg-red-500',
      high: 'bg-orange-500',
      medium: 'bg-yellow-500',
      low: 'bg-green-500'
    };
    return colors[severity] || 'bg-gray-500';
  };

  const getSeverityIcon = (severity) => {
    const icons = {
      'High': '🔴',
      'Medium': '🟠',
      'Low': '🟡'
    };
    return icons[severity] || '⚪';
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-sans">
      {/* Top Navigation */}
      <div className="bg-gray-900 border-b border-cyan-500 px-6 py-4 shadow-lg shadow-cyan-500/10">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Shield className="w-8 h-8 text-cyan-400" style={{filter: 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.6))'}} />
            <div>
              <h1 className="text-xl font-bold text-cyan-400 tracking-wide">LLM THREAT INTELLIGENCE & RESPONSE COMMAND</h1>
              <p className="text-xs text-gray-500 font-mono">LTRC — National AI Defense System</p>
            </div>
          </div>
          <div className="flex items-center space-x-6">
            <div className="text-right">
              <p className="text-sm font-mono text-gray-300">{time.toLocaleDateString('en-US', {weekday: 'short', year: 'numeric', month: 'short', day: 'numeric'})}</p>
              <p className="text-lg font-mono text-cyan-400">{time.toLocaleTimeString('en-IN')}</p>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-400">Connected</span>
              <span className="text-xs text-gray-500">| Last Sync: 2 min ago</span>
            </div>
            <div className="flex items-center space-x-2 bg-gray-950 px-3 py-2 rounded-lg border border-gray-800 shadow-inner">
              <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center text-xs font-bold shadow-lg shadow-cyan-500/30">AK</div>
              <span className="text-sm">Analyst: A. Karmore</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Secure Mode</span>
              <button 
                onClick={() => setSecureMode(!secureMode)}
                className={`w-12 h-6 rounded-full transition-all ${secureMode ? 'bg-green-500' : 'bg-gray-600'}`}
                style={secureMode ? {boxShadow: '0 0 20px rgba(34, 197, 94, 0.5)'} : {}}
              >
                <div className={`w-5 h-5 bg-white rounded-full transition-transform ${secureMode ? 'translate-x-6' : 'translate-x-1'}`}></div>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-4 p-4">
        {/* Left Panel - Threat Feed */}
        <div className="col-span-3 space-y-4">
          <div className="bg-gray-900 border border-cyan-500/50 rounded-lg p-4 shadow-lg shadow-cyan-500/5">
            <h2 className="text-sm font-bold text-cyan-400 mb-4 flex items-center">
              <Activity className="w-4 h-4 mr-2" style={{filter: 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.6))'}} />
              THREAT FEED & FILTERS
            </h2>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-gray-500 block mb-1 uppercase tracking-wider">Platform</label>
                <select 
                  value={selectedPlatform}
                  onChange={(e) => setSelectedPlatform(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded px-3 py-2 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none transition-all"
                  style={{boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.3)'}}
                >
                  <option>All Platforms</option>
                  {mockData.platforms.map(p => <option key={p}>{p}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-500 block mb-1 uppercase tracking-wider">Time Range</label>
                <select 
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                  className="w-full bg-gray-950 border border-gray-800 rounded px-3 py-2 text-sm text-gray-200 focus:border-cyan-500 focus:outline-none transition-all"
                  style={{boxShadow: 'inset 0 2px 4px rgba(0, 0, 0, 0.3)'}}
                >
                  <option>Last 1h</option>
                  <option>24h</option>
                  <option>7d</option>
                </select>
              </div>
            </div>
          </div>

          <div className="bg-gray-900 border border-cyan-500/50 rounded-lg p-4 h-[calc(100vh-280px)] overflow-hidden flex flex-col shadow-lg shadow-cyan-500/5">
            <h3 className="text-sm font-bold text-cyan-400 mb-3 flex items-center">
              <Bell className="w-4 h-4 mr-2 animate-pulse" style={{filter: 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.6))'}} />
              LIVE THREAT STREAM
            </h3>
            <div className="mb-3 grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="bg-gray-950 border border-red-500/50 rounded p-2 shadow-inner">
                <div className="text-gray-500">Active Threats</div>
                <div className="text-red-400 text-lg font-bold" style={{filter: 'drop-shadow(0 0 8px rgba(239, 68, 68, 0.6))'}}>{mockData.overview.active_threats}</div>
              </div>
              <div className="bg-gray-950 border border-yellow-500/50 rounded p-2 shadow-inner">
                <div className="text-gray-500">Critical</div>
                <div className="text-yellow-400 text-lg font-bold" style={{filter: 'drop-shadow(0 0 8px rgba(251, 191, 36, 0.6))'}}>{mockData.overview.critical_incidents}</div>
              </div>
            </div>
            <div className="space-y-2 overflow-y-auto flex-1">
              {mockData.threat_feed.map((feed, i) => (
                <div key={i} className="bg-gray-950 border border-gray-800 rounded p-3 hover:border-cyan-500/50 transition-all cursor-pointer">
                  <div className="flex items-start justify-between mb-1">
                    <span className={`${getSeverityColor(feed.severity)} w-2 h-2 rounded-full mt-1.5 animate-pulse`}></span>
                    <span className="text-xs text-gray-600 font-mono">[{feed.time}]</span>
                  </div>
                  <p className="text-xs text-gray-300 mb-2">{feed.event}</p>
                  <div className="flex justify-between text-xs">
                    <span className="text-cyan-400">{feed.source}</span>
                    <span className="text-gray-600">{feed.region}</span>
                  </div>
                </div>
              ))}
            </div>
            <button 
              onClick={handleRefresh}
              className="mt-3 w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white py-2 rounded text-sm font-semibold flex items-center justify-center space-x-2 transition-all shadow-lg shadow-cyan-500/20"
            >
              <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
              <span>Fetch Latest from Platforms</span>
            </button>
          </div>
        </div>

        {/* Center Panel - GNN Visualization */}
        <div className="col-span-6 space-y-4">
          <div className="bg-gray-900 border border-cyan-500/50 rounded-lg p-4 shadow-lg shadow-cyan-500/5" style={{height: '700px'}}>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-bold text-cyan-400 flex items-center">
                <Globe className="w-4 h-4 mr-2" style={{filter: 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.6))'}} />
                GNN CLUSTER GRAPH — DISINFORMATION NETWORKS
              </h2>
              <div className="flex space-x-2">
                <button className="bg-gray-950 border border-gray-800 hover:border-cyan-500 px-3 py-1.5 rounded text-xs text-gray-300 transition-all">
                  <RefreshCw className="w-3 h-3 inline mr-1" />
                  Re-run GNN
                </button>
                <button className="bg-gray-950 border border-gray-800 hover:border-purple-500 px-3 py-1.5 rounded text-xs text-gray-300 transition-all">
                  <Database className="w-3 h-3 inline mr-1" />
                  Export to Blockchain
                </button>
                <button className="bg-gray-950 border border-gray-800 hover:border-green-500 px-3 py-1.5 rounded text-xs text-gray-300 transition-all">
                  <Download className="w-3 h-3 inline mr-1" />
                  Download PDF
                </button>
              </div>
            </div>
            
            <div className="relative h-[calc(100%-60px)] bg-gray-950 rounded-lg border border-gray-800 overflow-hidden shadow-inner">
              <svg width="100%" height="100%" viewBox="0 0 800 600" className="absolute inset-0">
                <defs>
                  <radialGradient id="node-gradient-critical">
                    <stop offset="0%" stopColor="#EF4444" stopOpacity="0.9"/>
                    <stop offset="100%" stopColor="#DC2626" stopOpacity="0.4"/>
                  </radialGradient>
                  <radialGradient id="node-gradient-high">
                    <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.9"/>
                    <stop offset="100%" stopColor="#D97706" stopOpacity="0.4"/>
                  </radialGradient>
                  <radialGradient id="node-gradient-medium">
                    <stop offset="0%" stopColor="#22D3EE" stopOpacity="0.9"/>
                    <stop offset="100%" stopColor="#06B6D4" stopOpacity="0.4"/>
                  </radialGradient>
                  <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                    <feMerge>
                      <feMergeNode in="coloredBlur"/>
                      <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                  </filter>
                </defs>
                
                {/* Background grid */}
                <defs>
                  <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1F2937" strokeWidth="0.5" opacity="0.3"/>
                  </pattern>
                </defs>
                <rect width="800" height="600" fill="url(#grid)" />
                
                {/* Connections */}
                <line x1="240" y1="180" x2="400" y2="300" stroke="#22D3EE" strokeWidth="2" opacity="0.4" filter="url(#glow)"/>
                <line x1="560" y1="240" x2="400" y2="300" stroke="#F59E0B" strokeWidth="2" opacity="0.4" filter="url(#glow)"/>
                <line x1="320" y1="420" x2="400" y2="300" stroke="#EF4444" strokeWidth="2" opacity="0.4" filter="url(#glow)"/>
                <line x1="520" y1="450" x2="400" y2="300" stroke="#22D3EE" strokeWidth="2" opacity="0.4" filter="url(#glow)"/>
                <line x1="160" y1="360" x2="240" y2="180" stroke="#22D3EE" strokeWidth="1.5" opacity="0.3"/>
                <line x1="640" y1="360" x2="560" y2="240" stroke="#F59E0B" strokeWidth="1.5" opacity="0.3"/>
                
                {/* Main Cluster Nodes */}
                <g onMouseEnter={() => setHoveredCluster(mockData.gnn_clusters[0])} onMouseLeave={() => setHoveredCluster(null)} className="cursor-pointer">
                  <circle cx="400" cy="300" r="50" fill="url(#node-gradient-critical)" filter="url(#glow)">
                    <animate attributeName="r" values="50;55;50" dur="2s" repeatCount="indefinite"/>
                  </circle>
                  <text x="400" y="300" textAnchor="middle" dy=".3em" fill="white" fontSize="14" fontWeight="bold" fontFamily="monospace">492A-FX</text>
                  <text x="400" y="320" textAnchor="middle" fill="#FCA5A5" fontSize="10" fontFamily="monospace">CRITICAL</text>
                </g>
                
                <g onMouseEnter={() => setHoveredCluster(mockData.gnn_clusters[1])} onMouseLeave={() => setHoveredCluster(null)} className="cursor-pointer">
                  <circle cx="240" cy="180" r="38" fill="url(#node-gradient-high)" filter="url(#glow)">
                    <animate attributeName="r" values="38;43;38" dur="2.5s" repeatCount="indefinite"/>
                  </circle>
                  <text x="240" y="180" textAnchor="middle" dy=".3em" fill="white" fontSize="12" fontWeight="bold" fontFamily="monospace">213B-ZT</text>
                  <text x="240" y="195" textAnchor="middle" fill="#FED7AA" fontSize="9" fontFamily="monospace">HIGH</text>
                </g>
                
                {/* Secondary nodes */}
                <g className="cursor-pointer">
                  <circle cx="560" cy="240" r="32" fill="url(#node-gradient-medium)" filter="url(#glow)">
                    <animate attributeName="r" values="32;36;32" dur="3s" repeatCount="indefinite"/>
                  </circle>
                  <text x="560" y="240" textAnchor="middle" dy=".3em" fill="white" fontSize="11" fontWeight="bold" fontFamily="monospace">8F3C-ZX</text>
                </g>
                
                <g className="cursor-pointer">
                  <circle cx="320" cy="420" r="28" fill="url(#node-gradient-high)" filter="url(#glow)">
                    <animate attributeName="r" values="28;32;28" dur="2.2s" repeatCount="indefinite"/>
                  </circle>
                  <text x="320" y="420" textAnchor="middle" dy=".3em" fill="white" fontSize="10" fontWeight="bold" fontFamily="monospace">A12D-P</text>
                </g>
                
                <g className="cursor-pointer">
                  <circle cx="520" cy="450" r="26" fill="url(#node-gradient-medium)" filter="url(#glow)">
                    <animate attributeName="r" values="26;30;26" dur="2.8s" repeatCount="indefinite"/>
                  </circle>
                  <text x="520" y="450" textAnchor="middle" dy=".3em" fill="white" fontSize="10" fontWeight="bold" fontFamily="monospace">C9B7-Q</text>
                </g>
                
                {/* Small peripheral nodes */}
                <circle cx="160" cy="360" r="20" fill="url(#node-gradient-medium)" opacity="0.7" filter="url(#glow)">
                  <animate attributeName="r" values="20;23;20" dur="3.5s" repeatCount="indefinite"/>
                </circle>
                
                <circle cx="640" cy="360" r="20" fill="url(#node-gradient-high)" opacity="0.7" filter="url(#glow)">
                  <animate attributeName="r" values="20;23;20" dur="3.2s" repeatCount="indefinite"/>
                </circle>
                
                <circle cx="400" cy="120" r="18" fill="url(#node-gradient-medium)" opacity="0.6" filter="url(#glow)">
                  <animate attributeName="r" values="18;21;18" dur="4s" repeatCount="indefinite"/>
                </circle>
                
                <circle cx="100" cy="200" r="16" fill="url(#node-gradient-medium)" opacity="0.5">
                  <animate attributeName="opacity" values="0.5;0.8;0.5" dur="3s" repeatCount="indefinite"/>
                </circle>
                
                <circle cx="700" cy="480" r="16" fill="url(#node-gradient-high)" opacity="0.5">
                  <animate attributeName="opacity" values="0.5;0.8;0.5" dur="2.7s" repeatCount="indefinite"/>
                </circle>
              </svg>
              
              {hoveredCluster && (
                <div className="absolute top-4 left-4 bg-gray-900 border border-cyan-500 rounded-lg p-4 shadow-2xl z-10 min-w-[280px]" style={{boxShadow: '0 0 30px rgba(34, 211, 238, 0.3)'}}>
                  <div className="space-y-2 text-xs font-mono">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Cluster ID:</span>
                      <span className="text-cyan-400 font-bold">{hoveredCluster.id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Confidence:</span>
                      <span className="text-green-400">{(hoveredCluster.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Dominant Model:</span>
                      <span className="text-purple-400">{hoveredCluster.model}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Platforms:</span>
                      <span className="text-yellow-400">{hoveredCluster.platforms.join(', ')}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Propagation Speed:</span>
                      <span className="text-red-400">{hoveredCluster.speed}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Analytics */}
        <div className="col-span-3 space-y-4">
          <div className="bg-gray-900 border border-cyan-500/50 rounded-lg p-4 shadow-lg shadow-cyan-500/5">
            <h3 className="text-sm font-bold text-cyan-400 mb-3" style={{filter: 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.6))'}}>MODEL ATTRIBUTION</h3>
            <ResponsiveContainer width="100%" height={150}>
              <PieChart>
                <Pie
                  data={modelData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={60}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {modelData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-3 space-y-2">
              {Object.entries(mockData.models).map(([model, pct], i) => (
                <div key={model} className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 rounded" style={{backgroundColor: COLORS[i]}}></div>
                    <span className="text-gray-300">{model}</span>
                  </div>
                  <span className="text-gray-400 font-mono">{pct}%</span>
                </div>
              ))}
            </div>
            <div className="mt-4 bg-gray-950 rounded p-3 shadow-inner border border-gray-800">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="text-gray-600 border-b border-gray-800">
                    <th className="text-left pb-2">Model</th>
                    <th className="text-center pb-2">Conf</th>
                    <th className="text-right pb-2">Cases</th>
                  </tr>
                </thead>
                <tbody className="text-gray-300">
                  <tr className="border-b border-gray-800">
                    <td className="py-2">GPT-4</td>
                    <td className="text-center text-green-400">0.91</td>
                    <td className="text-right">342</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Claude 3</td>
                    <td className="text-center text-green-400">0.89</td>
                    <td className="text-right">215</td>
                  </tr>
                  <tr className="border-b border-gray-800">
                    <td className="py-2">Gemini 1.5</td>
                    <td className="text-center text-yellow-400">0.87</td>
                    <td className="text-right">188</td>
                  </tr>
                  <tr>
                    <td className="py-2">Perplexity</td>
                    <td className="text-center text-yellow-400">0.83</td>
                    <td className="text-right">105</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-gray-900 border border-purple-500/50 rounded-lg p-4 shadow-lg shadow-purple-500/5">
            <h3 className="text-sm font-bold text-purple-400 mb-2 flex items-center" style={{filter: 'drop-shadow(0 0 8px rgba(168, 85, 247, 0.6))'}}>
              <Lock className="w-4 h-4 mr-2" />
              BLOCKCHAIN VERIFICATION
            </h3>
            <p className="text-xs text-gray-500 mb-3">All validated LLM detections are cryptographically hashed for traceability.</p>
            <button 
              onClick={handleBlockchainEncode}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white py-2 rounded text-sm font-semibold transition-all mb-3 shadow-lg shadow-purple-500/20"
            >
              🧱 Encode with Blockchain
            </button>
            {blockchainHash && (
              <div className="bg-gray-950 rounded p-3 space-y-2 text-xs font-mono border border-purple-500/50 shadow-inner">
                <div className="flex justify-between">
                  <span className="text-gray-500">Block Hash:</span>
                  <span className="text-purple-400">{blockchainHash.hash}...</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Timestamp:</span>
                  <span className="text-cyan-400">{blockchainHash.timestamp} IST</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Status:</span>
                  <span className="text-green-400">Verified ✅</span>
                </div>
              </div>
            )}
          </div>

          <div className="bg-gray-900 border border-red-500/50 rounded-lg p-4 shadow-lg shadow-red-500/5">
            <h3 className="text-sm font-bold text-red-400 mb-3 flex items-center" style={{filter: 'drop-shadow(0 0 8px rgba(239, 68, 68, 0.6))'}}>
              <AlertTriangle className="w-4 h-4 mr-2" />
              AUTOMATED THREAT RISK
            </h3>
            <div className="flex items-center justify-center mb-3">
              <div className="relative w-32 h-32">
                <svg className="w-full h-full transform -rotate-90">
                  <circle cx="64" cy="64" r="56" stroke="#1a1a1a" strokeWidth="8" fill="none"/>
                  <circle 
                    cx="64" 
                    cy="64" 
                    r="56" 
                    stroke="#EF4444" 
                    strokeWidth="8" 
                    fill="none"
                    strokeDasharray={`${2 * Math.PI * 56 * 0.92} ${2 * Math.PI * 56}`}
                    style={{filter: 'drop-shadow(0 0 12px rgba(239, 68, 68, 0.8))'}}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="text-3xl font-bold text-red-400" style={{filter: 'drop-shadow(0 0 10px rgba(239, 68, 68, 0.8))'}}>{mockData.risk_level.score}%</span>
                  <span className="text-xs text-gray-500 uppercase tracking-wider">SEVERE</span>
                </div>
              </div>
            </div>
            <div className="space-y-2 text-xs font-mono bg-gray-950 rounded p-3 shadow-inner border border-gray-800">
              <div className="flex justify-between">
                <span className="text-gray-500">Dominant Model:</span>
                <span className="text-purple-400">GPT-4</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Region:</span>
                <span className="text-yellow-400">{mockData.risk_level.region}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Top Platform:</span>
                <span className="text-cyan-400">{mockData.risk_level.platform}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="px-4 pb-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-900 border border-cyan-500/50 rounded-lg p-4 shadow-lg shadow-cyan-500/5">
            <h3 className="text-sm font-bold text-cyan-400 mb-3 flex items-center" style={{filter: 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.6))'}}>
              <TrendingUp className="w-4 h-4 mr-2" />
              AI-GENERATED DISINFORMATION VOLUME (PAST 7 DAYS)
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="day" stroke="#9CA3AF" style={{fontSize: '12px'}} />
                <YAxis stroke="#9CA3AF" style={{fontSize: '12px'}} />
                <RechartsTooltip 
                  contentStyle={{backgroundColor: '#111827', border: '1px solid #22D3EE', borderRadius: '8px'}}
                  labelStyle={{color: '#9CA3AF'}}
                />
                <Line type="monotone" dataKey="value" stroke="#22D3EE" strokeWidth={3} dot={{fill: '#22D3EE', r: 5, strokeWidth: 2, stroke: '#111827'}} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-900 border border-cyan-500/50 rounded-lg p-4 shadow-lg shadow-cyan-500/5">
            <h3 className="text-sm font-bold text-cyan-400 mb-3" style={{filter: 'drop-shadow(0 0 8px rgba(34, 211, 238, 0.6))'}}>RECENT THEMATIC TOPICS</h3>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-600 border-b border-gray-800">
                  <th className="text-left pb-2 uppercase text-xs tracking-wider">Topic</th>
                  <th className="text-center pb-2 uppercase text-xs tracking-wider">Severity</th>
                  <th className="text-center pb-2 uppercase text-xs tracking-wider">Trend</th>
                  <th className="text-right pb-2 uppercase text-xs tracking-wider">Last Update</th>
                </tr>
              </thead>
              <tbody className="text-gray-300">
                {mockData.topics.map((topic, i) => (
                  <tr key={i} className="border-b border-gray-800 hover:bg-gray-950 transition-all">
                    <td className="py-3">{topic.topic}</td>
                    <td className="text-center">{getSeverityIcon(topic.severity)} {topic.severity}</td>
                    <td className="text-center">
                      <span className={topic.trend === 'Rising' ? 'text-red-400' : 'text-gray-500'}>
                        {topic.trend}
                      </span>
                    </td>
                    <td className="text-right text-gray-600 font-mono text-xs">13:{40-i*5}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="bg-gray-900 border-t border-cyan-500/50 px-6 py-3 shadow-lg">
        <div className="flex items-center justify-between text-xs">
          <p className="text-gray-600">© 2025 National AI Threat Response Unit | Blockchain Verified | Data classified under CyberSec Act 2025</p>
          <p className="text-red-400 animate-pulse font-mono" style={{filter: 'drop-shadow(0 0 8px rgba(239, 68, 68, 0.6))'}}>● SYSTEM MONITORING LIVE STREAMS FROM X / REDDIT / TELEGRAM / BLOGS</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
