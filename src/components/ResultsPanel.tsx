// Horizontal bar chart for feature importance
interface FeatureImportanceChartProps {
  featureImportance: Record<string, number>;
}

const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({ featureImportance }) => {
  if (!featureImportance || Object.keys(featureImportance).length === 0) return null;
  // Normalize scores for bar width
  const maxScore = Math.max(...Object.values(featureImportance));
  return (
    <div className="mt-4">
      <h4 className="font-medium text-white mb-2 text-sm">Feature Importance</h4>
      <div className="space-y-2">
        {Object.entries(featureImportance).map(([name, score]) => (
          <div key={name} className="flex items-center gap-2">
            <span className="w-32 text-xs text-gray-300 truncate">{name}</span>
            <div className="flex-1 bg-gray-600 rounded h-3 relative">
              <div
                className="bg-green-400 h-3 rounded"
                style={{ width: `${(score / maxScore) * 100}%` }}
                title={score.toFixed(2)}
              />
            </div>
            <span className="ml-2 text-xs text-gray-400 font-mono">{score.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
};
import React from 'react';
import { TrendingUp, Target, Layers, Clock, BarChart3, Activity, Thermometer, Zap } from 'lucide-react';

interface AnalyticsDataPoint {
  timestamp: number;
  progress: number;
  materialRemovalRate: number;
  powerConsumption: number;
  surfaceQuality: number;
  temperature: number;
  efficiency: number;
}

interface Prediction {
  materialRemovalRate: number;
  surfaceRoughness: number;
  dimensionalAccuracy: number;
  processingTime: number;
}

interface ResultsPanelProps {
  predictions: Record<string, Prediction>;
  currentParameters: any;
  analyticsData: AnalyticsDataPoint[];
  processMetrics: any;
  cuttingMethod: string;
}

const ResultsPanel: React.FC<ResultsPanelProps> = ({ 
  predictions, 
  currentParameters, 
  analyticsData, 
  processMetrics,
  cuttingMethod
}) => {
  // Method-specific metric labels
  const getMethodSpecificLabels = () => {
    switch (cuttingMethod) {
      case 'wire':
        return {
          powerLabel: 'Voltage (V)',
          speedLabel: 'Wire Speed (mm/min)',
          processName: 'Wire EDM'
        };
      case 'water':
        return {
          powerLabel: 'Pressure (PSI)',
          speedLabel: 'Cutting Speed (mm/min)',
          processName: 'Water Jet'
        };
      case 'laser':
        return {
          powerLabel: 'Laser Power (kW)',
          speedLabel: 'Cutting Speed (mm/min)',
          processName: 'Laser Cutting'
        };
      case 'cnc':
        return {
          powerLabel: 'Spindle Power (kW)',
          speedLabel: 'Feed Rate (mm/min)',
          processName: 'CNC Milling'
        };
      default:
        return {
          powerLabel: 'Power',
          speedLabel: 'Speed',
          processName: 'Cutting'
        };
    }
  };

  const methodLabels = getMethodSpecificLabels();

  const metrics = [
    { 
      key: 'materialRemovalRate', 
      label: 'Material Removal Rate', 
      unit: 'mm³/min', 
      icon: TrendingUp, 
      color: 'text-green-400',
      bgColor: 'bg-green-400/10' 
    },
    { 
      key: 'surfaceRoughness', 
      label: 'Surface Roughness', 
      unit: 'Ra (μm)', 
      icon: Layers, 
      color: 'text-blue-400',
      bgColor: 'bg-blue-400/10' 
    },
    { 
      key: 'dimensionalAccuracy', 
      label: 'Dimensional Accuracy', 
      unit: '±μm', 
      icon: Target, 
      color: 'text-purple-400',
      bgColor: 'bg-purple-400/10' 
    },
    { 
      key: 'processingTime', 
      label: 'Processing Time', 
      unit: 'min', 
      icon: Clock, 
      color: 'text-orange-400',
      bgColor: 'bg-orange-400/10' 
    },
  ] as const;

  const getBestModel = (metricKey: keyof Prediction) => {
    if (Object.keys(predictions).length === 0) return null;
    
    let bestModel = '';
    let bestValue = metricKey === 'surfaceRoughness' ? Infinity : -Infinity;
    
    Object.entries(predictions).forEach(([model, pred]) => {
      const value = pred[metricKey];
      if (metricKey === 'surfaceRoughness') {
        if (value < bestValue) {
          bestValue = value;
          bestModel = model;
        }
      } else {
        if (value > bestValue) {
          bestValue = value;
          bestModel = model;
        }
      }
    });
    
    return bestModel;
  };

  // Simple line chart component
  const LineChart: React.FC<{
    data: number[];
    label: string;
    color: string;
    unit: string;
    height?: number;
  }> = ({ data, label, color, unit, height = 60 }) => {
    if (data.length === 0) return null;
    
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min || 1;
    
    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * 100;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');
    
    return (
      <div className="bg-gray-700 p-3 rounded-lg">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-white">{label}</span>
          <span className="text-xs text-gray-400">
            {data[data.length - 1]?.toFixed(2)} {unit}
          </span>
        </div>
        <svg width="100%" height={height} className="overflow-visible">
          <polyline
            points={points}
            fill="none"
            stroke={color}
            strokeWidth="2"
            className="drop-shadow-sm"
          />
          <defs>
            <linearGradient id={`gradient-${label}`} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={color} stopOpacity="0.3" />
              <stop offset="100%" stopColor={color} stopOpacity="0.1" />
            </linearGradient>
          </defs>
          <polygon
            points={`0,${height} ${points} 100,${height}`}
            fill={`url(#gradient-${label})`}
          />
        </svg>
      </div>
    );
  };

  // Real-time metrics cards
  const MetricCard: React.FC<{
    title: string;
    value: string;
    change: string;
    icon: React.ComponentType<any>;
    color: string;
  }> = ({ title, value, change, icon: Icon, color }) => (
    <div className="bg-gray-700 p-4 rounded-lg border border-gray-600">
      <div className="flex items-center justify-between mb-2">
        <Icon className={`w-5 h-5 ${color}`} />
        <span className="text-xs text-gray-400">{change}</span>
      </div>
      <div className="text-lg font-bold text-white">{value}</div>
      <div className="text-sm text-gray-300">{title}</div>
    </div>
  );

  return (
    <div className="space-y-6">
      {/* Analytics Dashboard */}
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-xl">
        <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 sm:w-6 sm:h-6 text-blue-400" />
          {methodLabels.processName} Analytics Dashboard
        </h3>

        {/* Real-time Metrics Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <MetricCard
            title="Material Removal Rate"
            value={`${processMetrics.materialRemovalRate.toFixed(2)} mm³/min`}
            change="+2.3%"
            icon={Activity}
            color="text-green-400"
          />
          <MetricCard
            title={methodLabels.powerLabel.split(' ')[0]}
            value={`${processMetrics.powerConsumption.toFixed(1)} ${methodLabels.powerLabel.match(/\(([^)]+)\)/)?.[1] || 'kW'}`}
            change="-1.2%"
            icon={Zap}
            color="text-yellow-400"
          />
          <MetricCard
            title="Surface Quality"
            value={`${processMetrics.surfaceQuality.toFixed(1)} Ra`}
            change="+0.8%"
            icon={Layers}
            color="text-blue-400"
          />
          <MetricCard
            title="Process Efficiency"
            value={`${processMetrics.efficiency.toFixed(1)}%`}
            change="+5.1%"
            icon={TrendingUp}
            color="text-purple-400"
          />
        </div>

        {/* Real-time Charts */}
        {analyticsData.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <LineChart
              data={analyticsData.map(d => d.materialRemovalRate)}
              label="Material Removal Rate"
              color="#10b981"
              unit="mm³/min"
            />
            <LineChart
              data={analyticsData.map(d => d.powerConsumption)}
              label="Power Consumption"
              color="#f59e0b"
              unit="kW"
            />
            <LineChart
              data={analyticsData.map(d => d.surfaceQuality)}
              label="Surface Quality"
              color="#3b82f6"
              unit="Ra"
            />
            <LineChart
              data={analyticsData.map(d => d.efficiency)}
              label="Process Efficiency"
              color="#8b5cf6"
              unit="%"
            />
          </div>
        )}

        {/* Process Statistics */}
        <div className="bg-gray-700 p-4 rounded-lg">
          <h4 className="font-medium text-white mb-3 text-sm sm:text-base">{methodLabels.processName} Statistics</h4>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-xs sm:text-sm">
            <div>
              <span className="text-gray-400">{cuttingMethod === 'water' ? 'Avg. Pressure:' : 'Avg. Temperature:'}</span>
              <div className="text-orange-400 font-mono">
                {analyticsData.length > 0 
                  ? (analyticsData.reduce((sum, d) => sum + d.temperature, 0) / analyticsData.length).toFixed(1)
                  : '0.0'
                }{cuttingMethod === 'water' ? ' PSI' : '°C'}
              </div>
            </div>
            <div>
              <span className="text-gray-400">Peak {methodLabels.powerLabel.split(' ')[0]}:</span>
              <div className="text-yellow-400 font-mono">
                {analyticsData.length > 0 
                  ? Math.max(...analyticsData.map(d => d.powerConsumption)).toFixed(1)
                  : processMetrics.powerConsumption.toFixed(1)
                } {methodLabels.powerLabel.match(/\(([^)]+)\)/)?.[1] || 'kW'}
              </div>
            </div>
            <div>
              <span className="text-gray-400">Total Volume:</span>
              <div className="text-green-400 font-mono">
                {(processMetrics.materialRemovalRate * 10).toFixed(2)} mm³
              </div>
            </div>
            <div>
              <span className="text-gray-400">Uptime:</span>
              <div className="text-blue-400 font-mono">
                {analyticsData.length > 0 ? `${analyticsData.length}s` : '0s'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* AI Predictions Panel */}
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-xl">
      <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6 flex items-center gap-2">
        <TrendingUp className="w-5 h-5 sm:w-6 sm:h-6 text-green-400" />
        AI Model Predictions - {methodLabels.processName}
      </h3>

      {Object.keys(predictions).length === 0 ? (
        <div className="text-gray-400 text-center py-8">
          Train an AI model to see predictions
        </div>
      ) : (
        <div className="space-y-4 sm:space-y-6">
          {metrics.map(({ key, label, unit, icon: Icon, color, bgColor }) => (
            <div key={key} className={`p-3 sm:p-4 rounded-lg ${bgColor} border border-gray-600`}>
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-3 gap-2">
                <div className="flex items-center gap-2">
                  <Icon className={`w-4 h-4 sm:w-5 sm:h-5 ${color}`} />
                  <span className="font-medium text-white text-sm sm:text-base">{label}</span>
                </div>
                <span className="text-xs text-gray-400">Best: {getBestModel(key)}</span>
              </div>
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 sm:gap-3">
                {Object.entries(predictions).map(([model, prediction]) => {
                  // Simulate model health status
                  const healthStatus = Math.random() > 0.5 ? 'Optimal' : 'Needs Review';
                  const healthColor = healthStatus === 'Optimal' ? 'bg-green-400' : 'bg-yellow-400';
                  return (
                    <div key={model} className="bg-gray-700/50 p-2 rounded">
                      <div className="flex items-center justify-between mb-1">
                        <div className="text-xs text-gray-400">{model}</div>
                        <div className="flex items-center gap-1">
                          <span className={`inline-block w-2 h-2 rounded-full ${healthColor}`}></span>
                          <span className="text-xs text-gray-300">{healthStatus}</span>
                        </div>
                      </div>
                      <div className="font-mono text-xs sm:text-sm text-white">
                        {prediction[key].toFixed(2)} {unit}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}

          {/* Feature Importance Chart for ANN */}
          {predictions.ANN && (predictions.ANN as any).featureImportance && (
            <FeatureImportanceChart featureImportance={(predictions.ANN as any).featureImportance} />
          )}

          <div className="bg-gray-700 p-3 sm:p-4 rounded-lg">
            <h4 className="font-medium text-white mb-3 text-sm sm:text-base">{methodLabels.processName} Parameter Impact</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs sm:text-sm">
              <div>
                <span className="text-gray-400">{methodLabels.powerLabel} Impact:</span>
                <div className="w-full bg-gray-600 rounded-full h-2 mt-1">
                  <div 
                    className="bg-yellow-400 h-2 rounded-full" 
                    style={{ width: `${Math.min(100, (currentParameters.laserPower || 3) / (cuttingMethod === 'water' ? 900 : cuttingMethod === 'cnc' ? 50 : 30) * 100)}%` }}
                  />
                </div>
              </div>
              <div>
                <span className="text-gray-400">{methodLabels.speedLabel} Impact:</span>
                <div className="w-full bg-gray-600 rounded-full h-2 mt-1">
                  <div 
                    className="bg-blue-400 h-2 rounded-full" 
                    style={{ width: `${Math.min(100, (currentParameters.speed || 2900) / (cuttingMethod === 'laser' ? 25000 : cuttingMethod === 'cnc' ? 10000 : 5000) * 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
};

export default ResultsPanel;