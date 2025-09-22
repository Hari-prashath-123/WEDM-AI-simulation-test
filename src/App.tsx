import { trainSVM, trainANN, trainELM, trainGA, ModelResult } from './utils/aiModels';
import React, { useState, useCallback, useMemo, useEffect } from 'react';
import { Settings, Zap, Brain, BarChart3 } from 'lucide-react';
import ParameterPanel from './components/ParameterPanel';
import CuttingSimulation from './components/CuttingSimulation';
import AIModelPanel from './components/AIModelPanel';
import ResultsPanel from './components/ResultsPanel';
import CuttingMethodSelector from './components/CuttingMethodSelector';

import { loadEDMDataset, EDMTrainingData } from './utils/datasetLoader';


interface EDMParameters {
  material: string;
  grade: string;
  thickness: number;
  laserPower: number;
  speed: number;
  gasAndPressure: string;
  surfaceRoughness: number;
  deviation: number;
  kerfTaper: number;
  hazDepth: number;
  linearEnergy: number;
}

function App() {
  // Dataset state
  const [dataset, setDataset] = useState<{ trainData: EDMTrainingData[]; testData: EDMTrainingData[] } | null>(null);

  // Load dataset on mount
  useEffect(() => {
    loadEDMDataset().then((data) => setDataset(data));
  }, []);
  const [parameters, setParameters] = useState<EDMParameters>({
    material: 'Mild Steel',
    grade: 'S355JR',
    thickness: 4,
    laserPower: 3.0,
    speed: 2900,
    gasAndPressure: 'O₂ @ 0.40 bar',
    surfaceRoughness: 1.3,
    deviation: 0.225,
    kerfTaper: 0.03,
    hazDepth: 33,
    linearEnergy: 62.07,
  });

  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [trainedModels, setTrainedModels] = useState<Record<string, ModelResult>>({});
  const [predictions, setPredictions] = useState<Record<string, any>>({});
  const [activeTab, setActiveTab] = useState('parameters');
  const [cuttingSpeed, setCuttingSpeed] = useState(1.0); // Default speed multiplier
  const [cuttingMethod, setCuttingMethod] = useState('wire'); // Default to Wire EDM
  const [analyticsData, setAnalyticsData] = useState<Array<{
    timestamp: number;
    progress: number;
    materialRemovalRate: number;
    powerConsumption: number;
    surfaceQuality: number;
    temperature: number;
    efficiency: number;
  }>>([]);

  // Calculate process metrics based on current parameters
  const processMetrics = useMemo(() => {
    const cuttingEnergy = parameters.linearEnergy;
    const cuttingSpeed = parameters.speed / 1000; // Convert mm/min to m/min
    const powerConsumption = parameters.laserPower;
    const estimatedCostPerHour = powerConsumption * 0.15 + 20 + (parameters.thickness * 2);
    const materialRemovalRate = (parameters.speed * parameters.thickness) / 1000; // mm³/min
    const surfaceQuality = Math.max(0.1, parameters.surfaceRoughness);
    const precisionLevel = Math.max(0.001, parameters.deviation);
    const efficiency = Math.min(100, (parameters.speed * parameters.laserPower) / (parameters.thickness * 100));
    
    return {
      cuttingEnergy,
      cuttingSpeed,
      powerConsumption,
      estimatedCostPerHour,
      materialRemovalRate,
      surfaceQuality,
      precisionLevel,
      efficiency
    };
  }, [parameters]);

  const handleParameterChange = useCallback((key: keyof EDMParameters, value: number) => {
    setParameters(prev => ({ ...prev, [key]: value }));
    
    // Update predictions for all trained models
    const newPredictions: Record<string, any> = {};
    Object.entries(trainedModels).forEach(([modelType, model]) => {
      newPredictions[modelType] = model.predict({ ...parameters, [key]: value });
    });
    setPredictions(newPredictions);
  }, [trainedModels, parameters]);

  const handleToggleSimulation = () => {
    setIsSimulationRunning(prev => !prev);
  };

  const handleStopSimulation = () => {
    setIsSimulationRunning(false);
  };

  // Accepts (modelType, dataObj) from AIModelPanel
  const handleTrainModel = async (modelType: string, dataObj?: any) => {
    if (!dataset) return;
    let model: ModelResult;
    // Extract feature engineering flag if present
    const useFeatureEngineering = dataObj?.useFeatureEngineering ?? true;
    switch (modelType) {
      case 'SVM':
        model = await trainSVM(dataset);
        break;
      case 'ANN':
        model = await trainANN(dataset, dataObj, useFeatureEngineering);
        break;
      case 'ELM':
        model = await trainELM(dataset);
        break;
      case 'GA':
        model = await trainGA(dataset);
        break;
      default:
        return;
    }
    setTrainedModels(prev => ({ ...prev, [modelType]: model }));
    // Generate prediction for current parameters
    const prediction = model.predict(parameters);
    setPredictions(prev => ({ ...prev, [modelType]: prediction }));
  };

  // Update analytics data periodically
  React.useEffect(() => {
    if (isSimulationRunning) {
      const interval = setInterval(() => {
        const timestamp = Date.now();
        const progress = Math.random() * 100; // This would come from actual simulation
        const newDataPoint = {
          timestamp,
          progress,
          materialRemovalRate: processMetrics.materialRemovalRate + (Math.random() - 0.5) * 2,
          powerConsumption: processMetrics.powerConsumption + (Math.random() - 0.5) * 0.5,
          surfaceQuality: processMetrics.surfaceQuality + (Math.random() - 0.5) * 0.2,
          temperature: 150 + Math.random() * 100,
          efficiency: processMetrics.efficiency + (Math.random() - 0.5) * 10
        };
        
        setAnalyticsData(prev => {
          const updated = [...prev, newDataPoint];
          // Keep only last 50 data points
          return updated.slice(-50);
        });
      }, 1000);
      
      return () => clearInterval(interval);
    }
  }, [isSimulationRunning, processMetrics]);

  const tabs = [
    { id: 'parameters', label: 'Parameters', icon: Settings },
    { id: 'simulation', label: 'Simulation', icon: Zap },
    { id: 'ai', label: 'AI Models', icon: Brain },
    { id: 'results', label: 'Results', icon: BarChart3 },
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 shadow-lg border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 sm:w-10 sm:h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl sm:text-2xl font-bold text-white">Wire EDM Simulator</h1>
                <p className="text-sm sm:text-base text-gray-400 hidden sm:block">Advanced Machining Process Simulation & AI Analysis</p>
                <p className="text-xs text-gray-400 sm:hidden">AI-Powered EDM Analysis</p>
              </div>
            </div>
            <div className="flex items-center gap-2 sm:gap-4">
              <div className={`w-2 h-2 sm:w-3 sm:h-3 rounded-full ${isSimulationRunning ? 'bg-green-400' : 'bg-gray-600'}`} />
              <span className="text-xs sm:text-sm text-gray-400">
                {isSimulationRunning ? 'Running' : 'Stopped'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs - Scrollable on mobile */}
      <nav className="bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-1 overflow-x-auto scrollbar-hide">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex-shrink-0 px-4 sm:px-6 py-3 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap ${
                  activeTab === id
                    ? 'bg-gray-900 text-blue-400 border-b-2 border-blue-400'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                <Icon className="w-4 h-4 inline mr-2" />
                <span className="hidden sm:inline">{label}</span>
                <span className="sm:hidden">{label.split(' ')[0]}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 sm:py-8">
        <div className="space-y-6 sm:space-y-8">
          {activeTab === 'parameters' && (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 sm:gap-8">
              <div className="space-y-6">
                <CuttingMethodSelector
                  selectedMethod={cuttingMethod}
                  onMethodChange={setCuttingMethod}
                />
                <ParameterPanel
                  parameters={parameters}
                  cuttingMethod={cuttingMethod}
                  onParameterChange={handleParameterChange}
                />
              </div>
              <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-xl">
                <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Process Overview</h3>
                <div className="space-y-3 sm:space-y-4">
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Cutting Energy</span>
                    <span className="font-mono text-blue-400">
                      {processMetrics.cuttingEnergy.toFixed(2)} J/mm
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Cutting Speed</span>
                    <span className="font-mono text-green-400">
                      {processMetrics.cuttingSpeed.toFixed(1)} m/min
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Power Consumption</span>
                    <span className="font-mono text-yellow-400">
                      {processMetrics.powerConsumption.toFixed(1)} kW
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Material Removal Rate</span>
                    <span className="font-mono text-purple-400">
                      {processMetrics.materialRemovalRate.toFixed(2)} mm³/min
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Surface Roughness</span>
                    <span className="font-mono text-cyan-400">
                      {processMetrics.surfaceQuality.toFixed(2)} Ra (µm)
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Precision Level</span>
                    <span className="font-mono text-red-400">
                      ±{processMetrics.precisionLevel.toFixed(3)} mm
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Process Efficiency</span>
                    <span className="font-mono text-emerald-400">
                      {processMetrics.efficiency.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-2 sm:p-3 bg-gray-700 rounded text-sm sm:text-base">
                    <span className="text-gray-300">Estimated Cost/Hour</span>
                    <span className="font-mono text-orange-400">
                      ${processMetrics.estimatedCostPerHour.toFixed(2)}
                    </span>
                  </div>
                </div>
                
                {/* Process Quality Indicators */}
                <div className="mt-4 sm:mt-6">
                  <h4 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">Quality Indicators</h4>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs sm:text-sm mb-1">
                        <span className="text-gray-300">Cutting Precision</span>
                        <span className="text-blue-400">{Math.min(100, 100 - processMetrics.precisionLevel * 1000).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(100, 100 - processMetrics.precisionLevel * 1000)}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs sm:text-sm mb-1">
                        <span className="text-gray-300">Surface Quality</span>
                        <span className="text-green-400">{Math.max(0, 100 - processMetrics.surfaceQuality * 20).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${Math.max(0, 100 - processMetrics.surfaceQuality * 20)}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs sm:text-sm mb-1">
                        <span className="text-gray-300">Energy Efficiency</span>
                        <span className="text-yellow-400">{Math.min(100, processMetrics.efficiency).toFixed(0)}%</span>
                      </div>
                      <div className="w-full bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-yellow-500 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${Math.min(100, processMetrics.efficiency)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'simulation' && (
            <CuttingSimulation
              isRunning={isSimulationRunning}
              parameters={parameters}
              cuttingMethod={cuttingMethod}
              cuttingSpeed={cuttingSpeed}
              onCuttingSpeedChange={setCuttingSpeed}
              onToggleSimulation={handleToggleSimulation}
              onStopSimulation={handleStopSimulation}
            />
          )}

          {activeTab === 'ai' && (
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 sm:gap-8">
              <AIModelPanel
                onTrainModel={handleTrainModel}
                trainingResults={trainedModels}
              />
              <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-xl">
                <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6">Model Comparison</h3>
                {Object.keys(trainedModels).length === 0 ? (
                  <div className="text-gray-400 text-center py-8">
                    No models trained yet
                  </div>
                ) : (
                  <div className="space-y-4">
                    {Object.entries(trainedModels).map(([modelType, model]) => (
                      <div key={modelType} className="p-3 sm:p-4 bg-gray-700 rounded-lg">
                        <div className="flex justify-between items-center mb-2">
                          <span className="font-medium text-white text-sm sm:text-base">{modelType}</span>
                          <span className="text-xs sm:text-sm text-green-400">
                            {(model.rSquared * 100).toFixed(1)}% R²
                          </span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full"
                            style={{ width: `${model.rSquared * 100}%` }}
                          />
                        </div>
                        <div className="mt-2 text-xs text-gray-400">
                          RMSE: {model.rmse.toFixed(3)} | Training Time: {model.trainingTime}ms
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'results' && (
            <ResultsPanel
              predictions={predictions}
              currentParameters={parameters}
              analyticsData={analyticsData}
              processMetrics={processMetrics}
              cuttingMethod={cuttingMethod}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;