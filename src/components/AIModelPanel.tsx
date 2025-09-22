import React, { useState, useRef } from 'react';
import { Brain, Target, Zap, Dna, Upload, FileText, Database, BarChart2 } from 'lucide-react';
// FeatureImportanceChart component
interface FeatureImportanceChartProps {
  importanceData: { [key: string]: number };
}

const FeatureImportanceChart: React.FC<FeatureImportanceChartProps> = ({ importanceData }) => {
  // 1. Convert to array of { name, score }
  const data = Object.entries(importanceData || {}).map(([name, score]) => ({ name, score }));
  // 2. Sort descending by score
  data.sort((a, b) => b.score - a.score);
  if (!data.length) return null;
  // 3. Find max score for scaling
  const max = Math.max(...data.map(d => d.score));
  // 4. Render chart with title and bars
  return (
    <div className="bg-gray-700 p-4 rounded-lg mb-4">
      <div className="flex items-center gap-2 mb-3">
        <BarChart2 className="w-5 h-5 text-yellow-400" />
        <span className="font-semibold text-white text-sm">Feature Importance Analysis</span>
      </div>
      <div className="space-y-2">
        {data.map(({ name, score }) => (
          <div key={name} className="flex items-center gap-2">
            <span className="text-xs text-gray-300 w-28 truncate">{name}</span>
            <div className="flex-1 bg-gray-800 rounded h-3 relative">
              <div
                className="bg-yellow-400 h-3 rounded"
                style={{ width: `${max > 0 ? (score / max) * 100 : 0}%` }}
              />
              <span className="absolute right-2 top-0 text-xs text-yellow-200 font-mono">
                {score.toFixed(3)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

interface AIModelPanelProps {
  onTrainModel: (modelType: string, data: any) => void;
  trainingResults: Record<string, any>;
}

interface DatasetInfo {
  name: string;
  size: number;
  columns: string[];
  preview: any[];
}

const AIModelPanel: React.FC<AIModelPanelProps> = ({ onTrainModel, trainingResults }) => {
  // ANN hyperparameters
  const [annLearningRate, setAnnLearningRate] = useState(0.01);
  const [annEpochs, setAnnEpochs] = useState(50);
  const [annHiddenUnits, setAnnHiddenUnits] = useState(12);
  const [selectedModel, setSelectedModel] = useState('SVM');
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState<DatasetInfo | null>(null);
  const [useUploadedData, setUseUploadedData] = useState(false);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [useRealDataset, setUseRealDataset] = useState(true);
  const [useFeatureEngineering, setUseFeatureEngineering] = useState(true);

  const models = [
    { id: 'SVM', name: 'Support Vector Machine', icon: Target, color: 'text-blue-400' },
    { id: 'ANN', name: 'Artificial Neural Network', icon: Brain, color: 'text-green-400' },
    { id: 'ELM', name: 'Extreme Learning Machine', icon: Zap, color: 'text-yellow-400' },
    { id: 'GA', name: 'Genetic Algorithm', icon: Dna, color: 'text-purple-400' },
  ];

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setDatasetError(null);

    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      setDatasetError('Please upload a CSV file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const lines = text.split('\n').filter(line => line.trim());
        
        if (lines.length < 2) {
          setDatasetError('CSV file must contain at least a header and one data row');
          return;
        }

        const headers = lines[0].split(',').map(h => h.trim());
        const requiredColumns = ['Material', 'Thickness', 'Laser Power', 'Speed', 'Ra', 'Deviation'];
        
        const missingColumns = requiredColumns.filter(col => 
          !headers.some(h => h.toLowerCase().includes(col.toLowerCase()))
        );

        if (missingColumns.length > 0) {
          setDatasetError(`Missing required columns: ${missingColumns.join(', ')}`);
          return;
        }

        const data = lines.slice(1).map(line => {
          const values = line.split(',').map(v => v.trim());
          const row: any = {};
          headers.forEach((header, index) => {
            const value = values[index];
            row[header] = isNaN(Number(value)) ? value : Number(value);
          });
          return row;
        });

        const preview = data.slice(0, 5);
        
        setUploadedDataset({
          name: file.name,
          size: data.length,
          columns: headers,
          preview: data
        });

        setUseUploadedData(true);
      } catch (error) {
        setDatasetError('Error parsing CSV file. Please check the format.');
      }
    };

    reader.readAsText(file);
  };

  const handleTrainModel = async () => {
    setIsTraining(true);
    setTrainingProgress(0);

    // Simulate training progress
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          // Use the selected data source for training
          const useRealData = useRealDataset && !useUploadedData;
          let dataObj: any = {
            useRealData,
            uploadedData: useUploadedData ? uploadedDataset?.preview : null,
            useFeatureEngineering
          };
          if (selectedModel === 'ANN') {
            dataObj = {
              ...dataObj,
              annLearningRate,
              annEpochs,
              annHiddenUnits
            };
          }
          onTrainModel(selectedModel, dataObj);
          return 100;
        }
        return prev + 2;
      });
    }, 50);
  };

  const generateTrainingData = () => {
    // Generate synthetic training data
    const data = [];
    for (let i = 0; i < 100; i++) {
      data.push({
        voltage: 20 + Math.random() * 280,
        current: 1 + Math.random() * 49,
        pulseOn: 0.5 + Math.random() * 99.5,
        pulseOff: 1 + Math.random() * 199,
        materialRemovalRate: Math.random() * 10,
        surfaceRoughness: Math.random() * 5,
        accuracy: 0.8 + Math.random() * 0.2,
      });
    }
    return data;
  };

  const downloadSampleDataset = () => {
    // Use actual laser cutting parameters as sample data
    const sampleData = [
      ['Material', 'Grade', 'Thickness (mm)', 'Laser Power (kW)', 'Speed (mm/min)', 'Gas & Pressure', 'Ra (µm)', 'Deviation (mm)', 'Kerf Taper (mm)', 'HAZ Depth (µm)', 'Linear Energy (J/mm)'],
      ['Mild Steel', 'S355JR', '4', '3.0', '2900', 'O₂ @ 0.40 bar', '1.3', '0.225', '0.03', '33', '62.07'],
      ['Mild Steel', 'S355JR', '6', '3.9', '3240', 'O₂ @ 0.55 bar', '1.1', '0.096', '0.15', '205', '72.22'],
      ['Stainless Steel', 'AISI 304', '2', '3.0', '4000', 'N₂ @ 1.0 bar', '0.9', '0.080', '0.01', '20', '45.00'],
      ['Stainless Steel', 'AISI 304', '4', '4.0', '3500', 'N₂ @ 1.2 bar', '0.8', '0.090', '0.02', '30', '68.57'],
      ['Aluminium', 'Al-6061', '2', '2.5', '4500', 'N₂ @ 0.8 bar', '0.7', '0.070', '0.01', '15', '33.33'],
      ['Aluminium', 'Al-6061', '4', '3.5', '4000', 'N₂ @ 1.0 bar', '0.8', '0.080', '0.02', '20', '52.50'],
      ['Titanium', 'Ti6Al4V', '2', '3.0', '3000', 'N₂ @ 1.5 bar', '0.8', '0.090', '0.01', '25', '60.00'],
      ['Titanium', 'Ti6Al4V', '4', '4.0', '2500', 'N₂ @ 1.7 bar', '0.9', '0.100', '0.02', '35', '96.00']
    ];
    
    const csvContent = sampleData.map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'laser_cutting_parameters_sample.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-xl">
      <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
        <Brain className="w-6 h-6 text-green-400" />
        AI Model Training
      </h3>

  {/* Dataset Upload Section */}
  <div className="mb-6 p-4 bg-gray-700 rounded-lg">
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Database className="w-5 h-5 text-blue-400" />
          Training Dataset
        </h4>
        
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <Upload className="w-4 h-4" />
              Upload CSV Dataset
            </button>
            
            <button
              onClick={downloadSampleDataset}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg flex items-center gap-2 transition-colors"
            >
              <FileText className="w-4 h-4" />
              Download Sample
            </button>
            
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>


          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="useRealDataset"
                checked={useRealDataset}
                onChange={(e) => setUseRealDataset(e.target.checked)}
                disabled={useUploadedData}
                className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
              />
              <label htmlFor="useRealDataset" className="text-sm text-gray-300">
                Use built-in laser cutting dataset (78 samples)
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="useUploadedData"
                checked={useUploadedData}
                onChange={(e) => setUseUploadedData(e.target.checked)}
                disabled={!uploadedDataset}
                className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
              />
              <label htmlFor="useUploadedData" className="text-sm text-gray-300">
                Use uploaded dataset for training (overrides built-in dataset)
              </label>
            </div>
            <div className="flex items-center gap-2 mt-1">
              <input
                type="checkbox"
                id="useFeatureEngineering"
                checked={useFeatureEngineering}
                onChange={e => setUseFeatureEngineering(e.target.checked)}
                className="w-4 h-4 text-green-600 bg-gray-700 border-gray-600 rounded focus:ring-green-500"
              />
              <label htmlFor="useFeatureEngineering" className="text-sm text-green-300">
                Enable feature engineering (add interaction, power density, duty cycle)
              </label>
            </div>
          </div>

          {datasetError && (
            <div className="p-3 bg-red-900/50 border border-red-600 rounded-lg">
              <p className="text-red-400 text-sm">{datasetError}</p>
            </div>
          )}

          {uploadedDataset && (
            <div className="p-3 bg-green-900/50 border border-green-600 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-green-400 font-medium">{uploadedDataset.name}</span>
                <span className="text-sm text-gray-400">{uploadedDataset.size} samples</span>
              </div>
              
              <div className="text-sm text-gray-300 mb-2">
                Columns: {uploadedDataset.columns.join(', ')}
              </div>
              
              {uploadedDataset.preview.length > 0 && (
                <div className="mt-3">
                  <div className="text-sm text-gray-400 mb-2">Data Preview:</div>
                  <div className="bg-gray-800 p-2 rounded text-xs font-mono overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr>
                          {uploadedDataset.columns.slice(0, 6).map(col => (
                            <th key={col} className="text-left pr-4 text-gray-400">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {uploadedDataset.preview.slice(0, 3).map((row, i) => (
                          <tr key={i}>
                            {uploadedDataset.columns.slice(0, 6).map(col => (
                              <td key={col} className="pr-4 text-white">
                                {typeof row[col] === 'number' ? row[col].toFixed(2) : row[col]}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Model Selection */}
      <div className="grid grid-cols-2 gap-3 mb-6">
        {models.map(({ id, name, icon: Icon, color }) => (
          <button
            key={id}
            onClick={() => setSelectedModel(id)}
            className={`p-3 rounded-lg border-2 transition-all ${
              selectedModel === id
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-gray-600 hover:border-gray-500'
            }`}
          >
            <div className="flex items-center gap-2">
              <Icon className={`w-5 h-5 ${color}`} />
              <div className="text-left">
                <div className="font-medium text-white text-sm">{id}</div>
                <div className="text-xs text-gray-400">{name}</div>
              </div>
            </div>
          </button>
        ))}
      </div>

      {/* ANN Hyperparameters */}
      {selectedModel === 'ANN' && (
        <div className="mb-6 p-4 bg-gray-700 rounded-lg">
          <h4 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-green-400" />
            ANN Hyperparameters
          </h4>
          <div className="space-y-4">
            <div>
              <label htmlFor="annLearningRate" className="block text-sm text-gray-300 mb-1">Learning Rate</label>
              <input
                id="annLearningRate"
                type="number"
                min="0.0001"
                max="1"
                step="0.0001"
                value={annLearningRate}
                onChange={e => setAnnLearningRate(Number(e.target.value))}
                className="w-full bg-gray-800 text-white rounded px-3 py-2 border border-gray-600 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label htmlFor="annEpochs" className="block text-sm text-gray-300 mb-1">Epochs</label>
              <input
                id="annEpochs"
                type="number"
                min="1"
                max="500"
                step="1"
                value={annEpochs}
                onChange={e => setAnnEpochs(Number(e.target.value))}
                className="w-full bg-gray-800 text-white rounded px-3 py-2 border border-gray-600 focus:outline-none focus:border-blue-500"
              />
            </div>
            <div>
              <label htmlFor="annHiddenUnits" className="block text-sm text-gray-300 mb-1">Hidden Layer Units</label>
              <input
                id="annHiddenUnits"
                type="number"
                min="1"
                max="128"
                step="1"
                value={annHiddenUnits}
                onChange={e => setAnnHiddenUnits(Number(e.target.value))}
                className="w-full bg-gray-800 text-white rounded px-3 py-2 border border-gray-600 focus:outline-none focus:border-blue-500"
              />
            </div>
          </div>
        </div>
      )}
      {/* Training Controls */}
      <button
        onClick={handleTrainModel}
        disabled={isTraining}
        className="w-full px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg font-medium transition-colors mb-4"
      >
        {isTraining ? 'Training...' : `Train ${selectedModel} Model`}
        {useUploadedData && uploadedDataset && (
          <span className="text-xs block mt-1">
            Using {uploadedDataset.name} ({uploadedDataset.size} samples)
          </span>
        )}
        {!useUploadedData && useRealDataset && (
          <span className="text-xs block mt-1">
            Using built-in laser cutting dataset (78 samples)
          </span>
        )}
        {!useUploadedData && !useRealDataset && (
          <span className="text-xs block mt-1">
            <span className="text-sm text-gray-500">
            Using synthetic dataset (100 samples)
            </span>
          </span>
        )}
      </button>

      {isTraining && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-gray-300 mb-1">
            <span>Training Progress</span>
            <span>{trainingProgress.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-100"
              style={{ width: `${trainingProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Training Results */}
      {Object.entries(trainingResults).map(([modelKey, result]) => (
        <div key={modelKey} className="bg-gray-700 p-4 rounded-lg mb-4">
          <div className="flex items-center justify-between mb-2">
            <h4 className="font-medium text-white">{modelKey} Training Results</h4>
            <button
              className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm"
              onClick={() => {
                let dataObj: any = {
                  useRealData: useRealDataset,
                  uploadedData: useUploadedData ? uploadedDataset?.preview : null,
                  useFeatureEngineering
                };
                if (modelKey === 'ANN') {
                  dataObj = {
                    ...dataObj,
                    annLearningRate,
                    annEpochs,
                    annHiddenUnits
                  };
                }
                onTrainModel(modelKey, dataObj);
              }}
              disabled={isTraining}
            >
              Retrain
            </button>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Accuracy:</span>
              <span className="text-green-400 ml-2 font-mono">
                {result.accuracy !== undefined ? (result.accuracy * 100).toFixed(1) + '%' : (result.rSquared !== undefined ? (result.rSquared * 100).toFixed(1) + '%' : '--')}
              </span>
            </div>
            <div>
              <span className="text-gray-400">Training Time:</span>
              <span className="text-blue-400 ml-2 font-mono">
                {result.trainingTime}ms
              </span>
            </div>
            <div>
              <span className="text-gray-400">Samples:</span>
              <span className="text-yellow-400 ml-2 font-mono">
                {result.samples}
              </span>
            </div>
            <div>
              <span className="text-gray-400">RMSE:</span>
              <span className="text-orange-400 ml-2 font-mono">
                {result.rmse !== undefined ? result.rmse.toFixed(3) : '--'}
              </span>
            </div>
          </div>
          {useUploadedData && uploadedDataset && (
            <div className="mt-2 text-xs text-blue-400">
              Trained with uploaded dataset: {uploadedDataset.name}
            </div>
          )}
          {!useUploadedData && useRealDataset && (
            <div className="mt-2 text-xs text-green-400">
              Trained with built-in laser cutting dataset
            </div>
          )}
          {!useUploadedData && !useRealDataset && (
            <div className="mt-2 text-xs text-yellow-400">
              Trained with synthetic dataset
            </div>
          )}
        </div>
      ))}

      {/* Dataset Requirements */}
      <div className="mt-4 p-3 bg-gray-700/50 rounded-lg">
        <h5 className="text-sm font-medium text-white mb-2">Dataset Requirements:</h5>
        <ul className="text-xs text-gray-400 space-y-1">
          <li>• <strong>Built-in Dataset:</strong> 78 laser cutting parameter samples converted to EDM equivalents</li>
          <li>• <strong>Materials:</strong> Mild Steel, Stainless Steel, Aluminum, Titanium</li>
          <li>• <strong>Parameters:</strong> Material, Grade, Thickness, Laser Power, Speed, Gas & Pressure, Surface Roughness, Deviation, etc.</li>
          <li>• <strong>Real AI Algorithms:</strong> Actual SVM, ANN, ELM, and GA implementations with proper training</li>
          <li>• <strong>Training Process:</strong> Gradient descent, backpropagation, genetic evolution, and matrix operations</li>
          <li>• Upload your own dataset to override built-in data</li>
        </ul>
      </div>
    </div>
  );
};

export default AIModelPanel;