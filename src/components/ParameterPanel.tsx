import React from 'react';
import { Zap, Clock, Gauge, Wind } from 'lucide-react';
import { cuttingMethods } from '../types/cuttingMethods';

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

interface ParameterPanelProps {
  parameters: EDMParameters;
  cuttingMethod: string;
  onParameterChange: (key: keyof EDMParameters, value: number) => void;
}

const ParameterPanel: React.FC<ParameterPanelProps> = ({ parameters, cuttingMethod, onParameterChange }) => {
  const currentMethod = cuttingMethods[cuttingMethod];
  
  const materialOptions = ['Mild Steel', 'Stainless Steel', 'Aluminium', 'Titanium'];
  const gradeOptions = {
    'Mild Steel': ['S355JR'],
    'Stainless Steel': ['AISI 304'],
    'Aluminium': ['Al-6061'],
    'Titanium': ['Ti6Al4V']
  };
  const gasOptions = ['O₂ @ 0.40 bar', 'N₂ @ 1.0 bar', 'N₂ @ 1.2 bar', 'N₂ @ 1.5 bar'];

  // Dynamic parameter configs based on cutting method
  const getParameterConfigs = () => {
    if (!currentMethod) return [];
    
    const configs = [
      { 
        key: 'thickness', 
        label: `${currentMethod.parameters.thickness.label} (${currentMethod.parameters.thickness.unit})`, 
        min: currentMethod.parameters.thickness.min, 
        max: Math.min(currentMethod.parameters.thickness.max, 50), 
        step: 0.5, 
        icon: Gauge, 
        color: 'text-blue-400' 
      },
      { 
        key: 'laserPower', 
        label: `${currentMethod.parameters.power.label} (${currentMethod.parameters.power.unit})`, 
        min: currentMethod.parameters.power.min, 
        max: currentMethod.parameters.power.max, 
        step: cuttingMethod === 'laser' ? 0.1 : cuttingMethod === 'water' ? 1000 : cuttingMethod === 'cnc' ? 1 : 1, 
        icon: Zap, 
        color: 'text-yellow-400' 
      },
      { 
        key: 'speed', 
        label: `${currentMethod.parameters.speed.label} (${currentMethod.parameters.speed.unit})`, 
        min: currentMethod.parameters.speed.min, 
        max: currentMethod.parameters.speed.max, 
        step: 25, 
        icon: Wind, 
        color: 'text-green-400' 
      },
      { 
        key: 'surfaceRoughness', 
        label: 'Ra (µm)', 
        min: 0.1, 
        max: cuttingMethod === 'wire' ? 0.8 : cuttingMethod === 'laser' ? 6.3 : cuttingMethod === 'water' ? 6.3 : 3.2, 
        step: 0.1, 
        icon: Gauge, 
        color: 'text-cyan-400' 
      },
      { 
        key: 'deviation', 
        label: `Tolerance (${currentMethod.parameters.precision.unit})`, 
        min: currentMethod.parameters.precision.min, 
        max: currentMethod.parameters.precision.max, 
        step: 0.001, 
        icon: Gauge, 
        color: 'text-purple-400' 
      }
    ];

    // Add method-specific parameters
    if (currentMethod.parameters.specialParam1) {
      configs.push({
        key: 'hazDepth',
        label: `${currentMethod.parameters.specialParam1.label} (${currentMethod.parameters.specialParam1.unit})`,
        min: currentMethod.parameters.specialParam1.min,
        max: currentMethod.parameters.specialParam1.max,
        step: cuttingMethod === 'water' ? 0.1 : cuttingMethod === 'cnc' ? 100 : 1,
        icon: Gauge,
        color: 'text-red-400'
      });
    }

    if (currentMethod.parameters.specialParam2) {
      configs.push({
        key: 'linearEnergy',
        label: `${currentMethod.parameters.specialParam2.label} (${currentMethod.parameters.specialParam2.unit})`,
        min: currentMethod.parameters.specialParam2.min,
        max: currentMethod.parameters.specialParam2.max,
        step: cuttingMethod === 'water' ? 0.1 : cuttingMethod === 'cnc' ? 0.1 : 1,
        icon: Zap,
        color: 'text-pink-400'
      });
    }

    return configs;
  };

  const parameterConfigs = getParameterConfigs();

  return (
    <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-xl">
      <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6 flex items-center gap-2">
        <Gauge className="w-5 h-5 sm:w-6 sm:h-6 text-blue-400" />
        {currentMethod?.name || 'Cutting'} Parameters
      </h3>
      
      {/* Method-specific info banner */}
      {currentMethod && (
        <div className="mb-4 p-3 bg-gray-700/50 rounded-lg border border-gray-600">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">{currentMethod.icon}</span>
            <span className="font-medium text-white">{currentMethod.name}</span>
          </div>
          <div className="text-sm text-gray-300">
            Max Thickness: {currentMethod.capabilities.maxThickness}mm • 
            Precision: ±{currentMethod.capabilities.precision}mm • 
            Speed: {currentMethod.capabilities.cuttingSpeed}
          </div>
        </div>
      )}
      
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
        {/* Material Selection */}
        <div className="space-y-2">
          <label className="text-xs sm:text-sm font-medium text-gray-300 flex items-center gap-2">
            <Gauge className="w-3 h-3 sm:w-4 sm:h-4 text-blue-400" />
            Material
          </label>
          <select
            value={parameters.material}
            onChange={(e) => {
              const newMaterial = e.target.value;
              const newGrade = gradeOptions[newMaterial as keyof typeof gradeOptions][0];
              onParameterChange('material', newMaterial as any);
              onParameterChange('grade', newGrade as any);
            }}
            className="w-full px-3 py-2 bg-gray-700 text-white rounded text-sm border border-gray-600 focus:border-blue-500"
          >
            {materialOptions.filter(material => 
              !currentMethod || currentMethod.capabilities.materials.some(m => 
                m.toLowerCase().includes(material.toLowerCase().replace(' steel', '').replace('aluminium', 'aluminum'))
              )
            ).map(material => (
              <option key={material} value={material}>
                {material}
                {currentMethod && !currentMethod.capabilities.materials.some(m => 
                  m.toLowerCase().includes(material.toLowerCase().replace(' steel', '').replace('aluminium', 'aluminum'))
                ) && ' (Limited Support)'}
              </option>
            ))}
          </select>
        </div>

        {/* Grade Selection */}
        <div className="space-y-2">
          <label className="text-xs sm:text-sm font-medium text-gray-300 flex items-center gap-2">
            <Gauge className="w-3 h-3 sm:w-4 sm:h-4 text-green-400" />
            Grade
          </label>
          <select
            value={parameters.grade}
            onChange={(e) => onParameterChange('grade', e.target.value as any)}
            className="w-full px-3 py-2 bg-gray-700 text-white rounded text-sm border border-gray-600 focus:border-blue-500"
          >
            {gradeOptions[parameters.material as keyof typeof gradeOptions]?.map(grade => (
              <option key={grade} value={grade}>{grade}</option>
            ))}
          </select>
        </div>
        {parameterConfigs.map(({ key, label, min, max, step, icon: Icon, color }) => (
          <div key={key} className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs sm:text-sm font-medium text-gray-300 flex items-center gap-2">
                <Icon className={`w-3 h-3 sm:w-4 sm:h-4 ${color}`} />
                <span className="truncate">{label}</span>
              </label>
              <span className="text-xs sm:text-sm font-mono text-white bg-gray-700 px-2 py-1 rounded min-w-0 flex-shrink-0">
                {typeof parameters[key] === 'number' ? parameters[key] : ''}
              </span>
            </div>
            {typeof parameters[key] === 'number' && (
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={parameters[key] as number}
              onChange={(e) => onParameterChange(key, parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            />
            )}
            {typeof parameters[key] === 'number' && (
            <div className="flex justify-between text-xs text-gray-500">
              <span>{min}</span>
              <span>{max}</span>
            </div>
            )}
          </div>
        ))}

        {/* Gas & Pressure Selection - only for laser and some methods */}
        {(cuttingMethod === 'laser' || cuttingMethod === 'water') && (
        <div className="space-y-2 sm:col-span-2">
          <label className="text-xs sm:text-sm font-medium text-gray-300 flex items-center gap-2">
            <Wind className="w-3 h-3 sm:w-4 sm:h-4 text-cyan-400" />
            {cuttingMethod === 'laser' ? 'Assist Gas & Pressure' : 'Abrasive & Water Flow'}
          </label>
          <select
            value={parameters.gasAndPressure}
            onChange={(e) => onParameterChange('gasAndPressure', e.target.value as any)}
            className="w-full px-3 py-2 bg-gray-700 text-white rounded text-sm border border-gray-600 focus:border-blue-500"
          >
            {(cuttingMethod === 'laser' ? gasOptions : ['Garnet + Water', 'Aluminum Oxide + Water', 'Steel Grit + Water']).map(gas => (
              <option key={gas} value={gas}>{gas}</option>
            ))}
          </select>
        </div>
        )}
      </div>
    </div>
  );
};

export default ParameterPanel;