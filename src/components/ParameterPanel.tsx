import React from 'react';
import { Zap, Clock, Gauge, Wind } from 'lucide-react';
import { cuttingMethods } from '../types/cuttingMethods';

interface EDMParameters {
  // Only keep for typing, but UI will use pulseOnTime, pulseOffTime, current
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
  onParameterChange: (key: keyof EDMParameters, value: EDMParameters[keyof EDMParameters]) => void;
}

const ParameterPanel: React.FC<ParameterPanelProps> = ({ parameters, cuttingMethod, onParameterChange }) => {
  const currentMethod = cuttingMethods[cuttingMethod];

  // Dynamic parameter configs based on cutting method
  const getParameterConfigs = () => {
    if (!currentMethod) return [];
    // Only show configs for pulseOnTime, pulseOffTime, current
    return [
      {
        key: 'pulseOnTime',
        label: `${currentMethod.parameters.power.label} (${currentMethod.parameters.power.unit})`,
        min: currentMethod.parameters.power.min,
        max: currentMethod.parameters.power.max,
        step: 1,
        icon: Clock,
        color: 'text-yellow-400'
      },
      {
        key: 'pulseOffTime',
        label: `${currentMethod.parameters.speed.label} (${currentMethod.parameters.speed.unit})`,
        min: currentMethod.parameters.speed.min,
        max: currentMethod.parameters.speed.max,
        step: 1,
        icon: Clock,
        color: 'text-blue-400'
      },
      currentMethod.parameters.specialParam1 ? {
        key: 'current',
        label: `${currentMethod.parameters.specialParam1.label} (${currentMethod.parameters.specialParam1.unit})`,
        min: currentMethod.parameters.specialParam1.min,
        max: currentMethod.parameters.specialParam1.max,
        step: 1,
        icon: Zap,
        color: 'text-red-400'
      } : null
    ].filter(Boolean);
  };


  const parameterConfigs = getParameterConfigs();

  // Material options for Wire EDM
  const materialOptions = cuttingMethod === 'wire' ? (cuttingMethods['wire'].capabilities.materials) : [];

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

      {/* Material selection dropdown for Wire EDM */}
      {cuttingMethod === 'wire' && materialOptions.length > 0 && (
        <div className="mb-4">
          <label className="block text-xs sm:text-sm font-medium text-gray-300 mb-1">Material</label>
          <select
            className="w-full bg-gray-700 text-white rounded px-2 py-2"
            value={parameters.material}
            onChange={e => onParameterChange('material', e.target.value as EDMParameters['material'])}
          >
            {materialOptions.map((mat) => (
              <option key={mat} value={mat}>{mat}</option>
            ))}
          </select>
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
        {parameterConfigs.map((config) => {
          if (!config) return null;
          const { key, label, min, max, step, icon: Icon, color } = config;
          return (
            <div key={key} className="space-y-2">
              <div className="flex items-center justify-between">
                <label className="text-xs sm:text-sm font-medium text-gray-300 flex items-center gap-2">
                  <Icon className={`w-3 h-3 sm:w-4 sm:h-4 ${color}`} />
                  <span className="truncate">{label}</span>
                </label>
                <span className="text-xs sm:text-sm font-mono text-white bg-gray-700 px-2 py-1 rounded min-w-0 flex-shrink-0">
                  {typeof parameters[key as keyof EDMParameters] === 'number' ? parameters[key as keyof EDMParameters] : ''}
                </span>
              </div>
              {typeof parameters[key as keyof EDMParameters] === 'number' && (
              <input
                type="range"
                min={min}
                max={max}
                step={step}
                value={parameters[key as keyof EDMParameters] as number}
                onChange={(e) => onParameterChange(key as keyof EDMParameters, parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              )}
              {typeof parameters[key as keyof EDMParameters] === 'number' && (
              <div className="flex justify-between text-xs text-gray-500">
                <span>{min}</span>
                <span>{max}</span>
              </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ParameterPanel;