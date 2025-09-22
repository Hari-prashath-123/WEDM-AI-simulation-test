import React from 'react';
import { Zap, Droplets, Flame, Wrench } from 'lucide-react';
import { cuttingMethods, CuttingMethod } from '../types/cuttingMethods';

interface CuttingMethodSelectorProps {
  selectedMethod: string;
  onMethodChange: (method: string) => void;
}

const CuttingMethodSelector: React.FC<CuttingMethodSelectorProps> = ({
  selectedMethod,
  onMethodChange
}) => {
  const methodIcons = {
    wire: Zap,
    water: Droplets,
    laser: Flame,
    cnc: Wrench
  };

  const methodColors = {
    wire: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30',
    water: 'text-blue-400 bg-blue-400/10 border-blue-400/30',
    laser: 'text-red-400 bg-red-400/10 border-red-400/30',
    cnc: 'text-green-400 bg-green-400/10 border-green-400/30'
  };

  return (
    <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-xl">
      <h3 className="text-lg sm:text-xl font-bold text-white mb-4 sm:mb-6 flex items-center gap-2">
        <Wrench className="w-5 h-5 sm:w-6 sm:h-6 text-purple-400" />
        Cutting Method Selection
      </h3>

      {/* Method Selection Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 mb-6">
        {Object.entries(cuttingMethods).map(([key, method]) => {
          const Icon = methodIcons[key as keyof typeof methodIcons];
          const isSelected = selectedMethod === key;
          
          return (
            <button
              key={key}
              onClick={() => onMethodChange(key)}
              className={`p-3 sm:p-4 rounded-lg border-2 transition-all duration-200 ${
                isSelected
                  ? methodColors[key as keyof typeof methodColors]
                  : 'border-gray-600 bg-gray-700 hover:border-gray-500 hover:bg-gray-600'
              }`}
            >
              <div className="flex flex-col items-center gap-2">
                <Icon className={`w-6 h-6 sm:w-8 sm:h-8 ${
                  isSelected 
                    ? methodColors[key as keyof typeof methodColors].split(' ')[0]
                    : 'text-gray-400'
                }`} />
                <div className="text-center">
                  <div className={`font-medium text-sm sm:text-base ${
                    isSelected ? 'text-white' : 'text-gray-300'
                  }`}>
                    {method.name}
                  </div>
                  <div className="text-xs text-gray-400 mt-1 hidden sm:block">
                    {method.description}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Method Details */}
      {selectedMethod && cuttingMethods[selectedMethod] && (
        <div className="bg-gray-700 p-4 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            {React.createElement(methodIcons[selectedMethod as keyof typeof methodIcons], {
              className: `w-6 h-6 ${methodColors[selectedMethod as keyof typeof methodColors].split(' ')[0]}`
            })}
            <div>
              <h4 className="text-lg font-semibold text-white">
                {cuttingMethods[selectedMethod].name}
              </h4>
              <p className="text-sm text-gray-300">
                {cuttingMethods[selectedMethod].description}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Capabilities */}
            <div>
              <h5 className="font-medium text-white mb-2">Capabilities</h5>
              <div className="space-y-2 text-sm">
                <div>
                  <span className="text-gray-400">Max Thickness:</span>
                  <span className="text-green-400 ml-2">
                    {cuttingMethods[selectedMethod].capabilities.maxThickness} mm
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Precision:</span>
                  <span className="text-blue-400 ml-2">
                    ±{cuttingMethods[selectedMethod].capabilities.precision} mm
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Surface Finish:</span>
                  <span className="text-purple-400 ml-2">
                    {cuttingMethods[selectedMethod].capabilities.surfaceFinish}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Speed:</span>
                  <span className="text-yellow-400 ml-2">
                    {cuttingMethods[selectedMethod].capabilities.cuttingSpeed}
                  </span>
                </div>
              </div>
            </div>

            {/* Materials */}
            <div>
              <h5 className="font-medium text-white mb-2">Compatible Materials</h5>
              <div className="flex flex-wrap gap-1">
                {cuttingMethods[selectedMethod].capabilities.materials.map((material) => (
                  <span
                    key={material}
                    className="px-2 py-1 bg-gray-600 text-gray-200 rounded text-xs"
                  >
                    {material}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Advantages and Limitations */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
            <div>
              <h5 className="font-medium text-green-400 mb-2">Advantages</h5>
              <ul className="text-sm text-gray-300 space-y-1">
                {cuttingMethods[selectedMethod].capabilities.advantages.map((advantage, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-green-400 mt-1">•</span>
                    {advantage}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-red-400 mb-2">Limitations</h5>
              <ul className="text-sm text-gray-300 space-y-1">
                {cuttingMethods[selectedMethod].capabilities.limitations.map((limitation, index) => (
                  <li key={index} className="flex items-start gap-2">
                    <span className="text-red-400 mt-1">•</span>
                    {limitation}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CuttingMethodSelector;