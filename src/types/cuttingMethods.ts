export interface CuttingMethod {
  id: string;
  name: string;
  description: string;
  icon: string;
  parameters: CuttingParameters;
  capabilities: CuttingCapabilities;
}

export interface CuttingParameters {
  power: { min: number; max: number; unit: string; label: string };
  speed: { min: number; max: number; unit: string; label: string };
  precision: { min: number; max: number; unit: string; label: string };
  thickness: { min: number; max: number; unit: string; label: string };
  specialParam1?: { min: number; max: number; unit: string; label: string };
  specialParam2?: { min: number; max: number; unit: string; label: string };
}

export interface CuttingCapabilities {
  materials: string[];
  maxThickness: number;
  precision: number;
  surfaceFinish: string;
  cuttingSpeed: string;
  advantages: string[];
  limitations: string[];
}

export const cuttingMethods: Record<string, CuttingMethod> = {
  wire: {
    id: 'wire',
    name: 'Wire EDM',
    description: 'Electrical Discharge Machining with wire electrode',
    icon: '⚡',
    parameters: {
      power: { min: 20, max: 300, unit: 'V', label: 'Voltage' },
      speed: { min: 10, max: 500, unit: 'mm/min', label: 'Wire Speed' },
      precision: { min: 0.001, max: 0.01, unit: 'mm', label: 'Tolerance' },
      thickness: { min: 0.1, max: 300, unit: 'mm', label: 'Max Thickness' },
      specialParam1: { min: 1, max: 50, unit: 'A', label: 'Current' },
      specialParam2: { min: 0.5, max: 20, unit: 'L/min', label: 'Dielectric Flow' }
    },
    capabilities: {
      materials: ['Steel', 'Stainless Steel', 'Titanium', 'Aluminum', 'Copper', 'Carbide', 'Nickel'], // Added Nickel
      maxThickness: 300,
      precision: 0.001,
      surfaceFinish: 'Ra 0.1-0.8 μm',
      cuttingSpeed: 'Slow (10-500 mm/min)',
      advantages: ['Extremely high precision', 'No mechanical stress', 'Complex shapes', 'Hard materials'],
      limitations: ['Slow cutting speed', 'Conductive materials only', 'High operating cost']
    }
  },
  water: {
    id: 'water',
    name: 'Water Jet',
    description: 'High-pressure water cutting with abrasive',
    icon: '💧',
    parameters: {
      power: { min: 30000, max: 90000, unit: 'PSI', label: 'Pressure' },
      speed: { min: 50, max: 2000, unit: 'mm/min', label: 'Cutting Speed' },
      precision: { min: 0.05, max: 0.5, unit: 'mm', label: 'Tolerance' },
      thickness: { min: 0.5, max: 200, unit: 'mm', label: 'Max Thickness' },
      specialParam1: { min: 0.1, max: 2.0, unit: 'kg/min', label: 'Abrasive Flow' },
      specialParam2: { min: 0.5, max: 4.0, unit: 'L/min', label: 'Water Flow' }
    },
    capabilities: {
      materials: ['Steel', 'Stainless Steel', 'Aluminum', 'Titanium', 'Glass', 'Stone', 'Composites', 'Nickel'], // Added Nickel
      maxThickness: 200,
      precision: 0.05,
      surfaceFinish: 'Ra 1.6-6.3 μm',
      cuttingSpeed: 'Medium (50-2000 mm/min)',
      advantages: ['No heat affected zone', 'Any material', 'Thick sections', 'No tool wear'],
      limitations: ['Wet process', 'Noise', 'Abrasive disposal', 'Taper on thick materials']
    }
  },
  laser: {
    id: 'laser',
    name: 'Laser Cutting',
    description: 'High-power laser beam cutting',
    icon: '🔥',
    parameters: {
      power: { min: 1, max: 30, unit: 'kW', label: 'Laser Power' },
      speed: { min: 500, max: 25000, unit: 'mm/min', label: 'Cutting Speed' },
      precision: { min: 0.05, max: 0.3, unit: 'mm', label: 'Tolerance' },
      thickness: { min: 0.1, max: 50, unit: 'mm', label: 'Max Thickness' },
      specialParam1: { min: 0.5, max: 3.0, unit: 'bar', label: 'Gas Pressure' },
      specialParam2: { min: 50, max: 300, unit: 'J/mm', label: 'Linear Energy' }
    },
    capabilities: {
      materials: ['Steel', 'Stainless Steel', 'Aluminum', 'Copper', 'Plastics', 'Wood', 'Nickel'], // Added Nickel
      maxThickness: 50,
      precision: 0.05,
      surfaceFinish: 'Ra 0.8-6.3 μm',
      cuttingSpeed: 'Fast (500-25000 mm/min)',
      advantages: ['High speed', 'Good precision', 'Automation ready', 'Clean cuts'],
      limitations: ['Heat affected zone', 'Reflective materials', 'Thickness limited', 'Fume extraction needed']
    }
  },
  cnc: {
    id: 'cnc',
    name: 'CNC Milling',
    description: 'Computer Numerical Control machining',
    icon: '🔧',
    parameters: {
      power: { min: 1, max: 50, unit: 'kW', label: 'Spindle Power' },
      speed: { min: 100, max: 10000, unit: 'mm/min', label: 'Feed Rate' },
      precision: { min: 0.01, max: 0.1, unit: 'mm', label: 'Tolerance' },
      thickness: { min: 1, max: 500, unit: 'mm', label: 'Max Depth' },
      specialParam1: { min: 1000, max: 30000, unit: 'RPM', label: 'Spindle Speed' },
      specialParam2: { min: 0.1, max: 5.0, unit: 'mm', label: 'Tool Diameter' }
    },
    capabilities: {
      materials: ['Steel', 'Stainless Steel', 'Aluminum', 'Titanium', 'Plastics', 'Composites', 'Nickel'], // Added Nickel
      maxThickness: 500,
      precision: 0.01,
      surfaceFinish: 'Ra 0.4-3.2 μm',
      cuttingSpeed: 'Variable (100-10000 mm/min)',
      advantages: ['High precision', 'Complex 3D shapes', 'Good surface finish', 'Versatile'],
      limitations: ['Tool wear', 'Vibration', 'Setup time', 'Material waste']
    }
  }
};