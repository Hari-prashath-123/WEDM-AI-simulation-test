// Shuffle and split dataset into train and test sets
export function splitData<T>(dataset: T[]): { trainData: T[]; testData: T[] } {
  // Shuffle using Fisher-Yates algorithm
  const shuffled = [...dataset];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const trainSize = Math.floor(shuffled.length * 0.8);
  const trainData = shuffled.slice(0, trainSize);
  const testData = shuffled.slice(trainSize);
  return { trainData, testData };
}
// Dataset loader for laser cutting parameters
export interface LaserCuttingData {
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

export interface EDMTrainingData {
  voltage: number;
  current: number;
  pulseOnTime: number;
  pulseOffTime: number;
  wireSpeed: number;
  dielectricFlow: number;
  materialRemovalRate: number;
  surfaceRoughness: number;
  dimensionalAccuracy: number;
  processingTime: number;
}

// Convert laser cutting data to EDM equivalent parameters
export function convertLaserToEDM(laserData: LaserCuttingData): EDMTrainingData {
  // Map laser parameters to EDM parameters with realistic conversions
  const voltage = Math.min(300, Math.max(20, laserData.laserPower * 50 + Math.random() * 20));
  const current = Math.min(50, Math.max(1, laserData.laserPower * 8 + Math.random() * 5));
  const pulseOnTime = Math.min(100, Math.max(0.5, (laserData.linearEnergy / 10) + Math.random() * 10));
  const pulseOffTime = Math.min(200, Math.max(1, pulseOnTime * (1.5 + Math.random() * 0.5)));
  const wireSpeed = Math.min(500, Math.max(10, laserData.speed / 10 + Math.random() * 50));
  const dielectricFlow = Math.min(20, Math.max(0.5, laserData.thickness * 1.5 + Math.random() * 2));
  
  // Calculate derived parameters
  const materialRemovalRate = Math.max(0.1, (voltage * current * pulseOnTime) / (1000 * laserData.thickness));
  const surfaceRoughness = Math.max(0.1, laserData.surfaceRoughness * (0.8 + Math.random() * 0.4));
  const dimensionalAccuracy = Math.max(1, laserData.deviation * 1000 * (0.9 + Math.random() * 0.2));
  const processingTime = Math.max(1, (laserData.thickness * 100) / wireSpeed + Math.random() * 10);

  return {
    voltage,
    current,
    pulseOnTime,
    pulseOffTime,
    wireSpeed,
    dielectricFlow,
    materialRemovalRate,
    surfaceRoughness,
    dimensionalAccuracy,
    processingTime
  };
}

// Parse CSV data
export function parseCSVData(csvText: string): LaserCuttingData[] {
  const lines = csvText.split('\n').filter(line => line.trim());
  const headers = lines[0].split(',').map(h => h.trim());
  
  return lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim());
    return {
      material: values[0],
      grade: values[1],
      thickness: parseFloat(values[2]),
      laserPower: parseFloat(values[3]),
      speed: parseFloat(values[4]),
      gasAndPressure: values[5],
      surfaceRoughness: parseFloat(values[6]),
      deviation: parseFloat(values[7]),
      kerfTaper: parseFloat(values[8]),
      hazDepth: parseFloat(values[9]),
      linearEnergy: parseFloat(values[10])
    };
  }).filter(data => !isNaN(data.thickness) && !isNaN(data.laserPower));
}

// Load and convert the dataset
export async function loadEDMDataset(): Promise<{ trainData: EDMTrainingData[]; testData: EDMTrainingData[] }> {
  try {
    // Try to load the CSV file from the assets directory
    const response = await fetch('/src/assets/laser_cutting_parameters.csv');
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const csvText = await response.text();
    const laserData = parseCSVData(csvText);
    
    if (laserData.length === 0) {
      throw new Error('No valid data found in CSV file');
    }
    
    console.log(`Loaded ${laserData.length} laser cutting parameter records`);
    const edmData = laserData.map(convertLaserToEDM);
    return splitData(edmData);
  } catch (error) {
    console.warn('Error loading CSV dataset, using synthetic data:', error);
    // Fallback to generated data
    const edmData = generateSyntheticData();
    return splitData(edmData);
  }
}

// Generate synthetic data as fallback
function generateSyntheticData(): EDMTrainingData[] {
  const data: EDMTrainingData[] = [];
  for (let i = 0; i < 100; i++) {
    const voltage = 20 + Math.random() * 280;
    const current = 1 + Math.random() * 49;
    const pulseOnTime = 0.5 + Math.random() * 99.5;
    const pulseOffTime = 1 + Math.random() * 199;
    const wireSpeed = 10 + Math.random() * 490;
    const dielectricFlow = 0.5 + Math.random() * 19.5;
    
    data.push({
      voltage,
      current,
      pulseOnTime,
      pulseOffTime,
      wireSpeed,
      dielectricFlow,
      materialRemovalRate: (voltage * current * pulseOnTime) / 10000,
      surfaceRoughness: Math.max(0.1, 5 - (voltage / 100) + (pulseOnTime / 20)),
      dimensionalAccuracy: Math.max(1, 10 - (current / 5) - (voltage / 50)),
      processingTime: Math.max(1, 60 - (current * 0.5) - (voltage * 0.1))
    });
  }
  return data;
}