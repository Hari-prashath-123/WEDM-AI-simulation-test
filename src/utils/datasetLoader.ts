/**
 * Splits a dataset into k folds for cross-validation.
 * Each fold returns an object with trainData and testData arrays.
 * @param dataset The full dataset to split
 * @param k Number of folds
 */
export function getKFoldSplits<T>(dataset: T[], k: number): Array<{ trainData: T[]; testData: T[] }> {
  if (k < 2) throw new Error('k must be at least 2');
  // Shuffle dataset (Fisher-Yates)
  const shuffled = [...dataset];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  // Split into k folds
  const foldSize = Math.floor(shuffled.length / k);
  const folds: T[][] = [];
  let start = 0;
  for (let i = 0; i < k; i++) {
    const end = i === k - 1 ? shuffled.length : start + foldSize;
    folds.push(shuffled.slice(start, end));
    start = end;
  }
  // Build train/test splits for each fold
  const splits = folds.map((testData, i) => {
    const trainData = folds
      .filter((_, idx) => idx !== i)
      .reduce((acc, fold) => acc.concat(fold), [] as T[]);
    return { trainData, testData };
  });
  return splits;
}
// StandardScaler normalization for numerical features
/**
 * Scales numerical features in train and test sets using StandardScaler (z-score normalization).
 * Returns scaled data and the mean/std for each feature for future use.
 * @param train Array of LaserCuttingData (training set)
 * @param test Array of LaserCuttingData (testing set)
 */
export function scaleFeatures(
  train: LaserCuttingData[],
  test: LaserCuttingData[]
): {
  scaledTrain: LaserCuttingData[];
  scaledTest: LaserCuttingData[];
  mean: { [key: string]: number };
  std: { [key: string]: number };
} {
  // List of numerical feature keys
  const numKeys: (keyof LaserCuttingData)[] = [
    'thickness',
    'laserPower',
    'speed',
    'surfaceRoughness',
    'deviation',
    'kerfTaper',
    'hazDepth',
    'linearEnergy'
  ];
  // Compute mean and std from training set
  const mean: { [key: string]: number } = {};
  const std: { [key: string]: number } = {};
  numKeys.forEach(key => {
    const vals = train.map(row => Number(row[key]));
    const m = vals.reduce((a, b) => a + b, 0) / vals.length;
    mean[key] = m;
    std[key] = Math.sqrt(vals.reduce((sum, v) => sum + (v - m) ** 2, 0) / vals.length) || 1;
  });
  // Helper to scale a row
  function scaleRow(row: LaserCuttingData): LaserCuttingData {
    const scaled = { ...row };
    numKeys.forEach(key => {
      (scaled as any)[key] = (Number(row[key]) - mean[key]) / std[key];
    });
    return scaled;
  }
  return {
    scaledTrain: train.map(scaleRow),
    scaledTest: test.map(scaleRow),
    mean,
    std
  };
}
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
  const numericalIndices = [2, 3, 4, 6, 7, 8, 9, 10];
  let discarded = 0;
  const data = lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim());
    // Check for missing values
    if (values.length < 11 || numericalIndices.some(idx => values[idx] === '' || values[idx] == null)) {
      discarded++;
      return null;
    }
    // Parse and check numeric columns
    const parsed = {
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
    // If any numeric field is NaN, discard
    // Check each numeric property directly
    if (
      isNaN(parsed.thickness) ||
      isNaN(parsed.laserPower) ||
      isNaN(parsed.speed) ||
      isNaN(parsed.surfaceRoughness) ||
      isNaN(parsed.deviation) ||
      isNaN(parsed.kerfTaper) ||
      isNaN(parsed.hazDepth) ||
      isNaN(parsed.linearEnergy)
    ) {
      discarded++;
      return null;
    }
    return parsed;
  }).filter(row => row !== null) as LaserCuttingData[];
  if (discarded > 0) {
    console.log(`[parseCSVData] Discarded ${discarded} row(s) due to missing or invalid numeric values.`);
  }
  return data;
}

// Load and convert the dataset
export async function loadEDMDataset(): Promise<{ trainData: EDMTrainingData[]; testData: EDMTrainingData[]; mean: { [key: string]: number }; std: { [key: string]: number } }> {
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
  const { trainData: laserTrain, testData: laserTest } = splitData(laserData);
  // Scale features on LaserCuttingData
  const { scaledTrain, scaledTest, mean, std } = scaleFeatures(laserTrain, laserTest);
  // Convert to EDMTrainingData after scaling
  const edmTrain = scaledTrain.map(convertLaserToEDM);
  const edmTest = scaledTest.map(convertLaserToEDM);
  return { trainData: edmTrain, testData: edmTest, mean, std };
  } catch (error) {
    console.warn('Error loading CSV dataset, using synthetic data:', error);
    // Fallback to generated data
  // For synthetic fallback, just split and return (no scaling)
  const edmData = generateSyntheticData();
  const { trainData, testData } = splitData(edmData);
  return { trainData, testData, mean: {}, std: {} };
  }
}

// Generate synthetic data as fallback
function generateSyntheticData(): EDMTrainingData[] {
  const data: EDMTrainingData[] = [];
  for (let i = 0; i < 100; i++) {
    // Base parameters
    const voltage = 20 + Math.random() * 280; // 20-300V
    const current = 1 + Math.random() * 49;   // 1-50A
    const pulseOnTime = 0.5 + Math.random() * 99.5; // 0.5-100us
    const pulseOffTime = 1 + Math.random() * 199;   // 1-200us
    const wireSpeed = 10 + Math.random() * 490;     // 10-500 mm/min
    const dielectricFlow = 0.5 + Math.random() * 19.5; // 0.5-20 L/min

    // Material removal rate: strong dependence on voltage, current, pulseOnTime
    let mrr = 0.0005 * voltage * current * (pulseOnTime / (pulseOnTime + pulseOffTime));
    mrr += mrr * (Math.random() * 0.15 - 0.075); // ±7.5% noise
    mrr = Math.max(0.05, mrr);

    // Surface roughness: increases with pulseOnTime, decreases with voltage
    let surfaceRoughness = 1.5 + 0.03 * pulseOnTime - 0.01 * voltage;
    surfaceRoughness += surfaceRoughness * (Math.random() * 0.2 - 0.1); // ±10% noise
    surfaceRoughness = Math.max(0.1, surfaceRoughness);

    // Dimensional accuracy: better (lower) with moderate current and voltage
    let dimensionalAccuracy = 10 - 0.08 * current - 0.04 * voltage + 0.02 * Math.abs(current - 25);
    dimensionalAccuracy += dimensionalAccuracy * (Math.random() * 0.15 - 0.075); // ±7.5% noise
    dimensionalAccuracy = Math.max(1, dimensionalAccuracy);

    // Processing time: inversely related to MRR, with some dependence on wireSpeed
    let processingTime = 1000 / (mrr * wireSpeed / 100);
    processingTime += processingTime * (Math.random() * 0.15 - 0.075); // ±7.5% noise
    processingTime = Math.max(1, processingTime);

    data.push({
      voltage,
      current,
      pulseOnTime,
      pulseOffTime,
      wireSpeed,
      dielectricFlow,
      materialRemovalRate: mrr,
      surfaceRoughness,
      dimensionalAccuracy,
      processingTime
    });
  }
  return data;
}

/**
 * Augments EDMTrainingData by creating multiplier times more samples with ±5% random noise on each numerical feature.
 * @param data Array of EDMTrainingData
 * @param multiplier Number of times to increase the dataset size
 * @returns Augmented EDMTrainingData array
 */
export function augmentData(data: EDMTrainingData[], multiplier: number): EDMTrainingData[] {
  if (multiplier <= 1) return [...data];
  const augmented: EDMTrainingData[] = [];
  // List of numerical fields in EDMTrainingData
  const numericFields: (keyof EDMTrainingData)[] = [
    'voltage',
    'current',
    'pulseOnTime',
    'pulseOffTime',
    'wireSpeed',
    'dielectricFlow',
    'materialRemovalRate',
    'surfaceRoughness',
    'dimensionalAccuracy',
    'processingTime'
  ];
  for (let i = 0; i < multiplier; i++) {
    for (const sample of data) {
      const noisySample: EDMTrainingData = { ...sample };
      for (const key of numericFields) {
        const value = noisySample[key];
        if (typeof value === 'number') {
          const noise = (Math.random() * 0.1 - 0.05) * value;
          noisySample[key] = value + noise;
        }
      }
      augmented.push(noisySample);
    }
  }
  return augmented;
}

/**
 * Generates new EDMTrainingData points by interpolating between two random points from the training set.
 * For each new point, selects two random points and creates a weighted average.
 * @param data Array of EDMTrainingData
 * @param numNewPoints Number of new points to generate
 * @returns Array of new EDMTrainingData points
 */
export function generateAugmentedData(data: EDMTrainingData[], numNewPoints: number): EDMTrainingData[] {
  if (data.length < 2 || numNewPoints < 1) return [];
  const numericFields: (keyof EDMTrainingData)[] = [
    'voltage',
    'current',
    'pulseOnTime',
    'pulseOffTime',
    'wireSpeed',
    'dielectricFlow',
    'materialRemovalRate',
    'surfaceRoughness',
    'dimensionalAccuracy',
    'processingTime'
  ];
  const augmented: EDMTrainingData[] = [];
  for (let i = 0; i < numNewPoints; i++) {
    // Pick two random indices
    const idx1 = Math.floor(Math.random() * data.length);
    let idx2 = Math.floor(Math.random() * data.length);
    while (idx2 === idx1) idx2 = Math.floor(Math.random() * data.length);
    const a = data[idx1];
    const b = data[idx2];
    // Random weight between 0 and 1
    const w = Math.random();
    // Interpolate each numeric field
    const newPoint: EDMTrainingData = { ...a };
    for (const key of numericFields) {
      newPoint[key] = a[key] * w + b[key] * (1 - w);
    }
    augmented.push(newPoint);
  }
  return augmented;
}