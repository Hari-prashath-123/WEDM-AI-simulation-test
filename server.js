// GET endpoint to load the latest saved model
app.get('/api/load-model', async (req, res) => {
  try {
    // Find the latest model directory in /models
    const modelDirs = fs.readdirSync(modelsDir, { withFileTypes: true })
      .filter(dirent => dirent.isDirectory())
      .map(dirent => dirent.name)
      .sort((a, b) => fs.statSync(path.join(modelsDir, b)).mtimeMs - fs.statSync(path.join(modelsDir, a)).mtimeMs);
    if (modelDirs.length === 0) {
      return res.status(404).json({ error: 'No models found' });
    }
    const latestModelDir = modelDirs[0];
    const modelPath = `file://${path.join(modelsDir, latestModelDir)}`;
    // Load the model using tfjs-node
    const model = await tf.loadLayersModel(modelPath + '/model.json');
    // Get model JSON and weights
    const modelJson = model.toJSON();
    // Read weights file(s)
    const weightsManifestPath = path.join(modelsDir, latestModelDir, 'weights.bin');
    const weightsBuffer = fs.readFileSync(weightsManifestPath);
    // Send model JSON and weights as base64
    res.json({
      modelName: latestModelDir,
      modelJson,
      weights: weightsBuffer.toString('base64')
    });
  } catch (err) {
    console.error('Error loading model:', err);
    res.status(500).json({ error: 'Failed to load model', details: err.message });
  }
});
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
// Ensure models directory exists
const modelsDir = path.join(__dirname, 'models');
if (!fs.existsSync(modelsDir)) {
  fs.mkdirSync(modelsDir);
}
// POST endpoint to save a model
app.post('/api/save-model', async (req, res) => {
  const { modelName, modelData } = req.body;
  if (!modelName || !modelData) {
    return res.status(400).json({ error: 'modelName and modelData are required' });
  }
  try {
    // Recreate the model from the provided data (assume modelData is a JSON string or object)
    // If modelData is a JSON string, parse it
    const modelJson = typeof modelData === 'string' ? JSON.parse(modelData) : modelData;
    // Load model from JSON (assume modelData is in tfjs Layers format)
    const model = await tf.models.modelFromJSON(modelJson);
    const savePath = `file://${path.join(modelsDir, modelName)}`;
    await model.save(savePath);
    res.json({ success: true, message: `Model saved as ${modelName}` });
  } catch (err) {
    console.error('Error saving model:', err);
    res.status(500).json({ error: 'Failed to save model', details: err.message });
  }
});
const express = require('express');
const app = express();
const PORT = 3001;

app.use(express.json());

app.get('/', (req, res) => {
  res.json({ status: 'Server is running on port 3001' });
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
