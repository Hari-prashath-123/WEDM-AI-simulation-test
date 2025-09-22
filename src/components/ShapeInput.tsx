import React, { useState, useRef, useCallback } from 'react';
import { Upload, Pen, Grid3X3, Download, Trash2, Save } from 'lucide-react';

interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface ShapeInputProps {
  onShapeChange: (points: Point3D[]) => void;
  currentShape: string;
}

const ShapeInput: React.FC<ShapeInputProps> = ({ onShapeChange, currentShape }) => {
  const [inputMode, setInputMode] = useState<'preset' | 'upload' | 'draw' | 'coordinates'>('preset');
  const [coordinatePoints, setCoordinatePoints] = useState<Point3D[]>([]);
  const [newPoint, setNewPoint] = useState({ x: 0, y: 0, z: 0 });
  const [drawingPoints, setDrawingPoints] = useState<Point3D[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawingZ, setDrawingZ] = useState(10);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Preset shapes
  const presetShapes = {
    rectangle: generateRectangle3D(0, 0, 0, 80, 60, 20),
    circle: generateCircle3D(0, 0, 0, 40, 20, 32),
    star: generateStar3D(0, 0, 0, 35, 15, 20, 5),
    hexagon: generateHexagon3D(0, 0, 0, 40, 20)
  };

  function generateRectangle3D(centerX: number, centerY: number, centerZ: number, width: number, height: number, depth: number): Point3D[] {
    const points: Point3D[] = [];
    const w = width / 2, h = height / 2, d = depth / 2;
    
    points.push(
      { x: centerX - w, y: centerY - h, z: centerZ + d },
      { x: centerX + w, y: centerY - h, z: centerZ + d },
      { x: centerX + w, y: centerY + h, z: centerZ + d },
      { x: centerX - w, y: centerY + h, z: centerZ + d },
      { x: centerX - w, y: centerY - h, z: centerZ + d }
    );
    
    return points;
  }

  function generateCircle3D(centerX: number, centerY: number, centerZ: number, radius: number, depth: number, segments: number): Point3D[] {
    const points: Point3D[] = [];
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2;
      points.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        z: centerZ + depth / 2
      });
    }
    return points;
  }

  function generateStar3D(centerX: number, centerY: number, centerZ: number, outerRadius: number, innerRadius: number, depth: number, points: number): Point3D[] {
    const starPoints: Point3D[] = [];
    for (let i = 0; i < points * 2; i++) {
      const angle = (i / (points * 2)) * Math.PI * 2 - Math.PI / 2;
      const radius = i % 2 === 0 ? outerRadius : innerRadius;
      starPoints.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        z: centerZ + depth / 2
      });
    }
    starPoints.push(starPoints[0]);
    return starPoints;
  }

  function generateHexagon3D(centerX: number, centerY: number, centerZ: number, radius: number, depth: number): Point3D[] {
    const points: Point3D[] = [];
    for (let i = 0; i <= 6; i++) {
      const angle = (i / 6) * Math.PI * 2;
      points.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius,
        z: centerZ + depth / 2
      });
    }
    return points;
  }

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const content = e.target?.result as string;
        let points: Point3D[] = [];

        if (file.name.endsWith('.json')) {
          // JSON format
          const data = JSON.parse(content);
          points = data.points || data;
        } else if (file.name.endsWith('.csv')) {
          // CSV format
          const lines = content.split('\n').filter(line => line.trim());
          points = lines.slice(1).map(line => {
            const [x, y, z] = line.split(',').map(v => parseFloat(v.trim()));
            return { x: x || 0, y: y || 0, z: z || 0 };
          }).filter(p => !isNaN(p.x) && !isNaN(p.y) && !isNaN(p.z));
        } else if (file.name.endsWith('.txt')) {
          // Text format (x y z per line)
          const lines = content.split('\n').filter(line => line.trim());
          points = lines.map(line => {
            const [x, y, z] = line.split(/\s+/).map(v => parseFloat(v));
            return { x: x || 0, y: y || 0, z: z || 0 };
          }).filter(p => !isNaN(p.x) && !isNaN(p.y) && !isNaN(p.z));
        }

        if (points.length > 0) {
          onShapeChange(points);
        }
      } catch (error) {
        alert('Error parsing file. Please check the format.');
      }
    };
    reader.readAsText(file);
  };

  // Drawing canvas handlers
  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (inputMode !== 'draw') return;
    
    setIsDrawing(true);
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left - rect.width / 2) * 0.5;
    const y = (e.clientY - rect.top - rect.height / 2) * 0.5;
    const newPoints = [{ x, y, z: drawingZ }];
    setDrawingPoints(newPoints);
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing || inputMode !== 'draw') return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left - rect.width / 2) * 0.5;
    const y = (e.clientY - rect.top - rect.height / 2) * 0.5;
    
    setDrawingPoints(prev => [...prev, { x, y, z: drawingZ }]);
  };

  const handleCanvasMouseUp = () => {
    if (isDrawing && drawingPoints.length > 1) {
      // Close the shape by connecting to the first point
      const closedShape = [...drawingPoints, drawingPoints[0]];
      onShapeChange(closedShape);
    }
    setIsDrawing(false);
  };

  // Coordinate input handlers
  const addCoordinatePoint = () => {
    const updatedPoints = [...coordinatePoints, { ...newPoint }];
    setCoordinatePoints(updatedPoints);
    setNewPoint({ x: 0, y: 0, z: 0 });
  };

  const removeCoordinatePoint = (index: number) => {
    const updatedPoints = coordinatePoints.filter((_, i) => i !== index);
    setCoordinatePoints(updatedPoints);
    if (updatedPoints.length > 0) {
      onShapeChange(updatedPoints);
    }
  };

  const applyCoordinatePoints = () => {
    if (coordinatePoints.length > 0) {
      // Close the shape if it's not already closed
      const lastPoint = coordinatePoints[coordinatePoints.length - 1];
      const firstPoint = coordinatePoints[0];
      const needsClosure = lastPoint.x !== firstPoint.x || lastPoint.y !== firstPoint.y || lastPoint.z !== firstPoint.z;
      
      const finalPoints = needsClosure ? [...coordinatePoints, firstPoint] : coordinatePoints;
      onShapeChange(finalPoints);
    }
  };

  // Export current shape
  const exportShape = (format: 'json' | 'csv') => {
    const points = coordinatePoints.length > 0 ? coordinatePoints : presetShapes[currentShape as keyof typeof presetShapes] || [];
    
    let content = '';
    let filename = '';
    
    if (format === 'json') {
      content = JSON.stringify({ points }, null, 2);
      filename = `shape_${Date.now()}.json`;
    } else {
      content = 'x,y,z\n' + points.map(p => `${p.x},${p.y},${p.z}`).join('\n');
      filename = `shape_${Date.now()}.csv`;
    }
    
    const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Draw on canvas for visualization
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    for (let i = -100; i <= 100; i += 20) {
      ctx.beginPath();
      ctx.moveTo(centerX + i, 0);
      ctx.lineTo(centerX + i, canvas.height);
      ctx.moveTo(0, centerY + i);
      ctx.lineTo(canvas.width, centerY + i);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.moveTo(0, centerY);
    ctx.lineTo(canvas.width, centerY);
    ctx.stroke();

    // Draw current drawing
    if (drawingPoints.length > 1) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(centerX + drawingPoints[0].x, centerY + drawingPoints[0].y);
      
      for (let i = 1; i < drawingPoints.length; i++) {
        ctx.lineTo(centerX + drawingPoints[i].x, centerY + drawingPoints[i].y);
      }
      ctx.stroke();

      // Draw points
      ctx.fillStyle = '#ef4444';
      drawingPoints.forEach(point => {
        ctx.beginPath();
        ctx.arc(centerX + point.x, centerY + point.y, 4, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }, [drawingPoints]);

  return (
    <div className="bg-gray-700 p-4 rounded-lg">
      <h4 className="text-lg font-semibold text-white mb-4">Shape Input</h4>
      
      {/* Input Mode Selection */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 mb-4">
        <button
          onClick={() => setInputMode('preset')}
          className={`p-2 rounded text-sm flex items-center gap-2 ${
            inputMode === 'preset' ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
          }`}
        >
          <Grid3X3 className="w-4 h-4" />
          Preset
        </button>
        <button
          onClick={() => setInputMode('upload')}
          className={`p-2 rounded text-sm flex items-center gap-2 ${
            inputMode === 'upload' ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
          }`}
        >
          <Upload className="w-4 h-4" />
          Upload
        </button>
        <button
          onClick={() => setInputMode('draw')}
          className={`p-2 rounded text-sm flex items-center gap-2 ${
            inputMode === 'draw' ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
          }`}
        >
          <Pen className="w-4 h-4" />
          Draw
        </button>
        <button
          onClick={() => setInputMode('coordinates')}
          className={`p-2 rounded text-sm flex items-center gap-2 ${
            inputMode === 'coordinates' ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
          }`}
        >
          <Grid3X3 className="w-4 h-4" />
          X,Y,Z
        </button>
      </div>

      {/* Preset Shapes */}
      {inputMode === 'preset' && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {Object.keys(presetShapes).map((shape) => (
              <button
                key={shape}
                onClick={() => onShapeChange(presetShapes[shape as keyof typeof presetShapes])}
                className={`px-3 py-2 rounded text-sm transition-colors ${
                  currentShape === shape
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
                }`}
              >
                {shape.charAt(0).toUpperCase() + shape.slice(1)}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* File Upload */}
      {inputMode === 'upload' && (
        <div className="space-y-3">
          <div className="flex gap-2">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center gap-2"
            >
              <Upload className="w-4 h-4" />
              Upload Shape File
            </button>
            <button
              onClick={() => exportShape('json')}
              className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              JSON
            </button>
            <button
              onClick={() => exportShape('csv')}
              className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              CSV
            </button>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,.csv,.txt"
            onChange={handleFileUpload}
            className="hidden"
          />
          <div className="text-sm text-gray-400">
            Supported formats: JSON, CSV, TXT<br/>
            JSON: {`{"points": [{"x": 0, "y": 0, "z": 0}, ...]}`}<br/>
            CSV: x,y,z header with coordinate rows<br/>
            TXT: x y z coordinates per line
          </div>
        </div>
      )}

      {/* Drawing Canvas */}
      {inputMode === 'draw' && (
        <div className="space-y-3">
          <div className="flex items-center gap-4">
            <label className="text-sm text-gray-300">Z-Height:</label>
            <input
              type="number"
              value={drawingZ}
              onChange={(e) => setDrawingZ(parseFloat(e.target.value) || 0)}
              className="w-20 px-2 py-1 bg-gray-600 text-white rounded text-sm"
              step="0.1"
            />
            <button
              onClick={() => {
                setDrawingPoints([]);
                onShapeChange([]);
              }}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm flex items-center gap-1"
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </button>
          </div>
          <canvas
            ref={canvasRef}
            width={400}
            height={300}
            className="border border-gray-600 rounded cursor-crosshair w-full"
            onMouseDown={handleCanvasMouseDown}
            onMouseMove={handleCanvasMouseMove}
            onMouseUp={handleCanvasMouseUp}
            onMouseLeave={handleCanvasMouseUp}
          />
          <div className="text-sm text-gray-400">
            Click and drag to draw a shape. The shape will be automatically closed.
          </div>
        </div>
      )}

      {/* Coordinate Input */}
      {inputMode === 'coordinates' && (
        <div className="space-y-3">
          <div className="grid grid-cols-4 gap-2 items-end">
            <div>
              <label className="text-xs text-gray-300">X</label>
              <input
                type="number"
                value={newPoint.x}
                onChange={(e) => setNewPoint(prev => ({ ...prev, x: parseFloat(e.target.value) || 0 }))}
                className="w-full px-2 py-1 bg-gray-600 text-white rounded text-sm"
                step="0.1"
              />
            </div>
            <div>
              <label className="text-xs text-gray-300">Y</label>
              <input
                type="number"
                value={newPoint.y}
                onChange={(e) => setNewPoint(prev => ({ ...prev, y: parseFloat(e.target.value) || 0 }))}
                className="w-full px-2 py-1 bg-gray-600 text-white rounded text-sm"
                step="0.1"
              />
            </div>
            <div>
              <label className="text-xs text-gray-300">Z</label>
              <input
                type="number"
                value={newPoint.z}
                onChange={(e) => setNewPoint(prev => ({ ...prev, z: parseFloat(e.target.value) || 0 }))}
                className="w-full px-2 py-1 bg-gray-600 text-white rounded text-sm"
                step="0.1"
              />
            </div>
            <button
              onClick={addCoordinatePoint}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm"
            >
              Add
            </button>
          </div>

          {coordinatePoints.length > 0 && (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-300">Points ({coordinatePoints.length})</span>
                <div className="flex gap-2">
                  <button
                    onClick={applyCoordinatePoints}
                    className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm flex items-center gap-1"
                  >
                    <Save className="w-3 h-3" />
                    Apply
                  </button>
                  <button
                    onClick={() => setCoordinatePoints([])}
                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded text-sm flex items-center gap-1"
                  >
                    <Trash2 className="w-3 h-3" />
                    Clear
                  </button>
                </div>
              </div>
              <div className="max-h-32 overflow-y-auto bg-gray-800 rounded p-2">
                {coordinatePoints.map((point, index) => (
                  <div key={index} className="flex justify-between items-center text-sm py-1">
                    <span className="font-mono text-gray-300">
                      ({point.x.toFixed(1)}, {point.y.toFixed(1)}, {point.z.toFixed(1)})
                    </span>
                    <button
                      onClick={() => removeCoordinatePoint(index)}
                      className="text-red-400 hover:text-red-300"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ShapeInput;