import React, { useState } from 'react';
import { Shapes, Download, Upload, Star, Square, Circle, Hexagon } from 'lucide-react';

interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface ShapeTemplate {
  id: string;
  name: string;
  description: string;
  points: Point3D[];
  category: 'basic' | 'complex' | 'industrial' | 'custom';
  icon: React.ComponentType<any>;
}

interface ShapeLibraryProps {
  onShapeSelect: (points: Point3D[]) => void;
}

const ShapeLibrary: React.FC<ShapeLibraryProps> = ({ onShapeSelect }) => {
  const [selectedCategory, setSelectedCategory] = useState<'basic' | 'complex' | 'industrial' | 'custom'>('basic');

  // Generate complex shapes
  const generateGear = (teeth: number = 12, outerRadius: number = 40, innerRadius: number = 25): Point3D[] => {
    const points: Point3D[] = [];
    const toothAngle = (Math.PI * 2) / teeth;
    
    for (let i = 0; i <= teeth; i++) {
      const baseAngle = i * toothAngle;
      
      // Outer tooth point
      points.push({
        x: Math.cos(baseAngle) * outerRadius,
        y: Math.sin(baseAngle) * outerRadius,
        z: 10
      });
      
      // Inner tooth point
      if (i < teeth) {
        points.push({
          x: Math.cos(baseAngle + toothAngle * 0.3) * innerRadius,
          y: Math.sin(baseAngle + toothAngle * 0.3) * innerRadius,
          z: 10
        });
        
        points.push({
          x: Math.cos(baseAngle + toothAngle * 0.7) * innerRadius,
          y: Math.sin(baseAngle + toothAngle * 0.7) * innerRadius,
          z: 10
        });
      }
    }
    
    return points;
  };

  const generateSpiral = (turns: number = 3, maxRadius: number = 40): Point3D[] => {
    const points: Point3D[] = [];
    const totalPoints = turns * 20;
    
    for (let i = 0; i <= totalPoints; i++) {
      const t = i / totalPoints;
      const angle = t * turns * Math.PI * 2;
      const radius = t * maxRadius;
      
      points.push({
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        z: 10 + t * 5
      });
    }
    
    return points;
  };

  const generateKeyhole = (): Point3D[] => {
    const points: Point3D[] = [];
    
    // Circular part
    for (let i = 0; i <= 32; i++) {
      const angle = (i / 32) * Math.PI * 2;
      points.push({
        x: Math.cos(angle) * 15,
        y: Math.sin(angle) * 15 + 10,
        z: 10
      });
    }
    
    // Rectangular slot
    points.push({ x: -5, y: -10, z: 10 });
    points.push({ x: 5, y: -10, z: 10 });
    points.push({ x: 5, y: -5, z: 10 });
    points.push({ x: -5, y: -5, z: 10 });
    points.push({ x: -5, y: -10, z: 10 });
    
    return points;
  };

  const shapeTemplates: ShapeTemplate[] = [
    // Basic shapes
    {
      id: 'square',
      name: 'Square',
      description: 'Basic square shape',
      category: 'basic',
      icon: Square,
      points: [
        { x: -25, y: -25, z: 10 },
        { x: 25, y: -25, z: 10 },
        { x: 25, y: 25, z: 10 },
        { x: -25, y: 25, z: 10 },
        { x: -25, y: -25, z: 10 }
      ]
    },
    {
      id: 'circle',
      name: 'Circle',
      description: 'Perfect circle',
      category: 'basic',
      icon: Circle,
      points: Array.from({ length: 33 }, (_, i) => {
        const angle = (i / 32) * Math.PI * 2;
        return {
          x: Math.cos(angle) * 30,
          y: Math.sin(angle) * 30,
          z: 10
        };
      })
    },
    {
      id: 'triangle',
      name: 'Triangle',
      description: 'Equilateral triangle',
      category: 'basic',
      icon: Star,
      points: [
        { x: 0, y: -30, z: 10 },
        { x: 26, y: 15, z: 10 },
        { x: -26, y: 15, z: 10 },
        { x: 0, y: -30, z: 10 }
      ]
    },
    {
      id: 'hexagon',
      name: 'Hexagon',
      description: 'Regular hexagon',
      category: 'basic',
      icon: Hexagon,
      points: Array.from({ length: 7 }, (_, i) => {
        const angle = (i / 6) * Math.PI * 2;
        return {
          x: Math.cos(angle) * 30,
          y: Math.sin(angle) * 30,
          z: 10
        };
      })
    },
    
    // Complex shapes
    {
      id: 'gear',
      name: 'Gear',
      description: '12-tooth gear',
      category: 'complex',
      icon: Star,
      points: generateGear(12, 40, 25)
    },
    {
      id: 'spiral',
      name: 'Spiral',
      description: '3-turn spiral',
      category: 'complex',
      icon: Star,
      points: generateSpiral(3, 40)
    },
    {
      id: 'flower',
      name: 'Flower',
      description: '8-petal flower',
      category: 'complex',
      icon: Star,
      points: Array.from({ length: 161 }, (_, i) => {
        const t = (i / 160) * Math.PI * 2;
        const r = 20 + 15 * Math.sin(8 * t);
        return {
          x: Math.cos(t) * r,
          y: Math.sin(t) * r,
          z: 10
        };
      })
    },
    
    // Industrial shapes
    {
      id: 'keyhole',
      name: 'Keyhole',
      description: 'Standard keyhole shape',
      category: 'industrial',
      icon: Star,
      points: generateKeyhole()
    },
    {
      id: 'slot',
      name: 'Slot',
      description: 'Rectangular slot with rounded ends',
      category: 'industrial',
      icon: Square,
      points: [
        // Left semicircle
        ...Array.from({ length: 17 }, (_, i) => {
          const angle = Math.PI + (i / 16) * Math.PI;
          return {
            x: -20 + Math.cos(angle) * 10,
            y: Math.sin(angle) * 10,
            z: 10
          };
        }),
        // Right semicircle
        ...Array.from({ length: 17 }, (_, i) => {
          const angle = (i / 16) * Math.PI;
          return {
            x: 20 + Math.cos(angle) * 10,
            y: Math.sin(angle) * 10,
            z: 10
          };
        })
      ]
    }
  ];

  const categories = [
    { id: 'basic', name: 'Basic', icon: Square },
    { id: 'complex', name: 'Complex', icon: Star },
    { id: 'industrial', name: 'Industrial', icon: Shapes },
    { id: 'custom', name: 'Custom', icon: Upload }
  ] as const;

  const filteredShapes = shapeTemplates.filter(shape => shape.category === selectedCategory);

  const exportShapeLibrary = () => {
    const libraryData = {
      version: '1.0',
      shapes: shapeTemplates.map(shape => ({
        id: shape.id,
        name: shape.name,
        description: shape.description,
        category: shape.category,
        points: shape.points
      }))
    };
    
    const blob = new Blob([JSON.stringify(libraryData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'shape_library.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-700 p-4 rounded-lg">
      <div className="flex justify-between items-center mb-4">
        <h4 className="text-lg font-semibold text-white flex items-center gap-2">
          <Shapes className="w-5 h-5 text-purple-400" />
          Shape Library
        </h4>
        <button
          onClick={exportShapeLibrary}
          className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm flex items-center gap-1"
        >
          <Download className="w-3 h-3" />
          Export
        </button>
      </div>

      {/* Category Selection */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 mb-4">
        {categories.map(({ id, name, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setSelectedCategory(id)}
            className={`p-2 rounded text-sm flex items-center gap-2 ${
              selectedCategory === id ? 'bg-purple-600 text-white' : 'bg-gray-600 text-gray-300'
            }`}
          >
            <Icon className="w-4 h-4" />
            {name}
          </button>
        ))}
      </div>

      {/* Shape Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 max-h-64 overflow-y-auto">
        {filteredShapes.map((shape) => (
          <button
            key={shape.id}
            onClick={() => onShapeSelect(shape.points)}
            className="p-3 bg-gray-600 hover:bg-gray-500 rounded-lg text-left transition-colors"
          >
            <div className="flex items-center gap-2 mb-2">
              <shape.icon className="w-4 h-4 text-blue-400" />
              <span className="font-medium text-white text-sm">{shape.name}</span>
            </div>
            <p className="text-xs text-gray-300">{shape.description}</p>
            <p className="text-xs text-gray-400 mt-1">{shape.points.length} points</p>
          </button>
        ))}
      </div>

      {selectedCategory === 'custom' && (
        <div className="mt-4 p-3 bg-gray-800 rounded-lg">
          <p className="text-sm text-gray-300 mb-2">Custom shapes will appear here after creation.</p>
          <p className="text-xs text-gray-400">
            Use the Shape Input panel above to create custom shapes via upload, drawing, or coordinate entry.
          </p>
        </div>
      )}
    </div>
  );
};

export default ShapeLibrary;