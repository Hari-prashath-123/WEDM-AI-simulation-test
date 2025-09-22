import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, Square, RotateCcw, RotateCw, Move3D } from 'lucide-react';
import ShapeInput from './ShapeInput';
import ShapeLibrary from './ShapeLibrary';

interface SimulationProps {
  isRunning: boolean;
  parameters: any;
  cuttingMethod: string;
  cuttingSpeed: number;
  onCuttingSpeedChange: (speed: number) => void;
  onToggleSimulation: () => void;
  onStopSimulation: () => void;
}

interface Point3D {
  x: number;
  y: number;
  z: number;
}

interface Point2D {
  x: number;
  y: number;
}

const CuttingSimulation: React.FC<SimulationProps> = ({ 
  isRunning, 
  parameters, 
  cuttingMethod,
  cuttingSpeed,
  onCuttingSpeedChange,
  onToggleSimulation, 
  onStopSimulation 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const sparkParticles = useRef<Array<{
    x: number, y: number, z: number, 
    vx: number, vy: number, vz: number, 
    life: number, maxLife: number, size: number
  }>>([]);
  const cutProgress = useRef(0);
  const [isComplete, setIsComplete] = useState(false);
  const [showCutPiece, setShowCutPiece] = useState(false);
  const [cameraAngle, setCameraAngle] = useState({ x: 0.3, y: 0.5, z: 0 });
  const [autoRotate, setAutoRotate] = useState(true);
  
  // Mouse control states
  const [isDragging, setIsDragging] = useState(false);
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [cameraDistance, setCameraDistance] = useState(400);

  // Custom shape state
  const [customShape, setCustomShape] = useState<Point3D[] | null>(null);
  const [isUsingCustomShape, setIsUsingCustomShape] = useState(false);

  // 3D Camera and projection settings
  const camera = {
    distance: cameraDistance,
    fov: 60,
    near: 1,
    far: 1000
  };

  // Method-specific visual settings
  const getMethodVisuals = () => {
    switch (cuttingMethod) {
      case 'wire':
        return {
          toolColor: '#fbbf24', // Yellow wire
          sparkColor: '#ef4444', // Red sparks
          cutColor: '#ef4444',   // Red cut line
          toolWidth: 2,
          showSparks: true,
          showFluid: true,
          fluidColor: 'rgba(59, 130, 246, 0.3)', // Blue dielectric
          processName: 'Wire EDM'
        };
      case 'water':
        return {
          toolColor: '#06b6d4', // Cyan water jet
          sparkColor: '#ffffff', // White particles
          cutColor: '#06b6d4',   // Cyan cut line
          toolWidth: 4,
          showSparks: true,
          showFluid: true,
          fluidColor: 'rgba(6, 182, 212, 0.4)', // Cyan water
          processName: 'Water Jet'
        };
      case 'laser':
        return {
          toolColor: '#dc2626', // Red laser beam
          sparkColor: '#fbbf24', // Yellow sparks
          cutColor: '#dc2626',   // Red cut line
          toolWidth: 3,
          showSparks: true,
          showFluid: false,
          fluidColor: 'rgba(220, 38, 38, 0.2)', // Red heat zone
          processName: 'Laser Cutting'
        };
      case 'cnc':
        return {
          toolColor: '#64748b', // Gray tool
          sparkColor: '#f59e0b', // Amber chips
          cutColor: '#64748b',   // Gray cut line
          toolWidth: 6,
          showSparks: true,
          showFluid: false,
          fluidColor: 'rgba(100, 116, 139, 0.2)', // Gray coolant
          processName: 'CNC Milling'
        };
      default:
        return {
          toolColor: '#fbbf24',
          sparkColor: '#ef4444',
          cutColor: '#ef4444',
          toolWidth: 2,
          showSparks: true,
          showFluid: true,
          fluidColor: 'rgba(59, 130, 246, 0.3)',
          processName: 'Cutting'
        };
    }
  };

  const methodVisuals = getMethodVisuals();

  // Define 3D cutting shapes
  const shapes3D = {
    rectangle: generateRectangle3D(0, 0, 0, 80, 60, 20),
    circle: generateCircle3D(0, 0, 0, 40, 20, 32),
    star: generateStar3D(0, 0, 0, 35, 15, 20, 5),
    hexagon: generateHexagon3D(0, 0, 0, 40, 20)
  };

  const [currentShape, setCurrentShape] = useState<keyof typeof shapes3D | 'custom'>('rectangle');
  const cuttingPath3D = (currentShape === 'custom' && customShape && customShape.length > 0) 
    ? customShape 
    : shapes3D[currentShape === 'custom' ? 'rectangle' : currentShape as keyof typeof shapes3D];
  const totalPathLength = calculatePath3DLength(cuttingPath3D);

  function generateRectangle3D(centerX: number, centerY: number, centerZ: number, width: number, height: number, depth: number): Point3D[] {
    const points: Point3D[] = [];
    const w = width / 2, h = height / 2, d = depth / 2;
    
    // Top face cutting path
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

  function calculatePath3DLength(path: Point3D[]): number {
    if (!path || path.length < 2) {
      return 0;
    }
    
    let length = 0;
    for (let i = 1; i < path.length; i++) {
      const dx = path[i].x - path[i-1].x;
      const dy = path[i].y - path[i-1].y;
      const dz = path[i].z - path[i-1].z;
      length += Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    return length;
  }

  // 3D to 2D projection
  function project3D(point: Point3D, angle: { x: number, y: number, z: number }): Point2D {
    // Apply rotation transformations
    let { x, y, z } = point;
    
    // Rotate around X axis
    const cosX = Math.cos(angle.x);
    const sinX = Math.sin(angle.x);
    const y1 = y * cosX - z * sinX;
    const z1 = y * sinX + z * cosX;
    
    // Rotate around Y axis
    const cosY = Math.cos(angle.y);
    const sinY = Math.sin(angle.y);
    const x2 = x * cosY + z1 * sinY;
    const z2 = -x * sinY + z1 * cosY;
    
    // Rotate around Z axis
    const cosZ = Math.cos(angle.z);
    const sinZ = Math.sin(angle.z);
    const x3 = x2 * cosZ - y1 * sinZ;
    const y3 = x2 * sinZ + y1 * cosZ;
    
    // Perspective projection
    const distance = camera.distance;
    const scale = distance / (distance + z2);
    
    return {
      x: x3 * scale + 400, // Center on canvas
      y: y3 * scale + 200
    };
  }

  function getCurrentPosition3D(progress: number): Point3D {
    if (!cuttingPath3D || cuttingPath3D.length === 0) {
      return { x: 0, y: 0, z: 0 };
    }
    
    let currentLength = 0;
    const targetLength = (progress / 100) * totalPathLength;
    
    for (let i = 1; i < cuttingPath3D.length; i++) {
      const segmentStart = cuttingPath3D[i - 1];
      const segmentEnd = cuttingPath3D[i];
      const dx = segmentEnd.x - segmentStart.x;
      const dy = segmentEnd.y - segmentStart.y;
      const dz = segmentEnd.z - segmentStart.z;
      const segmentLength = Math.sqrt(dx * dx + dy * dy + dz * dz);
      
      if (currentLength + segmentLength >= targetLength) {
        const segmentProgress = (targetLength - currentLength) / segmentLength;
        return {
          x: segmentStart.x + dx * segmentProgress,
          y: segmentStart.y + dy * segmentProgress,
          z: segmentStart.z + dz * segmentProgress
        };
      }
      currentLength += segmentLength;
    }
    
    return cuttingPath3D[cuttingPath3D.length - 1];
  }

  // Mouse event handlers for improved 3D controls
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDragging(true);
    setAutoRotate(false); // Disable auto-rotate when user starts dragging
    const rect = e.currentTarget.getBoundingClientRect();
    setLastMousePos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const currentMousePos = {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };

    const deltaX = currentMousePos.x - lastMousePos.x;
    const deltaY = currentMousePos.y - lastMousePos.y;

    // Adjust sensitivity for smoother control
    const sensitivity = 0.005;

    setCameraAngle(prev => ({
      ...prev,
      y: prev.y + deltaX * sensitivity,
      x: prev.x + deltaY * sensitivity
    }));

    setLastMousePos(currentMousePos);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const zoomSensitivity = 0.1;
    const newDistance = Math.max(200, Math.min(800, cameraDistance + e.deltaY * zoomSensitivity));
    setCameraDistance(newDistance);
  };

  const resetSimulation = () => {
    cutProgress.current = 0;
    setIsComplete(false);
    setShowCutPiece(false);
    sparkParticles.current = [];
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const animate = () => {
      // Slower auto-rotate camera (reduced speed by 5x)
      if (autoRotate) {
        setCameraAngle(prev => ({
          ...prev,
          y: prev.y + 0.001 // Reduced from 0.005 to 0.001
        }));
      }

      // Clear canvas with gradient background
      const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
      gradient.addColorStop(0, '#0f172a');
      gradient.addColorStop(0.5, '#1e293b');
      gradient.addColorStop(1, '#334155');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw 3D grid
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.2)';
      ctx.lineWidth = 1;
      for (let i = -200; i <= 200; i += 40) {
        for (let j = -200; j <= 200; j += 40) {
          const p1 = project3D({ x: i, y: j, z: -50 }, cameraAngle);
          const p2 = project3D({ x: i, y: j, z: 50 }, cameraAngle);
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.stroke();
        }
      }

      // Draw workpiece as 3D block
      const workpieceVertices: Point3D[] = [
        { x: -100, y: -80, z: -30 }, { x: 100, y: -80, z: -30 },
        { x: 100, y: 80, z: -30 }, { x: -100, y: 80, z: -30 },
        { x: -100, y: -80, z: 30 }, { x: 100, y: -80, z: 30 },
        { x: 100, y: 80, z: 30 }, { x: -100, y: 80, z: 30 }
      ];

      const projectedVertices = workpieceVertices.map(v => project3D(v, cameraAngle));

      // Draw workpiece faces with proper depth sorting
      const faces = [
        { vertices: [0, 1, 2, 3], color: '#64748b', depth: -30 }, // bottom
        { vertices: [4, 5, 6, 7], color: '#94a3b8', depth: 30 },  // top
        { vertices: [0, 1, 5, 4], color: '#6b7280', depth: 0 },   // front
        { vertices: [2, 3, 7, 6], color: '#6b7280', depth: 0 },   // back
        { vertices: [1, 2, 6, 5], color: '#78716c', depth: 0 },   // right
        { vertices: [3, 0, 4, 7], color: '#78716c', depth: 0 }    // left
      ];

      // Sort faces by depth for proper rendering
      faces.sort((a, b) => a.depth - b.depth);

      faces.forEach(face => {
        ctx.fillStyle = face.color;
        ctx.beginPath();
        const firstVertex = projectedVertices[face.vertices[0]];
        ctx.moveTo(firstVertex.x, firstVertex.y);
        
        for (let i = 1; i < face.vertices.length; i++) {
          const vertex = projectedVertices[face.vertices[i]];
          ctx.lineTo(vertex.x, vertex.y);
        }
        ctx.closePath();
        ctx.fill();
        
        ctx.strokeStyle = '#475569';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Update cutting progress
      if (isRunning && cutProgress.current < 100) {
        const baseSpeed = Math.max(0.05, (parameters.speed || 2900) / 15000); // Further reduced base speed
        const speed = baseSpeed * cuttingSpeed; // Apply user speed multiplier
        cutProgress.current += speed;
        
        if (cutProgress.current >= 100) {
          cutProgress.current = 100;
          setIsComplete(true);
          setTimeout(() => setShowCutPiece(true), 1000);
        }
      }

      // Draw 3D cutting path
      if (cutProgress.current > 0) {
        const currentLength = (cutProgress.current / 100) * totalPathLength;
        let drawnLength = 0;
        
        ctx.strokeStyle = methodVisuals.cutColor;
        ctx.lineWidth = 3;
        ctx.shadowColor = methodVisuals.cutColor;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        
        let firstPoint = true;
        for (let i = 1; i < cuttingPath3D.length && drawnLength < currentLength; i++) {
          const segmentStart = cuttingPath3D[i - 1];
          const segmentEnd = cuttingPath3D[i];
          const dx = segmentEnd.x - segmentStart.x;
          const dy = segmentEnd.y - segmentStart.y;
          const dz = segmentEnd.z - segmentStart.z;
          const segmentLength = Math.sqrt(dx * dx + dy * dy + dz * dz);
          
          const projectedStart = project3D(segmentStart, cameraAngle);
          
          if (firstPoint) {
            ctx.moveTo(projectedStart.x, projectedStart.y);
            firstPoint = false;
          }
          
          if (drawnLength + segmentLength <= currentLength) {
            const projectedEnd = project3D(segmentEnd, cameraAngle);
            ctx.lineTo(projectedEnd.x, projectedEnd.y);
            drawnLength += segmentLength;
          } else {
            const remainingLength = currentLength - drawnLength;
            const segmentProgress = remainingLength / segmentLength;
            const partialEnd = {
              x: segmentStart.x + dx * segmentProgress,
              y: segmentStart.y + dy * segmentProgress,
              z: segmentStart.z + dz * segmentProgress
            };
            const projectedPartialEnd = project3D(partialEnd, cameraAngle);
            ctx.lineTo(projectedPartialEnd.x, projectedPartialEnd.y);
            break;
          }
        }
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Draw 3D cutting tool system
      const currentPos3D = getCurrentPosition3D(cutProgress.current);
      const toolTop = project3D({ x: currentPos3D.x, y: currentPos3D.y, z: 150 }, cameraAngle);
      const toolBottom = project3D({ x: currentPos3D.x, y: currentPos3D.y, z: -150 }, cameraAngle);
      const currentPos2D = project3D(currentPos3D, cameraAngle);

      // Tool guides (for wire and water jet)
      if (cuttingMethod === 'wire' || cuttingMethod === 'water') {
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(toolTop.x - 5, toolTop.y);
        ctx.lineTo(toolTop.x + 5, toolTop.y);
        ctx.moveTo(toolBottom.x - 5, toolBottom.y);
        ctx.lineTo(toolBottom.x + 5, toolBottom.y);
        ctx.stroke();
      }

      // Cutting tool
      ctx.strokeStyle = methodVisuals.toolColor;
      ctx.lineWidth = methodVisuals.toolWidth;
      ctx.shadowColor = methodVisuals.toolColor;
      ctx.shadowBlur = 6;
      ctx.beginPath();
      
      if (cuttingMethod === 'cnc') {
        // Draw spinning tool for CNC
        ctx.arc(currentPos2D.x, currentPos2D.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = methodVisuals.toolColor;
        ctx.fill();
      } else if (cuttingMethod === 'laser') {
        // Draw laser beam
        ctx.moveTo(currentPos2D.x, currentPos2D.y - 100);
        ctx.lineTo(currentPos2D.x, currentPos2D.y + 100);
      } else {
        // Draw wire or water jet
        ctx.moveTo(toolTop.x, toolTop.y);
        ctx.lineTo(toolBottom.x, toolBottom.y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Method-specific fluid/coolant visualization
      if (methodVisuals.showFluid && (parameters.laserPower || 3) > 3) {
        const fluidAlpha = (parameters.laserPower || 3) / 20;
        ctx.fillStyle = methodVisuals.fluidColor;
        ctx.beginPath();
        const radius = cuttingMethod === 'water' ? 35 : cuttingMethod === 'cnc' ? 20 : 25;
        ctx.arc(currentPos2D.x, currentPos2D.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      // Generate method-specific particles/sparks
      if (methodVisuals.showSparks && isRunning && cutProgress.current < 100 && Math.random() < (parameters.laserPower || 3) / 20) {
        const sparkCount = Math.floor((parameters.laserPower || 3) * 2);
        for (let i = 0; i < sparkCount; i++) {
          const spreadFactor = cuttingMethod === 'water' ? 40 : cuttingMethod === 'cnc' ? 25 : 30;
          sparkParticles.current.push({
            x: currentPos3D.x + (Math.random() - 0.5) * spreadFactor,
            y: currentPos3D.y + (Math.random() - 0.5) * spreadFactor,
            z: currentPos3D.z + (Math.random() - 0.5) * 20,
            vx: (Math.random() - 0.5) * (cuttingMethod === 'water' ? 15 : 12),
            vy: (Math.random() - 0.5) * (cuttingMethod === 'water' ? 15 : 12),
            vz: (Math.random() - 0.5) * (cuttingMethod === 'cnc' ? 12 : 8),
            life: 20 + Math.random() * 30,
            maxLife: 50,
            size: (cuttingMethod === 'water' ? 1 : 2) + Math.random() * 4
          });
        }
      }

      // Update and draw method-specific particles
      sparkParticles.current = sparkParticles.current.filter(particle => {
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.z += particle.vz;
        particle.vx *= 0.92;
        particle.vy *= 0.92;
        particle.vz *= 0.92;
        particle.life--;

        const alpha = Math.max(0, particle.life) / particle.maxLife;
        const size = Math.max(0, (particle.life / particle.maxLife) * particle.size);
        
        if (size > 0) {
          const projectedSpark = project3D({ x: particle.x, y: particle.y, z: particle.z }, cameraAngle);
          
          // Method-specific particle colors
          let sparkGradient;
          if (cuttingMethod === 'water') {
            sparkGradient = ctx.createRadialGradient(
              projectedSpark.x, projectedSpark.y, 0,
              projectedSpark.x, projectedSpark.y, size
            );
            sparkGradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
            sparkGradient.addColorStop(0.3, `rgba(6, 182, 212, ${alpha * 0.9})`);
            sparkGradient.addColorStop(0.7, `rgba(14, 165, 233, ${alpha * 0.6})`);
            sparkGradient.addColorStop(1, `rgba(59, 130, 246, ${alpha * 0.2})`);
          } else if (cuttingMethod === 'laser') {
            sparkGradient = ctx.createRadialGradient(
              projectedSpark.x, projectedSpark.y, 0,
              projectedSpark.x, projectedSpark.y, size
            );
            sparkGradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
            sparkGradient.addColorStop(0.3, `rgba(251, 191, 36, ${alpha * 0.9})`);
            sparkGradient.addColorStop(0.7, `rgba(239, 68, 68, ${alpha * 0.6})`);
            sparkGradient.addColorStop(1, `rgba(220, 38, 38, ${alpha * 0.2})`);
          } else if (cuttingMethod === 'cnc') {
            sparkGradient = ctx.createRadialGradient(
              projectedSpark.x, projectedSpark.y, 0,
              projectedSpark.x, projectedSpark.y, size
            );
            sparkGradient.addColorStop(0, `rgba(245, 158, 11, ${alpha})`);
            sparkGradient.addColorStop(0.3, `rgba(217, 119, 6, ${alpha * 0.9})`);
            sparkGradient.addColorStop(0.7, `rgba(180, 83, 9, ${alpha * 0.6})`);
            sparkGradient.addColorStop(1, `rgba(146, 64, 14, ${alpha * 0.2})`);
          } else {
            // Default wire EDM sparks
            sparkGradient = ctx.createRadialGradient(
              projectedSpark.x, projectedSpark.y, 0,
              projectedSpark.x, projectedSpark.y, size
            );
            sparkGradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
            sparkGradient.addColorStop(0.3, `rgba(255, 165, 0, ${alpha * 0.9})`);
            sparkGradient.addColorStop(0.7, `rgba(255, 69, 0, ${alpha * 0.6})`);
            sparkGradient.addColorStop(1, `rgba(139, 69, 19, ${alpha * 0.2})`);
          }
          
          ctx.fillStyle = sparkGradient;
          ctx.beginPath();
          ctx.arc(projectedSpark.x, projectedSpark.y, size, 0, Math.PI * 2);
          ctx.fill();
        }

        return particle.life > 0;
      });

      // Draw cut piece if complete
      if (isComplete && showCutPiece) {
        const time = Date.now() * 0.001;
        const floatOffset = {
          x: Math.sin(time) * 15 + Math.cos(time * 0.7) * 8,
          y: Math.cos(time) * 10 + Math.sin(time * 0.8) * 5,
          z: Math.sin(time * 0.5) * 10
        };
        
        const cutPieceAngle = {
          x: cameraAngle.x + Math.sin(time * 0.3) * 0.2,
          y: cameraAngle.y + Math.cos(time * 0.4) * 0.3,
          z: cameraAngle.z + Math.sin(time * 0.2) * 0.1
        };

        // Draw floating cut piece
        ctx.strokeStyle = '#a78bfa';
        ctx.lineWidth = 3;
        ctx.shadowColor = '#8b5cf6';
        ctx.shadowBlur = 15;
        ctx.beginPath();
        
        let firstCutPoint = true;
        cuttingPath3D.forEach(point => {
          const offsetPoint = {
            x: point.x + floatOffset.x + 200,
            y: point.y + floatOffset.y,
            z: point.z + floatOffset.z
          };
          const projected = project3D(offsetPoint, cutPieceAngle);
          
          if (firstCutPoint) {
            ctx.moveTo(projected.x, projected.y);
            firstCutPoint = false;
          } else {
            ctx.lineTo(projected.x, projected.y);
          }
        });
        
        ctx.closePath();
        ctx.fillStyle = 'rgba(167, 139, 250, 0.3)';
        ctx.fill();
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Enhanced 3D statistics panel - responsive positioning
      const panelWidth = Math.min(260, canvas.width * 0.32);
      const panelHeight = 180;
      const panelX = canvas.width - panelWidth - 20;
      const panelY = 40;

      const panelGradient = ctx.createLinearGradient(panelX, panelY, panelX, panelY + panelHeight);
      panelGradient.addColorStop(0, 'rgba(15, 23, 42, 0.95)');
      panelGradient.addColorStop(1, 'rgba(30, 41, 59, 0.95)');
      ctx.fillStyle = panelGradient;
      ctx.fillRect(panelX, panelY, panelWidth, panelHeight);
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 2;
      ctx.strokeRect(panelX, panelY, panelWidth, panelHeight);

      // Method-specific statistics
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 12px monospace';
      ctx.fillText(`${methodVisuals.processName.toUpperCase()} ANALYSIS`, panelX + 10, panelY + 20);
      
      ctx.font = '10px monospace';
      const stats = [
        `Material: ${parameters.material || 'Mild Steel'}`,
        `Thickness: ${parameters.thickness || 4} mm`,
        `${cuttingMethod === 'wire' ? 'Voltage' : cuttingMethod === 'water' ? 'Pressure' : cuttingMethod === 'cnc' ? 'Spindle Power' : 'Laser Power'}: ${parameters.laserPower || 3.0} ${cuttingMethod === 'water' ? 'kPSI' : cuttingMethod === 'wire' ? 'V' : 'kW'}`,
        `Cut Depth: ${(cutProgress.current * 0.6).toFixed(1)} mm`,
        `Material Volume: ${((parameters.speed || 2900) * (parameters.thickness || 4) * cutProgress.current / 100000).toFixed(2)} mm³`,
        `Surface Quality: ${(100 - (parameters.surfaceRoughness || 1.3) * 20).toFixed(1)}%`,
        `Progress: ${Math.min(100, cutProgress.current).toFixed(1)}%`,
        `Cut Speed: ${(parameters.speed || 2900)} mm/min`,
        `${cuttingMethod === 'wire' ? 'Dielectric Flow' : cuttingMethod === 'water' ? 'Abrasive Flow' : cuttingMethod === 'cnc' ? 'Tool Diameter' : 'Linear Energy'}: ${(parameters.linearEnergy || 62)} ${cuttingMethod === 'wire' ? 'L/min' : cuttingMethod === 'water' ? 'kg/min' : cuttingMethod === 'cnc' ? 'mm' : 'J/mm'}`,
        `Status: ${isComplete ? (showCutPiece ? 'Cut Complete' : 'Finishing...') : (isRunning ? `${methodVisuals.processName}...` : 'Ready')}`
      ];

      stats.forEach((stat, index) => {
        const colors = ['#60a5fa', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#fb7185', '#10b981', '#06b6d4'];
        ctx.fillStyle = colors[index % colors.length];
        ctx.fillText(stat, panelX + 10, panelY + 40 + index * 14);
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRunning, parameters, isComplete, showCutPiece, currentShape, cuttingPath3D, totalPathLength, cameraAngle, autoRotate, cameraDistance, cuttingMethod, methodVisuals]);

  // Update effect dependencies to include custom shape
  React.useEffect(() => {
    if (customShape && customShape.length > 1) {
      resetSimulation();
    }
  }, [customShape]);

  const handleReset = () => {
    resetSimulation();
    onStopSimulation();
  };

  const handleShapeChange = (shape: keyof typeof shapes3D) => {
    setCurrentShape(shape);
    setIsUsingCustomShape(false);
    resetSimulation();
    onStopSimulation();
  };

  const handleCustomShapeChange = (points: Point3D[]) => {
    if (points && points.length > 0) {
      setCustomShape(points);
      setCurrentShape('custom');
      setIsUsingCustomShape(true);
    } else {
      // Reset to default shape if empty points array
      setCustomShape(null);
      setCurrentShape('rectangle');
      setIsUsingCustomShape(false);
    }
    resetSimulation();
    onStopSimulation();
  };

  const handleCameraControl = (axis: 'x' | 'y' | 'z', delta: number) => {
    setCameraAngle(prev => ({
      ...prev,
      [axis]: prev[axis] + delta
    }));
    setAutoRotate(false); // Disable auto-rotate when using manual controls
  };

  const resetCamera = () => {
    setCameraAngle({ x: 0.3, y: 0.5, z: 0 });
    setCameraDistance(400);
    setAutoRotate(true);
  };

  return (
    <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-4">
        <h3 className="text-lg sm:text-xl font-bold text-white flex items-center gap-2">
          <Move3D className="w-5 h-5 sm:w-6 sm:h-6 text-blue-400" />
          3D {methodVisuals.processName} Simulation
        </h3>
        <div className="flex flex-wrap gap-2">
          <button
            onClick={onToggleSimulation}
            disabled={isComplete}
            className={`px-3 sm:px-4 py-2 rounded-lg flex items-center gap-2 transition-colors text-sm ${
              isComplete
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : isRunning 
                ? 'bg-orange-600 hover:bg-orange-700 text-white' 
                : 'bg-blue-600 hover:bg-blue-700 text-white'
            }`}
          >
            {isRunning ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? 'Pause' : 'Start'}
          </button>
          <button
            onClick={onStopSimulation}
            className="px-3 sm:px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg flex items-center gap-2 transition-colors text-sm"
          >
            <Square className="w-4 h-4" />
            Stop
          </button>
          <button
            onClick={handleReset}
            className="px-3 sm:px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg flex items-center gap-2 transition-colors text-sm"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      {/* Enhanced 3D Controls - Mobile Responsive */}
      <div className="mb-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Cutting Speed Control:</label>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-gray-400">Speed Multiplier</span>
              <span className="text-xs font-mono text-white bg-gray-700 px-2 py-1 rounded">
                {cuttingSpeed.toFixed(1)}x
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="3.0"
              step="0.1"
              value={cuttingSpeed}
              onChange={(e) => onCuttingSpeedChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>0.1x (Slow)</span>
              <span>3.0x (Fast)</span>
            </div>
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Preset Shapes:</label>
          <div className="grid grid-cols-2 sm:flex gap-2">
            {Object.keys(shapes3D).map((shape) => (
              <button
                key={shape}
                onClick={() => handleShapeChange(shape as keyof typeof shapes3D)}
                className={`px-2 sm:px-3 py-1 rounded text-xs sm:text-sm transition-colors ${
                  currentShape === shape && !isUsingCustomShape
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {shape.charAt(0).toUpperCase() + shape.slice(1)}
              </button>
            ))}
            {isUsingCustomShape && (
              <div className="px-2 sm:px-3 py-1 rounded text-xs sm:text-sm bg-purple-600 text-white">
                Custom ({customShape?.length || 0} pts)
              </div>
            )}
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Camera Controls:</label>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => handleCameraControl('x', -0.1)}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm"
              title="Rotate Up"
            >
              ↑
            </button>
            <button
              onClick={() => handleCameraControl('x', 0.1)}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm"
              title="Rotate Down"
            >
              ↓
            </button>
            <button
              onClick={() => handleCameraControl('y', -0.1)}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm"
              title="Rotate Left"
            >
              ←
            </button>
            <button
              onClick={() => handleCameraControl('y', 0.1)}
              className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded text-sm"
              title="Rotate Right"
            >
              →
            </button>
            <button
              onClick={resetCamera}
              className="px-2 sm:px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm"
              title="Reset Camera"
            >
              Reset
            </button>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">View Options:</label>
          <div className="flex flex-wrap gap-2 items-center">
            <button
              onClick={() => setAutoRotate(!autoRotate)}
              className={`px-2 sm:px-3 py-1 rounded text-sm transition-colors flex items-center gap-1 ${
                autoRotate ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-300'
              }`}
              title="Toggle Auto Rotation"
            >
              <RotateCw className="w-4 h-4" />
              <span className="hidden sm:inline">Auto</span>
            </button>
            <div className="text-xs text-gray-400 flex items-center">
              Zoom: {Math.round((800 - cameraDistance) / 6)}%
            </div>
          </div>
        </div>
      </div>

      {/* Shape Input Component */}
      <div className="mb-4">
        <ShapeInput 
          onShapeChange={handleCustomShapeChange}
          currentShape={currentShape}
        />
      </div>

      {/* Shape Library Component */}
      <div className="mb-4">
        <ShapeLibrary onShapeSelect={handleCustomShapeChange} />
      </div>

      {/* Instructions */}
      <div className="mb-4 p-3 bg-gray-700/50 rounded-lg">
        <div className="text-xs sm:text-sm text-gray-300">
          <strong>Mouse Controls:</strong> Click and drag to rotate • Scroll wheel to zoom • 
          Manual controls disable auto-rotation<br/>
          <strong>Shape Input:</strong> Use preset shapes, upload files (JSON/CSV/TXT), draw shapes, or enter X,Y,Z coordinates
        </div>
      </div>
      
      <div className="border-2 border-gray-600 rounded-lg overflow-hidden bg-gray-900">
        <canvas
          ref={canvasRef}
          width={800}
          height={400}
          className={`w-full h-auto ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
        />
      </div>
      
      <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4 text-xs sm:text-sm">
        <div className="bg-gray-700 p-2 sm:p-3 rounded">
          <div className="text-gray-300 mb-1">Cutting Status</div>
          <div className={`font-medium ${isComplete ? 'text-green-400' : isRunning ? 'text-blue-400' : 'text-yellow-400'}`}>
            {isComplete ? 'Complete' : isRunning ? `${methodVisuals.processName}` : 'Ready'}
          </div>
        </div>
        <div className="bg-gray-700 p-2 sm:p-3 rounded">
          <div className="text-gray-300 mb-1">Complexity</div>
          <div className="text-purple-400 font-mono">
            {currentShape === 'custom' ? `Custom (${customShape?.length || 0})` : 
             currentShape === 'circle' ? 'High' : 
             currentShape === 'star' ? 'Very High' : 
             currentShape === 'hexagon' ? 'Medium' : 'Low'}
          </div>
        </div>
        <div className="bg-gray-700 p-2 sm:p-3 rounded">
          <div className="text-gray-300 mb-1">Cut Volume</div>
          <div className="text-orange-400 font-mono">
            {((parameters.thickness || 4) * cutProgress.current * 0.5 * cuttingSpeed).toFixed(1)} mm³
          </div>
        </div>
        <div className="bg-gray-700 p-2 sm:p-3 rounded">
          <div className="text-gray-300 mb-1">Speed Multiplier</div>
          <div className="text-cyan-400 font-mono">
            {cuttingSpeed.toFixed(1)}x
          </div>
        </div>
        <div className="bg-gray-700 p-2 sm:p-3 rounded">
          <div className="text-gray-300 mb-1">Method</div>
          <div className="text-cyan-400 font-mono">
            {methodVisuals.processName}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CuttingSimulation;