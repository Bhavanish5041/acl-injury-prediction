"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Camera, StopCircle, Activity, ActivitySquare, Terminal, WifiOff } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

type Features = {
  left_knee_flexion: number;
  right_knee_flexion: number;
  left_knee_valgus: number;
  right_knee_valgus: number;
  left_hip_flexion: number;
  right_hip_flexion: number;
  knee_asymmetry: number;
  hip_asymmetry: number;
  joint_velocity: number;
};

type Telemetry = {
  risk_score: number;
  features: Features | null;
  explanations: string[];
  shap_values: Record<string, number>;
};

export default function Dashboard() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraSource, setCameraSource] = useState('0'); // '0' for webcam, or IP string
  const [frameData, setFrameData] = useState<string | null>(null);
  const [streamStatus, setStreamStatus] = useState('Idle');
  
  const [telemetry, setTelemetry] = useState<Telemetry>({
    risk_score: 0,
    features: null,
    explanations: [],
    shap_values: {}
  });

  const [showDebug, setShowDebug] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
      fetch('http://localhost:8000/camera/stop', { method: 'POST' }).catch(() => {});
    };
  }, []);

  const startCamera = async () => {
    try {
      const res = await fetch(`http://localhost:8000/camera/start?source=${encodeURIComponent(cameraSource)}`, { method: 'POST' });
      if (res.ok) {
        setIsStreaming(true);
        setStreamStatus('Connecting');
        connectWebSocket();
      }
    } catch (e) {
      console.error("Failed to start camera", e);
      alert("Failed to start backend camera. Is FastAPI running?");
    }
  };

  const stopCamera = async () => {
    try {
      await fetch('http://localhost:8000/camera/stop', { method: 'POST' });
      setIsStreaming(false);
      setFrameData(null);
      setStreamStatus('Idle');
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const connectWebSocket = () => {
    const ws = new WebSocket('ws://localhost:8000/ws/stream');
    ws.onopen = () => setStreamStatus('Live');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.image) {
        setFrameData(`data:image/jpeg;base64,${data.image}`);
      }
      if (data.telemetry) {
        setTelemetry(data.telemetry);
      }
    };
    ws.onerror = () => setStreamStatus('Connection issue');
    ws.onclose = () => {
      if (isStreaming) setStreamStatus('Disconnected');
    };
    wsRef.current = ws;
  };

  const shapData = Object.entries(telemetry.shap_values || {})
    .map(([key, value]) => ({
      name: key.replace(/_/g, ' '),
      value: Number(value),
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 5);

  const getRiskColor = (score: number) => {
    if (score < 30) return 'text-green-500';
    if (score < 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getRiskLabel = (score: number) => {
    if (score < 30) return 'Low';
    if (score < 60) return 'Moderate';
    return 'High';
  };

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6 font-sans">
      <header className="flex justify-between items-center mb-8 border-b border-slate-700 pb-4">
        <div className="flex items-center gap-3">
          <ActivitySquare className="text-blue-500 w-8 h-8" />
          <h1 className="text-2xl font-bold tracking-tight">ACL Injury Risk Assessor <span className="text-slate-400 text-sm ml-2 font-normal">Biomechanics AI</span></h1>
        </div>
        <div className="flex gap-4 items-center">
          <button onClick={() => setShowDebug(!showDebug)} className="text-slate-400 hover:text-white mr-4 transition flex items-center gap-2 text-sm">
            <Terminal className="w-4 h-4" /> Debug Mode
          </button>
          <input 
            type="text" 
            placeholder="Camera Source (0 or IP)"
            value={cameraSource}
            onChange={(e) => setCameraSource(e.target.value)}
            className="bg-slate-800 border border-slate-600 rounded px-3 py-2 text-sm w-64 focus:outline-none focus:border-blue-500"
          />
          {!isStreaming ? (
            <button onClick={startCamera} className="bg-blue-600 hover:bg-blue-700 transition px-4 py-2 rounded font-medium flex items-center gap-2">
              <Camera className="w-4 h-4" /> Start Feed
            </button>
          ) : (
            <button onClick={stopCamera} className="bg-red-600 hover:bg-red-700 transition px-4 py-2 rounded font-medium flex items-center gap-2">
              <StopCircle className="w-4 h-4" /> Stop Feed
            </button>
          )}
        </div>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Column: Video Feed & Real-time features */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden shadow-2xl relative aspect-video flex items-center justify-center">
            <div className="absolute left-4 top-4 z-10 flex items-center gap-2 rounded bg-slate-950/80 px-3 py-1.5 text-xs text-slate-200">
              {streamStatus === 'Live' ? <Activity className="h-3.5 w-3.5 text-green-400" /> : <WifiOff className="h-3.5 w-3.5 text-slate-400" />}
              {streamStatus}
            </div>
            {frameData ? (
              <img src={frameData} alt="Live Camera Feed" className="w-full h-full object-cover" />
            ) : (
              <div className="text-slate-500 flex flex-col items-center gap-4">
                <Camera className="w-16 h-16 opacity-50" />
                <p>Waiting for camera stream...</p>
              </div>
            )}
          </div>

          {/* Biomechanics Data Panel */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6">
             <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><Activity className="w-5 h-5 text-blue-400" /> Real-Time Kinematics</h2>
             {telemetry.features ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-900 p-4 rounded border border-slate-700">
                    <div className="text-slate-400 text-xs uppercase mb-1">L/R Knee Valgus</div>
                    <div className="text-lg font-bold text-blue-300">
                      {telemetry.features.left_knee_valgus.toFixed(1)}° / {telemetry.features.right_knee_valgus.toFixed(1)}°
                    </div>
                  </div>
                  <div className="bg-slate-900 p-4 rounded border border-slate-700">
                    <div className="text-slate-400 text-xs uppercase mb-1">L/R Knee Flexion</div>
                    <div className="text-lg font-bold text-blue-300">
                      {telemetry.features.left_knee_flexion.toFixed(1)}° / {telemetry.features.right_knee_flexion.toFixed(1)}°
                    </div>
                  </div>
                  <div className="bg-slate-900 p-4 rounded border border-slate-700">
                    <div className="text-slate-400 text-xs uppercase mb-1">Asymmetry Score</div>
                    <div className="text-lg font-bold text-blue-300">{telemetry.features.knee_asymmetry.toFixed(1)}</div>
                  </div>
                  <div className="bg-slate-900 p-4 rounded border border-slate-700">
                    <div className="text-slate-400 text-xs uppercase mb-1">Joint Velocity</div>
                    <div className="text-lg font-bold text-blue-300">{telemetry.features.joint_velocity.toFixed(2)}</div>
                  </div>
                </div>
             ) : (
                <p className="text-slate-500 text-sm">No pose detected.</p>
             )}
          </div>

          {/* Debug Panel (Toggleable) */}
          {showDebug && (
            <div className="bg-slate-950 rounded-xl border border-fuchsia-500/50 p-6 font-mono text-sm text-green-400">
              <h2 className="text-fuchsia-400 mb-2 font-bold uppercase tracking-widest border-b border-fuchsia-500/30 pb-2">Pipeline Diagnostics</h2>
              {telemetry.features ? (
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <h3 className="text-slate-400 underline mb-1">Pose Metrics</h3>
                    <p>L_Knee_Ang: {telemetry.features.left_knee_flexion.toFixed(2)}</p>
                    <p>R_Knee_Ang: {telemetry.features.right_knee_flexion.toFixed(2)}</p>
                    <p>L_Hip_Ang:  {telemetry.features.left_hip_flexion.toFixed(2)}</p>
                    <p>R_Hip_Ang:  {telemetry.features.right_hip_flexion.toFixed(2)}</p>
                  </div>
                  <div>
                    <h3 className="text-slate-400 underline mb-1">Movement Metrics</h3>
                    <p>C_Velocity: {telemetry.features.joint_velocity.toFixed(3)}</p>
                    <p>K_Asymmetry: {telemetry.features.knee_asymmetry.toFixed(2)}</p>
                    <p>H_Asymmetry: {telemetry.features.hip_asymmetry.toFixed(2)}</p>
                  </div>
                  <div>
                    <h3 className="text-slate-400 underline mb-1">Prediction Metrics</h3>
                    <p>Raw Output: {(telemetry.risk_score / 100).toFixed(4)}</p>
                    <p>Probability: {telemetry.risk_score.toFixed(1)}%</p>
                    <p>Class: {telemetry.risk_score > 50 ? 'HIGH (1)' : 'LOW (0)'}</p>
                  </div>
                </div>
              ) : (
                <p>Waiting for data frame...</p>
              )}
            </div>
          )}
        </div>

        {/* Right Column: AI Analysis */}
        <div className="flex flex-col gap-6">
          
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex flex-col items-center">
             <h2 className="text-lg font-semibold w-full text-left mb-6">Injury Probability</h2>
             
             <div className="relative w-48 h-48 flex items-center justify-center">
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="#1e293b" strokeWidth="10" />
                  <circle 
                    cx="50" cy="50" r="45" fill="none" 
                    stroke={telemetry.risk_score > 60 ? "#ef4444" : telemetry.risk_score > 30 ? "#eab308" : "#22c55e"} 
                    strokeWidth="10" 
                    strokeLinecap="round"
                    strokeDasharray={`${(telemetry.risk_score / 100) * 283} 283`} 
                    className="transition-all duration-500 ease-out"
                  />
                </svg>
                <div className="absolute flex flex-col items-center">
                  <span className={`text-4xl font-black ${getRiskColor(telemetry.risk_score)}`}>
                    {telemetry.risk_score.toFixed(0)}%
                  </span>
                  <span className="text-xs text-slate-400 font-medium uppercase mt-1">{getRiskLabel(telemetry.risk_score)} Risk</span>
                </div>
             </div>
          </div>

          {/* Explainable AI */}
          <div className="bg-slate-800 rounded-xl border border-slate-700 p-6 flex-grow flex flex-col">
            <h2 className="text-lg font-semibold mb-2">Explainable AI (SHAP)</h2>
            <p className="text-xs text-slate-400 mb-6">Feature contributions to the current risk score.</p>
            
            <div className="h-48 w-full">
              {shapData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={shapData} layout="vertical" margin={{ top: 0, right: 0, left: 10, bottom: 0 }}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" width={112} tick={{ fill: '#94a3b8', fontSize: 10 }} axisLine={false} tickLine={false} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                      itemStyle={{ color: '#f8fafc' }}
                      formatter={(value) => Number(value ?? 0).toFixed(3)}
                    />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {shapData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#22c55e'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full items-center justify-center rounded border border-slate-700 bg-slate-900 text-sm text-slate-500">
                  Waiting for SHAP values
                </div>
              )}
            </div>

            <div className="mt-6 pt-6 border-t border-slate-700">
               <h3 className="text-sm font-semibold mb-3 text-slate-300">Why was this score generated?</h3>
               <ul className="space-y-2">
                 {telemetry.explanations && telemetry.explanations.length > 0 ? (
                   telemetry.explanations.map((msg: string, i: number) => (
                     <li key={i} className="text-sm flex items-start gap-2">
                        <span className={`w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0 ${msg.includes('increases') ? 'bg-red-500' : 'bg-green-500'}`}></span>
                        <span className="text-slate-300">{msg}</span>
                     </li>
                   ))
                 ) : (
                   <li className="text-sm text-slate-500 italic">Waiting for pose data...</li>
                 )}
               </ul>
            </div>
          </div>
          
        </div>
      </main>
    </div>
  );
}
