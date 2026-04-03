import React, { useState, useEffect } from 'react';
import { Upload, BarChart2, Cpu, ArrowRight, BrainCircuit, Activity, Database, Settings, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { dataAPI, analysisAPI, modelAPI } from './api';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, Legend } from 'recharts';
import Plot from 'react-plotly.js';

function App() {
  const [step, setStep] = useState('upload'); // upload, eda, modeling, results
  const [dataset, setDataset] = useState(null);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [modelSettings, setModelSettings] = useState({ target_column: '', task_type: 'auto', test_size: 0.2, hyperparameter_tuning: true, algorithms: [] });
  const [suggestions, setSuggestions] = useState(null);
  const [modelResults, setModelResults] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setLoadingMsg('Ingesting Dataset & Performing Quality Checks...');
    try {
      const res = await dataAPI.uploadDataset(file);
      setDataset(res);
      setStep('eda');
      setLoadingMsg('Executing Deep Exploratory Data Analysis...');
      
      const edaRes = await analysisAPI.getAnalysis(res.dataset_id);
      setAnalysis(edaRes.eda_results);
      
      setLoadingMsg('Generating AutoML Insights...');
      const suggRes = await modelAPI.getSuggestions(res.dataset_id);
      setSuggestions(suggRes);
      if(suggRes.potential_targets && suggRes.potential_targets.length > 0) {
        setModelSettings(prev => ({...prev, target_column: suggRes.potential_targets[0].column}));
      }
      setLoadingMsg('');
    } catch (err) {
      console.error(err);
      alert('Failed: ' + (err.response?.data?.detail || err.message));
      setLoadingMsg('');
    }
  };

  const handleTrain = async () => {
    setLoadingMsg('Training AutoML Models via XGBoost & Optuna Pipelines. Please wait...');
    setStep('training');
    try {
      const results = await modelAPI.trainModel(dataset.dataset_id, modelSettings);
      setModelResults(results.training_results);
      setStep('results');
    } catch(err) {
      console.error("Training Error:", err);
      alert('Training failed.');
      setStep('eda');
    } finally {
      setLoadingMsg('');
    }
  };

  const Sidebar = () => (
    <div className="w-64 bg-surface border-r border-slate-800 flex flex-col pt-8 pb-4">
      <div className="px-6 mb-12 flex items-center gap-3">
         <BrainCircuit className="w-8 h-8 text-primary-500" />
         <span className="font-bold text-xl tracking-tight text-white">AI Studio</span>
      </div>
      <nav className="flex-1 px-4 space-y-2">
        <div className={`p-3 rounded-lg flex items-center gap-3 cursor-pointer ${['upload','eda','modeling','results'].includes(step) ? 'bg-primary-500/10 text-primary-400 border border-primary-500/20' : 'text-slate-400 hover:bg-slate-800'}`}>
          <Database className="w-5 h-5"/> Data Pipeline
        </div>
        <div className={`p-3 rounded-lg flex items-center gap-3 cursor-pointer ${step === 'modeling' || step === 'results' ? 'bg-primary-500/10 text-primary-400 border border-primary-500/20' : 'text-slate-400 hover:bg-slate-800'}`}>
          <Cpu className="w-5 h-5"/> Model Orchestration
        </div>
        <div className={`p-3 rounded-lg flex items-center gap-3 cursor-pointer ${step === 'results' ? 'bg-primary-500/10 text-primary-400 border border-primary-500/20' : 'text-slate-400 hover:bg-slate-800'}`}>
          <Activity className="w-5 h-5"/> Analytics Engine
        </div>
      </nav>
      <div className="px-6">
        <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
          <p className="text-xs text-slate-400 uppercase font-semibold mb-2">System Status</p>
          <div className="flex items-center gap-2 text-sm text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> Compute Online
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background flex text-slate-200 font-sans selection:bg-primary-500/30 overflow-hidden">
      <Sidebar />
      
      {/* Dynamic Background */}
      <div className="absolute top-[-10%] left-[20%] w-[40rem] h-[50rem] bg-indigo-600/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40rem] h-[40rem] bg-primary-600/10 rounded-full blur-[120px] pointer-events-none" />
      
      <main className="flex-1 flex flex-col h-screen overflow-y-auto z-10 relative">
        <header className="h-20 border-b border-slate-800/50 flex items-center justify-between px-10 bg-background/50 backdrop-blur-md sticky top-0 z-20">
           <div className="flex items-center gap-4">
              <span className="text-xs font-semibold px-2 py-1 rounded bg-slate-800 text-slate-400 border border-slate-700">Project / Default</span>
              <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-slate-100 to-slate-400">
                {step === 'upload' ? 'Data Ingestion' : step === 'eda' ? 'Statistical Overview' : step === 'modeling' ? 'AutoML Configuration' : step === 'results' ? 'Model Evaluation' : 'Processing...'}
              </h2>
           </div>
           {dataset && <div className="flex items-center gap-2 text-sm text-slate-400"><CheckCircle2 className="w-4 h-4 text-emerald-500"/> Dataset: <span className="text-white font-medium">{dataset.filename}</span></div>}
        </header>

        <div className="p-10 max-w-7xl mx-auto w-full">
          {loadingMsg && (
            <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center">
              <div className="w-24 h-24 mb-8">
                <svg className="animate-spin text-primary-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">{loadingMsg}</h2>
              <p className="text-slate-400">Performing matrix operations in isolated environment...</p>
            </div>
          )}

          {step === 'upload' && (
            <div className="flex flex-col items-center justify-center py-10 animate-in fade-in slide-in-from-bottom-8">
              <div className="text-center mb-12">
                <div className="inline-block p-4 rounded-2xl bg-gradient-to-tr from-primary-500/20 to-indigo-500/20 border border-primary-500/20 mb-6">
                   <Database className="w-12 h-12 text-primary-400" />
                </div>
                <h1 className="text-5xl font-extrabold tracking-tight mb-4 text-white">Intelligent Data Workspace</h1>
                <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed">
                  Upload raw data and let the ML Engine autonomously parse, clean, and extract high-dimensional insights using State-of-the-Art algorithms.
                </p>
              </div>
              
              <div className="glass-card w-full max-w-2xl group hover:border-primary-500/40 relative h-72 flex border-dashed border-2 border-slate-700 bg-surface/50">
                <button
                  id="test-upload-btn"
                  className="absolute top-2 right-2 opacity-10 bg-slate-800 text-xs text-white p-1 z-50 hover:opacity-100"
                  onClick={() => {
                    const csvContent = "id,age,income,score,target\n1,25,50000,80,A\n2,35,60000,85,B\n3,45,70000,90,A\n4,22,45000,75,B\n5,50,90000,95,A\n6,28,52000,82,B\n7,40,65000,88,A\n8,32,58000,81,B\n9,48,80000,92,A\n10,26,48000,78,B\n11,38,62000,86,A\n12,42,75000,91,B\n13,30,55000,84,A\n14,55,100000,98,B\n15,24,49000,79,A";
                    const blob = new Blob([csvContent], { type: 'text/csv' });
                    const file = new File([blob], 'dummy_data.csv', { type: 'text/csv' });
                    handleFileUpload({ target: { files: [file] } });
                  }}
                >
                  Mock E2E Upload
                </button>
                <input 
                  type="file" 
                  className="absolute inset-0 opacity-0 cursor-pointer z-10 w-full h-full"
                  onChange={handleFileUpload}
                  accept=".csv,.xlsx,.json,.parquet"
                />
                <div className="flex flex-col items-center justify-center w-full relative z-0 pointer-events-none">
                  <Upload className="w-16 h-16 text-slate-500 mb-6 group-hover:text-primary-400 group-hover:-translate-y-2 transition-all duration-500" />
                  <p className="text-xl font-bold text-slate-200 mb-2">Drop your dataset here</p>
                  <p className="text-slate-400 text-sm">Or click to browse standard formats (CSV, JSON, Parquet)</p>
                  
                  <div className="mt-8 flex gap-4 opacity-50 grayscale group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-700">
                    <span className="px-3 py-1 bg-slate-800 rounded text-xs border border-slate-700">pandas processing</span>
                    <span className="px-3 py-1 bg-slate-800 rounded text-xs border border-slate-700">sklearn pipeline</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {step === 'eda' && dataset && analysis && (
            <div className="animate-in fade-in slide-in-from-bottom-4 space-y-8">
              {/* Top Stats */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card flex items-center justify-between p-5">
                  <div>
                    <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1">Total Records</p>
                    <p className="text-3xl font-bold font-mono">{dataset.rows?.toLocaleString()}</p>
                  </div>
                  <div className="p-3 bg-blue-500/10 rounded-lg"><Database className="w-6 h-6 text-blue-400"/></div>
                </div>
                <div className="glass-card flex items-center justify-between p-5">
                  <div>
                    <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1">Feature Dimension</p>
                    <p className="text-3xl font-bold font-mono">{dataset.columns}</p>
                  </div>
                  <div className="p-3 bg-purple-500/10 rounded-lg"><BarChart2 className="w-6 h-6 text-purple-400"/></div>
                </div>
                <div className="glass-card flex items-center justify-between p-5">
                  <div>
                    <p className="text-slate-400 text-xs font-semibold uppercase tracking-wider mb-1">Missing Values</p>
                    <p className="text-3xl font-bold font-mono text-amber-400">{analysis.missing_value_analysis?.total_missing || 0}</p>
                  </div>
                  <div className="p-3 bg-amber-500/10 rounded-lg"><ShieldAlert className="w-6 h-6 text-amber-400"/></div>
                </div>
                <div className="glass-card bg-gradient-to-r from-primary-900/30 to-indigo-900/30 flex items-center p-5 justify-between border-primary-500/30">
                  <div>
                    <p className="text-primary-200 text-sm font-semibold mb-2">Ready for AutoML</p>
                    <button onClick={() => setStep('modeling')} className="btn-primary text-sm py-1.5 px-4 flex items-center gap-2 shadow-none border border-primary-500/50">
                      Configure Model <ArrowRight className="w-4 h-4"/>
                    </button>
                  </div>
                </div>
              </div>

              {/* Data Quality Report */}
              <div className="grid grid-cols-3 gap-8">
                <div className="col-span-2 glass-card">
                   <h3 className="text-lg font-bold mb-6 flex items-center gap-2"><Activity className="text-primary-400 w-5 h-5"/> Numeric Distributions</h3>
                   <div className="h-72">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={Object.entries(analysis.descriptive_statistics?.numeric_statistics || {}).map(([k,v]) => ({name: k.substring(0,8), count: v.count || 0}))}>
                          <defs>
                            <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="name" stroke="#64748b" tick={{fill: '#64748b'}} />
                          <YAxis stroke="#64748b" tick={{fill: '#64748b'}} />
                          <RechartsTooltip contentStyle={{backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px'}} />
                          <Area type="monotone" dataKey="count" stroke="#0ea5e9" fillOpacity={1} fill="url(#colorCount)" />
                        </AreaChart>
                      </ResponsiveContainer>
                   </div>
                </div>
                <div className="glass-card">
                  <h3 className="text-lg font-bold mb-4">Correlation Alerts</h3>
                  <div className="space-y-3 max-h-72 overflow-y-auto pr-2">
                    {analysis.correlation_analysis?.highly_correlated_pairs?.length > 0 ? 
                      analysis.correlation_analysis.highly_correlated_pairs.map((pair, idx) => (
                        <div key={idx} className="p-3 bg-slate-800/50 rounded-lg border border-slate-700/50 flex justify-between items-center hover:border-slate-600 transition-colors">
                          <div className="flex flex-col">
                            <span className="text-sm font-medium text-slate-300">{pair.feature1}</span>
                            <span className="text-xs text-slate-500">vs</span>
                            <span className="text-sm font-medium text-slate-300">{pair.feature2}</span>
                          </div>
                          <div className="text-orange-400 font-bold font-mono text-sm bg-orange-500/10 px-2 py-1 rounded">
                            {pair.correlation > 0 ? '+' : ''}{pair.correlation}
                          </div>
                        </div>
                      ))
                     : <p className="text-slate-500 italic">No highly correlated features found.</p>
                    }
                  </div>
                </div>
              </div>

              {/* Advanced Plotly Visualizations */}
              {analysis.visualizations && analysis.visualizations.length > 0 && (
                <div className="glass-card">
                  <h3 className="text-xl font-bold mb-6 flex items-center gap-2"><Activity className="text-primary-400 w-5 h-5"/> Advanced Visualizations</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                    {analysis.visualizations.map((vizJson, idx) => {
                      try {
                        let viz = typeof vizJson === 'string' ? JSON.parse(vizJson) : vizJson;
                        return (
                          <div key={idx} className="bg-slate-900/50 p-4 rounded-xl border border-slate-700/50 flex justify-center w-full overflow-hidden shadow-inner">
                            <Plot
                              data={viz.data}
                              layout={{
                                ...viz.layout, 
                                paper_bgcolor: 'transparent',
                                plot_bgcolor: 'transparent',
                                font: { color: '#cbd5e1' },
                                margin: { t: 40, r: 20, l: 40, b: 40 },
                                autosize: true
                              }}
                              useResizeHandler={true}
                              style={{ width: '100%', height: '100%', minHeight: '350px' }}
                              config={{ displayModeBar: false, responsive: true }}
                            />
                          </div>
                        );
                      } catch (e) {
                        return null;
                      }
                    })}
                  </div>
                </div>
              )}

            </div>
          )}

          {step === 'modeling' && (
            <div className="animate-in fade-in slide-in-from-right-8 max-w-4xl mx-auto">
              <div className="glass-card p-10 border-t-4 border-t-indigo-500">
                <div className="flex flex-col md:flex-row justify-between items-start gap-12">
                  <div className="flex-1 space-y-8">
                    <div>
                      <h2 className="text-3xl font-extrabold mb-2">Configure AutoML Pipeline</h2>
                      <p className="text-slate-400">Leveraging distributed XGBoost & Optuna trees.</p>
                    </div>

                    <div className="space-y-4">
                      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 focus-within:border-primary-500 transition-colors">
                        <label className="block text-sm font-semibold text-slate-300 mb-2 uppercase tracking-wider">Predictive Target</label>
                        <select 
                          className="w-full bg-slate-900 border border-slate-700 text-white rounded-lg p-3 outline-none focus:ring-2 focus:ring-primary-500/50"
                          value={modelSettings.target_column}
                          onChange={(e) => setModelSettings({...modelSettings, target_column: e.target.value})}
                        >
                          <option value="">Select Target Feature...</option>
                          {dataset.column_info.map(c => <option key={c.name} value={c.name}>{c.name} ({c.dtype})</option>)}
                        </select>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
                          <label className="block text-sm font-semibold text-slate-300 mb-2 uppercase tracking-wider">Task Type</label>
                          <select 
                            className="w-full bg-slate-900 border border-slate-700 text-white rounded-lg p-2.5 outline-none"
                            value={modelSettings.task_type}
                            onChange={(e) => setModelSettings({...modelSettings, task_type: e.target.value})}
                          >
                            <option value="auto">Auto-Detect</option>
                            <option value="classification">Classification</option>
                            <option value="regression">Regression</option>
                          </select>
                        </div>
                        <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
                          <label className="block text-sm font-semibold text-slate-300 mb-2 uppercase tracking-wider">Validation Split</label>
                          <input 
                            type="range" min="0.1" max="0.5" step="0.05" 
                            className="w-full mt-2 accent-primary-500" 
                            value={modelSettings.test_size}
                            onChange={(e) => setModelSettings({...modelSettings, test_size: parseFloat(e.target.value)})}
                          />
                          <div className="text-right text-sm text-primary-400 font-mono mt-1 pt-1">{modelSettings.test_size * 100}% Holdout</div>
                        </div>
                      </div>
                      
                      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 flex items-center justify-between cursor-pointer group hover:border-slate-500 transition-colors" onClick={() => setModelSettings({...modelSettings, hyperparameter_tuning: !modelSettings.hyperparameter_tuning})}>
                        <div>
                          <p className="font-semibold text-slate-200">Deep Hyperparameter Tuning (Optuna)</p>
                          <p className="text-sm text-slate-400">Autonomous search space optimization</p>
                        </div>
                        <div className={`w-12 h-6 rounded-full p-1 transition-colors ${modelSettings.hyperparameter_tuning ? 'bg-indigo-600' : 'bg-slate-700'}`}>
                          <div className={`w-4 h-4 bg-white rounded-full transition-transform ${modelSettings.hyperparameter_tuning ? 'translate-x-6' : ''}`}/>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="w-full md:w-80 bg-surface border border-slate-700 rounded-xl p-6 shadow-2xl relative overflow-hidden">
                    <div className="absolute top-0 right-0 w-32 h-32 bg-primary-500/10 blur-[50px] pointer-events-none"/>
                    <h3 className="font-bold text-white mb-4 flex items-center gap-2"><Settings className="w-4 h-4 text-primary-400"/> AI Recommendations</h3>
                    {suggestions && suggestions.potential_targets ? (
                      <div className="space-y-4">
                        {suggestions.potential_targets.slice(0,3).map(tgt => (
                          <div key={tgt.column} className="p-3 border border-emerald-500/30 bg-emerald-500/5 rounded-lg cursor-pointer hover:bg-emerald-500/10 transition-colors" onClick={() => setModelSettings({...modelSettings, target_column: tgt.column})}>
                            <div className="flex justify-between items-center mb-1">
                              <span className="font-mono text-emerald-400 text-sm font-bold">{tgt.column}</span>
                              <span className="text-[10px] uppercase bg-emerald-500/20 px-2 rounded text-emerald-300">Suggested</span>
                            </div>
                            <p className="text-xs text-slate-400">{tgt.unique_values} distinct values found.</p>
                          </div>
                        ))}
                      </div>
                    ) : <p className="text-sm text-slate-500">No strong recommendations detected.</p>}
                  </div>
                </div>
                
                <div className="mt-10 pt-6 border-t border-slate-800 flex justify-end">
                  <button onClick={handleTrain} disabled={!modelSettings.target_column} className="btn-primary text-lg px-10 py-3 flex items-center gap-2">
                     <Cpu className="w-5 h-5"/> Initiate Training Run
                  </button>
                </div>
              </div>
            </div>
          )}

          {step === 'results' && modelResults && (
             <div className="animate-in slide-in-from-bottom-8 fade-in flex flex-col gap-8 pb-12">
               <div className="flex justify-between items-end">
                <div>
                  <h2 className="text-4xl font-extrabold mb-2 text-white">Model Evaluation Report</h2>
                  <p className="text-slate-400">Winning architecture: <span className="text-emerald-400 font-mono font-bold bg-emerald-500/10 px-2 py-1 rounded">{modelResults.best_model}</span></p>
                </div>
                <button 
                  onClick={() => analysisAPI.generateReport(dataset.dataset_id, modelResults.model_id)}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg font-bold flex items-center gap-2 transition-all shadow-lg shadow-indigo-500/20"
                >
                  <Activity className="w-5 h-5"/> Download PDF Report
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                 {/* Main Metric Spotlight */}
                 <div className="col-span-1 md:col-span-2 glass-card bg-gradient-to-br from-surface to-slate-900 flex">
                   <div className="p-8 flex-1 border-r border-slate-800/50">
                      <p className="text-slate-400 font-semibold uppercase tracking-wider mb-2">Primary Metric ({modelResults.task_type === 'classification' ? 'Accuracy' : 'R² Score'})</p>
                      <div className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                        {((modelResults.model_results[modelResults.best_model]?.performance_metrics?.accuracy || 
                           modelResults.model_results[modelResults.best_model]?.performance_metrics?.r2_score || 0) * 100).toFixed(1)}%
                      </div>
                      <div className="mt-4 inline-flex items-center gap-1.5 text-sm text-emerald-400 bg-emerald-400/10 px-3 py-1 rounded-full">
                        <CheckCircle2 className="w-4 h-4"/> Exceeds Baseline Output
                      </div>
                   </div>
                   <div className="p-8 flex-1 flex items-center justify-center relative overflow-hidden">
                      {/* Decorative elements */}
                      <div className="absolute inset-0 opacity-10 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-400 via-transparent to-transparent"></div>
                      <div className="space-y-4 w-full z-10">
                        {['f1_score', 'precision', 'recall'].map(metric => {
                           const val = modelResults.model_results[modelResults.best_model]?.performance_metrics?.[metric];
                           if(val !== undefined) {
                             return (
                               <div key={metric} className="flex justify-between items-center border-b border-slate-700/50 pb-2">
                                 <span className="text-slate-400 capitalize">{metric.replace('_', ' ')}</span>
                                 <span className="font-mono text-white font-bold">{(val*100).toFixed(1)}%</span>
                               </div>
                             );
                           }
                           return null;
                        })}
                      </div>
                   </div>
                 </div>
                 
                 <div className="glass-card">
                   <h3 className="font-bold mb-4 flex items-center gap-2"><Cpu className="w-4 h-4 text-indigo-400"/> Competitor Grid</h3>
                   <div className="space-y-3">
                     {modelResults.model_comparison?.ranking?.map((r, i) => (
                       <div key={r.algorithm} className={`p-3 rounded-lg border ${i===0 ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-slate-700 bg-slate-800/50'}`}>
                         <div className="flex justify-between items-center">
                           <span className={`font-semibold ${i===0 ? 'text-emerald-400' : 'text-slate-300'}`}>{r.algorithm}</span>
                           <span className="font-mono text-sm">{(r.primary_metric*100).toFixed(1)}%</span>
                         </div>
                       </div>
                     ))}
                   </div>
                 </div>
              </div>

              {/* Feature Importance */}
              <div className="glass-card">
                 <h3 className="text-xl font-bold mb-6">SHAP / Feature Impact Analysis</h3>
                 <div className="h-80 w-full">
                   <ResponsiveContainer width="100%" height="100%">
                     <BarChart data={Object.entries(modelResults.model_results[modelResults.best_model]?.feature_importance || {}).slice(0, 8).map(([k,v]) => ({name: k.substring(0,10), importance: v}))} layout="vertical">
                       <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                       <XAxis type="number" stroke="#64748b" tick={{fill: '#64748b'}} />
                       <YAxis dataKey="name" type="category" width={100} stroke="#64748b" tick={{fill: '#cbd5e1', fontSize: 12}} />
                       <RechartsTooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px'}}/>
                       <Bar dataKey="importance" fill="#818cf8" radius={[0, 4, 4, 0]} barSize={20}>
                         {/* Subtle gradient styling via individual cell mappings if supported by recharts, otherwise flat color */}
                       </Bar>
                     </BarChart>
                   </ResponsiveContainer>
                 </div>
              </div>
           </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
