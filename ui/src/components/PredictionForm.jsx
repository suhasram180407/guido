import React, { useState } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8001'

export default function PredictionForm({onResult}){
  const [csv, setCsv] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [projectId, setProjectId] = useState('TCGA-LUAD')
  const [disease, setDisease] = useState('lung_adenocarcinoma')
  const [jsonPayload, setJsonPayload] = useState('')

  async function handleSubmit(e){
    e.preventDefault()
    setLoading(true); setError(null)
    try{
      // Minimal example: send CSV or sample payload to backend API
      // The API should return { prediction, features, report }
      const formData = new FormData()
      formData.append('project_id', projectId)
      formData.append('disease_name', disease)
      if(csv) formData.append('file', csv)
      if(!csv && jsonPayload) formData.append('json_payload', jsonPayload)
      const resp = await axios.post(`${API_BASE}/api/ui/predict`, formData)
      onResult(resp.data)
    }catch(err){
      setError('Failed to get prediction')
    }finally{ setLoading(false) }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 gap-3">
        <div className="grid sm:grid-cols-2 gap-3">
          <div>
            <label className="text-sm subtle">Project ID</label>
            <input className="w-full p-2 rounded" value={projectId} onChange={e=>setProjectId(e.target.value)} />
          </div>
          <div>
            <label className="text-sm subtle">Disease</label>
            <input className="w-full p-2 rounded" value={disease} onChange={e=>setDisease(e.target.value)} />
          </div>
        </div>
        <label className="text-sm subtle">Upload CSV (samples x features)</label>
        <div className="p-4 border-dashed border-2" style={{borderColor:'var(--border)'}}>
          <input type="file" accept="text/csv" onChange={e=>setCsv(e.target.files[0])} />
        </div>
        <div>
          <label className="text-sm subtle">Or paste JSON sample (object or array)</label>
          <textarea rows={4} className="w-full p-2 mt-2 rounded" placeholder='{"GENE1":1.2,"GENE2":0.4}' value={jsonPayload} onChange={e=>setJsonPayload(e.target.value)} />
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button type="submit" className="px-4 py-2 rounded" style={{background:'var(--accent)', color:'var(--text)'}} disabled={loading}>
          {loading ? 'Processing…' : 'Submit'}
        </button>
        {loading && <div className="subtle">This may take a moment…</div>}
      </div>
      {error && <div className="text-sm text-red-400">{error}</div>}
    </form>
  )
}
