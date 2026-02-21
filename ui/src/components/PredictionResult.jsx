import React from 'react'

export default function PredictionResult({prediction}){
  const pct = Math.round((prediction.prob || 0) * 100)
  return (
    <div>
      <div className="flex items-baseline gap-4">
        <div className="text-3xl font-semibold">{prediction.label || 'Prediction'}</div>
        <div className="text-sm subtle">Model: {prediction.model || 'primary'}</div>
      </div>

      <div className="mt-4">
        <div className="h-2 bg-surface rounded overflow-hidden" style={{border:`1px solid var(--border)`}}>
          <div style={{width:`${pct}%`, background:'var(--accent)', height:'100%'}} />
        </div>
        <div className="text-sm subtle mt-2">Confidence: {pct}%</div>
      </div>

      <div className="mt-3 text-xs subtle">Processing time: {prediction.time || '—'}</div>
    </div>
  )
}
