import React from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function FeatureChart({data}){
  if(!data || !Array.isArray(data) || data.length===0) return <div className="subtle">No feature importance available.</div>

  const top = data.slice(0, 15)
  const h = Math.max(180, top.length * 28)

  return (
    <div style={{height: h}}>
      <ResponsiveContainer>
        <BarChart data={top} layout="vertical" margin={{left: 80, right: 20, top: 5, bottom: 5}}>
          <XAxis type="number" tick={{fill:'var(--sub)', fontSize:11}} axisLine={false} />
          <YAxis type="category" dataKey="name" tick={{fill:'var(--text)', fontSize:11}} width={75} />
          <Tooltip contentStyle={{background:'var(--card)', border:'1px solid var(--border)', color:'var(--text)'}}/>
          <Bar dataKey="value" fill="#888" radius={[0,3,3,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
