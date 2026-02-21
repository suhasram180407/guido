import React, { useState } from 'react'

function renderReport(report) {
  if (!report) return '<p>No report available.</p>'
  if (typeof report === 'string') {
    try {
      // Try parsing in case it's a stringified JSON
      const parsed = JSON.parse(report)
      return formatObj(parsed)
    } catch {
      return `<p>${report}</p>`
    }
  }
  return formatObj(report)
}

function formatObj(obj) {
  if (obj.error) return `<p class="text-red-400">Audit error: ${obj.error}</p>`
  const sections = []
  for (const [key, val] of Object.entries(obj)) {
    const title = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    if (typeof val === 'string') {
      sections.push(`<h4 style="margin-top:1em;font-weight:600">${title}</h4><p>${val}</p>`)
    } else if (Array.isArray(val)) {
      const items = val.length ? val.map(v => `<li>${typeof v === 'object' ? JSON.stringify(v) : v}</li>`).join('') : '<li class="subtle">None</li>'
      sections.push(`<h4 style="margin-top:1em;font-weight:600">${title}</h4><ul>${items}</ul>`)
    } else if (typeof val === 'object' && val !== null) {
      const rows = Object.entries(val).map(([k, v]) => {
        const label = k.replace(/_/g, ' ')
        let cell
        if (Array.isArray(v)) {
          cell = v.length ? '<ul style="margin:0;padding-left:1.2em">' + v.map(item => `<li>${typeof item === 'object' ? JSON.stringify(item) : item}</li>`).join('') + '</ul>' : '<span class="subtle">None</span>'
        } else if (typeof v === 'object' && v !== null) {
          cell = Object.entries(v).map(([sk, sv]) => `<strong>${sk.replace(/_/g, ' ')}:</strong> ${sv}`).join('<br/>')
        } else {
          cell = String(v)
        }
        return `<tr><td style="padding:4px 8px;vertical-align:top;color:var(--sub);white-space:nowrap">${label}</td><td style="padding:4px 8px">${cell}</td></tr>`
      }).join('')
      sections.push(`<h4 style="margin-top:1em;font-weight:600">${title}</h4><table>${rows}</table>`)
    }
  }
  return sections.join('\n')
}

export default function ReportViewer({markdown}){
  const [open, setOpen] = useState(true)
  if(!markdown) return <div className="subtle">No report available.</div>

  const html = renderReport(markdown)

  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="font-medium">AI Report</div>
        <button className="text-sm subtle" onClick={()=>setOpen(s=>!s)}>{open ? 'Collapse' : 'Expand'}</button>
      </div>
      {open && (
        <div className="mt-4" style={{lineHeight:'1.6'}} dangerouslySetInnerHTML={{__html: html}} />
      )}
    </div>
  )
}
