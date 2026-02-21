import React from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8001'

export default function ExportPanel({data}){
  function downloadJSON(){
    const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'})
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'prediction.json'; a.click()
    URL.revokeObjectURL(url)
  }

  function downloadPDF(){
    // Call server PDF export endpoint and download returned base64 PDF
    if(!data || !data.report){
      alert('No report available to export')
      return
    }

    axios.post(`${API_BASE}/api/ui/export_pdf`, {report: data.report})
      .then(res=>{
        const b64 = res.data.pdf_base64
        const byteChars = atob(b64)
        const byteNumbers = new Array(byteChars.length)
        for (let i = 0; i < byteChars.length; i++) byteNumbers[i] = byteChars.charCodeAt(i)
        const byteArray = new Uint8Array(byteNumbers)
        const blob = new Blob([byteArray], {type:'application/pdf'})
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url; a.download = 'report.pdf'; a.click()
        URL.revokeObjectURL(url)
      })
      .catch(()=>{ alert('PDF export failed on server') })
  }

  return (
    <div className="flex gap-3 mt-4">
      <button onClick={downloadPDF} className="px-3 py-1 border rounded subtle" style={{borderColor:'var(--border)'}}>Download PDF</button>
      <button onClick={downloadJSON} className="px-3 py-1 border rounded subtle" style={{borderColor:'var(--border)'}}>Download JSON</button>
    </div>
  )
}
