import React from 'react'

export default function Navbar(){
  return (
    <header className="w-full border-b" style={{borderColor:'var(--border)'}}>
      <div className="max-w-3xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="text-lg font-medium">Madhav</div>
        <div className="flex items-center gap-3">
          <button aria-label="toggle-theme" className="px-3 py-1 text-sm rounded bg-surface border" style={{borderColor:'var(--border)'}}>Dark</button>
        </div>
      </div>
    </header>
  )
}
