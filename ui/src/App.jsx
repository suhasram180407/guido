import React, { useState, Suspense } from 'react'
// Navbar removed per UI design (not needed)
import PredictionForm from './components/PredictionForm'
import PredictionResult from './components/PredictionResult'
import FeatureChart from './components/FeatureChart'
const ReportViewer = React.lazy(() => import('./components/ReportViewer'))
import ExportPanel from './components/ExportPanel'

export default function App(){
  const [prediction, setPrediction] = useState(null)
  const [featureData, setFeatureData] = useState(null)
  const [reportMarkdown, setReportMarkdown] = useState(null)

  return (
    <div className="min-h-screen">

      {/* Landing hero: put an image at /landing.jpg (ui/public/landing.jpg) */}
      <section className="hero min-h-screen relative flex items-center justify-center text-center">
        <div className="quad-grid">
          <div className="quad q-tl">
            <div className="logo-container">
              <img src="/logo.svg" alt="Logo" className="hero-logo"/>
            </div>
          </div>
          <div className="quad q-tr" />
          <div className="quad q-bl"><img src="/quad-bl.jpg" alt="" className="quad-img"/></div>
          <div className="quad q-br" />
        </div>
        <div className="hero-overlay" />
        <div className="hero-top-right">
          <div className="hero-title">madhav</div>
        </div>

        <div className="hero-bottom-right">
          <button className="btn hero-explore" onClick={()=>{document.getElementById('app').scrollIntoView({behavior:'smooth'})}}>Explore</button>
        </div>

        <div className="hero-bottom-left">
          <div className="hero-subtitle">privacy-first patient reports | shap biomarkers | llm-backed biomedical audit</div>
        </div>
        
      </section>

      {/* Main app content */}
      <main id="app" className="min-h-screen max-w-4xl mx-auto py-10 px-4 main-app">
        <div className="space-y-6">
          <div className="card p-6 rounded-md fade-in">
            <h2 className="section-title">Input</h2>
            <PredictionForm onResult={(res)=>{ setPrediction(res.prediction); setFeatureData(res.features); setReportMarkdown(res.report) }} />
          </div>

          {prediction && (
            <div className="results-grid">
              <div className="space-y-6">
                <div className="card p-6 rounded-md fade-in prediction-result">
                  <PredictionResult prediction={prediction} />
                </div>

                <div className="card p-6 rounded-md fade-in">
                  <h3 className="text-lg font-medium">Feature Importance</h3>
                  <div className="feature-chart">
                    <FeatureChart data={featureData} />
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="card p-6 rounded-md fade-in">
                  <h3 className="text-lg font-medium">AI Generated Report</h3>
                  <Suspense fallback={<div className="subtle">Loading report…</div>}>
                    <div className="report-viewer">
                      <ReportViewer markdown={reportMarkdown} />
                    </div>
                  </Suspense>
                </div>
                <div className="card p-6 rounded-md fade-in">
                  <ExportPanel data={{prediction, features: featureData, report: reportMarkdown}} />
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
