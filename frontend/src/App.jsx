import { useState, useRef, useEffect } from 'react'
import './index.css'

const Tooltip = ({ text, children }) => (
  <div className="tooltip-wrapper">
    {children}
    <span className="info-icon">i</span>
    <div className="tooltip-text">{text}</div>
  </div>
);

function App() {
  const [text, setText] = useState('')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0;
  const isShortText = wordCount > 0 && wordCount < 50;

  const handleClear = () => {
    setText('');
    setResults(null);
    setError(null);
  }

  const handleCopyResults = () => {
    if (!results) return;
    const formatted = `DetectML Analysis
Verdict: ${results.classification.toUpperCase()}
AI Probability: ${results.ai_probability}%

Metrics:
- Perplexity: ${results.features.perplexity}
- Word Entropy: ${results.features.word_entropy}
- Burstiness: ${results.features.burstiness}
- Repetition: ${Math.round(results.features.repetition_rate * 100)}%

RAG Similarities:
- Neutral: ${Math.round(results.comparisons['Neutral']?.Semantic || 0)}%
- Passionate: ${Math.round(results.comparisons['Passionate']?.Semantic || 0)}%
- HighTemp: ${Math.round(results.comparisons['HighTemp']?.Semantic || 0)}%`;
    navigator.clipboard.writeText(formatted);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please paste an article to analyze.");
      return;
    }

    if (text.split(' ').length < 10) {
      setError("Text is too short. Please provide a substantial article.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResults(null);

    try {
      // In production, this would point to the hugging face spaces URL
      // For local dev, we point to localhost:7860
      const response = await fetch('http://localhost:7860/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setResults(data);
      }
    } catch (err) {
      setError("Failed to connect to the analysis engine. Make sure the backend is running.");
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:7860/api/extract_text', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setText(data.text);
      }
    } catch (err) {
      setError("Failed to extract text from file.");
      console.error(err);
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  const hasContent = isAnalyzing || results !== null;

  return (
    <div className="app-container">
      <header>
        <h1 className="title-gradient">DetectML</h1>
        <p className="subtitle">Detecting AI-written articles related to the Martial Law Era (1972-1981) in the Philippines</p>
      </header>

      <main className={`main-content ${hasContent ? 'active' : 'idle'}`}>
        {/* Left Column: Input */}
        <div className="glass-panel input-section">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
            <h2 style={{ margin: 0 }}>Article Submission</h2>
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <input
                type="file"
                accept=".txt,.pdf,.docx"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleFileUpload}
              />
              <button
                onClick={() => fileInputRef.current.click()}
                disabled={isUploading || isAnalyzing}
                style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-main)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '0.6rem 1rem', fontSize: '0.85rem', cursor: 'pointer', transition: 'all 0.2s' }}
                onMouseEnter={(e) => e.target.style.background = 'rgba(255,255,255,0.1)'}
                onMouseLeave={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'}
              >
                {isUploading ? '⚙️ Parsing...' : '📁 Upload File (PDF/DOCX/TXT)'}
              </button>
            </div>
          </div>

          <div className="textarea-wrapper" style={{ marginBottom: '0.5rem' }}>
            <textarea
              ref={textareaRef}
              placeholder="Paste the historical article or essay here..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', fontSize: '0.85rem', padding: '0 0.5rem' }}>
            <div style={{ color: isShortText ? 'var(--primary)' : 'var(--text-muted)', transition: 'color 0.3s' }}>
              <strong>{wordCount}</strong> words | <strong>{text.length}</strong> characters
              {isShortText && <span style={{ marginLeft: '0.5rem', fontStyle: 'italic' }}>⚠️ Too short for high accuracy</span>}
            </div>

            <button
              onClick={handleClear}
              style={{ background: 'transparent', color: 'var(--danger)', border: 'none', cursor: 'pointer', opacity: text ? 1 : 0.4, display: 'flex', alignItems: 'center', gap: '0.4rem', pointerEvents: text ? 'auto' : 'none' }}
            >
              <span>🗑️</span> Clear
            </button>
          </div>

          {error && <p style={{ color: 'var(--danger)', marginBottom: '1rem' }}>{error}</p>}
          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={isAnalyzing}
          >
            {isAnalyzing ? 'Analyzing Stylometry & Fetching Context...' : 'Run Forensic Analysis'}
          </button>
        </div>

        {/* Right Column: Results - Only renders when active */}
        {hasContent && (
          <div className="glass-panel results-section">
            {isAnalyzing ? (
              <div className="loader-container">
                <div className="spinner"></div>
                <div className="pulsing-text">Extracting Linguistic Metrics & Querying Vectors...</div>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: '1rem' }}>This usually takes ~10 seconds</p>
              </div>
            ) : results && (
              <div className="results-dashboard">
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem' }}>
                  <button
                    onClick={handleCopyResults}
                    style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--text-main)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', padding: '0.5rem 1rem', fontSize: '0.85rem', cursor: 'pointer', transition: 'all 0.2s', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
                    onMouseEnter={(e) => e.target.style.background = 'rgba(255,255,255,0.1)'}
                    onMouseLeave={(e) => e.target.style.background = 'rgba(255,255,255,0.05)'}
                  >
                    {copied ? '✅ Copied!' : '📋 Copy Results'}
                  </button>
                </div>
                <div className="score-card premium-score-card" style={{ marginTop: '-1rem' }}>
                  <div className={`verdict-badge ${results.is_ai ? 'badge-ai' : 'badge-human'}`}>
                    <div className="badge-icon">
                      {results.is_ai ? '🤖' : '✍️'}
                    </div>
                    <div className="badge-text">
                      <span className="badge-subtitle">Forensic Origin Verdict</span>
                      <h2 className="badge-title">{results.classification.toUpperCase()}</h2>
                    </div>
                  </div>

                  <div className="confidence-meter-container">
                    <div className="confidence-header">
                      <span>{results.is_ai ? 'AI Probability' : 'Human Probability'}</span>
                      <strong className={results.is_ai ? 'text-danger' : 'text-success'}>
                        {results.is_ai ? results.ai_probability : (100 - results.ai_probability).toFixed(2)}%
                      </strong>
                    </div>
                    <div className="confidence-track">
                      <div
                        className={`confidence-fill ${results.is_ai ? 'fill-ai' : 'fill-human'}`}
                        style={{ width: `${results.is_ai ? results.ai_probability : (100 - results.ai_probability)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                <h3>Key Metrics</h3>
                <div className="metrics-grid">
                  <div className="metric-item">
                    <span className="metric-label">
                      <Tooltip text="Measures text predictability. AI writes very safely (low perplexity), while humans use unexpected word combinations (high perplexity).">
                        Perplexity
                      </Tooltip>
                    </span>
                    <span className="metric-value">{results.features.perplexity}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">
                      <Tooltip text="Calculates vocabulary randomness. Humans naturally use a wider, chaotic variety of words (high entropy) compared to AI.">
                        Word Entropy
                      </Tooltip>
                    </span>
                    <span className="metric-value">{results.features.word_entropy}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">
                      <Tooltip text="Evaluates sentence length variation. Humans write in 'bursts' (mixed short/long phrases), whereas AI maintains robotic uniformity.">
                        Burstiness
                      </Tooltip>
                    </span>
                    <span className="metric-value">{results.features.burstiness}</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">
                      <Tooltip text="The rate of repeated n-grams. High repetition signals an AI's tendency to loop highly probable historical concepts.">
                        Repetition
                      </Tooltip>
                    </span>
                    <span className="metric-value">{Math.round(results.features.repetition_rate * 100)}%</span>
                  </div>
                </div>

                <div style={{ marginTop: '2rem' }}>
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>
                    <Tooltip text="Which of the RAG-generated articles is the submitted text most similar to using the four features above?">
                      Highest RAG Similarities
                    </Tooltip>
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                    {['Neutral', 'Passionate', 'HighTemp'].map(style => (
                      <div key={style} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.2)', padding: '0.8rem 1rem', borderRadius: '8px' }}>
                        <span style={{ display: 'flex' }}>
                          <Tooltip text={
                            style === 'Neutral' ? "Similarity bridging to an AI-generated historical article prompted for strict objectivity." :
                              style === 'Passionate' ? "Similarity bridging to an AI-generated historical article prompted for intense emotional variance." :
                                "Similarity bridging to an AI-generated historical article synthesized with maximum algorithmic creativity (High Temperature)."
                          }>
                            {style} Style
                          </Tooltip>
                        </span>
                        <strong>{Math.round(results.comparisons[style]?.Semantic || 0)}% Match</strong>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Created by Kenneth Olano &copy; {new Date().getFullYear()}.</p>
        <p>Special Problem Research • Institute of Computer Science, College of Arts and Science, University of the Philippines Los Baños</p>
      </footer>
    </div>
  )
}

export default App
