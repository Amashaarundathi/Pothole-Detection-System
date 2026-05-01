import React, { useEffect, useRef, useState } from "react";

const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/stream";

// ─── Audio engine (outside component — never stale) ────────────────────────
let _audioCtx = null;
let _lastBeep = 0;

function getAudioCtx() {
  if (!_audioCtx) _audioCtx = new AudioContext();
  if (_audioCtx.state === "suspended") _audioCtx.resume();
  return _audioCtx;
}

function playTone(ctx, freq, startTime, duration) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.type = "square";
  osc.frequency.setValueAtTime(freq, startTime);
  gain.gain.setValueAtTime(0.3, startTime);
  gain.gain.setValueAtTime(0.3, startTime + duration - 0.01);
  gain.gain.linearRampToValueAtTime(0.0, startTime + duration);
  osc.start(startTime);
  osc.stop(startTime + duration);
}

function playBeep(severity) {
  const now = Date.now();
  if (now - _lastBeep < 2000) return;
  _lastBeep = now;
  try {
    const ctx = getAudioCtx();
    const t = ctx.currentTime;
    if (severity === "High") {
      playTone(ctx, 880, t, 0.18);
      playTone(ctx, 880, t + 0.28, 0.18);
    } else if (severity === "Medium") {
      playTone(ctx, 520, t, 0.25);
    } else {
      playTone(ctx, 300, t, 0.25);
    }
  } catch (e) {
    console.warn("Beep error:", e);
  }
}

// ─── Icons (inline SVG) ────────────────────────────────────────────────────
const Icon = {
  Camera: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
      <circle cx="12" cy="13" r="4"/>
    </svg>
  ),
  Stop: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <rect x="3" y="3" width="18" height="18" rx="2"/>
    </svg>
  ),
  Upload: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polyline points="16 16 12 12 8 16"/>
      <line x1="12" y1="12" x2="12" y2="21"/>
      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
    </svg>
  ),
  Play: () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
  ),
  Alert: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  ),
  Target: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>
    </svg>
  ),
  Map: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <polygon points="3 6 9 3 15 6 21 3 21 18 15 21 9 18 3 21"/>
      <line x1="9" y1="3" x2="9" y2="18"/><line x1="15" y1="6" x2="15" y2="21"/>
    </svg>
  ),
  Zap: () => (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
    </svg>
  ),
  Wifi: () => (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M5 12.55a11 11 0 0 1 14.08 0"/><path d="M1.42 9a16 16 0 0 1 21.16 0"/>
      <path d="M8.53 16.11a6 6 0 0 1 6.95 0"/><line x1="12" y1="20" x2="12.01" y2="20"/>
    </svg>
  ),
};

// ─── Severity badge ────────────────────────────────────────────────────────
function SeverityBadge({ level }) {
  const colors = {
    High: { bg: "rgba(239,68,68,0.12)", border: "#ef4444", text: "#ef4444" },
    Medium: { bg: "rgba(245,158,11,0.12)", border: "#f59e0b", text: "#f59e0b" },
    Low: { bg: "rgba(34,197,94,0.12)", border: "#22c55e", text: "#22c55e" },
  };
  const c = colors[level] || colors.Low;
  return (
    <span style={{
      background: c.bg, border: `1px solid ${c.border}`, color: c.text,
      padding: "2px 8px", borderRadius: 4, fontSize: 10,
      fontFamily: "var(--font-mono)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em"
    }}>
      {level}
    </span>
  );
}

// ─── Stat card ─────────────────────────────────────────────────────────────
function StatCard({ label, value, color, icon }) {
  return (
    <div className="stat-card" style={{ flex: 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <span style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.08em" }}>
          {label}
        </span>
        <span style={{ color }}>{icon}</span>
      </div>
      <div style={{ fontSize: 32, fontWeight: 800, color, marginTop: 8, fontFamily: "var(--font-mono)" }}>
        {value}
      </div>
    </div>
  );
}

// ─── Main component ─────────────────────────────────────────────────────────
export default function PotholeDetectionFrontend() {
  const [running, setRunning] = useState(false);
  const [frame, setFrame] = useState("");
  const [detections, setDetections] = useState([]);
  const [file, setFile] = useState(null);
  const [uploaded, setUploaded] = useState("");
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [tab, setTab] = useState("camera");
  const [totalCount, setTotalCount] = useState(0);
  const [highCount, setHighCount] = useState(0);
  const [medCount, setMedCount] = useState(0);
  const [lowCount, setLowCount] = useState(0);
  const [log, setLog] = useState([]);
  const socketRef = useRef(null);
  const prevDetCount = useRef(0);

  // Status poll
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch(`${API}/api/status`);
        setStatus(await r.json());
      } catch {}
    };
    poll();
    const id = setInterval(poll, 3000);
    return () => clearInterval(id);
  }, []);

  // WebSocket
  useEffect(() => {
    if (!running) {
      socketRef.current?.close();
      return;
    }
    socketRef.current = new WebSocket(WS_URL);
    socketRef.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      setFrame(`data:image/jpeg;base64,${data.data}`);
      setDetections(data.detections || []);

      // Beep based on alert level sent from backend
      if (data.alert) playBeep(data.alert);

      const dets = data.detections || [];
      if (dets.length > prevDetCount.current) {
        const newDets = dets.slice(prevDetCount.current);
        setTotalCount(c => c + newDets.length);
        setHighCount(c => c + newDets.filter(d => d.severity === "High").length);
        setMedCount(c => c + newDets.filter(d => d.severity === "Medium").length);
        setLowCount(c => c + newDets.filter(d => d.severity === "Low").length);
        const d = dets[dets.length - 1];
        setLog(l => [{
          id: Date.now(), time: new Date().toLocaleTimeString(),
          severity: d.severity, location: d.location, confidence: d.confidence
        }, ...l].slice(0, 20));
      }
      prevDetCount.current = dets.length;
    };
    socketRef.current.onerror = () => {};
    return () => socketRef.current?.close();
  }, [running]);

  const startCamera = async () => {
    try {
      getAudioCtx(); // unlock AudioContext on user gesture
      await fetch(`${API}/api/detection/start?source=camera`, { method: "POST" });
      setRunning(true);
    } catch { alert("Cannot connect to backend. Make sure it's running on port 8000."); }
  };

  const stop = async () => {
    try { await fetch(`${API}/api/detection/stop`, { method: "POST" }); } catch {}
    setRunning(false);
    setFrame("");
    setDetections([]);
    setTotalCount(0);
    setHighCount(0);
    setMedCount(0);
    setLowCount(0);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${API}/api/upload-video`, { method: "POST", body: form });
      const data = await res.json();
      setUploaded(data.filename);
    } catch { alert("Upload failed."); }
    setUploading(false);
  };

  const startVideo = async () => {
    if (!uploaded) return;
    const form = new FormData();
    form.append("filename", uploaded);
    try {
      getAudioCtx(); // unlock AudioContext on user gesture
      await fetch(`${API}/api/detection/start-video`, { method: "POST", body: form });
      setRunning(true);
    } catch { alert("Cannot start video detection."); }
  };



  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }} className="grid-bg">
      {/* Header */}
      <header style={{
        borderBottom: "1px solid var(--border)",
        background: "rgba(10,10,15,0.95)",
        backdropFilter: "blur(12px)",
        position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{ maxWidth: 1400, margin: "0 auto", padding: "0 24px", height: 60, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 36, height: 36, background: "var(--accent)",
              borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center",
              boxShadow: "0 0 16px rgba(249,115,22,0.4)"
            }}>
              <Icon.Target />
            </div>
            <div>
              <div style={{ fontWeight: 800, fontSize: 16, letterSpacing: "-0.02em" }}>PotholeAI</div>
              <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Road Intelligence System</div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
            {running && (
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="live-dot" />
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--danger)", textTransform: "uppercase", letterSpacing: "0.08em" }}>Live</span>
              </div>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Icon.Wifi />
              <span style={{ fontSize: 11, color: status?.running ? "var(--safe)" : "var(--muted)", fontFamily: "var(--font-mono)" }}>
                {status?.model_loaded ? "MODEL READY" : "MOCK MODE"}
              </span>
            </div>
          </div>
        </div>
      </header>

      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "24px 24px" }}>
        {/* Stats row */}
        <div style={{ display: "flex", gap: 12, marginBottom: 24 }} className="animate-fade-up">
          <StatCard label="Detected Total" value={totalCount} color="var(--accent)" icon={<Icon.Target />} />
          <StatCard label="High Severity" value={highCount} color="var(--danger)" icon={<Icon.Alert />} />
          <StatCard label="Medium Severity" value={medCount} color="var(--warn)" icon={<Icon.Zap />} />
          <StatCard label="Low Severity" value={lowCount} color="var(--safe)" icon={<Icon.Map />} />
        </div>

        {/* Main grid */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: 20 }}>

          {/* Left: Video feed + controls */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Feed card */}
            <div className="card" style={{ overflow: "hidden" }}>
              <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ width: 6, height: 6, background: running ? "var(--danger)" : "var(--muted)", borderRadius: "50%", display: "inline-block" }} />
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--muted)" }}>
                    {running ? (tab === "camera" ? "Camera Feed" : "Video Feed") : "Feed Inactive"}
                  </span>
                </div>
                {running && (
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--muted)" }}>
                    {detections.length} object{detections.length !== 1 ? "s" : ""} detected
                  </span>
                )}
              </div>

              {/* Video area */}
              <div style={{ position: "relative", background: "var(--surface2)", minHeight: 400, display: "flex", alignItems: "center", justifyContent: "center" }}>
                {frame ? (
                  <>
                    <img src={frame} alt="feed" style={{ width: "100%", display: "block", maxHeight: 500, objectFit: "contain" }} />
                    <div className="scan-overlay" />
                  </>
                ) : (
                  <div style={{ textAlign: "center", color: "var(--muted)" }}>
                    <div style={{ width: 64, height: 64, border: "2px dashed var(--border)", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", margin: "0 auto 16px" }}>
                      <Icon.Camera />
                    </div>
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, textTransform: "uppercase", letterSpacing: "0.08em" }}>No signal</div>
                    <div style={{ fontSize: 12, marginTop: 6 }}>Start camera or upload a video</div>
                  </div>
                )}
              </div>
            </div>

            {/* Controls card */}
            <div className="card" style={{ padding: 18 }}>
              {/* Tabs */}
              <div style={{ display: "flex", gap: 4, marginBottom: 18, background: "var(--surface2)", padding: 4, borderRadius: 8 }}>
                {["camera", "video"].map(t => (
                  <button key={t} onClick={() => setTab(t)} style={{
                    flex: 1, padding: "8px 0", background: tab === t ? "var(--accent)" : "transparent",
                    border: "none", borderRadius: 6, cursor: "pointer",
                    color: tab === t ? "#000" : "var(--muted)",
                    fontFamily: "var(--font-mono)", fontSize: 11, fontWeight: 700,
                    textTransform: "uppercase", letterSpacing: "0.06em", transition: "all 0.2s"
                  }}>
                    {t === "camera" ? "📷 Camera" : "🎬 Video"}
                  </button>
                ))}
              </div>

              {tab === "camera" ? (
                <div style={{ display: "flex", gap: 10 }}>
                  <button className="btn-primary" onClick={startCamera} disabled={running} style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, opacity: running ? 0.5 : 1 }}>
                    <Icon.Camera /> Start Camera
                  </button>
                  <button className="btn-danger" onClick={stop} disabled={!running} style={{ opacity: running ? 1 : 0.4, display: "flex", alignItems: "center", gap: 8 }}>
                    <Icon.Stop /> Stop
                  </button>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                    <label className="file-label" htmlFor="videofile" style={{ flex: 1, textAlign: "center" }}>
                      <Icon.Upload />
                      {file ? ` ${file.name}` : " Choose Video File"}
                    </label>
                    <input id="videofile" type="file" accept="video/*" onChange={e => setFile(e.target.files[0])} />
                    <button className="btn-ghost" onClick={handleUpload} disabled={!file || uploading}>
                      {uploading ? "Uploading…" : "Upload"}
                    </button>
                  </div>
                  {uploaded && (
                    <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--safe)", padding: "8px 12px", background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.2)", borderRadius: 6 }}>
                      ✓ Ready: {uploaded}
                    </div>
                  )}
                  <div style={{ display: "flex", gap: 10 }}>
                    <button className="btn-primary" onClick={startVideo} disabled={!uploaded || running} style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 8, opacity: (!uploaded || running) ? 0.5 : 1 }}>
                      <Icon.Play /> Start Detection
                    </button>
                    <button className="btn-danger" onClick={stop} disabled={!running} style={{ opacity: running ? 1 : 0.4, display: "flex", alignItems: "center", gap: 8 }}>
                      <Icon.Stop /> Stop
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right panel */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* Live detections */}
            <div className="card" style={{ flex: 1 }}>
              <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--muted)" }}>
                  Live Detections
                </span>
                {detections.length > 0 && (
                  <span style={{ background: "rgba(239,68,68,0.15)", color: "var(--danger)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 12, padding: "1px 8px", fontSize: 10, fontFamily: "var(--font-mono)", fontWeight: 700 }}>
                    {detections.length}
                  </span>
                )}
              </div>
              <div style={{ padding: "4px 18px", maxHeight: 260, overflowY: "auto" }}>
                {detections.length === 0 ? (
                  <div style={{ padding: "32px 0", textAlign: "center", color: "var(--muted)", fontSize: 13 }}>
                    No potholes detected
                  </div>
                ) : detections.map((d, i) => (
                  <div key={d.id || i} className="detection-row">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 12, fontWeight: 700 }}>
                        #{d.id} — {d.location}
                      </span>
                      <SeverityBadge level={d.severity} />
                    </div>
                    <div style={{ display: "flex", gap: 16 }}>
                      <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                        📍 {d.distance}m
                      </span>
                      <span style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                        🎯 {Math.round((d.confidence || 0) * 100)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Event log */}
            <div className="card">
              <div style={{ padding: "14px 18px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--muted)" }}>
                  Detection Log
                </span>
                <button onClick={() => setLog([])} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono)" }}>
                  CLEAR
                </button>
              </div>
              <div style={{ maxHeight: 220, overflowY: "auto" }}>
                {log.length === 0 ? (
                  <div style={{ padding: "20px 18px", color: "var(--muted)", fontSize: 12, fontFamily: "var(--font-mono)" }}>
                    — Waiting for events —
                  </div>
                ) : log.map(entry => (
                  <div key={entry.id} style={{
                    padding: "8px 18px", borderBottom: "1px solid rgba(42,42,58,0.5)",
                    display: "flex", alignItems: "center", gap: 10, fontSize: 11,
                    fontFamily: "var(--font-mono)"
                  }}>
                    <span style={{ color: "var(--muted)", flexShrink: 0 }}>{entry.time}</span>
                    <SeverityBadge level={entry.severity} />
                    <span style={{ color: "var(--text)" }}>{entry.location}</span>
                    <span style={{ color: "var(--muted)", marginLeft: "auto" }}>{Math.round((entry.confidence || 0) * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>

            {/* System status */}
            <div className="card" style={{ padding: 16 }}>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--muted)", marginBottom: 12 }}>
                System Status
              </div>
              {[
                { label: "Backend", value: status ? "Connected" : "Offline", ok: !!status },
                { label: "Model", value: status?.model_loaded ? "YOLO11n Loaded" : "Mock Mode", ok: status?.model_loaded },
                { label: "Detection", value: running ? "Running" : "Idle", ok: running },
              ].map(row => (
                <div key={row.label} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
                  <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--muted)" }}>{row.label}</span>
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ width: 5, height: 5, borderRadius: "50%", background: row.ok ? "var(--safe)" : "var(--muted)", display: "inline-block" }} />
                    <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: row.ok ? "var(--safe)" : "var(--muted)" }}>{row.value}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
