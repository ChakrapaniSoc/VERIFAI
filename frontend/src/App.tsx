import { useCallback, useEffect, useState } from "react";
import { ArrowUpRight, CircleCheck, Info } from "lucide-react";
import { AIInsightsCard } from "./components/AIInsightsCard";
import { AnalysisResultCard } from "./components/AnalysisResultCard";
import { DashboardHeader, type ApiStatus } from "./components/DashboardHeader";
import { DownloadReportButton } from "./components/DownloadReportButton";
import { ForensicsCard } from "./components/ForensicsCard";
import { GradCamCard } from "./components/GradCamCard";
import { LimitationsCard } from "./components/LimitationsCard";
import { Sidebar } from "./components/Sidebar";
import { UploadCard } from "./components/UploadCard";
import { analyzeImage, checkApiHealth } from "./services/api";
import type { AnalysisResponse } from "./types/analysis";

const SECTIONS = ["home", "about", "features", "analytics", "help"];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [activeSection, setActiveSection] = useState("home");

  const refreshApiStatus = useCallback(async (signal?: AbortSignal) => {
    try {
      await checkApiHealth(signal);
      setApiStatus("connected");
    } catch {
      if (!signal?.aborted) setApiStatus("offline");
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshApiStatus(controller.signal);
    const interval = window.setInterval(() => void refreshApiStatus(), 30_000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [refreshApiStatus]);

  useEffect(() => {
    const elements = SECTIONS.map((section) => document.getElementById(section)).filter(
      (element): element is HTMLElement => element !== null,
    );
    if (!("IntersectionObserver" in window) || elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (window.scrollY < 120) {
          setActiveSection("home");
          return;
        }
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
        if (visible) setActiveSection(visible.target.id);
      },
      { rootMargin: "-15% 0px -65% 0px", threshold: [0, 0.1, 0.25, 0.5] },
    );
    elements.forEach((element) => observer.observe(element));
    return () => observer.disconnect();
  }, []);

  const handleFileSelected = useCallback((selected: File | null) => {
    setFile(selected);
    setResult(null);
    setAnalysisError("");
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file || isAnalyzing) return;
    setIsAnalyzing(true);
    setAnalysisError("");
    setResult(null);
    try {
      const analysis = await analyzeImage(file);
      setResult(analysis);
      setApiStatus("connected");
    } catch (error) {
      setAnalysisError(error instanceof Error ? error.message : "Analysis failed. Please try again.");
      void refreshApiStatus();
    } finally {
      setIsAnalyzing(false);
    }
  }, [file, isAnalyzing, refreshApiStatus]);

  return (
    <div className="app-shell">
      <Sidebar activeSection={activeSection} onNavigate={setActiveSection} />
      <main className="main-content">
        <div className="content-shell">
          <DashboardHeader apiStatus={apiStatus} />

          <section className="welcome-strip" id="about">
            <div className="welcome-copy">
              <span className="welcome-icon"><Info size={17} aria-hidden="true" /></span>
              <p>
                <strong>A clearer view of image signals.</strong>
                Review the model’s assessment alongside explainability and returned forensic context.
              </p>
            </div>
            <span className="welcome-tag">AI + FORENSICS</span>
          </section>

          <div className="dashboard-grid">
            <UploadCard
              file={file}
              isAnalyzing={isAnalyzing}
              analysisError={analysisError}
              onFileSelected={handleFileSelected}
              onAnalyze={() => void handleAnalyze()}
            />
            <AnalysisResultCard result={result} isLoading={isAnalyzing} />
            <GradCamCard result={result} isLoading={isAnalyzing} />
            <ForensicsCard result={result} />
            <AIInsightsCard result={result} />
            <LimitationsCard result={result} />
          </div>

          <footer className="dashboard-footer">
            <span>VERIFAI <i>·</i> EXPLAINABLE IMAGE SCREENING</span>
            {result && <DownloadReportButton result={result} />}
            <span className="footer-note">
              Results are screening estimates <ArrowUpRight size={13} aria-hidden="true" />
            </span>
          </footer>
          {result && (
            <p className="sr-only" role="status" aria-live="polite">
              <CircleCheck size={14} aria-hidden="true" /> Analysis complete.
            </p>
          )}
        </div>
      </main>
    </div>
  );
}
