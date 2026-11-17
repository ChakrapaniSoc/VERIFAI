import { Download } from "lucide-react";
import type { AnalysisResponse } from "../types/analysis";

interface DownloadReportButtonProps {
  result: AnalysisResponse;
}

function safeFilename(filename: string | undefined): string {
  const stem = filename?.split(/[\\/]/).pop()?.replace(/\.[^.]+$/, "");
  const cleanStem = stem?.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return cleanStem ? `verifai-report-${cleanStem}.json` : "verifai-analysis-report.json";
}

export function DownloadReportButton({ result }: DownloadReportButtonProps) {
  function downloadReport() {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = safeFilename(result.filename);
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <button className="button-secondary report-button" type="button" onClick={downloadReport}>
      <Download size={17} aria-hidden="true" />
      Download report
    </button>
  );
}
