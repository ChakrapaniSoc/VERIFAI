import { Bell, ShieldCheck } from "lucide-react";

export type ApiStatus = "checking" | "connected" | "offline";

interface DashboardHeaderProps {
  apiStatus: ApiStatus;
}

const statusLabels: Record<ApiStatus, string> = {
  checking: "Checking API",
  connected: "API connected",
  offline: "API unavailable",
};

export function DashboardHeader({ apiStatus }: DashboardHeaderProps) {
  return (
    <header className="dashboard-header" id="home">
      <div className="mobile-brand">
        <span className="brand-mark">
          <ShieldCheck size={20} aria-hidden="true" />
        </span>
        <strong>VERIFAI</strong>
      </div>
      <div className="header-copy">
        <span className="eyebrow">EXPLAINABLE IMAGE SCREENING</span>
        <h1>Image intelligence, made transparent.</h1>
        <p>Investigate an image with model analysis and forensic context.</p>
      </div>
      <div className="header-actions">
        <div
          className={`api-status api-status--${apiStatus}`}
          role="status"
          aria-live="polite"
        >
          <span className="status-dot" />
          {statusLabels[apiStatus]}
        </div>
        <span className="header-icon" aria-label="VERIFAI research preview">
          <Bell size={18} aria-hidden="true" />
        </span>
      </div>
    </header>
  );
}
