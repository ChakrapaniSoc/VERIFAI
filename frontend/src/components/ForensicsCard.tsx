import { AudioWaveform, BadgeInfo, Fingerprint, ScanLine } from "lucide-react";
import type { AnalysisResponse } from "../types/analysis";

interface ForensicPanelProps {
  title: string;
  icon: typeof BadgeInfo;
  value: Record<string, unknown> | undefined;
  hasResult: boolean;
}

function ForensicPanel({ title, icon: Icon, value, hasResult }: ForensicPanelProps) {
  return (
    <article className="forensic-panel">
      <div className="forensic-panel-heading">
        <span className="forensic-icon"><Icon size={16} aria-hidden="true" /></span>
        <h3>{title}</h3>
      </div>
      {value ? (
        <details className="forensic-details">
          <summary>View returned data</summary>
          <pre>{JSON.stringify(value, null, 2)}</pre>
        </details>
      ) : (
        <p className="data-unavailable">
          {hasResult
            ? `No ${title.toLowerCase()} data was returned.`
            : `${title} data will appear here after analysis.`}
        </p>
      )}
    </article>
  );
}

interface ForensicsCardProps {
  result: AnalysisResponse | null;
}

export function ForensicsCard({ result }: ForensicsCardProps) {
  const forensics = result?.forensics;

  return (
    <section className="card forensics-card" aria-labelledby="forensics-title">
      <div className="card-heading">
        <div>
          <span className="section-kicker">04 / IMAGE CONTEXT</span>
          <h2 id="forensics-title">Forensic analysis</h2>
        </div>
        <span className="heading-icon heading-icon--lavender">
          <Fingerprint size={19} aria-hidden="true" />
        </span>
      </div>
      <div className="forensic-panels">
        <ForensicPanel
          title="Metadata"
          icon={BadgeInfo}
          value={forensics?.metadata}
          hasResult={result !== null}
        />
        <ForensicPanel
          title="Frequency analysis"
          icon={AudioWaveform}
          value={forensics?.frequency}
          hasResult={result !== null}
        />
        <ForensicPanel
          title="Provenance"
          icon={ScanLine}
          value={forensics?.provenance}
          hasResult={result !== null}
        />
      </div>
      <p className="forensics-footnote">
        These returned indicators provide context; they do not prove an image’s origin.
      </p>
    </section>
  );
}
