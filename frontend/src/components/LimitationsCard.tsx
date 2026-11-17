import { ShieldAlert } from "lucide-react";
import type { AnalysisResponse } from "../types/analysis";

interface LimitationsCardProps {
  result: AnalysisResponse | null;
}

export function LimitationsCard({ result }: LimitationsCardProps) {
  const limitations = result?.limitations ?? [];

  return (
    <section className="card limitations-card" aria-labelledby="limitations-title">
      <div className="limitations-heading">
        <span className="limitations-icon"><ShieldAlert size={20} aria-hidden="true" /></span>
        <div>
          <span className="section-kicker">IMPORTANT CONTEXT</span>
          <h2 id="limitations-title">Detection has limitations</h2>
        </div>
      </div>
      <p className="disclaimer">
        AI detection is not definitive proof of an image’s authenticity or origin. Treat results as a screening signal, not a final determination.
      </p>
      {limitations.length > 0 && (
        <ul className="limitations-list">
          {limitations.map((limitation, index) => (
            <li key={`${index}-${limitation}`}>{limitation}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
