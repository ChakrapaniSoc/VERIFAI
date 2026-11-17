import { Lightbulb, MessageSquareText } from "lucide-react";
import type { AnalysisResponse } from "../types/analysis";
import { isRecord } from "../types/analysis";

interface AIInsightsCardProps {
  result: AnalysisResponse | null;
}

function returnedText(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

export function AIInsightsCard({ result }: AIInsightsCardProps) {
  const prediction = result?.prediction;
  const forensics = result?.forensics;
  const items = [
    returnedText(prediction?.explanation),
    returnedText(isRecord(forensics?.metadata) ? forensics.metadata.interpretation : undefined),
    returnedText(isRecord(forensics?.frequency) ? forensics.frequency.note : undefined),
    returnedText(isRecord(forensics?.provenance) ? forensics.provenance.message : undefined),
  ].filter((item): item is string => Boolean(item));

  return (
    <section className="card insights-card" aria-labelledby="insights-title">
      <div className="card-heading">
        <div>
          <span className="section-kicker">05 / INTERPRETATION</span>
          <h2 id="insights-title">AI insights</h2>
        </div>
        <span className="heading-icon heading-icon--purple">
          <Lightbulb size={19} aria-hidden="true" />
        </span>
      </div>
      {items.length > 0 ? (
        <ul className="insight-list">
          {items.map((item, index) => (
            <li key={`${index}-${item}`}>
              <span className="insight-bullet"><MessageSquareText size={15} aria-hidden="true" /></span>
              <p>{item}</p>
            </li>
          ))}
        </ul>
      ) : (
        <div className="insights-empty">
          <span className="empty-orb"><Lightbulb size={20} aria-hidden="true" /></span>
          <p>
            {result
              ? "No additional explanation or forensic notes were included in this response."
              : "Model explanations and returned forensic notes will appear here after analysis."}
          </p>
        </div>
      )}
      <p className="insights-source">Insights reflect only information returned by the analysis API.</p>
    </section>
  );
}
