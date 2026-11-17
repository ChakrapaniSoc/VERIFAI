import { Activity, Check, CircleHelp, ScanSearch } from "lucide-react";
import type { AnalysisResponse, DetectionClass } from "../types/analysis";

const classLabels: Record<DetectionClass, string> = {
  ai_generated: "AI generated",
  real: "Real",
};

function getConfidence(result: AnalysisResponse): number | undefined {
  const { prediction } = result;
  if (typeof prediction.confidence === "number" && Number.isFinite(prediction.confidence)) {
    return prediction.confidence;
  }
  const className: DetectionClass | undefined =
    prediction.label === "ai_generated" || prediction.label === "real"
      ? prediction.label
      : undefined;
  const score = className ? prediction.probabilities[className] : undefined;
  return typeof score === "number" ? score : undefined;
}

function formatPercent(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(1)}%`
    : "—";
}

function formatLabel(label: string): string {
  if (label === "ai_generated" || label === "real") return classLabels[label];
  return label.replace(/_/g, " ");
}

interface AnalysisResultCardProps {
  result: AnalysisResponse | null;
  isLoading: boolean;
}

export function AnalysisResultCard({ result, isLoading }: AnalysisResultCardProps) {
  const prediction = result?.prediction;
  const confidence = result ? getConfidence(result) : undefined;
  const predictedClass =
    prediction?.label === "ai_generated" || prediction?.label === "real"
      ? prediction.label
      : undefined;

  return (
    <section className="card result-card" id="analytics" aria-labelledby="result-title">
      <div className="card-heading">
        <div>
          <span className="section-kicker">02 / MODEL ANALYSIS</span>
          <h2 id="result-title">Detection result</h2>
        </div>
        <span className="heading-icon heading-icon--purple">
          <ScanSearch size={19} aria-hidden="true" />
        </span>
      </div>

      {isLoading ? (
        <div className="result-empty result-loading" role="status">
          <span className="loading-orb"><Activity size={21} /></span>
          <strong>Analyzing your image</strong>
          <span>The model is evaluating the submitted image.</span>
          <span className="loading-track"><span /></span>
        </div>
      ) : !result || !prediction ? (
        <div className="result-empty">
          <span className="empty-orb"><CircleHelp size={22} aria-hidden="true" /></span>
          <strong>Your analysis will appear here</strong>
          <span>Upload an image to view its model assessment and class scores.</span>
        </div>
      ) : (
        <div className="result-content">
          <div className={`prediction-banner prediction-banner--${predictedClass ?? "unknown"}`}>
            <div className="prediction-symbol">
              <Check size={20} aria-hidden="true" />
            </div>
            <div className="prediction-copy">
              <span>MODEL PREDICTION</span>
              <strong>{formatLabel(prediction.label)}</strong>
            </div>
            {prediction.model && <span className="model-chip">{prediction.model.replace(/_/g, " ")}</span>}
          </div>

          <div className="confidence-block">
            <div className="confidence-topline">
              <div>
                <span className="metric-label">PREDICTED CLASS CONFIDENCE</span>
                <div className="confidence-value">{formatPercent(confidence)}</div>
              </div>
              <span className="confidence-caption">Model score</span>
            </div>
            {confidence !== undefined && (
              <div
                className="confidence-track"
                role="meter"
                aria-label="Predicted class confidence"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(confidence * 100)}
              >
                <span style={{ width: `${Math.min(100, Math.max(0, confidence * 100))}%` }} />
              </div>
            )}
            <p>Confidence is the predicted class probability, not calibrated certainty.</p>
          </div>

          <div className="probability-block">
            <span className="metric-label">CLASS PROBABILITIES</span>
            {(["ai_generated", "real"] as const).map((className) => {
              const score = prediction.probabilities[className];
              return (
                <div className="probability-row" key={className}>
                  <div className="probability-label">
                    <span>{classLabels[className]}</span>
                    <strong>{formatPercent(score)}</strong>
                  </div>
                  {score !== undefined && (
                    <div className={`probability-track probability-track--${className}`}>
                      <span style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          {prediction.explanation && <p className="model-explanation">{prediction.explanation}</p>}
        </div>
      )}
    </section>
  );
}
