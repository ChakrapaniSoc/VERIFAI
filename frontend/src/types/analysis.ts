export type DetectionClass = "ai_generated" | "real";

export interface Prediction extends Record<string, unknown> {
  label: string;
  probabilities: Partial<Record<DetectionClass, number>>;
  confidence?: number;
  model?: string;
  explanation?: string;
  device?: string;
}

export interface Explainability extends Record<string, unknown> {
  gradcam_png_base64?: string | null;
  note?: string;
}

export interface Forensics extends Record<string, unknown> {
  metadata?: Record<string, unknown>;
  frequency?: Record<string, unknown>;
  provenance?: Record<string, unknown>;
}

export interface AnalysisResponse extends Record<string, unknown> {
  filename?: string;
  prediction: Prediction;
  explainability?: Explainability;
  forensics?: Forensics;
  limitations?: string[];
}

function isOptionalString(value: unknown): boolean {
  return value === undefined || typeof value === "string";
}

function isOptionalProbability(value: unknown): boolean {
  return (
    value === undefined ||
    (typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1)
  );
}

export function isAnalysisResponse(value: unknown): value is AnalysisResponse {
  if (!isRecord(value) || !isRecord(value.prediction)) return false;

  const prediction = value.prediction;
  if (
    typeof prediction.label !== "string" ||
    !isRecord(prediction.probabilities) ||
    !isOptionalProbability(prediction.probabilities.ai_generated) ||
    !isOptionalProbability(prediction.probabilities.real) ||
    !isOptionalProbability(prediction.confidence) ||
    !isOptionalString(value.filename) ||
    !isOptionalString(prediction.model) ||
    !isOptionalString(prediction.explanation) ||
    !isOptionalString(prediction.device)
  ) {
    return false;
  }

  if (value.explainability !== undefined) {
    if (!isRecord(value.explainability)) return false;
    const explainability = value.explainability;
    if (
      (explainability.gradcam_png_base64 !== undefined &&
        explainability.gradcam_png_base64 !== null &&
        typeof explainability.gradcam_png_base64 !== "string") ||
      !isOptionalString(explainability.note)
    ) {
      return false;
    }
  }

  if (value.forensics !== undefined) {
    if (!isRecord(value.forensics)) return false;
    for (const key of ["metadata", "frequency", "provenance"]) {
      if (value.forensics[key] !== undefined && !isRecord(value.forensics[key])) {
        return false;
      }
    }
  }

  return (
    value.limitations === undefined ||
    (Array.isArray(value.limitations) &&
      value.limitations.every((limitation) => typeof limitation === "string"))
  );
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
