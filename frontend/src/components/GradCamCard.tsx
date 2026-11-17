import { Focus, ImageOff } from "lucide-react";
import type { AnalysisResponse } from "../types/analysis";

interface GradCamCardProps {
  result: AnalysisResponse | null;
  isLoading: boolean;
}

export function GradCamCard({ result, isLoading }: GradCamCardProps) {
  const cam = result?.explainability?.gradcam_png_base64;
  const note = result?.explainability?.note;

  return (
    <section className="card gradcam-card" id="features" aria-labelledby="gradcam-title">
      <div className="card-heading">
        <div>
          <span className="section-kicker">03 / EXPLAINABILITY</span>
          <h2 id="gradcam-title">Model attention</h2>
        </div>
        <span className="heading-icon heading-icon--blue">
          <Focus size={19} aria-hidden="true" />
        </span>
      </div>

      {isLoading ? (
        <div className="visual-empty">
          <span className="skeleton skeleton-image" />
          <span className="skeleton skeleton-line" />
        </div>
      ) : cam ? (
        <div className="gradcam-result">
          <div className="gradcam-image-wrap">
            <img src={`data:image/png;base64,${cam}`} alt="Grad-CAM attention overlay for the analyzed image" />
            <span className="image-badge"><Focus size={13} /> Grad-CAM overlay</span>
          </div>
          <p className="explanation-note">
            {note ?? "Grad-CAM highlights regions that influenced the model's output. It does not identify verified manipulated pixels."}
          </p>
        </div>
      ) : (
        <div className="visual-empty visual-empty--message">
          <span className="empty-orb"><ImageOff size={21} aria-hidden="true" /></span>
          <strong>{result ? "Visualization unavailable" : "No visualization yet"}</strong>
          <span>
            {result
              ? note ?? "The API did not return a Grad-CAM image for this analysis."
              : "The Grad-CAM overlay will appear here after an image is analyzed."}
          </span>
        </div>
      )}
    </section>
  );
}
