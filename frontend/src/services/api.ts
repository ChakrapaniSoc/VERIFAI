import { isAnalysisResponse, isRecord, type AnalysisResponse } from "../types/analysis";

export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
const MAX_IMAGE_PIXELS = 40_000_000;

const MIME_SIGNATURES: ReadonlyArray<{
  mime: string;
  matches: (bytes: Uint8Array) => boolean;
}> = [
  {
    mime: "image/jpeg",
    matches: (bytes) => bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff,
  },
  {
    mime: "image/png",
    matches: (bytes) =>
      bytes[0] === 0x89 &&
      bytes[1] === 0x50 &&
      bytes[2] === 0x4e &&
      bytes[3] === 0x47 &&
      bytes[4] === 0x0d &&
      bytes[5] === 0x0a &&
      bytes[6] === 0x1a &&
      bytes[7] === 0x0a,
  },
  {
    mime: "image/webp",
    matches: (bytes) =>
      bytes[0] === 0x52 &&
      bytes[1] === 0x49 &&
      bytes[2] === 0x46 &&
      bytes[3] === 0x46 &&
      bytes[8] === 0x57 &&
      bytes[9] === 0x45 &&
      bytes[10] === 0x42 &&
      bytes[11] === 0x50,
  },
  {
    mime: "image/bmp",
    matches: (bytes) => bytes[0] === 0x42 && bytes[1] === 0x4d,
  },
];

export async function validateImageFile(file: File): Promise<void> {
  if (file.size === 0) {
    throw new Error("This file is empty. Choose an image to continue.");
  }
  if (file.size > MAX_IMAGE_BYTES) {
    throw new Error("Images must be 12 MB or smaller.");
  }

  const signature = new Uint8Array(await file.slice(0, 12).arrayBuffer());
  const detected = MIME_SIGNATURES.find(({ matches }) => matches(signature))?.mime;
  if (!detected) {
    throw new Error("This file is not a supported JPEG, PNG, WEBP, or BMP image.");
  }
  if (file.type && file.type !== detected) {
    throw new Error("The file extension or type does not match its image content.");
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    throw new Error("This image could not be decoded. Choose a valid image file.");
  }

  try {
    if (bitmap.width * bitmap.height > MAX_IMAGE_PIXELS) {
      throw new Error("Image dimensions exceed the supported limit.");
    }
  } finally {
    bitmap.close();
  }
}

export async function checkApiHealth(signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${API_URL}/health`, { signal });
  const payload: unknown = await readJson(response);
  if (!response.ok || !isRecord(payload) || payload.status !== "ok") {
    throw new Error("VERIFAI API is unavailable.");
  }
}

export async function analyzeImage(file: File): Promise<AnalysisResponse> {
  const body = new FormData();
  body.append("file", file);

  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/analyze`, { method: "POST", body });
  } catch {
    throw new Error("Could not reach the VERIFAI API. Check your connection and try again.");
  }

  const payload: unknown = await readJson(response);
  if (!response.ok) {
    const detail = isRecord(payload) && typeof payload.detail === "string" ? payload.detail : "";
    throw new Error(detail || "Analysis failed. Please try again.");
  }
  if (!isAnalysisResponse(payload)) {
    throw new Error("The API returned an unexpected analysis response.");
  }
  return payload;
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new Error(`The API returned an invalid response (HTTP ${response.status}).`);
  }
}
