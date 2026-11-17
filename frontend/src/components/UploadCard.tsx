import { useEffect, useRef, useState, type DragEvent } from "react";
import { FileImage, ImagePlus, LoaderCircle, UploadCloud } from "lucide-react";
import { MAX_IMAGE_BYTES, validateImageFile } from "../services/api";

interface UploadCardProps {
  file: File | null;
  isAnalyzing: boolean;
  analysisError: string;
  onFileSelected: (file: File | null) => void;
  onAnalyze: () => void;
}

const ACCEPTED_TYPES = "image/jpeg,image/png,image/webp,image/bmp";

export function UploadCard({
  file,
  isAnalyzing,
  analysisError,
  onFileSelected,
  onAnalyze,
}: UploadCardProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const selectionSequence = useRef(0);
  const [previewUrl, setPreviewUrl] = useState("");
  const [fileError, setFileError] = useState("");
  const [isValidating, setIsValidating] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function handleFile(candidate: File | undefined) {
    if (!candidate) return;
    const sequence = ++selectionSequence.current;
    setFileError("");
    setIsValidating(true);
    onFileSelected(null);

    try {
      await validateImageFile(candidate);
      if (sequence === selectionSequence.current) onFileSelected(candidate);
    } catch (error) {
      if (sequence === selectionSequence.current) {
        setFileError(error instanceof Error ? error.message : "Could not read this image.");
      }
    } finally {
      if (sequence === selectionSequence.current) setIsValidating(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    void handleFile(event.dataTransfer.files[0]);
  }

  const fileSize = file
    ? file.size < 1024 * 1024
      ? `${(file.size / 1024).toFixed(0)} KB`
      : `${(file.size / (1024 * 1024)).toFixed(1)} MB`
    : "";

  return (
    <section className="card upload-card" id="help" aria-labelledby="upload-title">
      <div className="card-heading">
        <div>
          <span className="section-kicker">01 / INPUT</span>
          <h2 id="upload-title">Add an image</h2>
        </div>
        <span className="heading-icon heading-icon--blue">
          <ImagePlus size={19} aria-hidden="true" />
        </span>
      </div>

      <div
        className={`dropzone${isDragging ? " is-dragging" : ""}${previewUrl ? " has-preview" : ""}`}
        onDragEnter={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={(event) => {
          if (
            !event.relatedTarget ||
            !(event.relatedTarget instanceof Node) ||
            !event.currentTarget.contains(event.relatedTarget)
          ) {
            setIsDragging(false);
          }
        }}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <div className="image-preview">
            <img src={previewUrl} alt="Preview of the selected image" />
            <div className="preview-overlay">
              <span className="preview-file-icon">
                <FileImage size={16} aria-hidden="true" />
              </span>
              <span className="preview-name" title={file?.name}>
                {file?.name}
                <small>{fileSize}</small>
              </span>
            </div>
          </div>
        ) : (
          <div className="dropzone-empty">
            <span className="upload-illustration">
              {isValidating ? (
                <LoaderCircle className="spin" size={27} aria-hidden="true" />
              ) : (
                <UploadCloud size={27} aria-hidden="true" />
              )}
            </span>
            <strong>{isValidating ? "Checking your image…" : "Drop an image here"}</strong>
            <span>or browse from your device</span>
            <span className="file-format-note">JPG/JPEG, PNG, WEBP or BMP · up to 20 MB</span>
          </div>
        )}
        <input
          ref={inputRef}
          className="file-input-hidden"
          type="file"
          accept={ACCEPTED_TYPES}
          aria-label="Choose an image"
          onChange={(event) => {
            void handleFile(event.currentTarget.files?.[0]);
            event.currentTarget.value = "";
          }}
        />
        <button
          className="browse-button"
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={isValidating || isAnalyzing}
        >
          Browse files
        </button>
      </div>

      {(fileError || analysisError) && (
        <p className="inline-error" role="alert">
          {fileError || analysisError}
        </p>
      )}

      <button
        className="button-primary analyze-button"
        type="button"
        onClick={onAnalyze}
        disabled={!file || isValidating || isAnalyzing}
      >
        {isAnalyzing ? (
          <>
            <LoaderCircle className="spin" size={17} aria-hidden="true" />
            Analyzing image
          </>
        ) : (
          <>
            Analyze image
            <span aria-hidden="true">→</span>
          </>
        )}
      </button>
      <p className="privacy-note">
        Your image is sent to the VERIFAI API for analysis and is not stored by this frontend.
      </p>
      <span className="sr-only">Maximum file size: {MAX_IMAGE_BYTES / (1024 * 1024)} MB.</span>
    </section>
  );
}
