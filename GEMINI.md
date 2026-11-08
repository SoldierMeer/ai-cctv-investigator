# AI CCTV Investigator — Engineering Constitution (DL-Fall2026)

## 1. System Architecture & 5-Stage Pipeline Invariants
1. **Stage 1 (`apps/vision-pipeline`):** YOLOv8 + ByteTrack extracts 3.0-second pedestrian tracklets (`TRACKLET_WIN_SEC = 3.0`) and selects top-3 sharpest crops via Laplacian variance.
2. **Stage 2 (`apps/dl-core`):** OpenAI CLIP (`ViT-B/16` + Text Transformer) projects images and text into a shared 512-D vector space (`EMBED_DIM = 512`). All output embeddings MUST be L2-normalized (`norm == 1.0`).
3. **Stage 3 (`apps/dl-core`):** IRRA cross-modal alignment with a 4-stripe horizontal attention head (Head, Torso, Legs, Bag) trained with `InfoNCE + SDM Loss`. Early CLIP layers stay frozen so fine-tuning completes in 3–4 hours on a single Kaggle P100 or Colab T4 GPU.
4. **Stage 4 (`apps/api-server`):** FAISS `IndexHNSWFlat(512, 32, METRIC_INNER_PRODUCT)` stores normalized tracklet vectors and metadata.
5. **Stage 5 (`apps/api-server` + `apps/web-dashboard`):** FastAPI serves search (`POST /api/v1/search`) and HTTP 206 byte-range video streams (`GET /api/v1/video/{cam_id}`) to the Next.js dashboard. End-to-end query latency MUST remain `< 400 ms`.

## 2. Strict File Boundaries During Sprints
- Always check `packages/shared-contracts/` before modifying any schema or API payload.
- Never hardcode file paths; use relative paths anchored to the repository root `data/` directory.
- Every new function in `vision-pipeline`, `dl-core`, or `api-server` must include a corresponding `pytest` unit test.
