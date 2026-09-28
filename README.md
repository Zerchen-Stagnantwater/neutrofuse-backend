# neutrofuse-api

The fusion pipeline and REST API for [NeutroFuse](https://neutrofuse.vercel.app) — a free, open-source multi-focus image fusion tool.

**Frontend:** [neutrofuse-web](https://github.com/YOUR_ORG/neutrofuse-web)  
**Research report:** [NeutroFuse_Report.docx](./NeutroFuse_Report.docx)  
**Live demo:** https://neutrofuse.vercel.app

---

## What's in this repo

```
neutrofuse/          The verified fusion package (195 tests)
├── core/              Patch decomposition, sharpness, neutrosophic T/I/F, hypergraph
├── fusion/              Decision rule, smooth composition
├── metrics/               MI, SF, Qabf, SSIM, evaluate_fusion()
├── data/                    Lytro + MFI-WHU dataset loaders
├── viz/                       Heatmaps, decision maps, comparison panels
└── pipeline.py                  run_fusion() — the single entry point

experiments/         Ablation runner, GFF baseline, statistical comparison
├── ablation.py
├── statistics.py
└── baselines/

app/                 FastAPI service wrapping the pipeline
├── main.py            /fuse endpoint, error shaping, CORS
├── fusion_service.py    Resize cap, shape matching, JPEG encoding
└── config.py            Env-var driven configuration

pipeline_tests/      The original 195-test suite (190 offline, 5 network)
tests/               API-layer tests (25 tests, HTTP contract verification)
```

---

## API

### `POST /fuse`

Fuse two multi-focus images. Accepts `multipart/form-data`:

| Field | Type | Description |
|---|---|---|
| `image_a` | File | First source image (JPEG/PNG/WebP, max 12MB) |
| `image_b` | File | Second source image, same framing |

Returns the fused image as `image/jpeg`. On error:
```json
{ "error": "Human-readable message" }
```

### `GET /health`

Returns `{"status": "ok"}` — used by deploy platforms for uptime checks.

### Example

```bash
curl -X POST https://your-api-host/fuse \
  -F "image_a=@photo_a.jpg" \
  -F "image_b=@photo_b.jpg" \
  --output fused.jpg
```

---

## Running locally

```bash
git clone https://github.com/YOUR_ORG/neutrofuse-api
cd neutrofuse-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The API will be at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Running tests

```bash
# All 215 tests (fast, no network)
pytest tests/ pipeline_tests/ -m "not network"

# Including live Lytro/MFI-WHU dataset downloads
pytest tests/ pipeline_tests/
```

---

## Deploying

The API requires a persistent process (Python + OpenCV). It won't run on Vercel's serverless runtime. Options:

**Render / Railway / Fly.io (recommended for open-source):**
Point any of these at the repo root and they'll pick up the `Dockerfile` automatically.

**Docker manually:**
```bash
docker build -t neutrofuse-api .
docker run -p 8000:8000 neutrofuse-api
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `NEUTROFUSE_MAX_LONG_EDGE` | `800` | Server-side resize ceiling. 800px ≈ 4s/request; 1200px ≈ 10s. |
| `NEUTROFUSE_MAX_UPLOAD_BYTES` | `12582912` | 12MB per image |
| `NEUTROFUSE_ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. Set to your frontend URL in production. |

---

## The method

NeutroFuse models per-patch source confidence using neutrosophic logic (Truth, Indeterminacy, Falsity) derived from Laplacian-variance sharpness. Ambiguous patches are resolved by consensus over a hyperedge-similarity hypergraph rather than local comparison alone.

Tested against naive averaging and Guided Filter Fusion on 20 real Lytro pairs. The honest result: clear win on mutual information (20/20 pairs), marginal win on edge preservation (13/20, p=0.031), no demonstrated advantage on raw sharpness. The hyperedge-aggregation mechanism specifically trades a small amount of pixel-level sharpness for better structural coherence — verified mechanistically, not just statistically. Full details in the [research report](./NeutroFuse_Report.docx).

Two real bugs were found and fixed while building this (not just the parts that went well): a normalization collapse under log-normal sharpness distributions, and an upstream OpenCV `guidedFilter` defect affecting half of all tested real pairs. Both are documented in the report and in the codebase.

---

## Contributing

Issues and PRs welcome. See `.github/pull_request_template.md` for what a PR should include. The test suite (215 tests total) should pass before any merge. If a change shifts the headline metrics, the PR description should explain why — the research report is the baseline.

---

## License

[MIT](./LICENSE)
