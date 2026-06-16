# Hosting

GitHub Actions is **not** a hosting platform — its runners are ephemeral (a job is
killed after at most 6 hours). So we use it two ways:

1. A **temporary live demo** (good for sharing a link for a few hours).
2. A **test-then-deploy** pipeline to a real always-on host (Render).

GitHub Pages can't run this app — it serves static files only, and we need the
Python/ffmpeg backend. (The AI tier is client-side WebGPU, but the free tier and
rendering are not.)

---

## 1. Temporary live demo (Cloudflare tunnel)

Workflow: `.github/workflows/demo-tunnel.yml` (manual trigger).

1. Push the repo to GitHub.
2. Actions tab → **Live demo (Cloudflare tunnel)** → **Run workflow**. Optionally set
   how many minutes to stay up (default 60, max 330).
3. Open the running job → the **Summary** shows a public `https://….trycloudflare.com`
   link. (It's also printed in the "Open tunnel and print URL" step log.)

⚠️ Ephemeral: the link stops working when the job ends, it's a single runner, and
free-tier (audio) processing runs on the runner's CPU. Use it for demos, not real
traffic.

---

## 2. Real free hosting on Render (auto-deploy)

Render runs our `Dockerfile` (ffmpeg included) on a free plan. The free plan spins
down after ~15 min idle and cold-starts on the next request — fine for a hobby MVP.

### One-time setup

1. Create a free account at <https://render.com> and connect your GitHub.
2. **New → Blueprint**, pick this repo. Render reads `render.yaml` and creates the
   `montage-clipper` web service. With `autoDeploy: true`, Render redeploys on every
   push — you're done, and the `deploy-render.yml` step below is optional.

### Optional: deploy from GitHub Actions (gate deploy on tests)

If you'd rather have Actions run the tests first and then tell Render to deploy:

1. In Render: the service → **Settings → Deploy Hook** → copy the URL.
2. In GitHub: repo **Settings → Secrets and variables → Actions → New repository
   secret**, name it `RENDER_DEPLOY_HOOK_URL`, paste the hook URL.
3. `.github/workflows/deploy-render.yml` now runs `pytest` on every push and, if it
   passes, POSTs the hook to trigger the deploy. Without the secret, the deploy step
   skips gracefully (the workflow still passes).

   To avoid deploying twice, set `autoDeploy: false` in `render.yaml` if you use the
   Actions hook (or just keep Render's native auto-deploy and ignore the hook).

### Alternatives

Any Docker host works the same way — e.g. **Fly.io** (`fly launch` from the
Dockerfile, deploy via `superfly/flyctl-actions` with a `FLY_API_TOKEN` secret) or
**Railway**. Render is recommended here purely for the simplest free Docker path.
