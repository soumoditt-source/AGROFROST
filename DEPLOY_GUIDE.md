# 🚀 Deployment Guide: The Efficient Way

For the **Gemini Cloud Hackathon**, speed and reliability are key. We have architected this project for **Zero-Ops Deployment** using Vercel.

## Why Vercel?
*   **Hybrid Runtimes**: Supports both Python (Backend/AI) and React (Frontend) in a single repo.
*   **Global CDN**: Serves your assets (images, JS) from the edge, making the app feel "Flash" fast.
*   **Free Tier**: Entirely free for hacakthon projects.

## ⚡ Step-by-Step Deployment

### 1. Push Code
Ensure your latest code is on GitHub (we are handling this).
```bash
git push origin main
```

### 2. Import to Vercel
1.  Go to [vercel.com/new](https://vercel.com/new).
2.  Select your repository: `soumoditt-source/AGROFROST`.
3.  **Framework Preset**: Select `Vite`.
4.  **Root Directory**: Leave as `./`.

### 3. Configure Environment Variables
Add the following in the Vercel Dashboard project settings:
*   `GEMINI_API_KEY`: Your Google AI Studio Key.

### 4. Deploy
Click **Deploy**. Vercel will:
*   Build the React Frontend (`npm run build`).
*   Deploy the Python API as Serverless Functions (`api/index.py`).
*   Route functionality automatically via `vercel.json`.

---

## 🛠 Troubleshooting
*   **404 on API**: Ensure `vercel.json` has the `rewrites` rule pointing `/api/(.*)` to `api/index.py`. (Verified: ✅)
*   **Build Fail**: Check Vercel logs. Usually due to missing dependencies in `requirements.txt`. (Verified: ✅)
