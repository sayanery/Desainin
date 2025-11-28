import os
import random
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# buat si DIFFUSERS 
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import uuid

load_dotenv()
app = Flask(__name__)

FIGMA_TOKEN = os.getenv("FIGMA_TOKEN")
PEXELS_TOKEN = os.getenv("PEXELS_TOKEN")
FREEPIK_API_KEY = os.getenv("FREEPIK_API_KEY")


# LOAD MODEL STABLE DIFFUSION
print("Loading Stable Diffusion model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
    ).to(device)

    try:
        pipe.enable_attention_slicing()
    except:
        pass

    SD_READY = True
    print("Stable Diffusion ready on:", device)

except Exception as e:
    print("FAILED TO LOAD MODEL:", e)
    SD_READY = False

def search_figma(query):
    if not FIGMA_TOKEN: 
        return []

    try:
        url = "https://api.figma.com/v1/community/search"
        headers = {"X-Figma-Token": FIGMA_TOKEN}
        params = {"query": query, "page_size": 18}

        r = requests.get(url, headers=headers, params=params)
        r.raise_for_status()

        items = r.json().get("meta", {}).get("files", [])

        return [{
            "title": item.get("name"),
            "img": item.get("thumbnailUrl"),
            "source": "figma",
            "category": "ui",
            "url": f"https://www.figma.com/community/file/{item.get('key')}",
            "popular": item.get("like_count", 0),
            "latest": item.get("created_at", 0)
        } for item in items]

    except Exception as e:
        print("FIGMA ERROR:", e)
        return []


def search_pexels(query):
    if not PEXELS_TOKEN:
        return []

    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_TOKEN},
            params={"query": query, "per_page": 18}
        )
        r.raise_for_status()

        return [{
            "title": f"{query.title()} Inspiration",
            "img": item["src"]["medium"],
            "source": "pexels",
            "category": "landing",
            "url": item["url"],
            "popular": random.randint(10, 500),
            "latest": random.randint(1, 100)
        } for item in r.json().get("photos", [])]

    except Exception as e:
        print("PEXELS ERROR:", e)
        return []


def search_freepik(query):
    if not FREEPIK_API_KEY:
        return []

    try:
        r = requests.get(
            "https://api.freepik.com/v1/resources",
            headers={"X-Freepik-API-Key": FREEPIK_API_KEY},
            params={
                "term": query,
                "page": 1,
                "limit": 18,
                "filters[content_type][vector]": 1,
                "filters[content_type][psd]": 1
            }
        )
        r.raise_for_status()

        data = r.json().get("data", [])

        return [{
            "title": item.get("title", "Freepik Asset"),
            "img": item.get("image", {}).get("source", {}).get("url"),
            "source": "freepik",
            "category": "mobile",
            "url": item.get("url"),
            "popular": item.get("downloads", 0),
            "latest": item.get("published_at", 0)
        } for item in data]

    except Exception as e:
        print("FREEPIK ERROR:", e)
        return []

def local_generate_image(prompt):
    try:
        result = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=28,
            guidance_scale=7.5
        )
        img = result.images[0]

        static_path = os.path.join(app.root_path, "static")
        os.makedirs(static_path, exist_ok=True)

        file_id = f"{uuid.uuid4()}.png"
        img.save(os.path.join(static_path, file_id))

        return f"/static/{file_id}"

    except Exception as e:
        print("Local SD ERROR:", e)
        return None

@app.route("/inspirasi")
def inspirasi():
    
    query = request.args.get("q", "").strip()
    sorting = request.args.get("sort", "")

    results = []

    # Jika user melakukan pencarian
    if query:
        print("🔎 SEARCH:", query)

        results += search_figma(query)
        results += search_pexels(query)
        results += search_freepik(query)

    else:
        # Default topic ketika user membuka halaman inspirasi
        topics = ["ui design", "dashboard", "business", "mobile app", "landing page"]

        # Ambil data dari berbagai sumber
        for t in topics:
            results += search_figma(t)
            results += search_freepik(t)

        results += search_pexels("creative design")

        random.shuffle(results)

    # Sorting
    if sorting == "latest":
        results = sorted(results, key=lambda x: x.get("latest", 0), reverse=True)

    elif sorting == "popular":
        results = sorted(results, key=lambda x: x.get("popular", 0), reverse=True)

    return render_template("inspirasi.html", designs=results, query=query)


@app.route("/search")
def api_search():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    results = []
    results += search_figma(query)
    results += search_pexels(query)
    results += search_freepik(query)

    return jsonify(results)


@app.route("/desainin")
def desainin():
    return render_template("desainin.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.json
        prompt = data.get("message")

        IMAGE_KEYWORDS = ["buatkan", "generate", "gambar", "poster", "logo", "desain"]

        if any(k in prompt.lower() for k in IMAGE_KEYWORDS):
            url = local_generate_image(prompt)
            if url:
                return jsonify({"reply": "Siap! Ini desain yang berhasil saya buat:", "image_url": url})
            else:
                return jsonify({"reply": "Maaf, gambar gagal dibuat."})

        return jsonify({"reply": "Baik! Ada permintaan lain tentang desain?"})

    except Exception as e:
        print("Chat ERROR:", e)
        return jsonify({"reply": "Server error."}), 500

@app.route("/")
def home():
    return render_template("home.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)