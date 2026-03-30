# Zali Backend — API Documentation

Base URL: `http://localhost:8000`

Interactive Swagger docs: [`http://localhost:8000/docs`](http://localhost:8000/docs)

---

## Endpoints

### 1. Health Check

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/health` |
| **Description** | Verify the API is running. |

**Response** `200 OK`
```json
{
  "status": "ok"
}
```

**Example (JavaScript Fetch)**:
```js
const res = await fetch('http://localhost:8000/health')
const data = await res.json()
// { status: "ok" }
```

---

### 2. System Info

| | |
|---|---|
| **Method** | `GET` |
| **URL** | `/info` |
| **Description** | Returns the resolved file paths for the loaded models. Useful for debugging configuration. |

**Response** `200 OK`
```json
{
  "plant_model": "/absolute/path/to/models-cleaned/best_plant_classifier_clean.pth",
  "disease_models_path": "/absolute/path/to/models-cleaned"
}
```

---

### 3. Classify Plant Species

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/predict_plant_class` |
| **Description** | Accepts an image file and returns the predicted plant species and their confidence scores. |
| **Content-Type** | `multipart/form-data` |

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` (image) | ✅ | The plant image to classify. Must be a valid image (`image/jpeg`, `image/png`, etc.) |

**Response** `200 OK`

Returns a dictionary mapping each plant class name to its confidence score (float between 0 and 1). The top 5 most probable classes are returned.

```json
{
  "Tomato": 0.9234,
  "Apple": 0.0431,
  "Potato": 0.0215,
  "Corn_(maize)": 0.0089,
  "Pepper__bell": 0.0031
}
```

**Possible Plant Classes**

| Class Key | Description |
|---|---|
| `Apple` | Apple plant |
| `Tomato` | Tomato plant |
| `Potato` | Potato plant |
| `Corn_(maize)` | Corn / Maize plant |
| `Pepper__bell` | Bell Pepper plant |

**Error Responses**

| Status | Detail | Cause |
|---|---|---|
| `400` | `"File must be an image"` | Uploaded file is not an image type |
| `422` | Validation error | No file provided in the request |

**Example (JavaScript Fetch)**:
```js
const formData = new FormData()
formData.append('file', imageFile) // imageFile is a File object

const res = await fetch('http://localhost:8000/predict_plant_class', {
  method: 'POST',
  body: formData,
})
const predictions = await res.json()
// { "Tomato": 0.92, "Apple": 0.04, ... }
```

---

### 4. Predict Plant Disease (Main Endpoint)

| | |
|---|---|
| **Method** | `POST` |
| **URL** | `/predict_plant_desease` |
| **Description** | The main inference endpoint. Accepts an image, identifies the plant species, then runs the plant-specific disease classifier. Returns disease probabilities for that plant. |
| **Content-Type** | `multipart/form-data` |

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` (image) | ✅ | The plant image to analyze. Must be a valid image. |

**Response** `200 OK`

Returns a dictionary mapping each disease class name to its confidence score. The number of classes depends on the identified plant species.

```json
{
  "Tomato_Early_blight": 0.8712,
  "Tomato_Late_blight": 0.0834,
  "Tomato_healthy": 0.0215,
  "Tomato_Bacterial_spot": 0.0123,
  "Tomato_Leaf_Mold": 0.0116
}
```

**Pipeline**

The request goes through a two-stage hierarchical pipeline:

```
1. [Plant Classifier]  → Identifies which plant (e.g., "Tomato")
2. [Disease Classifier] → Runs the disease model specific to that plant
```

**Possible Disease Classes by Plant**

<details>
<summary><strong>Apple</strong> (4 classes)</summary>

- `Apple___Black_rot`
- `Apple___Cedar_apple_rust`
- `Apple___Apple_scab`
- `Apple___healthy`

</details>

<details>
<summary><strong>Tomato</strong> (10 classes)</summary>

- `Tomato_Bacterial_spot`
- `Tomato_Early_blight`
- `Tomato_Late_blight`
- `Tomato_Leaf_Mold`
- `Tomato_Septoria_leaf_spot`
- `Tomato_Spider_mites_Two_spotted_spider_mite`
- `Tomato__Target_Spot`
- `Tomato__Tomato_YellowLeaf__Curl_Virus`
- `Tomato__Tomato_mosaic_virus`
- `Tomato_healthy`

</details>

<details>
<summary><strong>Potato</strong> (3 classes)</summary>

- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`

</details>

<details>
<summary><strong>Corn (Maize)</strong> (4 classes)</summary>

- `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
- `Corn_(maize)___Common_rust_`
- `Corn_(maize)___Northern_Leaf_Blight`
- `Corn_(maize)___healthy`

</details>

<details>
<summary><strong>Pepper (Bell)</strong> (2 classes)</summary>

- `Pepper__bell___Bacterial_spot`
- `Pepper__bell___healthy`

</details>

**Error Responses**

| Status | Detail | Cause |
|---|---|---|
| `400` | `"File must be an image"` | Uploaded file is not an image type |
| `422` | Validation error | No file provided in the request |

**Example (JavaScript Fetch)**:
```js
const formData = new FormData()
formData.append('file', imageFile)

const res = await fetch('http://localhost:8000/predict_plant_desease', {
  method: 'POST',
  body: formData,
})

if (!res.ok) {
  throw new Error(`API error: ${res.status}`)
}

const diseaseScores = await res.json()
// e.g. { "Tomato_Early_blight": 0.87, "Tomato_healthy": 0.02, ... }

// Get the top prediction:
const topDisease = Object.entries(diseaseScores).sort((a, b) => b[1] - a[1])[0]
// ["Tomato_Early_blight", 0.87]
```

---

## Integration Notes for Frontend

### CORS
The backend currently allows all origins (`*`). No special headers are needed during development.

### Image Format Requirements
- Accepted MIME types: `image/jpeg`, `image/png`, `image/webp`, etc. (any `image/*` type)
- Images are resized and center-cropped to **224×224** internally.

### Reading Results
The response from `/predict_plant_desease` is a flat object of `{ "className": confidence }`. To display results:
1. Sort entries by confidence (descending).
2. The top entry is the predicted disease.
3. Confidence is a float `0.0 – 1.0` — multiply by 100 for a percentage.

### Recommended Helper Function
```ts
// Returns sorted predictions as [{ name, confidence }]
function parsePredictions(data: Record<string, number>) {
  return Object.entries(data)
    .map(([name, confidence]) => ({ name, confidence }))
    .sort((a, b) => b.confidence - a.confidence)
}
```

---

## API Summary Table

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/info` | Model path info |
| `POST` | `/predict_plant_class` | Classify plant species only |
| `POST` | `/predict_plant_desease` | Full hierarchical disease prediction |
