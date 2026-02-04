# API Documentation

## Authentication

All API requests require an API key in the header:
```
x-api-key: YOUR_API_KEY
```

## Endpoints

### 1. Health Check

**GET** `/health`

Response:
```json
{
  "status": "healthy",
  "model_type": "hybrid",
  "device": "cuda"
}
```

### 2. Voice Detection

**POST** `/api/voice-detection`

Request:
```json
{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "base64_encoded_audio..."
}
```

Response:
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Deep learning analysis detected synthetic voice patterns"
}
```

## Error Responses

### 401 Unauthorized
```json
{
  "status": "error",
  "message": "Invalid API key"
}
```

### 400 Bad Request
```json
{
  "status": "error",
  "message": "Audio processing error: Invalid base64 encoding"
}
```

## Rate Limits

- 100 requests per minute per API key
- Maximum audio size: 10MB