/**
 * API Service
 *
 * This service handles all communication with the backend API.
 */

// Base API URL - configurable via environment variable
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

/**
 * Generic request handler with error handling
 */
async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;

  // Default headers
  const headers = {
    "Content-Type": "application/json",
    ...options.headers,
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers,
    });

    // Handle non-JSON responses
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const data = await response.json();

      // Check for error responses
      if (!response.ok) {
        throw new Error(data.error || "An unexpected error occurred");
      }

      return data;
    } else {
      // Handle non-JSON responses like images
      if (!response.ok) {
        throw new Error("An unexpected error occurred");
      }

      return response;
    }
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);
    throw error;
  }
}

/**
 * API Service Object
 */
const api = {
  /**
   * Check API status
   */
  getStatus: () => request("/api/status"),

  /**
   * Get all frames
   */
  getFrames: () => request("/api/frames"),

  /**
   * Get a specific frame by ID
   */
  getFrame: (frameId) => request(`/api/frames/${frameId}`),

  /**
   * Ask a question about a specific frame
   */
  askQuestion: (frameId, question) =>
    request("/api/ask", {
      method: "POST",
      body: JSON.stringify({
        frameId,
        question,
      }),
    }),

  /**
   * Get list of available cameras
   */
  getCameras: () => request("/api/cameras"),

  /**
   * Start a recording session
   */
  startRecording: (cameraId, options = {}) =>
    request("/api/recording/start", {
      method: "POST",
      body: JSON.stringify({
        cameraId,
        ...options,
      }),
    }),

  /**
   * Proxy a request to avoid CORS issues
   */
  proxy: (url, method = "GET", data = null) => {
    const options = {
      method,
    };

    if (data && (method === "POST" || method === "PUT")) {
      options.body = JSON.stringify(data);
    }

    return request(`/proxy/${url}`, options);
  },
};

export default api;
