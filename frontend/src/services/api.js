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
   * Get all sessions
   */
  getSessions: () => request("/api/sessions"),

  /**
   * Get details of a specific session
   */
  getSessionDetails: (sessionId) => request(`/api/sessions/${sessionId}`),

  /**
   * Get frames from a specific session
   */
  getSessionFrames: (sessionId) => request(`/api/sessions/${sessionId}/frames`),

  /**
   * Get frame image
   * This returns the URL for the image, not the response
   */
  getFrameImage: (frameId) => `${API_BASE_URL}/api/frames/${frameId}/image`,

  /**
   * Get the latest frame image URL for live feed display
   */
  getLatestFrameUrl: () => {
    // Generate a unique timestamp to prevent browser caching
    const timestamp = Date.now();
    // Return the complete URL that can be directly used in an img src attribute
    return `${API_BASE_URL}/api/latest-frame?t=${timestamp}`;
  },

  /**
   * Get the latest detections from the live feed
   */
  getLatestDetections: () => request("/api/latest-detections"),

  /**
   * Start the live feed with given parameters
   */
  startLiveFeed: (params = {}) =>
    request("/api/control", {
      method: "POST",
      body: JSON.stringify({
        command: "start_live",
        ...params,
      }),
    }),

  /**
   * Stop the live feed
   */
  stopLiveFeed: () =>
    request("/api/control", {
      method: "POST",
      body: JSON.stringify({
        command: "stop_live",
      }),
    }),

  /**
   * Pause the analysis (freeze processing but keep camera active)
   */
  pauseAnalysis: () =>
    request("/api/control", {
      method: "POST",
      body: JSON.stringify({
        command: "pause",
      }),
    }),

  /**
   * Resume the analysis
   */
  resumeAnalysis: () =>
    request("/api/control", {
      method: "POST",
      body: JSON.stringify({
        command: "resume",
      }),
    }),

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
        frame_id: frameId,
        question: question,
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

  /**
   * Check if a camera is available before starting the live feed
   */
  checkCamera: (cameraId = 0) =>
    request("/api/cameras/check", {
      method: "POST",
      body: JSON.stringify({
        camera_id: cameraId,
      }),
    }),
};

export default api;
