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

  // Add CORS mode
  const requestOptions = {
    ...options,
    headers,
    mode: "cors", // Explicitly set CORS mode
    credentials: "omit", // Changed from "same-origin" to "omit" for simpler CORS
  };

  try {
    console.log(
      `Fetching ${url} with method: ${requestOptions.method || "GET"}`
    );

    const response = await fetch(url, requestOptions);

    // Handle non-JSON responses
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const data = await response.json();

      // Check for error responses
      if (!response.ok) {
        console.error(
          `API returned error status ${response.status}: ${
            data.error || "Unknown error"
          }`
        );
        throw new Error(data.error || `API error: ${response.status}`);
      }

      return data;
    } else {
      // Handle non-JSON responses like images
      if (!response.ok) {
        console.error(`API returned error status ${response.status}`);
        throw new Error(`API error: ${response.status}`);
      }

      return response;
    }
  } catch (error) {
    console.error(`API Error (${endpoint}):`, error);

    // Enhanced error logging
    if (error.name === "TypeError" && error.message === "Failed to fetch") {
      console.error(
        "Network error - this could be due to CORS policy or server unavailability"
      );
      console.error("Check that backend server is running at:", API_BASE_URL);
    }

    throw error;
  }
}

/**
 * API Service Object
 */
const api = {
  /**
   * Simple ping to check if the backend is available
   */
  ping: () => request("/api/ping"),

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
  askQuestion: async (frameId, question, conversationHistory = []) => {
    try {
      // Try the standard request approach first
      return await request("/api/ask", {
        method: "POST",
        body: JSON.stringify({
          frame_id: frameId,
          question: question,
          conversation_history: conversationHistory,
        }),
      });
    } catch (error) {
      console.warn(
        "Standard request to /api/ask failed, trying direct fetch as fallback:",
        error
      );

      // Direct fetch as fallback if the request function fails (e.g., due to CORS)
      const response = await fetch(`${API_BASE_URL}/api/ask`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          frame_id: frameId,
          question: question,
          conversation_history: conversationHistory,
        }),
        mode: "cors",
        credentials: "omit",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `API error: ${response.status}`);
      }

      return await response.json();
    }
  },

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
   * Chat with the AI about a specific frame
   * Uses a simplified endpoint to avoid CORS issues
   */
  chatWithFrame: async (frameId, question, conversationHistory = []) => {
    console.log(
      `Chatting about frame ${frameId}, direct fetch to /chat endpoint`
    );

    try {
      // Direct fetch implementation to avoid complex CORS handling
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          frame_id: frameId,
          question: question,
          conversation_history: conversationHistory,
        }),
        mode: "cors",
        credentials: "omit",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage =
          errorData.error || `Server error: ${response.status}`;
        console.error(`Chat error: ${errorMessage}`);
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error("Chat endpoint error:", error);
      throw error;
    }
  },

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

  /**
   * Perform a semantic search on frames within a session
   */
  semanticFrameSearch: async (sessionId, query) => {
    console.log(
      `Performing semantic search in session ${sessionId} with query: "${query}"`
    );

    try {
      // Direct fetch implementation to avoid complex CORS handling
      const response = await fetch(
        `${API_BASE_URL}/api/frames/semantic-search`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            session_id: sessionId,
            query: query,
          }),
          mode: "cors",
          credentials: "omit",
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage =
          errorData.error || `Server error: ${response.status}`;
        console.error(`Semantic search error: ${errorMessage}`);
        throw new Error(errorMessage);
      }

      return await response.json();
    } catch (error) {
      console.error("Semantic search endpoint error:", error);
      throw error;
    }
  },
};

export default api;
