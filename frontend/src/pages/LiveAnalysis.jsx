import React, { useState, useEffect, useRef } from "react";
import {
  FiVideo,
  FiClock,
  FiAlertCircle,
  FiPause,
  FiPlay,
  FiRefreshCw,
  FiInfo,
  FiWifi,
  FiWifiOff,
} from "react-icons/fi";

// Mock data for development - replace with real WebSocket or API data
const MOCK_DETECTIONS = [
  { id: 1, timestamp: "10:45:23", label: "USB Drive", confidence: 0.98 },
  { id: 2, timestamp: "10:45:22", label: "Person", confidence: 0.95 },
  { id: 3, timestamp: "10:45:21", label: "Monitor", confidence: 0.92 },
  { id: 4, timestamp: "10:45:20", label: "Keyboard", confidence: 0.88 },
  { id: 5, timestamp: "10:45:19", label: "Hard Drive", confidence: 0.97 },
];

const FORENSIC_KEYWORDS = ["usb", "drive", "monitor", "storage", "device"];

// API endpoints
const API_BASE_URL = "http://localhost:5000/api";

const DetectionItem = ({ timestamp, label, confidence }) => {
  const isForensicItem = FORENSIC_KEYWORDS.some((keyword) =>
    label.toLowerCase().includes(keyword)
  );

  return (
    <div
      className={`
        p-4 rounded-lg border transition-all duration-200 hover:shadow-md
        ${
          isForensicItem
            ? "bg-red-50 border-red-100"
            : "bg-white border-gray-100"
        }
      `}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={`
            p-2 rounded-lg
            ${isForensicItem ? "bg-red-100" : "bg-gray-100"}
          `}
          >
            <FiAlertCircle
              className={`w-4 h-4 ${
                isForensicItem ? "text-red-600" : "text-gray-600"
              }`}
            />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900">{label}</p>
            <div className="flex items-center gap-2">
              <FiClock className="w-3 h-3 text-gray-400" />
              <p className="text-xs text-gray-500">{timestamp}</p>
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm font-medium text-gray-900">
            {(confidence * 100).toFixed(1)}%
          </div>
          <p className="text-xs text-gray-500">confidence</p>
        </div>
      </div>
    </div>
  );
};

const LiveAnalysis = () => {
  const [detections, setDetections] = useState(MOCK_DETECTIONS);
  const [isVideoActive, setIsVideoActive] = useState(true);
  const [isConnected, setIsConnected] = useState(false);
  const [usingFallback, setUsingFallback] = useState(false);
  const [fps, setFps] = useState(0);
  const [resolution, setResolution] = useState("Loading...");
  const [analysisStatus, setAnalysisStatus] = useState({
    is_running: false,
    frame_count: 0,
    unique_frames: 0,
    elapsed_time: 0,
    current_detections: 0,
  });
  const [isLoading, setIsLoading] = useState(false);

  const imageRef = useRef(null);
  const wsRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const statusIntervalRef = useRef(null);
  const fallbackIntervalRef = useRef(null);

  // Fetch latest frame via HTTP API (fallback method)
  const fetchLatestFrameViaHttp = async () => {
    if (!isVideoActive) return;

    try {
      // Fetch latest frame as a blob
      const response = await fetch(`${API_BASE_URL}/latest-frame`);
      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        if (imageRef.current) {
          imageRef.current.src = imageUrl;

          // Update resolution when first frame is received
          if (!resolution || resolution === "Loading...") {
            imageRef.current.onload = () => {
              if (imageRef.current) {
                setResolution(
                  `${imageRef.current.naturalWidth}x${imageRef.current.naturalHeight}`
                );
              }
            };
          }

          // Calculate and update FPS
          frameCountRef.current++;
          const now = Date.now();
          const elapsed = now - lastTimeRef.current;

          if (elapsed >= 1000) {
            // Update FPS every second
            setFps(Math.round((frameCountRef.current * 1000) / elapsed));
            frameCountRef.current = 0;
            lastTimeRef.current = now;
          }
        }

        // Also fetch latest detections
        const detectionsResponse = await fetch(
          `${API_BASE_URL}/latest-detections`
        );
        if (detectionsResponse.ok) {
          const detectionsData = await detectionsResponse.json();

          if (detectionsData && detectionsData.length > 0) {
            const timestamp = new Date().toLocaleTimeString();

            const newDetections = detectionsData.map((det) => ({
              id: Date.now() + Math.random(), // Generate unique ID
              timestamp: timestamp,
              label: det.label,
              confidence: det.confidence,
            }));

            setDetections((prev) => {
              // Combine new detections with previous ones, limit to 10
              const combined = [...newDetections, ...prev];
              return combined.slice(0, 10);
            });
          }
        }
      }
    } catch (error) {
      console.error("Error fetching latest frame via HTTP:", error);
    }
  };

  // Fetch analysis status from API
  const fetchAnalysisStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      if (response.ok) {
        const data = await response.json();
        setAnalysisStatus(data);
      }
    } catch (error) {
      console.error("Error fetching analysis status:", error);
    }
  };

  // Control analysis (pause/resume)
  const controlAnalysis = async (command) => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/control`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ command }),
      });

      if (response.ok) {
        // Refresh status after control
        await fetchAnalysisStatus();
      } else {
        console.error("Error controlling analysis:", await response.text());
      }
    } catch (error) {
      console.error("Error controlling analysis:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs
      .toString()
      .padStart(2, "0")}`;
  };

  // Periodically fetch analysis status
  useEffect(() => {
    fetchAnalysisStatus();

    // Set up interval to fetch status every 2 seconds
    statusIntervalRef.current = setInterval(fetchAnalysisStatus, 2000);

    return () => {
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, []);

  // Connect to WebSocket server
  useEffect(() => {
    let reconnectTimer = null;
    let reconnectAttempts = 0;
    const MAX_RECONNECT_ATTEMPTS = 5; // Reduced from 10 to switch to fallback mode faster
    const INITIAL_RECONNECT_DELAY = 1000;
    const MAX_MESSAGE_ERRORS = 3;
    let messageErrorCount = 0;

    // WebSocket setup
    const connectWebSocket = () => {
      try {
        console.log("Attempting to connect to WebSocket server...");

        // Close existing connection if any
        if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) {
          wsRef.current.close();
        }

        // Create new WebSocket connection
        const ws = new WebSocket("ws://localhost:8765");
        wsRef.current = ws;

        // Connection opened
        ws.onopen = () => {
          console.log("WebSocket connection established");
          setIsConnected(true);
          setUsingFallback(false);
          reconnectAttempts = 0; // Reset reconnect attempts on successful connection
          messageErrorCount = 0; // Reset message error count

          // Clear fallback interval if it exists
          if (fallbackIntervalRef.current) {
            clearInterval(fallbackIntervalRef.current);
            fallbackIntervalRef.current = null;
          }

          // Send a ping every 20 seconds to keep the connection alive
          const pingInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
              try {
                ws.send(JSON.stringify({ type: "ping" }));
              } catch (err) {
                console.warn("Failed to send ping:", err);
                clearInterval(pingInterval);
              }
            } else {
              clearInterval(pingInterval);
            }
          }, 20000);

          // Store the interval so we can clear it on cleanup
          ws.pingInterval = pingInterval;
        };

        // Connection closed
        ws.onclose = (event) => {
          console.log(
            `WebSocket connection closed: Code: ${event.code}, Reason: ${
              event.reason || "No reason provided"
            }`
          );
          setIsConnected(false);

          // Clear ping interval if it exists
          if (ws.pingInterval) {
            clearInterval(ws.pingInterval);
          }

          // Start fallback method after a few failed attempts or immediately if code 1011 (server error)
          const shouldUseFallback =
            reconnectAttempts >= 2 || event.code === 1011;

          if (shouldUseFallback && !fallbackIntervalRef.current) {
            console.log("Starting fallback HTTP method for frame updates");
            setUsingFallback(true);
            fallbackIntervalRef.current = setInterval(
              fetchLatestFrameViaHttp,
              500
            );
          }

          // Only attempt to reconnect if we haven't exceeded the max attempts
          if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
            const delay = Math.min(
              INITIAL_RECONNECT_DELAY * Math.pow(1.5, reconnectAttempts),
              10000 // Max 10 seconds between retries
            );
            console.log(
              `Attempting to reconnect in ${delay / 1000} seconds...`
            );

            reconnectTimer = setTimeout(() => {
              reconnectAttempts++;
              connectWebSocket();
            }, delay);
          } else {
            console.warn(
              "Max reconnection attempts reached. Using HTTP fallback only."
            );
            // Ensure fallback is active
            if (!fallbackIntervalRef.current) {
              setUsingFallback(true);
              fallbackIntervalRef.current = setInterval(
                fetchLatestFrameViaHttp,
                500
              );
            }
          }
        };

        // Error handling
        ws.onerror = (error) => {
          // Just log the error - onclose will be called after this
          console.error("WebSocket error occurred");
        };

        // Message handling
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            // Handle connection confirmation message
            if (data.status === "connected") {
              console.log("Server connection confirmed:", data.message);
              return;
            }

            // Update the image with the received frame
            if (data.frame && imageRef.current) {
              imageRef.current.src = `data:image/jpeg;base64,${data.frame}`;

              // Update resolution when first frame is received
              if (!resolution || resolution === "Loading...") {
                imageRef.current.onload = () => {
                  if (imageRef.current) {
                    setResolution(
                      `${imageRef.current.naturalWidth}x${imageRef.current.naturalHeight}`
                    );
                  }
                };
              }

              // Calculate and update FPS
              frameCountRef.current++;
              const now = Date.now();
              const elapsed = now - lastTimeRef.current;

              if (elapsed >= 1000) {
                // Update FPS every second
                setFps(Math.round((frameCountRef.current * 1000) / elapsed));
                frameCountRef.current = 0;
                lastTimeRef.current = now;
              }
            }

            // Update detections list
            if (data.detections && data.detections.length > 0) {
              const timestamp = new Date().toLocaleTimeString();

              const newDetections = data.detections.map((det) => ({
                id: Date.now() + Math.random(), // Generate unique ID
                timestamp: timestamp,
                label: det.label,
                confidence: det.confidence,
              }));

              setDetections((prev) => {
                // Combine new detections with previous ones, limit to 10
                const combined = [...newDetections, ...prev];
                return combined.slice(0, 10);
              });
            }

            // Reset error count on successful message processing
            messageErrorCount = 0;
          } catch (error) {
            console.error("Error parsing WebSocket message:", error);
            messageErrorCount++;

            // If we have too many message errors in a row, close and reconnect
            if (
              messageErrorCount >= MAX_MESSAGE_ERRORS &&
              ws.readyState === WebSocket.OPEN
            ) {
              console.warn("Too many message errors, reconnecting...");
              ws.close(3000, "Too many message errors");
            }
          }
        };
      } catch (error) {
        console.error("Error setting up WebSocket:", error);
        setIsConnected(false);

        // Start fallback method after a few failed attempts
        if (reconnectAttempts >= 2 && !fallbackIntervalRef.current) {
          console.log(
            "Starting fallback HTTP method for frame updates due to setup error"
          );
          setUsingFallback(true);
          fallbackIntervalRef.current = setInterval(
            fetchLatestFrameViaHttp,
            500
          );
        }

        // Attempt to reconnect on error
        if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
          const delay = Math.min(
            INITIAL_RECONNECT_DELAY * Math.pow(1.5, reconnectAttempts),
            10000 // Max 10 seconds between retries
          );
          reconnectTimer = setTimeout(() => {
            reconnectAttempts++;
            connectWebSocket();
          }, delay);
        } else if (!fallbackIntervalRef.current) {
          // Ensure fallback is active
          setUsingFallback(true);
          fallbackIntervalRef.current = setInterval(
            fetchLatestFrameViaHttp,
            500
          );
        }
      }
    };

    // Initial connection
    connectWebSocket();

    // Clean up WebSocket connection and timers
    return () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }

      if (fallbackIntervalRef.current) {
        clearInterval(fallbackIntervalRef.current);
        fallbackIntervalRef.current = null;
      }

      if (wsRef.current) {
        // Clear ping interval if it exists
        if (wsRef.current.pingInterval) {
          clearInterval(wsRef.current.pingInterval);
        }

        try {
          if (wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.close();
          }
        } catch (err) {
          console.warn("Error closing WebSocket:", err);
        }
        wsRef.current = null;
      }
    };
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          Live Forensics Feed
        </h1>
        <p className="mt-1 text-gray-500">
          Real-time object detection and forensic analysis
        </p>
      </div>

      {/* Analysis Status Bar */}
      <div className="bg-white shadow rounded-lg p-4">
        <div className="flex flex-wrap items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center gap-2">
              <div
                className={`p-2 rounded-full ${
                  analysisStatus.is_running ? "bg-green-100" : "bg-yellow-100"
                }`}
              >
                {analysisStatus.is_running ? (
                  <FiRefreshCw className="w-5 h-5 text-green-600 animate-spin" />
                ) : (
                  <FiInfo className="w-5 h-5 text-yellow-600" />
                )}
              </div>
              <div>
                <h3 className="text-sm font-medium">Analysis Status</h3>
                <p className="text-xs text-gray-500">
                  {analysisStatus.is_running ? "Running" : "Paused"}
                </p>
              </div>
            </div>

            <div>
              <h3 className="text-sm font-medium">Total Frames</h3>
              <p className="text-xs text-gray-500">
                {analysisStatus.frame_count}
              </p>
            </div>

            <div>
              <h3 className="text-sm font-medium">Unique Keyframes</h3>
              <p className="text-xs text-gray-500">
                {analysisStatus.unique_frames}
              </p>
            </div>

            <div>
              <h3 className="text-sm font-medium">Elapsed Time</h3>
              <p className="text-xs text-gray-500">
                {formatTime(analysisStatus.elapsed_time)}
              </p>
            </div>
          </div>

          <div className="flex space-x-2">
            <button
              onClick={() =>
                controlAnalysis(analysisStatus.is_running ? "pause" : "resume")
              }
              disabled={isLoading}
              className={`px-3 py-1 rounded-md text-sm flex items-center gap-1 ${
                analysisStatus.is_running
                  ? "bg-yellow-50 text-yellow-700 hover:bg-yellow-100"
                  : "bg-green-50 text-green-700 hover:bg-green-100"
              } transition-colors`}
            >
              {isLoading ? (
                <FiRefreshCw className="w-4 h-4 animate-spin" />
              ) : analysisStatus.is_running ? (
                <>
                  <FiPause className="w-4 h-4" />
                  <span>Pause</span>
                </>
              ) : (
                <>
                  <FiPlay className="w-4 h-4" />
                  <span>Resume</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Video Feed Section */}
        <div className="xl:col-span-2">
          <div className="bg-white rounded-lg border border-gray-100 overflow-hidden">
            <div className="p-4 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-50 rounded-lg">
                    <FiVideo className="w-5 h-5 text-blue-600" />
                  </div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    Primary Camera Feed
                  </h2>
                </div>
                <div className="flex items-center gap-2">
                  <div className="flex items-center gap-1">
                    <div
                      className={`w-2 h-2 ${
                        isConnected
                          ? "bg-red-500 animate-pulse"
                          : usingFallback
                          ? "bg-yellow-500"
                          : "bg-gray-400"
                      } rounded-full`}
                    ></div>
                    <span
                      className={`text-xs font-medium ${
                        isConnected
                          ? "text-red-500"
                          : usingFallback
                          ? "text-yellow-500"
                          : "text-gray-400"
                      }`}
                    >
                      {isConnected
                        ? "LIVE (WebSocket)"
                        : usingFallback
                        ? "LIVE (HTTP Fallback)"
                        : "DISCONNECTED"}
                    </span>
                    {isConnected ? (
                      <FiWifi className="w-3 h-3 text-green-500" />
                    ) : (
                      <FiWifiOff className="w-3 h-3 text-gray-400" />
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Video Player - Replaced with WebSocket stream */}
            <div className="aspect-video bg-gray-900 relative">
              <img
                ref={imageRef}
                alt="YOLO Detection Feed"
                className="w-full h-full object-contain"
                style={{ display: isVideoActive ? "block" : "none" }}
              />
              {!isVideoActive && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <p>Video paused</p>
                </div>
              )}

              {/* Video Controls Overlay */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-4">
                <div className="flex items-center justify-between text-white">
                  <div className="flex items-center gap-4">
                    <button
                      onClick={() => setIsVideoActive(!isVideoActive)}
                      className="p-2 hover:bg-white/20 rounded-full transition-colors"
                    >
                      {isVideoActive ? (
                        <>
                          <FiPause className="w-4 h-4" /> Pause
                        </>
                      ) : (
                        <>
                          <FiPlay className="w-4 h-4" /> Play
                        </>
                      )}
                    </button>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="px-2 py-1 bg-white/20 rounded">
                      {resolution}
                    </span>
                    <span className="px-2 py-1 bg-white/20 rounded">
                      {fps} FPS
                    </span>
                    {usingFallback && (
                      <span className="px-2 py-1 bg-yellow-500/70 rounded text-xs">
                        Fallback Mode
                      </span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Detection Feed Section */}
        <div className="xl:col-span-1">
          <div className="bg-white rounded-lg border border-gray-100 p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                Detection Feed
              </h3>
              <span className="text-sm text-gray-500">Last 10 detections</span>
            </div>

            <div className="space-y-3 max-h-[600px] overflow-y-auto">
              {detections.length > 0 ? (
                detections.map((detection) => (
                  <DetectionItem key={detection.id} {...detection} />
                ))
              ) : (
                <div className="p-4 text-center text-gray-500">
                  No detections yet
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveAnalysis;
