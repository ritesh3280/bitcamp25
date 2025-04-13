import React, { useState, useEffect, useRef } from "react";
import {
  FiVideo,
  FiClock,
  FiAlertCircle,
  FiPause,
  FiPlay,
  FiRefreshCw,
  FiInfo,
  FiWifiOff,
  FiPower,
  FiSquare,
} from "react-icons/fi";
import api from "../services/api"; // Import the API service

// Mock data for development - replace with real API data
const MOCK_DETECTIONS = [
  { id: 1, timestamp: "10:45:23", label: "USB Drive", confidence: 0.98 },
  { id: 2, timestamp: "10:45:22", label: "Person", confidence: 0.95 },
  { id: 3, timestamp: "10:45:21", label: "Monitor", confidence: 0.92 },
  { id: 4, timestamp: "10:45:20", label: "Keyboard", confidence: 0.88 },
  { id: 5, timestamp: "10:45:19", label: "Hard Drive", confidence: 0.97 },
];

const FORENSIC_KEYWORDS = ["usb", "drive", "monitor", "storage", "device"];

// API endpoints
// const API_BASE_URL = "http://localhost:5000/api";

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
  const [isLiveFeedRunning, setIsLiveFeedRunning] = useState(false);

  const imageRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());
  const statusIntervalRef = useRef(null);
  const frameIntervalRef = useRef(null);

  // Fetch latest frame via HTTP API
  const fetchLatestFrame = async () => {
    // Only fetch frames if the live feed is running
    if (!isLiveFeedRunning) return;

    try {
      // Generate a URL with a timestamp to prevent caching
      const imageUrl = api.getLatestFrameUrl();
      console.log("Fetching image from:", imageUrl);

      // Create a new image element to force reload
      const img = new Image();

      // Set up onload handler before setting src
      img.onload = () => {
        console.log(
          "Image loaded successfully, dimensions:",
          img.width,
          "x",
          img.height
        );

        // Update the display image
        if (imageRef.current) {
          imageRef.current.src = img.src;

          // Update resolution when we first get an image
          if (!resolution || resolution === "Loading...") {
            setResolution(`${img.width}x${img.height}`);
          }
        }

        // Calculate and update FPS
        frameCountRef.current++;
        const now = Date.now();
        const elapsed = now - lastTimeRef.current;

        if (elapsed >= 1000) {
          // Update FPS every second
          const currentFps = Math.round(
            (frameCountRef.current * 1000) / elapsed
          );
          console.log("FPS calculated:", currentFps);
          setFps(currentFps);
          frameCountRef.current = 0;
          lastTimeRef.current = now;
        }
      };

      img.onerror = (error) => {
        console.error("Failed to load image:", error);
      };

      // Set the src to start loading the image
      img.src = imageUrl;

      // Also fetch latest detections using the API service (less frequently)
      if (frameCountRef.current % 3 === 0) {
        try {
          const detectionsData = await api.getLatestDetections();

          if (detectionsData && detectionsData.length > 0) {
            console.log("Got detections:", detectionsData.length);
            const timestamp = new Date().toLocaleTimeString();

            const newDetections = detectionsData.map((det) => ({
              id: Date.now() + Math.random(),
              timestamp: timestamp,
              label: det.label,
              confidence: det.confidence,
            }));

            setDetections((prev) => {
              const combined = [...newDetections, ...prev];
              return combined.slice(0, 10);
            });
          }
        } catch (detectionsError) {
          console.error("Error fetching latest detections:", detectionsError);
        }
      }
    } catch (error) {
      console.error("Error in fetchLatestFrame:", error);
    }
  };

  // Fetch analysis status from API
  const fetchAnalysisStatus = async () => {
    try {
      const data = await api.getStatus();
      setAnalysisStatus(data);
    } catch (error) {
      console.error("Error fetching analysis status:", error);
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

  // Set up intervals to fetch status and frames
  useEffect(() => {
    // Initial data fetch
    fetchAnalysisStatus();

    // Set up status interval - check every 2 seconds
    statusIntervalRef.current = setInterval(fetchAnalysisStatus, 2000);

    // Check if live feed is already running on mount
    const checkInitialStatus = async () => {
      try {
        const status = await api.getStatus();

        // If there's activity, consider the feed as running
        const isActive = status.frame_count > 0 || status.is_running;
        setIsLiveFeedRunning(isActive);

        // If active, start the frame fetching
        if (isActive && !frameIntervalRef.current) {
          console.log("Feed already active, starting frame fetching");
          frameIntervalRef.current = setInterval(fetchLatestFrame, 50);
        }
      } catch (error) {
        console.error("Error checking initial status:", error);
      }
    };

    checkInitialStatus();

    // Cleanup on unmount
    return () => {
      console.log("Component unmounting, cleaning up intervals");

      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }

      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }

      // Release any object URLs we've created
      if (imageRef.current && imageRef.current.src) {
        URL.revokeObjectURL(imageRef.current.src);
      }
    };
  }, []);

  // Watch for live feed status changes to control frame fetching
  useEffect(() => {
    console.log("Live feed status changed:", isLiveFeedRunning);

    if (isLiveFeedRunning) {
      // Start fetching frames when live feed starts
      if (!frameIntervalRef.current) {
        console.log("Starting frame interval");
        frameIntervalRef.current = setInterval(fetchLatestFrame, 50); // 20 FPS

        // Fetch one frame immediately
        fetchLatestFrame();
      }
    } else {
      // Stop fetching frames when live feed stops
      if (frameIntervalRef.current) {
        console.log("Stopping frame interval");
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
    }
  }, [isLiveFeedRunning]);

  // Start the live feed
  const startLiveFeed = async () => {
    setIsLoading(true);
    try {
      console.log("Starting live feed...");

      // First check if the camera is available
      const cameraCheck = await api.checkCamera(0);
      console.log("Camera check result:", cameraCheck);

      if (!cameraCheck.available) {
        console.error("Camera not available:", cameraCheck.error);
        alert(`Camera not available: ${cameraCheck.error || "Unknown error"}`);
        setIsLoading(false);
        return;
      }

      // Camera is available, start the live feed
      const response = await api.startLiveFeed({ camera_id: 0 });
      console.log("Start live feed response:", response);

      if (response && response.status === "live_feed_started") {
        console.log("Live feed started successfully");
        setIsLiveFeedRunning(true);

        // Update resolution if available from camera check
        if (cameraCheck.resolution) {
          setResolution(cameraCheck.resolution);
        }

        // Ensure we're fetching frames by triggering one fetch immediately
        fetchLatestFrame();

        // Refresh status
        await fetchAnalysisStatus();

        // Force a small delay to ensure the backend is ready
        setTimeout(() => {
          if (!frameIntervalRef.current) {
            console.log("Setting up frame interval after delay");
            frameIntervalRef.current = setInterval(fetchLatestFrame, 50);
          }
        }, 500);
      } else {
        console.error("Failed to start live feed:", response);
        alert("Failed to start live feed. See console for details.");
      }
    } catch (error) {
      console.error("Error starting live feed:", error);
      alert(`Error starting live feed: ${error.message || "Unknown error"}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Stop the live feed
  const stopLiveFeed = async () => {
    setIsLoading(true);
    try {
      console.log("Stopping live feed...");

      // Stop the live feed
      const response = await api.stopLiveFeed();
      console.log("Stop live feed response:", response);

      if (response && response.status === "live_feed_stopped") {
        console.log("Live feed stopped successfully");
        setIsLiveFeedRunning(false);

        // Reset UI state
        setDetections([]); // Clear existing detections
        setFps(0); // Reset FPS counter

        // Clear the image source
        if (imageRef.current) {
          console.log("Clearing image source");
          imageRef.current.src = ""; // Clear the image
        }

        // Make sure video display is stopped
        if (frameIntervalRef.current) {
          console.log("Clearing frame interval");
          clearInterval(frameIntervalRef.current);
          frameIntervalRef.current = null;
        }

        // Refresh status to make sure UI reflects correct state
        await fetchAnalysisStatus();
      } else {
        console.error("Failed to stop live feed:", response);
        alert("Failed to stop live feed. See console for details.");
      }
    } catch (error) {
      console.error("Error stopping live feed:", error);
      alert(`Error stopping live feed: ${error.message || "Unknown error"}`);
    } finally {
      setIsLoading(false);
    }
  };

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
                  {analysisStatus.is_running ? "Running" : "Stopped"}
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
            {/* Start/Stop Live Feed Buttons */}
            {isLiveFeedRunning ? (
              <button
                onClick={stopLiveFeed}
                disabled={isLoading}
                className="px-3 py-1 rounded-md text-sm flex items-center gap-1 bg-red-50 text-red-700 hover:bg-red-100 transition-colors"
              >
                {isLoading ? (
                  <FiRefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <>
                    <FiSquare className="w-4 h-4" />
                    <span>Stop Feed</span>
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={startLiveFeed}
                disabled={isLoading}
                className="px-3 py-1 rounded-md text-sm flex items-center gap-1 bg-blue-50 text-blue-700 hover:bg-blue-100 transition-colors"
              >
                {isLoading ? (
                  <FiRefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <>
                    <FiPower className="w-4 h-4" />
                    <span>Start Feed</span>
                  </>
                )}
              </button>
            )}
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
                    <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                    <span className="text-xs font-medium text-yellow-500">
                      LIVE (HTTP API)
                    </span>
                    <FiWifiOff className="w-3 h-3 text-gray-400" />
                  </div>
                </div>
              </div>
            </div>

            {/* Video Player */}
            <div className="aspect-video bg-gray-900 relative">
              <img
                ref={imageRef}
                alt="YOLO Detection Feed"
                className="w-full h-full object-contain"
                style={{
                  maxWidth: "100%",
                  maxHeight: "100%",
                  display: isLiveFeedRunning ? "block" : "none",
                }}
                onLoad={(e) => {
                  // Update resolution when first frame is loaded
                  if (!resolution || resolution === "Loading...") {
                    setResolution(
                      `${e.target.naturalWidth}x${e.target.naturalHeight}`
                    );
                  }
                  // Log successful load
                  console.log("Image displayed successfully");
                }}
                onError={(e) => {
                  console.error("Error displaying image:", e);
                }}
              />
              {!isLiveFeedRunning && (
                <div className="absolute inset-0 flex items-center justify-center text-white">
                  <p>Live feed is not active. Click "Start Feed" to begin.</p>
                </div>
              )}

              {/* Video Controls Overlay */}
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/50 to-transparent p-4">
                <div className="flex items-center justify-between text-white">
                  <div className="flex items-center gap-4">
                    {/* Replace play/pause button with status indicator */}
                    <div className="flex items-center gap-2">
                      {isLiveFeedRunning ? (
                        <>
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                          <span className="text-sm">Live Feed Active</span>
                        </>
                      ) : (
                        <>
                          <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                          <span className="text-sm">Feed Inactive</span>
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <span className="px-2 py-1 bg-white/20 rounded">
                      {resolution}
                    </span>
                    <span className="px-2 py-1 bg-white/20 rounded">
                      {fps} FPS
                    </span>
                    <span className="px-2 py-1 bg-yellow-500/70 rounded text-xs">
                      HTTP API (20 FPS)
                    </span>
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
