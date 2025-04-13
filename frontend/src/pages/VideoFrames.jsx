import React, { useState, useRef, useEffect } from "react";
import {
  FiSearch,
  FiImage,
  FiMessageSquare,
  FiX,
  FiClock,
  FiFilter,
  FiSend,
} from "react-icons/fi";
import api from "../services/api";

// Mock data for development
const MOCK_FRAMES = [
  {
    id: "frame_001",
    timestamp: "2024-03-24 10:45:23",
    thumbnail: "https://placehold.co/400x225",
    detections: ["Person", "Laptop", "Chair"],
    confidence: 0.95,
  },
  {
    id: "frame_002",
    timestamp: "2024-03-24 10:45:24",
    thumbnail: "https://placehold.co/400x225",
    detections: ["Person", "Phone", "Table"],
    confidence: 0.92,
  },
  {
    id: "frame_003",
    timestamp: "2024-03-24 10:45:25",
    thumbnail: "https://placehold.co/400x225",
    detections: ["Person", "Door", "Bag"],
    confidence: 0.88,
  },
];

const FrameModal = ({ frame, onClose }) => {
  if (!frame) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
        <div className="p-4 border-b border-gray-100 flex justify-between items-center">
          <h3 className="text-lg font-semibold text-gray-900">Frame Details</h3>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <FiX className="w-5 h-5 text-gray-500" />
          </button>
        </div>
        <div className="p-4">
          <img
            src={frame.thumbnail}
            alt={`Frame ${frame.id}`}
            className="w-full h-auto rounded-lg"
          />
          <div className="mt-4 space-y-2">
            <p className="text-sm text-gray-500">
              <span className="font-medium text-gray-900">Frame ID:</span>{" "}
              {frame.id}
            </p>
            <p className="text-sm text-gray-500">
              <span className="font-medium text-gray-900">Timestamp:</span>{" "}
              {frame.timestamp}
            </p>
            <p className="text-sm text-gray-500">
              <span className="font-medium text-gray-900">Detections:</span>{" "}
              {frame.detections.join(", ")}
            </p>
            <p className="text-sm text-gray-500">
              <span className="font-medium text-gray-900">Confidence:</span>{" "}
              {(frame.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const ChatMessage = ({ isUser, content, timestamp }) => (
  <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
    <div className={`max-w-[85%] ${isUser ? "order-1" : "order-none"}`}>
      <div
        className={`
          px-4 py-2.5 
          ${
            isUser
              ? "bg-indigo-600 text-white rounded-2xl rounded-tr-sm"
              : "bg-gray-100 text-gray-900 rounded-2xl rounded-tl-sm"
          }
          shadow-sm
        `}
      >
        <p className="text-sm leading-relaxed">{content}</p>
      </div>
      <div
        className={`flex items-center gap-1 mt-1 ${
          isUser ? "justify-end" : "justify-start"
        }`}
      >
        <FiClock className="w-3 h-3 text-gray-400" />
        <p className="text-xs text-gray-400">{timestamp}</p>
      </div>
    </div>
  </div>
);

const AskPopup = ({ frame, onClose, triggerRect }) => {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      content:
        "Hi! I can help you analyze this frame. What would you like to know?",
      isUser: false,
      timestamp: "Just now",
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const popupRef = useRef(null);
  const chatHistoryRef = useRef(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });

  useEffect(() => {
    if (triggerRect && popupRef.current) {
      const popup = popupRef.current;
      const popupRect = popup.getBoundingClientRect();
      const viewportHeight = window.innerHeight;
      const viewportWidth = window.innerWidth;

      // Try to position below the frame first
      let top = triggerRect.bottom + 16;
      let left = triggerRect.left;

      // If it would go off the bottom, position it above
      if (top + 500 > viewportHeight - 20) {
        top = Math.max(20, viewportHeight - 520);
      }

      // If it would go off the right, align to right edge with margin
      if (left + 400 > viewportWidth - 20) {
        left = viewportWidth - 420;
      }

      setPosition({ top, left });
    }
  }, [triggerRect]);

  const handleClickOutside = (event) => {
    if (popupRef.current && !popupRef.current.contains(event.target)) {
      onClose();
    }
  };

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  const handleSendMessage = async () => {
    if (!question.trim() || isLoading) return;

    // Add user message
    const userMessage = {
      content: question,
      isUser: true,
      timestamp: "Just now",
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setQuestion("");
    setIsLoading(true);

    try {
      // Send the question to the API
      const response = await api.askQuestion(frame.id, question);

      // Add AI response
      setMessages([
        ...newMessages,
        {
          content: response.answer,
          isUser: false,
          timestamp: "Just now",
        },
      ]);
    } catch (error) {
      // Handle error
      setMessages([
        ...newMessages,
        {
          content:
            "Sorry, I'm having trouble processing your question. Please try again later.",
          isUser: false,
          timestamp: "Just now",
        },
      ]);
      console.error("Error asking question:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  useEffect(() => {
    // Scroll to bottom when messages change
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages]);

  if (!frame) return null;

  return (
    <div
      ref={popupRef}
      style={{
        top: `${position.top}px`,
        left: `${position.left}px`,
      }}
      className="
        fixed z-50 
        w-[400px] h-[500px] 
        bg-gray-50
        rounded-xl 
        shadow-lg shadow-gray-200/80
        border border-gray-200
        flex flex-col
        backdrop-blur-sm
        backdrop-saturate-150
      "
    >
      {/* Header */}
      <div
        className="
        px-4 py-3 
        bg-white 
        border-b border-gray-100 
        flex items-center justify-between 
        shrink-0
        rounded-t-xl
      "
      >
        <div className="flex items-center gap-3">
          <div className="relative">
            <img
              src={frame.thumbnail}
              alt={`Frame ${frame.id}`}
              className="w-10 h-10 object-cover rounded-lg shadow-sm"
            />
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-indigo-600 rounded-full flex items-center justify-center">
              <FiMessageSquare className="w-2.5 h-2.5 text-white" />
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-900">
              Frame Analysis
            </h3>
            <p className="text-xs text-gray-500 flex items-center gap-1">
              <span>{frame.id}</span>
              <span className="block w-1 h-1 rounded-full bg-gray-300" />
              <span>Active</span>
            </p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="
            p-1.5 
            text-gray-400 hover:text-gray-600
            hover:bg-gray-100 
            rounded-lg 
            transition-colors
          "
        >
          <FiX className="w-4 h-4" />
        </button>
      </div>

      {/* Chat History */}
      <div
        ref={chatHistoryRef}
        className="
          flex-1 
          overflow-y-auto 
          px-4 py-6
          space-y-4
          bg-gradient-to-b from-gray-50/50 to-white
        "
      >
        {messages.map((message, index) => (
          <ChatMessage key={index} {...message} />
        ))}

        {isLoading && (
          <div className="flex justify-start mb-3">
            <div className="max-w-[85%]">
              <div className="bg-gray-100 text-gray-900 rounded-2xl rounded-tl-sm shadow-sm px-4 py-2.5">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.4s" }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div
        className="
        p-4 
        border-t border-gray-100 
        bg-white
        rounded-b-xl
        shrink-0
      "
      >
        <div className="flex items-end gap-2">
          <div className="relative flex-1">
            <textarea
              rows={2}
              placeholder="Ask about this frame..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              className="
                w-full 
                resize-none 
                px-4 py-3 
                text-sm 
                text-gray-900
                placeholder-gray-400
                bg-gray-50
                border border-gray-200 
                rounded-xl
                focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500
                transition-shadow
                disabled:opacity-70
              "
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!question.trim() || isLoading}
            className="
              inline-flex items-center justify-center 
              p-3
              text-white 
              bg-indigo-600 hover:bg-indigo-700
              rounded-xl
              shadow-sm
              transition-all
              hover:shadow
              disabled:opacity-50
              disabled:cursor-not-allowed
            "
          >
            <FiSend className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

const VideoFrames = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [askFrame, setAskFrame] = useState(null);
  const [askButtonRect, setAskButtonRect] = useState(null);
  const [frames, setFrames] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch frames from the API
  useEffect(() => {
    const fetchFrames = async () => {
      try {
        setIsLoading(true);
        const response = await api.getFrames();
        setFrames(response.frames);
      } catch (error) {
        console.error("Error fetching frames:", error);
        setError("Failed to load frames. Please try again later.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchFrames();
  }, []);

  const handleAskClick = (frame, event) => {
    event.stopPropagation(); // Prevent event bubbling
    const button = event.currentTarget;
    const rect = button.getBoundingClientRect();
    setAskButtonRect(rect);
    setAskFrame(frame);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-2xl font-bold text-gray-900">Video Frames</h1>
          <p className="mt-1 text-gray-500">
            Review and analyze detected video frames
          </p>
        </div>

        {/* Search and Filters */}
        <div className="mb-6 flex gap-4">
          <div className="flex-1 relative">
            <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search frames..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <button className="px-4 py-2 text-gray-700 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 flex items-center gap-2">
            <FiFilter className="w-5 h-5" />
            Filters
          </button>
        </div>

        {/* Frames List */}
        <div className="space-y-3">
          {isLoading ? (
            <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mb-4"></div>
              <p className="text-gray-500">Loading frames...</p>
            </div>
          ) : error ? (
            <div className="bg-white rounded-lg border border-red-200 p-8 text-center">
              <p className="text-red-500">{error}</p>
              <button
                onClick={() => window.location.reload()}
                className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg"
              >
                Retry
              </button>
            </div>
          ) : frames.length === 0 ? (
            <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
              <p className="text-gray-500">No frames found</p>
            </div>
          ) : (
            frames.map((frame) => (
              <div
                key={frame.id}
                className="bg-white rounded-lg border border-gray-200 hover:shadow-sm transition-shadow duration-200"
              >
                <div className="flex items-center p-3 gap-4">
                  {/* Thumbnail */}
                  <div className="w-40 flex-shrink-0">
                    <img
                      src={frame.thumbnail}
                      alt={`Frame ${frame.id}`}
                      className="w-full h-24 object-cover rounded-lg"
                    />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                      <div>
                        <h3 className="text-sm font-medium text-gray-900">
                          {frame.id}
                        </h3>
                        <div className="mt-1 flex items-center gap-1.5 text-sm text-gray-500">
                          <FiClock className="w-3.5 h-3.5" />
                          {frame.timestamp}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => setSelectedFrame(frame)}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-50 rounded-md hover:bg-gray-100 transition-colors"
                        >
                          <FiImage className="w-4 h-4" />
                          Show Frame
                        </button>
                        <button
                          onClick={(e) => handleAskClick(frame, e)}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-white bg-indigo-600 rounded-md hover:bg-indigo-700 transition-colors"
                        >
                          <FiMessageSquare className="w-4 h-4" />
                          Ask Anything
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Frame Modal */}
      {selectedFrame && (
        <FrameModal
          frame={selectedFrame}
          onClose={() => setSelectedFrame(null)}
        />
      )}

      {/* Ask Popup */}
      {askFrame && (
        <AskPopup
          frame={askFrame}
          triggerRect={askButtonRect}
          onClose={() => {
            setAskFrame(null);
            setAskButtonRect(null);
          }}
        />
      )}
    </div>
  );
};

export default VideoFrames;
