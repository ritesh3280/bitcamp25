import React, { useState, useRef, useEffect } from "react";
import {
  FiSearch,
  FiImage,
  FiMessageSquare,
  FiX,
  FiClock,
  FiFilter,
  FiSend,
  FiFolder,
  FiChevronRight,
  FiCamera,
  FiRefreshCw,
  FiCheckCircle,
  FiAlertCircle,
  FiZap,
  FiAward,
} from "react-icons/fi";
import api from "../services/api";

// Get API_BASE_URL from the environment or use default
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:5000";

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
          <div className="relative">
            <img
              src={frame.thumbnail}
              alt={`Frame ${frame.id}`}
              className="w-full h-auto rounded-lg"
              onError={(e) => {
                console.error("Error loading image:", e);
                e.target.src =
                  "https://placehold.co/600x400?text=Image+Not+Found";
                e.target.alt = "Image failed to load";
              }}
            />
          </div>
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
              {frame.objects ? frame.objects.join(", ") : "None"}
            </p>
            <p className="text-sm text-gray-500">
              <span className="font-medium text-gray-900">Confidence:</span>{" "}
              {frame.confidence
                ? `${(frame.confidence * 100).toFixed(1)}%`
                : "N/A"}
            </p>

            {frame.llm_description && (
              <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
                <h4 className="text-sm font-medium text-gray-900 mb-2">
                  AI Description:
                </h4>
                {frame.llm_description.description ? (
                  <p className="text-sm text-gray-700">
                    {frame.llm_description.description}
                  </p>
                ) : frame.llm_description.error ? (
                  <p className="text-sm text-red-500">
                    {frame.llm_description.error}
                  </p>
                ) : (
                  <p className="text-sm text-gray-500">
                    No description generated
                  </p>
                )}
                {frame.llm_description.timestamp && (
                  <p className="text-xs text-gray-500 mt-2">
                    Generated at:{" "}
                    {new Date(frame.llm_description.timestamp).toLocaleString()}
                  </p>
                )}
              </div>
            )}
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
        <p className="text-sm leading-relaxed whitespace-pre-wrap">{content}</p>
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

const TypingIndicator = () => (
  <div className="flex justify-start mb-3">
    <div className="max-w-[85%]">
      <div className="bg-gray-100 text-gray-900 rounded-2xl rounded-tl-sm shadow-sm px-4 py-3">
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
);

const AskPopup = ({ frame, onClose, triggerRect }) => {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([
    {
      content:
        "Hi! I'm GPT-4o. I can help analyze this frame by describing what I see and answering your questions about it. What would you like to know?",
      isUser: false,
      timestamp: new Date().toLocaleTimeString(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [error, setError] = useState("");
  const popupRef = useRef(null);
  const chatHistoryRef = useRef(null);
  const inputRef = useRef(null);
  const [position, setPosition] = useState({ top: 0, left: 0 });

  useEffect(() => {
    if (triggerRect && popupRef.current) {
      const popup = popupRef.current;
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

    // Focus the input when the popup opens
    if (inputRef.current) {
      setTimeout(() => {
        inputRef.current.focus();
      }, 100);
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

    setError(""); // Clear any previous errors

    // Add user message
    const userMessage = {
      content: question,
      isUser: true,
      timestamp: new Date().toLocaleTimeString(),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setQuestion("");
    setIsLoading(true);

    try {
      console.log("Asking about frame:", frame.id, "Question:", question);

      // Use the new chatWithFrame method instead of askQuestion
      const response = await api.chatWithFrame(
        frame.id,
        question,
        conversationHistory
      );

      if (response && response.answer) {
        // Add AI response
        setMessages([
          ...newMessages,
          {
            content: response.answer,
            isUser: false,
            timestamp: new Date().toLocaleTimeString(),
          },
        ]);

        // Update conversation history for next message
        setConversationHistory(response.conversation || []);
      } else {
        throw new Error("Invalid response from API");
      }
    } catch (error) {
      console.error("Error asking question:", error);

      // Set error state
      setError(
        error.message ||
          "Failed to connect to the AI service. Please try again later."
      );

      // Add error message to chat
      setMessages([
        ...newMessages,
        {
          content:
            "Sorry, I encountered an error processing your question. Please try again or check the console for details.",
          isUser: false,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    } finally {
      setIsLoading(false);

      // Focus the input after sending
      if (inputRef.current) {
        inputRef.current.focus();
      }
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
        bg-white
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
              onError={(e) => {
                e.target.src = "https://placehold.co/100x100?text=Frame";
              }}
            />
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-indigo-600 rounded-full flex items-center justify-center">
              <FiMessageSquare className="w-2.5 h-2.5 text-white" />
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-900">
              Chat with GPT-4o
            </h3>
            <p className="text-xs text-gray-500 flex items-center gap-1">
              <span>{frame.id}</span>
              <span className="block w-1 h-1 rounded-full bg-gray-300" />
              <span>Frame Analysis</span>
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
          px-4 py-4
          space-y-2
          bg-white
        "
      >
        {messages.map((message, index) => (
          <ChatMessage key={index} {...message} />
        ))}

        {isLoading && <TypingIndicator />}

        {error && (
          <div className="bg-red-50 text-red-600 rounded-lg p-2 text-xs mt-2 mb-2">
            <div className="flex items-center">
              <FiAlertCircle className="mr-1" />
              <span>{error}</span>
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
              ref={inputRef}
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
            <div className="absolute bottom-2 right-3 text-xs text-gray-400">
              {isLoading ? "Processing..." : "Enter to send"}
            </div>
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
            {isLoading ? (
              <FiRefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <FiSend className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// Super Search result item component
const SearchResultItem = ({ result, onSelect }) => {
  return (
    <div
      className="border border-gray-200 rounded-lg p-4 mb-4 hover:shadow-md transition-shadow cursor-pointer bg-white"
      onClick={() => onSelect(result)}
    >
      <div className="flex items-start gap-4">
        <div className="shrink-0 w-20 h-20 bg-gray-100 rounded-lg overflow-hidden">
          <img
            src={api.getFrameImage(result.id)}
            alt={`Frame ${result.id}`}
            className="w-full h-full object-cover"
            onError={(e) => {
              e.target.src = "https://placehold.co/80x80?text=Frame";
            }}
          />
        </div>

        <div className="flex-1">
          <div className="flex justify-between items-start">
            <h3 className="text-sm font-medium text-gray-900">{result.id}</h3>
            <span className="text-xs text-gray-500">{result.timestamp}</span>
          </div>

          <div className="mt-1">
            <div className="flex items-center gap-1 mb-1">
              <FiAward className="text-yellow-500 w-3 h-3" />
              <span className="text-xs font-medium text-gray-700">
                {(result.similarity * 100).toFixed(1)}% match
              </span>
            </div>
          </div>

          <div className="mt-1">
            <p className="text-sm text-gray-600">{result.explanation}</p>
          </div>

          <div className="flex flex-wrap gap-1 mt-2">
            {result.objects.slice(0, 3).map((obj, idx) => (
              <span
                key={idx}
                className="px-2 py-0.5 text-xs bg-gray-100 text-gray-800 rounded-full"
              >
                {obj}
              </span>
            ))}
            {result.objects.length > 3 && (
              <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-500 rounded-full">
                +{result.objects.length - 3}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Super Search popup component
const SuperSearchPopup = ({ sessionId, onClose }) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState("");
  const popupRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    // Focus the input when the popup opens
    if (inputRef.current) {
      setTimeout(() => {
        inputRef.current.focus();
      }, 100);
    }
  }, []);

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

  const handleSemanticSearch = async () => {
    if (!searchQuery.trim() || isSearching) return;

    setError("");
    setIsSearching(true);

    try {
      const response = await api.semanticFrameSearch(sessionId, searchQuery);
      if (response && response.results) {
        setSearchResults(response.results);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (error) {
      console.error("Error performing semantic search:", error);
      setError(
        error.message ||
          "Failed to perform search. Please try again with a different query."
      );
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSemanticSearch();
    }
  };

  const handleResultSelect = (result) => {
    // Find the matching frame in the session and highlight it
    // This function is passed from the parent component via onSelectFrame
    if (onClose) {
      onClose(result);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4">
      <div
        ref={popupRef}
        className="bg-white rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col"
      >
        <div className="p-4 border-b border-gray-100 flex justify-between items-center">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-indigo-100">
              <FiZap className="w-5 h-5 text-indigo-600" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">
              Super Search
            </h2>
          </div>
          <button
            onClick={() => onClose()}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <FiX className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        <div className="p-4 border-b border-gray-100">
          <div className="flex items-stretch gap-2">
            <div className="relative flex-1">
              <input
                ref={inputRef}
                type="text"
                className="w-full pl-10 pr-4 py-3 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                placeholder="Describe what you're looking for..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isSearching}
              />
              <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            </div>
            <button
              onClick={handleSemanticSearch}
              disabled={!searchQuery.trim() || isSearching}
              className="px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              {isSearching ? (
                <>
                  <FiRefreshCw className="w-4 h-4 animate-spin" />
                  <span>Searching...</span>
                </>
              ) : (
                <>
                  <FiZap className="w-4 h-4" />
                  <span>Search</span>
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="mt-2 p-2 bg-red-50 text-red-600 rounded-lg text-sm flex items-center gap-2">
              <FiAlertCircle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          )}

          <p className="mt-2 text-xs text-gray-500">
            Use natural language to describe frames you're looking for. For
            example: "person near a computer" or "frames showing storage
            devices"
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
          {isSearching ? (
            <div className="flex flex-col items-center justify-center h-64">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600 mb-4"></div>
              <p className="text-gray-500">
                Finding the most relevant frames...
              </p>
            </div>
          ) : searchResults.length > 0 ? (
            <div>
              <div className="mb-4">
                <h3 className="text-sm font-medium text-gray-700">
                  Top Results
                </h3>
                <p className="text-xs text-gray-500">
                  Found {searchResults.length} frames matching your query
                </p>
              </div>
              {searchResults.map((result) => (
                <SearchResultItem
                  key={result.id}
                  result={result}
                  onSelect={handleResultSelect}
                />
              ))}
            </div>
          ) : searchQuery ? (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <FiSearch className="w-10 h-10 text-gray-300 mb-4" />
              <p className="text-gray-500 mb-1">No results found</p>
              <p className="text-sm text-gray-400">
                Try different search terms or check if the session has frames
                with descriptions
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-center">
              <FiZap className="w-10 h-10 text-indigo-200 mb-4" />
              <p className="text-gray-500 mb-1">
                Enter a search query to find relevant frames
              </p>
              <p className="text-sm text-gray-400">
                The AI will find frames that semantically match your description
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const VideoFrames = () => {
  const [activeTab, setActiveTab] = useState("sessions");
  const [sessions, setSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [sessionFrames, setSessionFrames] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [chatFrame, setChatFrame] = useState(null);
  const [askPopupVisible, setAskPopupVisible] = useState(false);
  const [askPopupTriggerRect, setAskPopupTriggerRect] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [filteredFrames, setFilteredFrames] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");
  const [expandedFrameId, setExpandedFrameId] = useState(null);
  const [superSearchOpen, setSuperSearchOpen] = useState(false);

  // Fetch sessions from the API
  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    setIsLoading(true);
    setErrorMessage("");
    try {
      // Use the API service
      const data = await api.getSessions();
      setSessions(data.sessions || []);
    } catch (error) {
      console.error("Error fetching sessions:", error);
      setErrorMessage("Failed to load sessions. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch frames for a specific session
  const fetchSessionFrames = async (sessionId) => {
    setIsLoading(true);
    setErrorMessage("");
    try {
      // Use the API service
      const data = await api.getSessionFrames(sessionId);
      setSessionFrames(data.frames || []);
      setFilteredFrames(data.frames || []);
    } catch (error) {
      console.error("Error fetching session frames:", error);
      setErrorMessage("Failed to load frames. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Handle session selection
  const handleSessionSelect = (session) => {
    setSelectedSession(session);
    fetchSessionFrames(session.id);
    setActiveTab("frames");
  };

  // Handle search
  useEffect(() => {
    if (searchQuery.trim() === "") {
      setFilteredFrames(sessionFrames);
    } else {
      const lowerCaseQuery = searchQuery.toLowerCase();
      const filtered = sessionFrames.filter((frame) => {
        // Search by objects, timestamp, or id
        return (
          frame.objects.some((obj) =>
            obj.toLowerCase().includes(lowerCaseQuery)
          ) ||
          frame.timestamp.toLowerCase().includes(lowerCaseQuery) ||
          frame.id.toLowerCase().includes(lowerCaseQuery)
        );
      });
      setFilteredFrames(filtered);
    }
  }, [searchQuery, sessionFrames]);

  const handleAskClick = (frame, event) => {
    console.log("Ask about frame:", frame.id);
    event.stopPropagation(); // Prevent event bubbling

    // Prepare frame with all necessary data for the chat popup
    const frameForChat = {
      ...frame,
      detections: frame.objects,
      thumbnail: api.getFrameImage(frame.id),
    };

    // Use the chatFrame state instead of selectedFrame
    setChatFrame(frameForChat);
    setAskPopupTriggerRect(event.currentTarget.getBoundingClientRect());
    setAskPopupVisible(true);
  };

  const handleFrameClick = (frame) => {
    console.log("Showing frame:", frame);
    // Make sure we include all necessary properties for the modal
    const frameForModal = {
      ...frame,
      // Convert objects to detections format if needed by FrameModal
      detections: frame.objects,
      thumbnail: api.getFrameImage(frame.id),
      // Ensure llm_description is included
      llm_description: frame.llm_description,
    };
    setSelectedFrame(frameForModal);
  };

  // Toggle expanded row
  const toggleFrameDescription = (frameId) => {
    if (expandedFrameId === frameId) {
      setExpandedFrameId(null);
    } else {
      setExpandedFrameId(frameId);
    }
  };

  // Handle selection of a frame from super search
  const handleSuperSearchFrameSelect = (result) => {
    // Find the matching frame in the session frames
    const frameId = result.id;
    const matchingFrame = sessionFrames.find((frame) => frame.id === frameId);

    if (matchingFrame) {
      // Highlight and scroll to this frame
      setExpandedFrameId(frameId);

      // Also show the frame in the modal
      handleFrameClick(matchingFrame);
    }

    // Close the super search popup
    setSuperSearchOpen(false);
  };

  return (
    <div className="flex flex-col p-4 h-full">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Video Frames</h1>
        <div className="flex items-center space-x-2">
          <button
            onClick={fetchSessions}
            className="px-3 py-2 rounded-lg text-sm text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors flex items-center gap-2"
          >
            <FiRefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>
      </div>

      {errorMessage && (
        <div className="mb-4 p-3 bg-red-50 text-red-600 rounded-lg flex items-center gap-2">
          <FiAlertCircle className="w-5 h-5" />
          {errorMessage}
        </div>
      )}

      <div className="flex mb-4 border-b border-gray-200">
        <button
          className={`px-4 py-2 text-sm font-medium relative ${
            activeTab === "sessions"
              ? "text-indigo-600 border-indigo-600"
              : "text-gray-500 hover:text-gray-700"
          }`}
          onClick={() => setActiveTab("sessions")}
        >
          Sessions
          {activeTab === "sessions" && (
            <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600"></div>
          )}
        </button>
        {selectedSession && (
          <button
            className={`px-4 py-2 text-sm font-medium relative ${
              activeTab === "frames"
                ? "text-indigo-600 border-indigo-600"
                : "text-gray-500 hover:text-gray-700"
            }`}
            onClick={() => setActiveTab("frames")}
          >
            Frames
            {activeTab === "frames" && (
              <div className="absolute bottom-0 left-0 w-full h-0.5 bg-indigo-600"></div>
            )}
          </button>
        )}
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        </div>
      ) : (
        <div className="flex-1 overflow-hidden flex flex-col">
          {activeTab === "sessions" && (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 overflow-auto">
              {sessions.length === 0 ? (
                <div className="col-span-3 py-16 flex flex-col items-center justify-center text-gray-500">
                  <FiFolder className="w-12 h-12 mb-4 opacity-50" />
                  <p className="text-lg font-medium mb-1">No Sessions Found</p>
                  <p className="text-sm">
                    Run a video analysis first to create a session.
                  </p>
                </div>
              ) : (
                sessions.map((session) => (
                  <div
                    key={session.id}
                    className="bg-white rounded-xl shadow-sm border border-gray-200 p-4 hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => handleSessionSelect(session)}
                  >
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center">
                        <div className="w-10 h-10 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600">
                          <FiCamera className="w-5 h-5" />
                        </div>
                        <div className="ml-3">
                          <h3 className="text-sm font-medium text-gray-900">
                            {session.name}
                          </h3>
                          <p className="text-xs text-gray-500">
                            {new Date(session.created).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <FiChevronRight className="w-5 h-5 text-gray-400" />
                    </div>
                    <div className="flex items-center text-xs text-gray-500 mt-2">
                      <div className="flex items-center">
                        <FiImage className="w-4 h-4 mr-1" />
                        <span>{session.frame_count} frames</span>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          )}

          {activeTab === "frames" && selectedSession && (
            <div className="flex flex-col flex-1 overflow-hidden">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex-1">
                  <div className="relative">
                    <input
                      type="text"
                      className="w-full pl-9 pr-4 py-2 bg-gray-100 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                      placeholder="Search frames by objects, timestamp..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                    <FiSearch className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  </div>
                </div>
                <div className="ml-4 flex gap-2">
                  <button
                    onClick={() => setSuperSearchOpen(true)}
                    className="px-3 py-2 rounded-lg text-sm text-white bg-indigo-600 hover:bg-indigo-700 transition-colors flex items-center gap-2"
                  >
                    <FiZap className="w-4 h-4" />
                    Super Search
                  </button>

                  <button
                    onClick={() => {
                      setSelectedSession(null);
                      setSessionFrames([]);
                      setActiveTab("sessions");
                    }}
                    className="px-3 py-2 rounded-lg text-sm text-gray-700 bg-gray-100 hover:bg-gray-200 transition-colors"
                  >
                    Back to Sessions
                  </button>
                </div>
              </div>

              <div className="mb-2">
                <h2 className="text-lg font-medium text-gray-900">
                  {selectedSession.name}
                </h2>
                <p className="text-sm text-gray-500">
                  {new Date(selectedSession.created).toLocaleString()} •{" "}
                  {sessionFrames.length} frames
                </p>
              </div>

              <div className="flex-1 overflow-hidden">
                {filteredFrames.length === 0 ? (
                  <div className="py-16 flex flex-col items-center justify-center text-gray-500 h-full">
                    <FiImage className="w-12 h-12 mb-4 opacity-50" />
                    <p className="text-lg font-medium mb-1">No Frames Found</p>
                    <p className="text-sm">
                      No frames matching your search criteria.
                    </p>
                  </div>
                ) : (
                  <div className="bg-white rounded-lg border border-gray-200 overflow-hidden h-full">
                    <div className="overflow-auto h-full">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50 sticky top-0 z-10">
                          <tr>
                            <th
                              scope="col"
                              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                            >
                              Frame ID
                            </th>
                            <th
                              scope="col"
                              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                            >
                              Timestamp
                            </th>
                            <th
                              scope="col"
                              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                            >
                              Objects
                            </th>
                            <th
                              scope="col"
                              className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                            >
                              Confidence
                            </th>
                            <th
                              scope="col"
                              className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider"
                            >
                              Actions
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {filteredFrames.map((frame) => (
                            <React.Fragment key={frame.id}>
                              <tr
                                className={`hover:bg-gray-50 ${
                                  expandedFrameId === frame.id
                                    ? "bg-indigo-50"
                                    : ""
                                }`}
                                onClick={() => toggleFrameDescription(frame.id)}
                              >
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                  <div className="flex items-center">
                                    <div
                                      className={`mr-2 transition-transform duration-200 ${
                                        expandedFrameId === frame.id
                                          ? "rotate-90"
                                          : ""
                                      }`}
                                    >
                                      <svg
                                        xmlns="http://www.w3.org/2000/svg"
                                        width="16"
                                        height="16"
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                      >
                                        <polyline points="9 18 15 12 9 6"></polyline>
                                      </svg>
                                    </div>
                                    {frame.id}
                                  </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                  {frame.timestamp}
                                </td>
                                <td className="px-6 py-4 text-sm text-gray-500">
                                  <div className="flex flex-wrap gap-1">
                                    {frame.objects
                                      .slice(0, 3)
                                      .map((object, idx) => (
                                        <span
                                          key={idx}
                                          className="px-2 py-0.5 text-xs bg-gray-100 text-gray-800 rounded-full"
                                        >
                                          {object}
                                        </span>
                                      ))}
                                    {frame.objects.length > 3 && (
                                      <span className="px-2 py-0.5 text-xs bg-gray-100 text-gray-500 rounded-full">
                                        +{frame.objects.length - 3}
                                      </span>
                                    )}
                                  </div>
                                </td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                  {Math.round(frame.confidence * 100)}%
                                </td>
                                <td
                                  className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium"
                                  onClick={(e) => e.stopPropagation()}
                                >
                                  <div className="flex justify-end gap-2">
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleFrameClick(frame);
                                      }}
                                      className="text-indigo-600 hover:text-indigo-800 px-2 py-1 rounded hover:bg-indigo-50 flex items-center gap-1"
                                    >
                                      <FiImage className="w-4 h-4" />
                                      Show Frame
                                    </button>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleAskClick(frame, e);
                                      }}
                                      className="text-indigo-600 hover:text-indigo-800 px-2 py-1 rounded hover:bg-indigo-50 flex items-center gap-1"
                                    >
                                      <FiMessageSquare className="w-4 h-4" />
                                      Ask
                                    </button>
                                  </div>
                                </td>
                              </tr>
                              {expandedFrameId === frame.id && (
                                <tr>
                                  <td
                                    colSpan="5"
                                    className="px-6 py-4 bg-indigo-50"
                                  >
                                    <div className="text-sm text-gray-800">
                                      <h4 className="font-medium mb-2">
                                        AI Description:
                                      </h4>
                                      {frame.llm_description ? (
                                        <>
                                          <p className="text-gray-600 bg-white p-3 rounded border border-gray-200">
                                            {frame.llm_description
                                              .description ? (
                                              frame.llm_description.description
                                            ) : frame.llm_description.error ? (
                                              <span className="text-red-500">
                                                {frame.llm_description.error}
                                              </span>
                                            ) : (
                                              "No description generated"
                                            )}
                                          </p>
                                          {frame.llm_description.timestamp && (
                                            <p className="text-xs text-gray-500 mt-2">
                                              Generated at:{" "}
                                              {new Date(
                                                frame.llm_description.timestamp
                                              ).toLocaleString()}
                                            </p>
                                          )}
                                        </>
                                      ) : (
                                        <p className="text-gray-600 bg-white p-3 rounded border border-gray-200">
                                          No AI description available for this
                                          frame.
                                        </p>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              )}
                            </React.Fragment>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {selectedFrame && (
        <FrameModal
          frame={selectedFrame}
          onClose={() => setSelectedFrame(null)}
        />
      )}

      {askPopupVisible && chatFrame && (
        <AskPopup
          frame={chatFrame}
          onClose={() => {
            setAskPopupVisible(false);
            setChatFrame(null);
          }}
          triggerRect={askPopupTriggerRect}
        />
      )}

      {superSearchOpen && selectedSession && (
        <SuperSearchPopup
          sessionId={selectedSession.id}
          onClose={handleSuperSearchFrameSelect}
        />
      )}
    </div>
  );
};

export default VideoFrames;
