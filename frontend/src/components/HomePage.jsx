import React from "react";
import {
  FiPlayCircle,
  FiClock,
  FiAlertCircle,
  FiLayers,
  FiActivity,
} from "react-icons/fi";

const HomePage = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <span className="text-xl font-bold text-indigo-600">
                ForensicAI
              </span>
              <div className="hidden sm:ml-8 sm:flex sm:space-x-8">
                <a
                  href="/"
                  className="text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 border-indigo-500"
                >
                  Home
                </a>
                <a
                  href="/live-analysis"
                  className="text-gray-500 hover:text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300"
                >
                  Live Analysis
                </a>
                <a
                  href="/dashboard"
                  className="text-gray-500 hover:text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300"
                >
                  Dashboard
                </a>
                <a
                  href="/keyframes"
                  className="text-gray-500 hover:text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300"
                >
                  Keyframes
                </a>
                <a
                  href="/contact"
                  className="text-gray-500 hover:text-gray-900 inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300"
                >
                  Contact
                </a>
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="relative bg-white overflow-hidden">
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 bg-white sm:pb-16 md:pb-20 lg:max-w-2xl lg:w-full lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
              <div className="sm:text-center lg:text-left">
                <h1 className="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                  <span className="block">From Footage to Forensics —</span>
                  <span className="block text-indigo-600">
                    AI-Powered Scene Intelligence
                  </span>
                </h1>
                <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                  Turn live video feeds into actionable forensic timelines with
                  object detection and LLM summarization.
                </p>
                <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                  <div className="rounded-md shadow">
                    <a
                      href="#"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 md:py-4 md:text-lg md:px-10"
                    >
                      Get Started
                    </a>
                  </div>
                  <div className="mt-3 sm:mt-0 sm:ml-3">
                    <a
                      href="#"
                      className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-600 bg-indigo-100 hover:bg-indigo-200 md:py-4 md:text-lg md:px-10"
                    >
                      <FiPlayCircle className="mr-2" /> Watch Demo
                    </a>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              Powerful Features for Forensic Analysis
            </h2>
          </div>

          <div className="mt-10">
            <div className="grid grid-cols-1 gap-10 sm:grid-cols-2 lg:grid-cols-4">
              {/* Feature 1 */}
              <div className="relative bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                <div className="text-center">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white mx-auto">
                    <FiLayers className="h-6 w-6" />
                  </div>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">
                    Real-Time Object Detection
                  </h3>
                  <p className="mt-2 text-base text-gray-500">
                    Advanced AI algorithms for instant object recognition and
                    tracking in video streams.
                  </p>
                </div>
              </div>

              {/* Feature 2 */}
              <div className="relative bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                <div className="text-center">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white mx-auto">
                    <FiClock className="h-6 w-6" />
                  </div>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">
                    Keyframe Summarization with LLM
                  </h3>
                  <p className="mt-2 text-base text-gray-500">
                    Intelligent scene analysis and summarization powered by
                    large language models.
                  </p>
                </div>
              </div>

              {/* Feature 3 */}
              <div className="relative bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                <div className="text-center">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white mx-auto">
                    <FiAlertCircle className="h-6 w-6" />
                  </div>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">
                    Red Flag & Anomaly Detection
                  </h3>
                  <p className="mt-2 text-base text-gray-500">
                    Automated detection of suspicious activities and behavioral
                    anomalies.
                  </p>
                </div>
              </div>

              {/* Feature 4 */}
              <div className="relative bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow">
                <div className="text-center">
                  <div className="flex items-center justify-center h-12 w-12 rounded-md bg-indigo-500 text-white mx-auto">
                    <FiActivity className="h-6 w-6" />
                  </div>
                  <h3 className="mt-4 text-lg font-medium text-gray-900">
                    Interactive Forensics Timeline
                  </h3>
                  <p className="mt-2 text-base text-gray-500">
                    Visual timeline interface for intuitive evidence review and
                    analysis.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-gray-800">
        <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div>
              <h3 className="text-white text-lg font-semibold mb-4">
                ForensicAI
              </h3>
              <p className="text-gray-400">
                Empowering investigators with AI-powered video analysis tools.
              </p>
            </div>
            <div>
              <h3 className="text-white text-lg font-semibold mb-4">
                Quick Links
              </h3>
              <ul className="space-y-2">
                <li>
                  <a href="/about" className="text-gray-400 hover:text-white">
                    About Us
                  </a>
                </li>
                <li>
                  <a href="/privacy" className="text-gray-400 hover:text-white">
                    Privacy Policy
                  </a>
                </li>
                <li>
                  <a href="/terms" className="text-gray-400 hover:text-white">
                    Terms of Service
                  </a>
                </li>
              </ul>
            </div>
            <div>
              <h3 className="text-white text-lg font-semibold mb-4">Contact</h3>
              <p className="text-gray-400">Email: contact@forensicai.com</p>
              <div className="mt-4 flex space-x-4">
                <a href="#" className="text-gray-400 hover:text-white">
                  <span className="sr-only">Twitter</span>
                  <svg
                    className="h-6 w-6"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
                  </svg>
                </a>
                <a href="#" className="text-gray-400 hover:text-white">
                  <span className="sr-only">GitHub</span>
                  <svg
                    className="h-6 w-6"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      fillRule="evenodd"
                      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                      clipRule="evenodd"
                    />
                  </svg>
                </a>
              </div>
            </div>
          </div>
          <div className="mt-8 border-t border-gray-700 pt-8 text-center">
            <p className="text-gray-400">
              &copy; 2024 ForensicAI. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HomePage;
