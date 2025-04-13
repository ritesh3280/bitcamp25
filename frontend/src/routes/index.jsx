import React from "react";
import { Navigate } from "react-router-dom";
import MainLayout from "../layouts/MainLayout";
import Dashboard from "../pages/Dashboard";
import HomePage from "../components/HomePage";

// Lazy load other pages
const Anomalies = React.lazy(() => import("../pages/Anomalies"));
const LiveAnalysis = React.lazy(() => import("../pages/LiveAnalysis"));
const Timeline = React.lazy(() => import("../pages/Timeline"));
const Settings = React.lazy(() => import("../pages/Settings"));
const VideoFrames = React.lazy(() => import("../pages/VideoFrames"));

// Loading component for lazy-loaded routes
const LoadingFallback = () => (
  <div className="flex items-center justify-center h-full">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
  </div>
);

const routes = [
  {
    path: "/",
    element: <HomePage />,
  },
  {
    path: "/dashboard",
    element: <MainLayout />,
    children: [
      { index: true, element: <Dashboard /> },
      {
        path: "anomalies",
        element: (
          <React.Suspense fallback={<LoadingFallback />}>
            <Anomalies />
          </React.Suspense>
        ),
      },
      {
        path: "live",
        element: (
          <React.Suspense fallback={<LoadingFallback />}>
            <LiveAnalysis />
          </React.Suspense>
        ),
      },
      {
        path: "timeline",
        element: (
          <React.Suspense fallback={<LoadingFallback />}>
            <Timeline />
          </React.Suspense>
        ),
      },
      {
        path: "settings",
        element: (
          <React.Suspense fallback={<LoadingFallback />}>
            <Settings />
          </React.Suspense>
        ),
      },
      {
        path: "frames",
        element: (
          <React.Suspense fallback={<LoadingFallback />}>
            <VideoFrames />
          </React.Suspense>
        ),
      },
    ],
  },
  {
    path: "*",
    element: <Navigate to="/" replace />,
  },
];

export default routes;
