import React, { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Sidebar from "../components/ui/Sidebar";
import Topbar from "../components/ui/Topbar";

const MainLayout = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const location = useLocation();

  // Get current page title based on route
  const getPageTitle = () => {
    const path = location.pathname;
    if (path === "/dashboard") return "Dashboard Overview";
    return path
      .split("/")
      .pop()
      .replace(/-/g, " ")
      .replace(/\b\w/g, (l) => l.toUpperCase());
  };

  const handleLogout = () => {
    // Implement logout logic
    console.log("Logging out...");
  };

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Topbar
          pageTitle={getPageTitle()}
          onMenuClick={() => setIsSidebarOpen(true)}
          isDarkMode={isDarkMode}
          onThemeToggle={() => setIsDarkMode(!isDarkMode)}
          onLogout={handleLogout}
        />

        <main className="flex-1 overflow-y-auto bg-gray-50 p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
