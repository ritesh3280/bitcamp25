import React from "react";
import { Link, useLocation } from "react-router-dom";
import {
  FiHome,
  FiAlertTriangle,
  FiVideo,
  FiClock,
  FiSettings,
  FiX,
  FiImage,
} from "react-icons/fi";

const navigationItems = [
  { to: "/dashboard", icon: FiHome, label: "Home" },
  { to: "/dashboard/anomalies", icon: FiAlertTriangle, label: "Anomalies" },
  { to: "/dashboard/live", icon: FiVideo, label: "Live Analysis" },
  { to: "/dashboard/frames", icon: FiImage, label: "Video Frames" },
  { to: "/dashboard/timeline", icon: FiClock, label: "Timeline" },
  { to: "/dashboard/settings", icon: FiSettings, label: "Settings" },
];

const SidebarLink = ({ to, icon: Icon, label, isActive }) => (
  <Link
    to={to}
    className={`flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 hover:bg-indigo-600 hover:text-white
      ${
        isActive
          ? "bg-indigo-600 text-white shadow-lg shadow-indigo-500/30"
          : "text-gray-500 hover:text-white"
      }`}
  >
    <Icon className="w-5 h-5" />
    <span className="hidden lg:block font-medium">{label}</span>
  </Link>
);

const Sidebar = ({ isOpen, onClose }) => {
  const location = useLocation();

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 lg:hidden z-20"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed lg:static inset-y-0 left-0 w-64 transform 
          ${isOpen ? "translate-x-0" : "-translate-x-full"}
          lg:translate-x-0 transition-transform duration-200 ease-in-out z-30
          bg-white border-r border-gray-100 shadow-sm`}
      >
        {/* Logo */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-gray-100">
          <Link to="/" className="flex items-center gap-2">
            <span className="text-xl font-bold text-indigo-600">
              Forensic Vision
            </span>
          </Link>
          <button
            onClick={onClose}
            className="lg:hidden text-gray-500 hover:text-gray-700"
          >
            <FiX className="w-6 h-6" />
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="mt-6 px-4 space-y-1">
          {navigationItems.map((item) => (
            <SidebarLink
              key={item.to}
              {...item}
              isActive={item.to === location.pathname}
            />
          ))}
        </nav>
      </aside>
    </>
  );
};

export default Sidebar;
