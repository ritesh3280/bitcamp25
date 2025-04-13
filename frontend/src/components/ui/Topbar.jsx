import React from "react";
import {
  FiMenu,
  FiSun,
  FiMoon,
  FiBell,
  FiChevronDown,
  FiLogOut,
  FiUser,
  FiSettings,
} from "react-icons/fi";
import { Link } from "react-router-dom";

const UserDropdown = ({ isOpen, setIsOpen, onLogout }) => (
  <div className="relative">
    <button
      onClick={() => setIsOpen(!isOpen)}
      className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 transition-colors duration-200"
    >
      <div className="w-8 h-8 bg-indigo-600 rounded-full flex items-center justify-center">
        <span className="text-sm font-medium text-white">JD</span>
      </div>
      <span className="hidden md:block text-sm font-medium text-gray-700">
        John Doe
      </span>
      <FiChevronDown className="w-4 h-4 text-gray-500" />
    </button>

    {isOpen && (
      <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg py-1 z-50 border border-gray-100">
        <Link
          to="/dashboard/profile"
          className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
        >
          <FiUser className="w-4 h-4" />
          Profile
        </Link>
        <Link
          to="/dashboard/settings"
          className="flex items-center gap-2 px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
        >
          <FiSettings className="w-4 h-4" />
          Settings
        </Link>
        <hr className="my-1 border-gray-100" />
        <button
          onClick={onLogout}
          className="flex items-center gap-2 px-4 py-2 text-sm text-red-600 hover:bg-gray-50 w-full text-left"
        >
          <FiLogOut className="w-4 h-4" />
          Log out
        </button>
      </div>
    )}
  </div>
);

const Topbar = ({
  pageTitle,
  onMenuClick,
  isDarkMode,
  onThemeToggle,
  onLogout,
}) => {
  const [isUserDropdownOpen, setIsUserDropdownOpen] = React.useState(false);

  return (
    <header className="sticky top-0 z-10 bg-white border-b border-gray-100 shadow-sm">
      <div className="flex items-center justify-between h-16 px-6">
        <div className="flex items-center gap-4">
          <button
            onClick={onMenuClick}
            className="lg:hidden text-gray-500 hover:text-gray-700"
          >
            <FiMenu className="w-6 h-6" />
          </button>
          <h1 className="text-xl font-semibold text-gray-800">{pageTitle}</h1>
        </div>

        <div className="flex items-center gap-4">
          <button
            className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors duration-200"
            onClick={onThemeToggle}
          >
            {isDarkMode ? (
              <FiSun className="w-5 h-5" />
            ) : (
              <FiMoon className="w-5 h-5" />
            )}
          </button>

          <button className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors duration-200 relative">
            <FiBell className="w-5 h-5" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>

          <UserDropdown
            isOpen={isUserDropdownOpen}
            setIsOpen={setIsUserDropdownOpen}
            onLogout={onLogout}
          />
        </div>
      </div>
    </header>
  );
};

export default Topbar;
