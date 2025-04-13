import React, { useState } from "react";
import { FiSave, FiAlertCircle } from "react-icons/fi";

const SettingSection = ({ title, children }) => (
  <div className="bg-white rounded-lg border border-gray-100 p-6">
    <h2 className="text-lg font-semibold text-gray-900 mb-4">{title}</h2>
    <div className="space-y-4">{children}</div>
  </div>
);

const Settings = () => {
  const [settings, setSettings] = useState({
    notifications: true,
    emailAlerts: true,
    darkMode: false,
    autoRecord: true,
    storageLimit: "1000",
    retentionDays: "30",
    sensitivity: "75",
    apiKey: "••••••••••••••••",
  });

  const handleChange = (key, value) => {
    setSettings((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle settings update
    console.log("Settings updated:", settings);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="mt-1 text-gray-500">
          Manage your system preferences and configurations
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Notifications */}
        <SettingSection title="Notifications">
          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-900">
                Push Notifications
              </label>
              <p className="text-sm text-gray-500">
                Receive alerts for important events
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={settings.notifications}
                onChange={(e) =>
                  handleChange("notifications", e.target.checked)
                }
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-sm font-medium text-gray-900">
                Email Alerts
              </label>
              <p className="text-sm text-gray-500">
                Receive daily summary and critical alerts
              </p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                checked={settings.emailAlerts}
                onChange={(e) => handleChange("emailAlerts", e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-indigo-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
            </label>
          </div>
        </SettingSection>

        {/* System Configuration */}
        <SettingSection title="System Configuration">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">
                Storage Limit (GB)
              </label>
              <input
                type="number"
                value={settings.storageLimit}
                onChange={(e) => handleChange("storageLimit", e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-1">
                Data Retention (Days)
              </label>
              <input
                type="number"
                value={settings.retentionDays}
                onChange={(e) => handleChange("retentionDays", e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">
              Detection Sensitivity (%)
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.sensitivity}
              onChange={(e) => handleChange("sensitivity", e.target.value)}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>
        </SettingSection>

        {/* API Configuration */}
        <SettingSection title="API Configuration">
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">
              API Key
            </label>
            <div className="flex gap-2">
              <input
                type="password"
                value={settings.apiKey}
                onChange={(e) => handleChange("apiKey", e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              <button
                type="button"
                className="px-4 py-2 text-sm font-medium text-indigo-600 bg-indigo-50 rounded-lg hover:bg-indigo-100"
              >
                Generate New
              </button>
            </div>
            <p className="mt-1 text-xs text-gray-500 flex items-center gap-1">
              <FiAlertCircle className="w-3 h-3" />
              Keep this key secure and never share it
            </p>
          </div>
        </SettingSection>

        {/* Save Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            className="flex items-center gap-2 px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            <FiSave className="w-4 h-4" />
            Save Changes
          </button>
        </div>
      </form>
    </div>
  );
};

export default Settings;
