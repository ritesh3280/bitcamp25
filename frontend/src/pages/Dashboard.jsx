import React from "react";
import { FiUsers, FiAlertCircle, FiClock, FiVideo } from "react-icons/fi";
import StatCard from "../components/ui/cards/StatCard";
import StatusCard from "../components/ui/cards/StatusCard";

const ActivityItem = ({ camera, location, time }) => (
  <div className="flex items-center gap-4 py-3 border-b border-gray-100 last:border-0">
    <div className="relative">
      <div className="w-2 h-2 rounded-full bg-indigo-600"></div>
      <div className="absolute top-0 -left-0.5 w-3 h-3 rounded-full bg-indigo-600 animate-ping opacity-75"></div>
    </div>
    <div className="flex-1 min-w-0">
      <p className="text-sm font-medium text-gray-900 truncate">
        Anomaly detected in Camera {camera}
      </p>
      <p className="text-sm text-gray-500 truncate">{location}</p>
    </div>
    <span className="text-sm text-gray-500 whitespace-nowrap">{time}</span>
  </div>
);

const Dashboard = () => {
  const stats = [
    {
      icon: FiAlertCircle,
      label: "Anomalies Detected",
      value: "24",
      trend: 12,
    },
    { icon: FiVideo, label: "Active Streams", value: "8", trend: 0 },
    { icon: FiUsers, label: "Persons of Interest", value: "156", trend: -5 },
    { icon: FiClock, label: "Hours Analyzed", value: "1,284", trend: 8 },
  ];

  const recentActivity = [
    { camera: "1", location: "Main Entrance", time: "2m ago" },
    { camera: "3", location: "Parking Lot - Section A", time: "5m ago" },
    { camera: "2", location: "Loading Dock", time: "12m ago" },
    { camera: "5", location: "Storage Area", time: "25m ago" },
    { camera: "4", location: "Employee Exit", time: "34m ago" },
  ];

  return (
    <div className="space-y-6">
      {/* Welcome Message */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
        <h2 className="text-2xl font-bold text-gray-900">Welcome back, John</h2>
        <p className="mt-1 text-gray-500">
          Here's what's happening across your surveillance system
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Activity */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">
              Recent Activity
            </h2>
            <button className="text-sm font-medium text-indigo-600 hover:text-indigo-700">
              View all
            </button>
          </div>
          <div className="space-y-4">
            {recentActivity.map((activity, i) => (
              <ActivityItem key={i} {...activity} />
            ))}
          </div>
        </div>

        {/* System Status */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            System Status
          </h2>
          <div className="space-y-4">
            <StatusCard
              title="AI Model Status"
              status="Operating normally"
              color="green"
            />
            <StatusCard
              title="Stream Processing"
              status="8/8 streams active"
              color="blue"
            />
            <StatusCard
              title="Storage Usage"
              status="1.2TB / 2TB (60%)"
              color="purple"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
