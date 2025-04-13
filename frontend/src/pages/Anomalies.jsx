import React from "react";
import { FiAlertCircle, FiFilter, FiDownload } from "react-icons/fi";

const AnomalyItem = ({
  id,
  camera,
  location,
  time,
  type,
  severity,
  status,
}) => (
  <div className="bg-white rounded-lg border border-gray-100 p-4 hover:shadow-md transition-shadow duration-200">
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div
          className={`p-2 rounded-lg ${
            severity === "high"
              ? "bg-red-50"
              : severity === "medium"
              ? "bg-yellow-50"
              : "bg-blue-50"
          }`}
        >
          <FiAlertCircle
            className={`w-5 h-5 ${
              severity === "high"
                ? "text-red-600"
                : severity === "medium"
                ? "text-yellow-600"
                : "text-blue-600"
            }`}
          />
        </div>
        <div>
          <h3 className="text-sm font-medium text-gray-900">Camera {camera}</h3>
          <p className="text-sm text-gray-500">{location}</p>
        </div>
      </div>
      <span
        className={`px-2 py-1 text-xs font-medium rounded-full ${
          status === "resolved"
            ? "bg-green-100 text-green-800"
            : status === "investigating"
            ? "bg-yellow-100 text-yellow-800"
            : "bg-red-100 text-red-800"
        }`}
      >
        {status}
      </span>
    </div>
    <div className="mt-4">
      <p className="text-sm text-gray-600">{type}</p>
      <p className="text-xs text-gray-500 mt-1">{time}</p>
    </div>
  </div>
);

const Anomalies = () => {
  const anomalies = [
    {
      id: 1,
      camera: "1",
      location: "Main Entrance",
      time: "2 hours ago",
      type: "Unauthorized Access Attempt",
      severity: "high",
      status: "investigating",
    },
    {
      id: 2,
      camera: "3",
      location: "Parking Lot - Section A",
      time: "4 hours ago",
      type: "Suspicious Behavior",
      severity: "medium",
      status: "pending",
    },
    {
      id: 3,
      camera: "2",
      location: "Loading Dock",
      time: "6 hours ago",
      type: "Object Left Behind",
      severity: "low",
      status: "resolved",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Anomalies</h1>
          <p className="mt-1 text-gray-500">
            Review and manage detected anomalies
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50">
            <FiFilter className="w-4 h-4" />
            Filter
          </button>
          <button className="flex items-center gap-2 px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50">
            <FiDownload className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg border border-gray-100 p-4">
          <h3 className="text-sm font-medium text-gray-900">Total Anomalies</h3>
          <p className="mt-2 text-2xl font-bold text-gray-900">24</p>
          <p className="mt-1 text-sm text-gray-500">Last 24 hours</p>
        </div>
        <div className="bg-white rounded-lg border border-gray-100 p-4">
          <h3 className="text-sm font-medium text-gray-900">High Priority</h3>
          <p className="mt-2 text-2xl font-bold text-red-600">8</p>
          <p className="mt-1 text-sm text-gray-500">
            Requires immediate attention
          </p>
        </div>
        <div className="bg-white rounded-lg border border-gray-100 p-4">
          <h3 className="text-sm font-medium text-gray-900">
            Under Investigation
          </h3>
          <p className="mt-2 text-2xl font-bold text-yellow-600">12</p>
          <p className="mt-1 text-sm text-gray-500">Being reviewed</p>
        </div>
        <div className="bg-white rounded-lg border border-gray-100 p-4">
          <h3 className="text-sm font-medium text-gray-900">Resolved</h3>
          <p className="mt-2 text-2xl font-bold text-green-600">4</p>
          <p className="mt-1 text-sm text-gray-500">Cleared anomalies</p>
        </div>
      </div>

      {/* Anomalies List */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {anomalies.map((anomaly) => (
          <AnomalyItem key={anomaly.id} {...anomaly} />
        ))}
      </div>
    </div>
  );
};

export default Anomalies;
