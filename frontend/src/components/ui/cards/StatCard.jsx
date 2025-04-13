import React from "react";

const StatCard = ({ icon: Icon, label, value, trend }) => (
  <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-6 hover:shadow-md transition-shadow duration-200">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600">{label}</p>
        <p className="mt-1 text-2xl font-bold text-gray-900">{value}</p>
      </div>
      <div className="p-3 bg-indigo-50 rounded-lg">
        <Icon className="w-6 h-6 text-indigo-600" />
      </div>
    </div>
    <div className="mt-4 flex items-center">
      <span
        className={`inline-flex items-center px-2 py-0.5 rounded-full text-sm font-medium ${
          trend >= 0 ? "text-green-800 bg-green-100" : "text-red-800 bg-red-100"
        }`}
      >
        {trend >= 0 ? "↑" : "↓"} {Math.abs(trend)}%
      </span>
      <span className="ml-2 text-sm text-gray-500">vs last week</span>
    </div>
  </div>
);

export default StatCard;
