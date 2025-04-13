import React, { useState } from "react";
import { FiClock, FiFilter, FiDownload, FiChevronRight } from "react-icons/fi";

const TimelineEvent = ({ time, camera, location, type, description }) => (
  <div className="relative pb-8">
    <div className="absolute left-4 -ml-0.5 mt-1.5 top-4 h-full w-0.5 bg-gray-200"></div>
    <div className="relative flex items-start space-x-3">
      <div className="relative">
        <div className="h-8 w-8 rounded-full bg-indigo-50 flex items-center justify-center ring-8 ring-white">
          <FiClock className="h-5 w-5 text-indigo-600" />
        </div>
      </div>
      <div className="min-w-0 flex-1 bg-white rounded-lg border border-gray-100 shadow-sm p-4">
        <div className="flex justify-between items-center mb-1">
          <div className="text-sm font-medium text-gray-900">
            Camera {camera}
          </div>
          <time className="text-sm text-gray-500">{time}</time>
        </div>
        <div className="text-sm text-gray-500 mb-2">{location}</div>
        <div className="text-sm font-medium text-gray-900 mb-1">{type}</div>
        <p className="text-sm text-gray-600">{description}</p>
      </div>
    </div>
  </div>
);

const Timeline = () => {
  const [selectedDate, setSelectedDate] = useState("today");
  const [selectedCamera, setSelectedCamera] = useState("all");

  const events = [
    {
      time: "2 hours ago",
      camera: "1",
      location: "Main Entrance",
      type: "Unauthorized Access",
      description:
        "Individual attempted to access restricted area without proper credentials.",
    },
    {
      time: "4 hours ago",
      camera: "3",
      location: "Parking Lot - Section A",
      type: "Suspicious Activity",
      description:
        "Multiple individuals observed loitering in restricted parking area.",
    },
    {
      time: "6 hours ago",
      camera: "2",
      location: "Loading Dock",
      type: "Object Detection",
      description:
        "Unattended package detected in loading area for extended period.",
    },
    {
      time: "8 hours ago",
      camera: "4",
      location: "Storage Area",
      type: "Movement Detection",
      description: "After-hours movement detected in storage facility.",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Event Timeline</h1>
          <p className="mt-1 text-gray-500">
            Chronological view of all detected events
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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white rounded-lg border border-gray-100 p-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Time Range
          </label>
          <select
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="block w-full rounded-lg border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="today">Today</option>
            <option value="yesterday">Yesterday</option>
            <option value="week">Last 7 Days</option>
            <option value="month">Last 30 Days</option>
          </select>
        </div>
        <div className="bg-white rounded-lg border border-gray-100 p-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Camera
          </label>
          <select
            value={selectedCamera}
            onChange={(e) => setSelectedCamera(e.target.value)}
            className="block w-full rounded-lg border border-gray-300 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="all">All Cameras</option>
            <option value="1">Camera 1 - Main Entrance</option>
            <option value="2">Camera 2 - Loading Dock</option>
            <option value="3">Camera 3 - Parking Lot</option>
            <option value="4">Camera 4 - Storage Area</option>
          </select>
        </div>
      </div>

      {/* Timeline */}
      <div className="bg-white rounded-lg border border-gray-100 p-6">
        <div className="flow-root">
          <ul className="space-y-6">
            {events.map((event, index) => (
              <li key={index}>
                <TimelineEvent {...event} />
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Timeline;
