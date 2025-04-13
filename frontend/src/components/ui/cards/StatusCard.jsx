import React from "react";

const StatusCard = ({ title, status, color }) => (
  <div className={`p-4 rounded-lg bg-${color}-50 border border-${color}-100`}>
    <p className={`text-sm font-medium text-${color}-800`}>{title}</p>
    <p className={`mt-1 text-sm text-${color}-600`}>{status}</p>
  </div>
);

export default StatusCard;
