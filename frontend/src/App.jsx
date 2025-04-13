import React from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import routes from "./routes";
import "./App.css";

function App() {
  const renderRoutes = (routes) => {
    return routes.map((route) => {
      if (route.children) {
        return (
          <Route key={route.path} path={route.path} element={route.element}>
            {renderRoutes(route.children)}
          </Route>
        );
      }
      return (
        <Route
          key={route.path || "index"}
          path={route.path}
          index={route.index}
          element={route.element}
        />
      );
    });
  };

  return (
    <Router>
      <Routes>{renderRoutes(routes)}</Routes>
    </Router>
  );
}

export default App;
