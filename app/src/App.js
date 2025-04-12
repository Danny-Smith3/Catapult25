import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/home.js';
import Viewer from './pages/viewer.js';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/viewer" element={<Viewer />} />
      </Routes>
    </Router>
  );
}

export default App;
