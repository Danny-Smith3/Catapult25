import React, { useState, useEffect } from 'react';
import './errorpopup.css'; // Import the CSS file

function ErrorPopup({ message, duration = 2000 }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (message || message !== '') {
      setIsVisible(true);

      const timer = setTimeout(() => {
        setIsVisible(false);
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [message, duration]);

  return (
    isVisible && (
      <div className="error-popup">
        {message}
      </div>
    )
  );
}

export default ErrorPopup;