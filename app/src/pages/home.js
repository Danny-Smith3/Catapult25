import React, { useState } from 'react';
import './home.css';
import SearchBar from '../components/searchbar';
import '@fortawesome/fontawesome-free/css/all.min.css';
import { useNavigate } from 'react-router-dom';
import ErrorPopup from '../components/errorpopup';
import BullitLogo from '../assets/bullit256.png';

const API_BASE = "https://catapult25.onrender.com"

const getData = (ticker) => fetch(`${API_BASE}/stock/${ticker}`)
        .then((response) => response.json())
        .then((data) => {
            return data;
        })
        .catch((error) => {
            return null;
        });

function Home() {
  const [ticker, setTicker] = useState();
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const navigate = useNavigate();

  const search = async (ticker) => {
    if (ticker === undefined || ticker === "") {
      return;
    }
    else {
      setLoading(true);
      const readData = await getData(ticker);
      if (readData === null || readData['error'] != null || readData['open'] === null) {
          await handleError(ticker);
          setTicker("");
          setLoading(false);
          return;
      }
      const data = readData;
      setLoading(false);
      
      navigate('/viewer', {
        state:  {stockTicker: ticker, stockData: data} 
      });
    }
  }

  const handleError = async (ticker) => {
    setErrorMessage('An error occurred! ' + ticker + ' is not a valid ticker.');
    setTimeout(() => {
      setErrorMessage('');
    }, 3000);
  };

  return (
    <div className="home-app">
      <div className="home-app-header">
        <img src={BullitLogo} alt='logo'/>
        <h1 className="home-app-name">BULL-IT</h1>
        {loading ? <div className="loading-circle"></div> : <div className='home-search-bar'> 
            <div className='home-ticker-search'>
              <SearchBar value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} onSearch={search} />
            </div>
            <div>
              <button className='home-search-button' onClick={(e) => search(ticker)}>
                <i className="fas fa-search"></i>
              </button>
          </div>
        </div>}
        <ErrorPopup message={errorMessage} />
      </div>
    </div>
  );
}

export default Home;
