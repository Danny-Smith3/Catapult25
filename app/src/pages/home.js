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

/*const getPredict = (ticker) => fetch(`${API_BASE}/predictor/${ticker}`)
        .then((res) => res.json())
        .then((data) => {
            return data
        })
        .catch((error) => {
            return null
        })*/

const getPredict = {
  x: [
      '2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05',
      '2025-01-06', '2025-01-07', '2025-01-08', '2025-01-09', '2025-01-10',
      '2025-01-11', '2025-01-12', '2025-01-13', '2025-01-14', '2025-01-15'
    ],
  y: [
      150.23, 152.88, 151.76, 153.21, 154.10,
      153.75, 155.12, 156.03, 154.80, 153.90,
      155.30, 156.50, 157.20, 156.85, 158.00
    ]
}

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
      const readPredict = getPredict;
      if (readData === null || readData['error'] != null || readData['open'] === null) {
          await handleError(ticker);
          setTicker("");
          setLoading(false);
          return;
      }
      if (readPredict === null || readPredict.x.length === 0 || readPredict.y.length === 0) {
        await handleError(ticker);
        setTicker("");
        setLoading(false);
        return;
      }
      const data = readData;
      const predict = readPredict;
      setLoading(false);
      
      navigate('/viewer', {
        state:  {stockTicker: ticker, stockData: data, stockPredict: predict} 
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
