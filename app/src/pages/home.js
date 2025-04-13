import React, { useState } from 'react';
import './home.css';
import SearchBar from '../components/searchbar';
import '@fortawesome/fontawesome-free/css/all.min.css';
import { useNavigate } from 'react-router-dom';

const API_BASE = "https://catapult25.onrender.com"

const getData = (ticker) => fetch(`${API_BASE}/stock/${ticker}`)
        .then((response) => response.json())
        .then((data) => {
            return data;
        });

function Home() {
  const [ticker, setTicker] = useState();
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const search = async (ticker) => {
    if (ticker === undefined || ticker === "") {
      return;
    }
    else {
      setLoading(true);
      const data = await getData(ticker);
      setLoading(false);
      navigate('/viewer', {
        state:  {stockTicker: ticker, stockData: data} 
      });
    }
  }

  return (
    <div className="home-app">
      <div className="home-app-header">
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
      </div>
    </div>
  );
}

export default Home;
