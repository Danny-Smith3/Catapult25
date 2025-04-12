import React, { useState } from 'react';
import './home.css';
import SearchBar from '../components/searchbar';
import '@fortawesome/fontawesome-free/css/all.min.css';
import { useNavigate } from 'react-router-dom';

function Home() {
  const [ticker, setTicker] = useState();
  const navigate = useNavigate();

  const search = (ticker) => {
    if (ticker === undefined || ticker === "") {
      return;
    }
    console.log("Searched for: ", ticker);
    navigate('/viewer', {
        state: { stockTicker: ticker }
    });
  }

  return (
    <div className="home-app">
      <div className="home-app-header">
        <h1 className="home-app-name">NAME</h1>
        <div className='home-search-bar'> 
          <div className='home-ticker-search'>
            <SearchBar value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} onSearch={search} />
          </div>
          <div>
            <button className='home-search-button' onClick={(e) => search(ticker)}>
              <i className="fas fa-search"></i>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Home;
