import React, { useState } from 'react';
import './viewer.css';
import SearchBar from '../components/searchbar';
import { useNavigate, useLocation } from 'react-router-dom';

const Viewer = () => {
    const [ticker, setTicker] = useState();
    const navigate = useNavigate();
    const location = useLocation();
    const {stockTicker} = location.state;

    const search = (ticker) => {
        if (ticker === undefined || ticker === "") {
            return;
        }
        setTicker("");
        console.log("Searched for: ", ticker);
        navigate('/viewer', {
            state: { stockTicker: ticker }
        });
    }

    return (
        <div className='viewer viewer-header'>

            <div className='viewer-top-bar'>
                <h1 className='viewer-app-name' onClick={() => {navigate('/')}}>NAME</h1>
                <div className='viewer-search-bar'>
                    <div className='viewer-ticker-search'>
                        <SearchBar value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} onSearch={search}/>
                    </div>
                    <div>
                        <button className='viewer-search-button' onClick={(e) => search(ticker)}>
                        <i className="fas fa-search"></i>
                        </button>
                    </div>
                </div>
                
            </div>
            <div>
                <h1>{stockTicker}</h1>
                <p>Welcome to the {stockTicker} page!</p>
            </div>
        </div>

    );
};

export default Viewer;