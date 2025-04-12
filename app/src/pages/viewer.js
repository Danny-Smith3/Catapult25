import React, { useState } from 'react';
import './viewer.css';
import SearchBar from '../components/searchbar';
import { useNavigate, useLocation } from 'react-router-dom';

const Viewer = () => {
    const [ticker, setTicker] = useState();

    const navigate = useNavigate();
    const location = useLocation();
    const stockTicker = location.state.stockTicker;
    const data = location.state.stockData;

    const search = (ticker) => {
        if (ticker === undefined || ticker === "") {
            return;
        }
        setTicker("");
        navigate('/viewer', {
            state: { stockTicker: ticker }
        });
    }

    return (
        <div className='viewer viewer-header'>
            <div className='viewer-top-bar'>
                <h1 className='viewer-app-name' onClick={() => {navigate('/')}}>BULL-IT</h1>
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
            <div className='viewer-content'>
                <div className='viewer-sidebar'>
                    <h1 className='viewer-title viewer-label-text'>${stockTicker}</h1>
                    <h3 className='viewer-label-text'>{data['name']}</h3>
                    <div className='viewer-stats'>
                        <div className='viewer-stats-buff'></div>
                        <div className='viewer-stats-label'>
                            <p className='viewer-label-text'>Open:</p>
                            <p className='viewer-label-text'>High:</p>
                            <p className='viewer-label-text'>Low:</p>
                            <p className='viewer-label-text'>Mkt cap:</p>
                            <p className='viewer-label-text'>P/E ratio:</p>
                            <p className='viewer-label-text'>Div yield:</p>
                            <p className='viewer-label-text'>52-wk high:</p>
                            <p className='viewer-label-text'>52-wk low:</p>
                        </div>
                        <div className='viewer-stats-vals'>
                        <p className='viewer-label-text'>{parseFloat(data['open'].toFixed(2))}</p>
                            <p className='viewer-label-text'>{parseFloat(data['high'].toFixed(2))}</p>
                            <p className='viewer-label-text'>{parseFloat(data['low'].toFixed(2))}</p>
                            <p className='viewer-label-text'>{data['market_cap']}T</p>
                            <p className='viewer-label-text'>{parseFloat(data['pe_ratio'].toFixed(2))}</p>
                            <p className='viewer-label-text'>0.{data['div_yield']}%</p>
                            <p className='viewer-label-text'>{data['52_week_high']}</p>
                            <p className='viewer-label-text'>{data['52_week_low']}</p>
                        </div>
                        <div className='viewer-stats-buff'></div>
                    </div>
                    <div className='viewer-sidebar-buff'></div>
                    <h3 className='viewer-label-text'>Predicted Value:</h3>
                    <div className='viewer-stats'>
                        <div className='viewer-stats-buff'></div>
                        <div className='viewer-stats-label'>
                            <p className='viewer-label-text'>1D:</p>
                            <p className='viewer-label-text'>1W:</p>
                            <p className='viewer-label-text'>1M:</p>
                            <p className='viewer-label-text'>3M:</p>
                            <p className='viewer-label-text'>1Y:</p>
                        </div>
                        <div className='viewer-stats-vals'>
                            <p className={100 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>100</p>
                            <p className={110 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>110</p>
                            <p className={120 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>120</p>
                            <p className={130 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>130</p>
                            <p className={140 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>140</p>
                        </div>
                        <div className='viewer-stats-buff'></div>
                    </div>
                </div>
                <div className='viewer-charts'>
                    <div className='viewer-graph'>
                        <p>{stockTicker} Graph</p>
                    </div>
                    <div className='viewer-data'>
                        <p>{stockTicker} Data</p>
                    </div>
                </div>
            </div>
        </div>

    );
};

export default Viewer;