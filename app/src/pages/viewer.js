import React, { useRef, useState } from 'react';
import { useEffect } from 'react';
import './viewer.css';
import SearchBar from '../components/searchbar';
import { useNavigate, useLocation } from 'react-router-dom';
import ErrorPopup from '../components/errorpopup';
import BullitLogo from '../assets/bullit64.png';
import StockChart from '../components/stockChart';

const API_BASE = "https://catapult25.onrender.com"

const getData = (ticker) => fetch(`${API_BASE}/stock/${ticker}`)
        .then((response) => response.json())
        .then((data) => {
            return data;
        })
        .catch((error) => {
            return null;
        });

const getPredict = (ticker) => fetch(`${API_BASE}/predictor/${ticker}`)
        .then((res) => res.json())
        .then((data) => {
            return data
        })
        .catch((error) => {
            return null
        })

/*const getPredict = {
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
}*/

const today = new Date().toISOString().split('T')[0];

/*const today = '2025-01-10'*/

const getSentiment = (ticker) => fetch(`${API_BASE}/sentiment/${ticker}`)
    .then((response) => response.json())
    .then((data) => {
        return data;
    })
    .catch((error) => {
        return null;
    });

const Viewer = () => {
    const [ticker, setTicker] = useState();
    const [loading, setLoading] = useState(false);
    const [sentimentLoading, setSentimentLoading] = useState(true);
    const [predictionLoading, setPredictionLoading] = useState(true);
    const [errorMessage, setErrorMessage] = useState('');
    const [sentiment, setSentiment] = useState({'average_sentiment': 0.5, 'generated_conclusion': 'No sentiment data available.', 'articles_analyzed': 0});
    const [predict, setPredict] = useState({'x': [], 'y': []});

    const navigate = useNavigate();
    const location = useLocation();
    const stockTicker = location.state?.stockTicker;

    let data = location.state.stockData;
    const didFetchSentiment = useRef(false);
    const previousTicker = useRef(null);

    useEffect(() => {
        if (!stockTicker) {
            return
        }

        if (previousTicker.current !== stockTicker) {
            didFetchSentiment.current = false;
            previousTicker.current = stockTicker;
        }

        if (didFetchSentiment.current) {
            return
        }

        didFetchSentiment.current = true;

        const fetchSentiment = async () => {
            setSentimentLoading(true);
            const readSentiment = await getSentiment(stockTicker);
            if (readSentiment === null || readSentiment['error'] != null || readSentiment['average_sentiment'] === null) {
                setSentiment({'average_sentiment': 0.5, 'generated_conclusion': 'No sentiment data available.', 'articles_analyzed': 0});
                setSentimentLoading(false);
                return;
            }
            setSentiment(readSentiment);
            setSentimentLoading(false);
        };

        const fetchPrediction = async () => {
            setPredictionLoading(true);
            const readSentiment = await getPredict(stockTicker);
            if (readSentiment === null || readSentiment['error'] != null || readSentiment['average_sentiment'] === null) {
                setPredict({'x': [], 'y': []});
                setPredictionLoading(false);
                return;
            }
            setSentiment(readSentiment);
            setPredictionLoading(false);
        };
    
        if (stockTicker) {
            fetchPrediction();
            fetchSentiment();
        }
    }, [stockTicker]);

    const search = async (ticker) => {
        if (ticker === undefined || ticker === "") {
            return;
        }
        setLoading(true);
        const readData = await getData(ticker);
        console.log(readData);
        if (readData === null || readData['error'] != null || readData['open'] === null) {
            await handleError(ticker);
            setTicker("");
            setLoading(false);
            return;
        }
        data = readData;
        setTicker("");
        setLoading(false);
        setSentimentLoading(true);
        navigate('/viewer', {
            state:  {stockTicker: ticker, stockData: data} 
        });
    }

    const handleError = async (ticker) => {
        setErrorMessage('An error occurred! ' + ticker + ' is not a valid ticker.');
        setTimeout(() => {
          setErrorMessage('');
        }, 3000);
      };

    return (
        <div className='viewer viewer-header'>
            <div className='viewer-top-bar'>
                <div className='viewer-logo'>
                    <h1 className='viewer-app-name' onClick={() => {navigate('/')}}>BULL-IT</h1>
                    <img src={BullitLogo} alt='logo' onClick={() => {navigate('/')}}/>
                </div>
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
            {loading ? <div className="loading-circle"></div> : <div className='viewer-content'>
                <div className='viewer-sidebar'>
                    <h1 className='viewer-title viewer-label-text'>${stockTicker}</h1>
                    <h3 className='viewer-label-text'>{data['name:']}</h3>
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
                        {predictionLoading ? <div className="loading-circle"></div> : <div className='viewer-stats-vals'>
                            <p className={100 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>100</p>
                            <p className={110 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>110</p>
                            <p className={120 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>120</p>
                            <p className={130 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>130</p>
                            <p className={140 > data['open'] ? 'viewer-label-text viewer-value-green' : 'viewer-label-text viewer-value-red'}>140</p>
                        </div>}
                        <div className='viewer-stats-buff'></div>
                    </div>
                </div>
                <div className='viewer-charts'>
                    {predictionLoading ? <div className="loading-circle"></div> : <div className='viewer-graph'>
                        <StockChart xData={predict['dates']} yData={predict['prices']} splitDate={today}/>
                    </div>}
                    <div className='viewer-data'>
                        <h2 className='viewer-data-title'>{data['name:']} Sentiment</h2>
                        {sentimentLoading ? <div className="loading-circle"></div> : <div className='viewer-sentiment'>
                            <div className='viewer-ratings'>
                                <div className='viewer-ratings-label'>
                                    <p className='viewer-label-text'>Raw Score: {parseFloat(sentiment['average_sentiment'].toFixed(3))}</p>
                                    <p className='viewer-label-text'>Grade: {sentiment['average_sentiment'] < 0.2 ? 'F' : 
                                                                      sentiment['average_sentiment'] < 0.4 ? 'D' :
                                                                      sentiment['average_sentiment'] < 0.6 ? 'C' :
                                                                      sentiment['average_sentiment'] < 0.8 ? 'B' : 'A'}</p>
                                </div>
                            </div>
                            <div className='viewer-conclusion'>
                                <p className='viewer-label-text'>Conclusion:</p>
                                <div className='viewer-conclusion-text'><p className='viewer-label-text'>{sentiment['generated_conclusion']}</p></div>
                            </div>
                        </div>}
                    </div>
                </div>
                <ErrorPopup message={errorMessage} />
            </div>}
        </div>

    );
};

export default Viewer;