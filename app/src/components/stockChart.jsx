import React from "react";
import Plot from "react-plotly.js";

const StockChart = ({ xData, yData, splitDate }) => {
    // Default to current date if splitDate is not provided
    const splitDateObj = splitDate ? new Date(splitDate) : new Date();
    
    // Split the data into "historical" (before splitDate) and "prediction" (after splitDate)
    const historicalIndices = [];
    const predictionIndices = [];
    
    xData.forEach((dateStr, index) => {
      const currentDate = new Date(dateStr);
      if (currentDate <= splitDateObj) {
        historicalIndices.push(index);
      } else {
        predictionIndices.push(index);
      }
    });
    
    // Create the arrays for historical data
    const historicalX = historicalIndices.map(i => xData[i]);
    const historicalY = historicalIndices.map(i => yData[i]);
    
    // Create the arrays for prediction data
    const predictionX = predictionIndices.map(i => xData[i]);
    const predictionY = predictionIndices.map(i => yData[i]);
    
    // If prediction starts right after historical, add the last historical point 
    // to the prediction data to connect the lines
    if (historicalIndices.length > 0 && predictionIndices.length > 0) {
      const lastHistoricalIndex = historicalIndices[historicalIndices.length - 1];
      predictionX.unshift(xData[lastHistoricalIndex]);
      predictionY.unshift(yData[lastHistoricalIndex]);
    }
    
    return (
      <Plot
        data={[
          // Historical data (solid line)
          {
            x: historicalX,
            y: historicalY,
            type: "scatter",
            mode: "lines+markers",
            marker: { color: "green" },
            line: { color: "green", width: 2 },
            name: "Historical",
          },
          // Prediction data (dashed line)
          {
            x: predictionX,
            y: predictionY,
            type: "scatter",
            mode: "lines+markers",
            marker: { color: "green" },
            line: { color: "green", width: 2, dash: "dash" },
            name: "Prediction",
          },
        ]}
        layout={{
          title: "Interactive Stock Price Chart",
          xaxis: { title: "Date" },
          yaxis: { title: "Price (USD)" },
          hovermode: "x unified",
          margin: { t: 50, l: 50, r: 50, b: 50 },
          plot_bgcolor: "#0f0f0f",
          paper_bgcolor: "#0f0f0f",
          font: { color: "#00ff88" },
          showlegend: true,
          legend: { x: 0, y: 1, font: { color: "#00ff88" } },
        }}
        config={{ responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    );
  };

export default StockChart;
