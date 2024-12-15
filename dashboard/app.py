# 

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List
import queue

class DashboardState:
    """Singleton class to maintain dashboard state."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.alert_queue = queue.Queue()
            cls._instance.market_data = {}
            cls._instance.alerts_history = []
        return cls._instance

def create_price_chart(symbol: str, data: pd.DataFrame):
    """Create price chart with volume bars."""
    fig = make_subplots(rows=2, cols=1, shared_xaxis=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume bars
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol} Price and Volume',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis2_title='Volume'
    )
    
    return fig

def create_metrics_chart(data: pd.DataFrame):
    """Create chart for technical metrics."""
    fig = make_subplots(rows=3, cols=1, shared_xaxis=True,
                       vertical_spacing=0.05)
    
    # Volatility
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['volatility'],
            name='Volatility'
        ),
        row=1, col=1
    )
    
    # Momentum
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['momentum'],
            name='Momentum'
        ),
        row=2, col=1
    )
    
    # Relative Volume
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['relative_volume'],
            name='Relative Volume'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=600,
        title='Technical Metrics',
        showlegend=True
    )
    
    return fig

def create_alerts_table(alerts: List[Dict]):
    """Create alerts summary table."""
    if not alerts:
        return pd.DataFrame()
        
    df = pd.DataFrame(alerts)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=False)
    
    return df[['timestamp', 'symbol', 'alert_type', 'severity', 'message']]

def main():
    st.set_page_config(
        page_title="Market Monitor",
        page_icon="📈",
        layout="wide"
    )
    
    state = DashboardState()
    
    # Sidebar
    st.sidebar.title("Market Monitor")
    selected_symbol = st.sidebar.selectbox(
        "Select Symbol",
        options=list(state.market_data.keys())
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.title("Market Overview")
        
        if selected_symbol in state.market_data:
            data = state.market_data[selected_symbol]
            
            # Price chart
            st.plotly_chart(
                create_price_chart(selected_symbol, data),
                use_container_width=True
            )
            
            # Metrics chart
            st.plotly_chart(
                create_metrics_chart(data),
                use_container_width=True
            )
    
    with col2:
        st.title("Alerts")
        
        # Recent alerts
        st.subheader("Recent Alerts")
        alerts_df = create_alerts_table(state.alerts_history[-50:])  # Last 50 alerts
        if not alerts_df.empty:
            st.dataframe(alerts_df, height=400)
        else:
            st.info("No recent alerts")
        
        # Alert statistics
        st.subheader("Alert Statistics")
        if state.alerts_history:
            alerts_df = pd.DataFrame(state.alerts_history)
            
            # Alert counts by type
            st.write("Alerts by Type")
            alert_counts = alerts_df['alert_type'].value_counts()
            st.bar_chart(alert_counts)
            
            # Alerts by severity
            st.write("Alerts by Severity")
            severity_counts = alerts_df['severity'].value_counts()
            st.bar_chart(severity_counts)

def update_dashboard_data():
    """Update dashboard data periodically."""
    state = DashboardState()
    
    while True:
        # Process new alerts
        while not state.alert_queue.empty():
            alert = state.alert_queue.get()
            state.alerts_history.append(alert)
            
            # Keep last 1000 alerts
            if len(state.alerts_history) > 1000:
                state.alerts_history = state.alerts_history[-1000:]
        
        # Update market data
        # This would be updated by your monitoring system
        
        time.sleep(1)  # Update every second

if __name__ == "__main__":
    import threading
    
    # Start update thread
    update_thread = threading.Thread(
        target=update_dashboard_data,
        daemon=True
    )
    update_thread.start()
    
    # Run dashboard
    main()
