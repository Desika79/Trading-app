"""
Enhanced FastAPI Main Application for AI-Powered Trading System
Includes dynamic TP/SL forecasting and advanced signal generation
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from loguru import logger
import uvicorn

from ..config.settings import settings
from ..data.providers import DataProviderManager
from ..ml.signal_engine import AdvancedSignalEngine, SignalValidator, AdvancedSignal
from ..ml.tp_sl_predictor import TPSLPredictor, TPSLOptimizer
from ..indicators.technical import calculate_all_indicators
from .models import *

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Signal Engine with Dynamic TP/SL Forecasting",
    description="Advanced Forex & V75 Trading System with 90%+ accuracy targeting",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
data_manager = DataProviderManager()
signal_engine = AdvancedSignalEngine()
signal_validator = SignalValidator()
tp_sl_predictor = TPSLPredictor()
tp_sl_optimizer = TPSLOptimizer()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove dead connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Active signals storage
active_signals: List[AdvancedSignal] = []
signal_performance_log: List[Dict[str, Any]] = []


@app.on_event("startup")
async def startup_event():
    """Initialize connections and load models on startup"""
    logger.info("Starting AI-Powered Signal Engine...")
    
    # Connect to data providers
    connection_results = await data_manager.connect_all()
    logger.info(f"Data provider connections: {connection_results}")
    
    # Try to load pre-trained TP/SL models
    try:
        tp_sl_predictor.load_model("models/tp_sl_model.joblib")
        logger.info("Loaded pre-trained TP/SL models")
    except:
        logger.info("No pre-trained models found, will use rule-based TP/SL")
    
    # Start background signal generation
    asyncio.create_task(background_signal_generation())


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    logger.info("Shutting down AI Signal Engine...")
    await data_manager.disconnect_all()


async def background_signal_generation():
    """Background task for continuous signal generation"""
    while True:
        try:
            # Generate signals for default symbols every 5 minutes
            symbols = settings.DEFAULT_SYMBOLS[:5]  # Monitor top 5 symbols
            timeframe = "15m"  # Use 15-minute timeframe
            
            new_signals = await signal_engine.batch_generate_signals(symbols, timeframe, data_manager)
            
            # Validate and filter signals
            validated_signals = []
            for signal in new_signals:
                validation = signal_validator.validate_signal(signal)
                if validation['is_valid']:
                    validated_signals.append(signal)
            
            # Add to active signals
            active_signals.extend(validated_signals)
            
            # Keep only recent signals (last 100)
            active_signals[:] = active_signals[-100:]
            
            # Broadcast new signals via WebSocket
            for signal in validated_signals:
                await manager.broadcast(json.dumps({
                    "type": "new_signal",
                    "data": signal.to_dict()
                }))
            
            if validated_signals:
                logger.info(f"Generated {len(validated_signals)} new validated signals")
            
            # Wait 5 minutes before next generation
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Error in background signal generation: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with system overview"""
    return """
    <html>
        <head>
            <title>AI-Powered Signal Engine</title>
        </head>
        <body>
            <h1>🤖 AI-Powered Signal Engine</h1>
            <h2>Dynamic TP/SL Forecasting for Forex & V75</h2>
            <p><strong>Status:</strong> ✅ Active</p>
            <p><strong>Version:</strong> 2.0.0</p>
            <p><strong>Target Win Rate:</strong> 90%+</p>
            
            <h3>🔗 Quick Links</h3>
            <ul>
                <li><a href="/docs">📚 API Documentation</a></li>
                <li><a href="/signals/live">📊 Live Signals</a></li>
                <li><a href="/monitor">📈 System Monitor</a></li>
                <li><a href="/optimization/summary">⚙️ Optimization Status</a></li>
            </ul>
            
            <h3>📡 WebSocket Real-time Updates</h3>
            <p>Connect to: <code>ws://localhost:8000/ws</code></p>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Enhanced health check with signal engine status"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "signal_engine": {
            "active": True,
            "total_signals_generated": signal_engine.performance_metrics['total_signals'],
            "avg_confidence": signal_engine.performance_metrics.get('avg_confidence', 0),
            "active_signals_count": len(active_signals)
        },
        "data_providers": {
            name: provider.is_connected 
            for name, provider in data_manager.providers.items()
        },
        "tp_sl_predictor": {
            "trained": tp_sl_predictor.is_trained,
            "optimizer_status": tp_sl_optimizer.get_optimization_summary()['status']
        }
    }


# Enhanced Signal Generation Endpoints
@app.post("/signals/generate/advanced")
async def generate_advanced_signal(symbol: str, timeframe: str = "15m"):
    """Generate advanced AI-powered signal with dynamic TP/SL"""
    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = await data_manager.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Generate signal
        signal = await signal_engine.generate_signal(data, symbol, timeframe)
        
        if not signal:
            return {"message": "No signal generated", "symbol": symbol, "timestamp": datetime.now()}
        
        # Validate signal
        validation = signal_validator.validate_signal(signal)
        
        response = {
            "signal": signal.to_dict(),
            "validation": validation,
            "generated_at": datetime.now()
        }
        
        # Add to active signals if valid
        if validation['is_valid']:
            active_signals.append(signal)
            # Broadcast via WebSocket
            await manager.broadcast(json.dumps({
                "type": "new_signal",
                "data": signal.to_dict()
            }))
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating advanced signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signals/live")
async def get_live_signals():
    """Get all active live signals"""
    return {
        "total_signals": len(active_signals),
        "signals": [signal.to_dict() for signal in active_signals[-20:]],  # Last 20 signals
        "timestamp": datetime.now()
    }


@app.get("/signals/{signal_id}")
async def get_signal_details(signal_id: str):
    """Get detailed information about a specific signal"""
    signal = next((s for s in active_signals if s.signal_id == signal_id), None)
    
    if not signal:
        raise HTTPException(status_code=404, detail="Signal not found")
    
    return {
        "signal": signal.to_dict(),
        "additional_analysis": {
            "market_structure_analysis": signal.market_structure,
            "fibonacci_levels": signal.fibonacci_levels,
            "alternative_tp_sl_levels": signal.alternative_levels,
            "risk_assessment": signal.risk_metrics
        }
    }


# TP/SL Prediction Endpoints
@app.post("/tp-sl/predict")
async def predict_tp_sl(symbol: str, direction: str, entry_price: float, timeframe: str = "15m"):
    """Predict optimal TP/SL levels for a given trade setup"""
    try:
        # Get current market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = await data_manager.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Create signal info
        signal_info = {
            'direction': direction.lower(),
            'entry_price': entry_price,
            'confidence': 0.8,  # Default confidence
            'timestamp': datetime.now()
        }
        
        # Predict TP/SL
        tp_sl_result = tp_sl_predictor.predict_optimal_tp_sl(data, signal_info)
        
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "predictions": tp_sl_result,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error predicting TP/SL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tp-sl/optimize")
async def optimize_tp_sl_parameters(
    background_tasks: BackgroundTasks,
    symbol: str,
    timeframe: str = "1h",
    days_back: int = 90
):
    """Optimize TP/SL parameters using historical data"""
    try:
        # Start optimization in background
        background_tasks.add_task(
            run_tp_sl_optimization,
            symbol,
            timeframe,
            days_back
        )
        
        return {
            "message": "TP/SL optimization started",
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Error starting TP/SL optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_tp_sl_optimization(symbol: str, timeframe: str, days_back: int):
    """Background task for TP/SL optimization"""
    try:
        logger.info(f"Starting TP/SL optimization for {symbol}")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        data = await data_manager.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            logger.error(f"No data for optimization: {symbol}")
            return
        
        # Generate sample signals for optimization (simplified)
        sample_signals = []
        for i in range(50, len(data), 20):  # Sample every 20 bars
            sample_signals.append({
                'entry_price': data['close'].iloc[i],
                'direction': 'buy' if i % 2 == 0 else 'sell',
                'timestamp': data.index[i],
                'atr': data['close'].iloc[i] * 0.01  # Simplified ATR
            })
        
        # Run optimization
        optimization_result = tp_sl_optimizer.optimize_tp_sl_combinations(
            data, sample_signals
        )
        
        logger.info(f"TP/SL optimization completed for {symbol}")
        logger.info(f"Best combination: TP {optimization_result['best_combination']['tp_multiplier']}x, "
                   f"SL {optimization_result['best_combination']['sl_multiplier']}x")
        
    except Exception as e:
        logger.error(f"TP/SL optimization failed: {e}")


@app.get("/optimization/summary")
async def get_optimization_summary():
    """Get TP/SL optimization summary"""
    return tp_sl_optimizer.get_optimization_summary()


@app.get("/optimization/heatmap")
async def get_optimization_heatmap():
    """Get TP/SL optimization heatmap data"""
    if not tp_sl_optimizer.optimization_results:
        raise HTTPException(status_code=404, detail="No optimization results available")
    
    return {
        "heatmap_data": tp_sl_optimizer.optimization_results['heatmap_data'],
        "optimization_date": tp_sl_optimizer.optimization_results['optimization_date']
    }


# Market Structure Analysis Endpoints
@app.post("/analysis/market-structure")
async def analyze_market_structure(symbol: str, timeframe: str = "1h"):
    """Perform comprehensive market structure analysis"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = await data_manager.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Calculate indicators
        data_with_indicators = calculate_all_indicators(data)
        
        # Analyze market structure
        market_structure = await signal_engine._analyze_market_structure(data_with_indicators)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis": market_structure,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Real-time WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time signal updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(30)  # Send update every 30 seconds
            
            status_update = {
                "type": "status_update",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "active_signals": len(active_signals),
                    "engine_performance": signal_engine.get_performance_summary()
                }
            }
            
            await manager.send_personal_message(json.dumps(status_update), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Enhanced Monitoring Dashboard
@app.get("/monitor")
async def get_enhanced_monitor():
    """Get comprehensive system monitoring data"""
    try:
        # Get current market data for top symbols
        current_data = {}
        for symbol in settings.DEFAULT_SYMBOLS[:5]:
            live_data = await data_manager.get_live_data(symbol)
            if live_data:
                current_data[symbol] = live_data
        
        # Signal engine performance
        engine_performance = signal_engine.get_performance_summary()
        
        # Recent signal statistics
        recent_signals = [s for s in active_signals 
                         if (datetime.now() - s.timestamp).total_seconds() < 3600]
        
        signal_stats = {
            "total_active": len(active_signals),
            "recent_1h": len(recent_signals),
            "avg_confidence": np.mean([s.confidence_score for s in recent_signals]) if recent_signals else 0,
            "avg_rr_ratio": np.mean([s.rr_ratio for s in recent_signals if s.rr_ratio > 0]) if recent_signals else 0,
            "direction_distribution": {
                "buy": len([s for s in recent_signals if s.direction == 'buy']),
                "sell": len([s for s in recent_signals if s.direction == 'sell'])
            }
        }
        
        return {
            "timestamp": datetime.now(),
            "market_data": current_data,
            "signal_engine": {
                "performance": engine_performance,
                "statistics": signal_stats,
                "tp_sl_predictor_trained": tp_sl_predictor.is_trained
            },
            "data_providers": {
                name: provider.is_connected 
                for name, provider in data_manager.providers.items()
            },
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Error getting monitor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Signal Performance Tracking
@app.post("/signals/{signal_id}/update")
async def update_signal_performance(signal_id: str, outcome: str, actual_exit_price: float):
    """Update signal performance with actual trade outcome"""
    try:
        signal = next((s for s in active_signals if s.signal_id == signal_id), None)
        
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        # Calculate actual performance
        entry_price = signal.entry_price
        
        if signal.direction == 'buy':
            actual_pnl = actual_exit_price - entry_price
        else:
            actual_pnl = entry_price - actual_exit_price
        
        # Store performance data
        performance_record = {
            "signal_id": signal_id,
            "symbol": signal.pair,
            "direction": signal.direction,
            "entry_price": entry_price,
            "predicted_tp": signal.take_profit,
            "predicted_sl": signal.stop_loss,
            "actual_exit_price": actual_exit_price,
            "actual_pnl": actual_pnl,
            "outcome": outcome,  # 'tp_hit', 'sl_hit', 'manual_close'
            "confidence_score": signal.confidence_score,
            "predicted_rr": signal.rr_ratio,
            "timestamp": datetime.now()
        }
        
        signal_performance_log.append(performance_record)
        
        return {
            "message": "Signal performance updated",
            "performance": performance_record
        }
        
    except Exception as e:
        logger.error(f"Error updating signal performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance/summary")
async def get_performance_summary():
    """Get overall signal performance summary"""
    if not signal_performance_log:
        return {"message": "No performance data available"}
    
    total_signals = len(signal_performance_log)
    successful_signals = len([p for p in signal_performance_log if p['actual_pnl'] > 0])
    
    win_rate = (successful_signals / total_signals) * 100 if total_signals > 0 else 0
    
    avg_pnl = np.mean([p['actual_pnl'] for p in signal_performance_log])
    total_pnl = sum(p['actual_pnl'] for p in signal_performance_log)
    
    return {
        "total_signals": total_signals,
        "successful_signals": successful_signals,
        "win_rate": round(win_rate, 2),
        "avg_pnl": round(avg_pnl, 6),
        "total_pnl": round(total_pnl, 6),
        "performance_by_symbol": _calculate_symbol_performance(),
        "performance_by_direction": _calculate_direction_performance()
    }


def _calculate_symbol_performance():
    """Calculate performance by symbol"""
    symbol_performance = {}
    
    for record in signal_performance_log:
        symbol = record['symbol']
        if symbol not in symbol_performance:
            symbol_performance[symbol] = {
                'total': 0,
                'successful': 0,
                'total_pnl': 0
            }
        
        symbol_performance[symbol]['total'] += 1
        if record['actual_pnl'] > 0:
            symbol_performance[symbol]['successful'] += 1
        symbol_performance[symbol]['total_pnl'] += record['actual_pnl']
    
    # Calculate win rates
    for symbol in symbol_performance:
        data = symbol_performance[symbol]
        data['win_rate'] = (data['successful'] / data['total']) * 100 if data['total'] > 0 else 0
    
    return symbol_performance


def _calculate_direction_performance():
    """Calculate performance by direction"""
    direction_performance = {'buy': {'total': 0, 'successful': 0, 'total_pnl': 0},
                           'sell': {'total': 0, 'successful': 0, 'total_pnl': 0}}
    
    for record in signal_performance_log:
        direction = record['direction']
        direction_performance[direction]['total'] += 1
        if record['actual_pnl'] > 0:
            direction_performance[direction]['successful'] += 1
        direction_performance[direction]['total_pnl'] += record['actual_pnl']
    
    # Calculate win rates
    for direction in direction_performance:
        data = direction_performance[direction]
        data['win_rate'] = (data['successful'] / data['total']) * 100 if data['total'] > 0 else 0
    
    return direction_performance


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.enhanced_main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )