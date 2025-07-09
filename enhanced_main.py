"""
Enhanced Main Entry Point for AI-Powered Signal Engine
Demonstrates dynamic TP/SL forecasting and 90%+ win rate targeting
"""
import asyncio
import argparse
import sys
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np

from src.config.settings import settings
from src.data.providers import DataProviderManager
from src.ml.signal_engine import AdvancedSignalEngine, SignalValidator
from src.ml.tp_sl_predictor import TPSLPredictor, TPSLOptimizer
from src.api.enhanced_main import app


async def run_advanced_signal_demo():
    """Run demonstration of the advanced AI signal engine"""
    logger.info("🚀 AI-Powered Signal Engine Demo Starting...")
    logger.info("=" * 60)
    
    # Initialize components
    data_manager = DataProviderManager()
    signal_engine = AdvancedSignalEngine()
    signal_validator = SignalValidator()
    tp_sl_predictor = TPSLPredictor()
    
    # Connect to data providers
    await data_manager.connect_all()
    
    try:
        # Demo 1: Generate advanced signals
        logger.info("📊 DEMO 1: Advanced Signal Generation")
        logger.info("-" * 40)
        
        symbols = ["EURUSD", "V75", "GBPUSD"]
        timeframe = "15m"
        
        for symbol in symbols:
            logger.info(f"\n🔍 Analyzing {symbol}...")
            
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = await data_manager.get_historical_data(symbol, timeframe, start_date, end_date)
            
            if not data.empty:
                # Generate advanced signal
                signal = await signal_engine.generate_signal(data, symbol, timeframe)
                
                if signal:
                    # Validate signal
                    validation = signal_validator.validate_signal(signal)
                    
                    logger.info(f"✅ Signal Generated for {symbol}:")
                    logger.info(f"   Direction: {signal.direction.upper()}")
                    logger.info(f"   Entry Price: {signal.entry_price:.5f}")
                    logger.info(f"   Take Profit: {signal.take_profit:.5f}")
                    logger.info(f"   Stop Loss: {signal.stop_loss:.5f}")
                    logger.info(f"   Confidence Score: {signal.confidence_score}%")
                    logger.info(f"   Risk/Reward Ratio: {signal.rr_ratio:.2f}")
                    logger.info(f"   Strategy: {signal.strategy}")
                    logger.info(f"   Validation: {'✅ VALID' if validation['is_valid'] else '❌ INVALID'}")
                    
                    # Show market structure analysis
                    logger.info(f"   Market Structure:")
                    logger.info(f"     - Trend: {signal.market_structure.get('trend_direction', 'N/A')}")
                    logger.info(f"     - Volatility: {signal.volatility:.4f}")
                    logger.info(f"     - Session: {signal.time_analysis.get('session', 'N/A')}")
                    
                    # Show alternative TP/SL levels
                    logger.info(f"   Alternative Levels:")
                    conservative_tp = signal.alternative_levels.get('conservative_tp', [])
                    aggressive_tp = signal.alternative_levels.get('aggressive_tp', [])
                    if conservative_tp:
                        logger.info(f"     - Conservative TP: {conservative_tp[0]:.5f} - {conservative_tp[1]:.5f}")
                    if aggressive_tp:
                        logger.info(f"     - Aggressive TP: {aggressive_tp[0]:.5f} - {aggressive_tp[1]:.5f}")
                
                else:
                    logger.info(f"❌ No signal generated for {symbol}")
            else:
                logger.info(f"❌ No data available for {symbol}")
        
        # Demo 2: TP/SL Prediction
        logger.info("\n" + "=" * 60)
        logger.info("📈 DEMO 2: TP/SL Prediction Engine")
        logger.info("-" * 40)
        
        # Example prediction for manual trade setup
        symbol = "EURUSD"
        direction = "buy"
        entry_price = 1.0950  # Example entry price
        
        data = await data_manager.get_historical_data(symbol, "1h", start_date, end_date)
        
        if not data.empty:
            signal_info = {
                'direction': direction,
                'entry_price': entry_price,
                'confidence': 0.85,
                'timestamp': datetime.now()
            }
            
            tp_sl_result = tp_sl_predictor.predict_optimal_tp_sl(data, signal_info)
            
            logger.info(f"\n🎯 TP/SL Prediction for {symbol}:")
            logger.info(f"   Entry Price: {entry_price:.5f}")
            logger.info(f"   Direction: {direction.upper()}")
            logger.info(f"   Optimal TP: {tp_sl_result['optimal_take_profit']:.5f}")
            logger.info(f"   Optimal SL: {tp_sl_result['optimal_stop_loss']:.5f}")
            logger.info(f"   Risk/Reward: {tp_sl_result['risk_reward_ratio']:.2f}")
            logger.info(f"   Confidence: {tp_sl_result['confidence']:.2%}")
            
            # Show all calculated levels
            all_levels = tp_sl_result.get('all_levels', {})
            if all_levels:
                logger.info(f"\n   📊 All Calculated Levels:")
                tp_levels = all_levels.get('take_profit', {})
                sl_levels = all_levels.get('stop_loss', {})
                
                logger.info(f"   Take Profit Options:")
                for method, price in tp_levels.items():
                    logger.info(f"     - {method}: {price:.5f}")
                
                logger.info(f"   Stop Loss Options:")
                for method, price in sl_levels.items():
                    logger.info(f"     - {method}: {price:.5f}")
        
        # Demo 3: TP/SL Optimization
        logger.info("\n" + "=" * 60)
        logger.info("⚙️ DEMO 3: TP/SL Optimization")
        logger.info("-" * 40)
        
        tp_sl_optimizer = TPSLOptimizer()
        
        # Generate sample signals for optimization
        sample_signals = []
        if not data.empty:
            for i in range(50, min(len(data), 200), 10):  # Sample signals
                sample_signals.append({
                    'entry_price': data['close'].iloc[i],
                    'direction': 'buy' if i % 2 == 0 else 'sell',
                    'timestamp': data.index[i],
                    'atr': data['close'].iloc[i] * 0.01  # Simplified ATR
                })
        
        if sample_signals:
            logger.info(f"🔧 Running optimization with {len(sample_signals)} sample signals...")
            
            optimization_result = tp_sl_optimizer.optimize_tp_sl_combinations(
                data, sample_signals,
                tp_range=(1.0, 3.0),
                sl_range=(0.5, 1.5),
                step_size=0.25
            )
            
            best = optimization_result['best_combination']
            logger.info(f"\n✅ Optimization Results:")
            logger.info(f"   Best TP Multiplier: {best['tp_multiplier']}x")
            logger.info(f"   Best SL Multiplier: {best['sl_multiplier']}x")
            logger.info(f"   Expected Win Rate: {best['win_rate']:.1f}%")
            logger.info(f"   Expected Profit Factor: {best['profit_factor']:.2f}")
            logger.info(f"   Total Combinations Tested: {len(optimization_result['all_results'])}")
            
            # Show top 5 combinations
            top_combinations = sorted(optimization_result['all_results'], 
                                    key=lambda x: x['score'], reverse=True)[:5]
            
            logger.info(f"\n   🏆 Top 5 Combinations:")
            for i, combo in enumerate(top_combinations, 1):
                logger.info(f"   {i}. TP:{combo['tp_multiplier']}x SL:{combo['sl_multiplier']}x "
                           f"Win:{combo['win_rate']:.1f}% PF:{combo['profit_factor']:.2f}")
        
        # Demo 4: Real-time Signal Generation
        logger.info("\n" + "=" * 60)
        logger.info("🔄 DEMO 4: Real-time Signal Generation")
        logger.info("-" * 40)
        
        logger.info("Generating signals for multiple symbols concurrently...")
        
        symbols = settings.DEFAULT_SYMBOLS[:3]  # Top 3 symbols
        signals = await signal_engine.batch_generate_signals(symbols, "15m", data_manager)
        
        if signals:
            logger.info(f"\n✅ Generated {len(signals)} real-time signals:")
            
            for signal in signals:
                validation = signal_validator.validate_signal(signal)
                status = "✅ VALID" if validation['is_valid'] else "⚠️ WARNING"
                
                logger.info(f"\n   🎯 {signal.pair} - {signal.direction.upper()} @ {signal.entry_price:.5f}")
                logger.info(f"      Confidence: {signal.confidence_score}% | R:R: {signal.rr_ratio:.2f} | {status}")
                logger.info(f"      TP: {signal.take_profit:.5f} | SL: {signal.stop_loss:.5f}")
                logger.info(f"      Strategy: {signal.strategy}")
                
                if validation['warnings']:
                    logger.info(f"      Warnings: {', '.join(validation['warnings'])}")
        else:
            logger.info("❌ No signals generated in current market conditions")
        
        # Demo 5: Performance Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 DEMO 5: Engine Performance Summary")
        logger.info("-" * 40)
        
        performance = signal_engine.get_performance_summary()
        
        logger.info(f"\n🎯 Signal Engine Performance:")
        logger.info(f"   Total Signals Generated: {performance['total_signals_generated']}")
        logger.info(f"   Average Confidence Score: {performance['avg_confidence_score']:.1f}%")
        logger.info(f"   Average Risk/Reward Ratio: {performance['avg_risk_reward_ratio']:.2f}")
        logger.info(f"   Recent Signals (1h): {performance['recent_signals']}")
        
        logger.info(f"\n⚙️ Strategy Weights:")
        for strategy, weight in performance['strategy_weights'].items():
            logger.info(f"   - {strategy}: {weight:.1%}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 AI-Powered Signal Engine Demo Complete!")
        logger.info("Ready for 90%+ win rate targeting! 🚀")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    
    finally:
        await data_manager.disconnect_all()


async def run_live_monitoring():
    """Run live signal monitoring demonstration"""
    logger.info("📡 Starting Live Signal Monitoring...")
    
    data_manager = DataProviderManager()
    signal_engine = AdvancedSignalEngine()
    signal_validator = SignalValidator()
    
    await data_manager.connect_all()
    
    try:
        symbols = ["EURUSD", "V75", "GBPUSD"]
        
        while True:
            logger.info(f"\n🔍 Scanning {', '.join(symbols)} at {datetime.now().strftime('%H:%M:%S')}")
            
            # Generate signals for all symbols
            signals = await signal_engine.batch_generate_signals(symbols, "5m", data_manager)
            
            if signals:
                for signal in signals:
                    validation = signal_validator.validate_signal(signal)
                    
                    if validation['is_valid']:
                        logger.info(f"🚨 NEW SIGNAL ALERT!")
                        logger.info(f"   {signal.pair} {signal.direction.upper()} @ {signal.entry_price:.5f}")
                        logger.info(f"   Confidence: {signal.confidence_score}% | R:R: {signal.rr_ratio:.2f}")
                        logger.info(f"   TP: {signal.take_profit:.5f} | SL: {signal.stop_loss:.5f}")
            else:
                logger.info("   No valid signals detected")
            
            # Wait 1 minute before next scan
            await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Live monitoring stopped")
    except Exception as e:
        logger.error(f"Live monitoring error: {e}")
    finally:
        await data_manager.disconnect_all()


def run_api_server():
    """Run the enhanced FastAPI server"""
    import uvicorn
    
    logger.info("🚀 Starting Enhanced AI Signal Engine API Server...")
    uvicorn.run(
        "src.api.enhanced_main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=True
    )


def main():
    """Enhanced main function with AI signal engine demonstrations"""
    parser = argparse.ArgumentParser(description="AI-Powered Signal Engine with Dynamic TP/SL Forecasting")
    parser.add_argument(
        "command",
        choices=["demo", "live", "api", "signals", "optimize"],
        help="Command to run"
    )
    parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Trading symbol (default: EURUSD)"
    )
    parser.add_argument(
        "--timeframe",
        default="15m",
        choices=settings.DEFAULT_TIMEFRAMES,
        help="Data timeframe (default: 15m)"
    )
    
    args = parser.parse_args()
    
    # Configure enhanced logging
    logger.add(
        settings.LOG_FILE,
        rotation="10 MB",
        retention="30 days",
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    logger.info("🤖 AI-Powered Signal Engine Starting...")
    logger.info(f"Command: {args.command}")
    logger.info(f"Target Win Rate: 90%+")
    logger.info(f"Features: Dynamic TP/SL, Market Structure Analysis, ML Predictions")
    
    try:
        if args.command == "demo":
            asyncio.run(run_advanced_signal_demo())
        elif args.command == "live":
            asyncio.run(run_live_monitoring())
        elif args.command == "api":
            run_api_server()
        elif args.command == "signals":
            # Quick signal generation for specific symbol
            asyncio.run(generate_single_signal(args.symbol, args.timeframe))
        elif args.command == "optimize":
            # Quick TP/SL optimization
            asyncio.run(run_quick_optimization(args.symbol, args.timeframe))
        
    except KeyboardInterrupt:
        logger.info("⏹️ Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


async def generate_single_signal(symbol: str, timeframe: str):
    """Generate a single signal for testing"""
    logger.info(f"🎯 Generating signal for {symbol} ({timeframe})...")
    
    data_manager = DataProviderManager()
    signal_engine = AdvancedSignalEngine()
    
    await data_manager.connect_all()
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = await data_manager.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if not data.empty:
            signal = await signal_engine.generate_signal(data, symbol, timeframe)
            
            if signal:
                logger.info(f"✅ Signal Generated!")
                logger.info(f"Direction: {signal.direction.upper()}")
                logger.info(f"Entry: {signal.entry_price:.5f}")
                logger.info(f"TP: {signal.take_profit:.5f}")
                logger.info(f"SL: {signal.stop_loss:.5f}")
                logger.info(f"Confidence: {signal.confidence_score}%")
                logger.info(f"R:R Ratio: {signal.rr_ratio:.2f}")
                
                # Print in the required JSON format
                output = {
                    "signal_id": signal.signal_id,
                    "pair": signal.pair,
                    "timeframe": signal.timeframe,
                    "direction": signal.direction,
                    "entry_price": signal.entry_price,
                    "take_profit": signal.take_profit,
                    "stop_loss": signal.stop_loss,
                    "strategy": signal.strategy,
                    "confidence_score": signal.confidence_score,
                    "rr_ratio": signal.rr_ratio
                }
                
                logger.info("\n📋 Signal Output (JSON):")
                import json
                print(json.dumps(output, indent=2))
            else:
                logger.info("❌ No signal generated")
        else:
            logger.info("❌ No data available")
    
    finally:
        await data_manager.disconnect_all()


async def run_quick_optimization(symbol: str, timeframe: str):
    """Run quick TP/SL optimization for testing"""
    logger.info(f"⚙️ Running TP/SL optimization for {symbol} ({timeframe})...")
    
    from src.ml.tp_sl_predictor import TPSLOptimizer
    
    data_manager = DataProviderManager()
    optimizer = TPSLOptimizer()
    
    await data_manager.connect_all()
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        
        data = await data_manager.get_historical_data(symbol, timeframe, start_date, end_date)
        
        if not data.empty:
            # Generate sample signals
            signals = []
            for i in range(50, len(data), 20):
                signals.append({
                    'entry_price': data['close'].iloc[i],
                    'direction': 'buy' if i % 2 == 0 else 'sell',
                    'timestamp': data.index[i],
                    'atr': data['close'].iloc[i] * 0.01
                })
            
            result = optimizer.optimize_tp_sl_combinations(data, signals)
            best = result['best_combination']
            
            logger.info(f"✅ Optimization Complete!")
            logger.info(f"Best TP Multiplier: {best['tp_multiplier']}x")
            logger.info(f"Best SL Multiplier: {best['sl_multiplier']}x")
            logger.info(f"Win Rate: {best['win_rate']:.1f}%")
            logger.info(f"Profit Factor: {best['profit_factor']:.2f}")
        else:
            logger.info("❌ No data available for optimization")
    
    finally:
        await data_manager.disconnect_all()


if __name__ == "__main__":
    main()