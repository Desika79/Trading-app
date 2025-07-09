"""
Advanced AI-Powered Signal Engine
Generates high-accuracy trading signals with dynamic TP/SL forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from loguru import logger
import json
from dataclasses import dataclass, asdict

from ..strategies.base import BaseStrategy, TradeSignal
from ..indicators.technical import TechnicalIndicators, calculate_all_indicators
from .tp_sl_predictor import TPSLPredictor, MarketStructureAnalyzer
from ..config.settings import settings


@dataclass
class AdvancedSignal:
    """Enhanced signal with comprehensive TP/SL analysis"""
    signal_id: str
    pair: str
    timeframe: str
    direction: str
    entry_price: float
    take_profit: float
    stop_loss: float
    strategy: str
    confidence_score: float
    rr_ratio: float
    timestamp: datetime
    
    # Additional metadata
    atr_value: float
    volatility: float
    market_structure: Dict[str, Any]
    fibonacci_levels: Dict[str, float]
    swing_levels: Dict[str, float]
    volume_analysis: Dict[str, float]
    time_analysis: Dict[str, Any]
    risk_metrics: Dict[str, float]
    ml_predictions: Dict[str, Any]
    alternative_levels: Dict[str, List[float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary format"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert signal to JSON format"""
        data = self.to_dict()
        # Convert datetime to string for JSON serialization
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data, indent=2)


class AdvancedSignalEngine:
    """AI-powered signal engine with dynamic TP/SL forecasting"""
    
    def __init__(self):
        self.tp_sl_predictor = TPSLPredictor()
        self.market_analyzer = MarketStructureAnalyzer()
        self.indicators = TechnicalIndicators()
        
        # Strategy weights for ensemble signals
        self.strategy_weights = {
            'rsi_divergence': 0.25,
            'atr_breakout': 0.20,
            'bollinger_squeeze': 0.15,
            'ma_cross': 0.15,
            'market_structure': 0.25
        }
        
        # Performance tracking
        self.signal_history: List[AdvancedSignal] = []
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'avg_accuracy': 0.0,
            'avg_rr_ratio': 0.0
        }
        
        logger.info("Advanced Signal Engine initialized")
    
    async def generate_signal(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Optional[AdvancedSignal]:
        """Generate comprehensive trading signal with AI-powered TP/SL"""
        try:
            if len(data) < 100:  # Need sufficient data
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Calculate all technical indicators
            data_with_indicators = calculate_all_indicators(data)
            
            # Analyze market structure
            market_structure = await self._analyze_market_structure(data_with_indicators)
            
            # Generate individual strategy signals
            strategy_signals = await self._generate_strategy_signals(data_with_indicators, market_structure)
            
            # Calculate ensemble signal
            ensemble_result = self._calculate_ensemble_signal(strategy_signals)
            
            if not ensemble_result or ensemble_result['confidence'] < settings.MIN_WIN_RATE:
                return None
            
            # Extract signal information
            signal_info = {
                'direction': ensemble_result['direction'],
                'entry_price': data_with_indicators['close'].iloc[-1],
                'confidence': ensemble_result['confidence'],
                'timestamp': data_with_indicators.index[-1]
            }
            
            # Predict optimal TP/SL using AI
            tp_sl_result = self.tp_sl_predictor.predict_optimal_tp_sl(data_with_indicators, signal_info)
            
            # Perform additional analysis
            additional_analysis = await self._perform_additional_analysis(data_with_indicators, signal_info)
            
            # Create advanced signal
            signal = AdvancedSignal(
                signal_id=self._generate_signal_id(symbol, timeframe),
                pair=symbol,
                timeframe=timeframe,
                direction=signal_info['direction'],
                entry_price=signal_info['entry_price'],
                take_profit=tp_sl_result['optimal_take_profit'],
                stop_loss=tp_sl_result['optimal_stop_loss'],
                strategy=ensemble_result['primary_strategy'],
                confidence_score=int(ensemble_result['confidence'] * 100),
                rr_ratio=tp_sl_result['risk_reward_ratio'],
                timestamp=signal_info['timestamp'],
                atr_value=additional_analysis['atr_value'],
                volatility=additional_analysis['volatility'],
                market_structure=market_structure,
                fibonacci_levels=additional_analysis['fibonacci_levels'],
                swing_levels=additional_analysis['swing_levels'],
                volume_analysis=additional_analysis['volume_analysis'],
                time_analysis=additional_analysis['time_analysis'],
                risk_metrics=additional_analysis['risk_metrics'],
                ml_predictions=tp_sl_result['ml_predictions'],
                alternative_levels=additional_analysis['alternative_levels']
            )
            
            # Store signal for performance tracking
            self.signal_history.append(signal)
            self._update_performance_metrics()
            
            logger.info(f"Generated signal: {signal.pair} {signal.direction} @ {signal.entry_price:.5f}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def _analyze_market_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure and identify key patterns"""
        try:
            # Find swing points
            swing_points = self.market_analyzer.find_swing_points(data)
            
            # Detect structure breaks
            structure_breaks = self.market_analyzer.detect_market_structure_breaks(data, swing_points)
            
            # Analyze trend
            current_price = data['close'].iloc[-1]
            sma_20 = data['sma_20'].iloc[-1] if 'sma_20' in data.columns else current_price
            sma_50 = data['sma_50'].iloc[-1] if 'sma_50' in data.columns else current_price
            
            trend_direction = 'neutral'
            if current_price > sma_20 > sma_50:
                trend_direction = 'bullish'
            elif current_price < sma_20 < sma_50:
                trend_direction = 'bearish'
            
            # Calculate trend strength
            trend_strength = abs(current_price - sma_20) / sma_20 if sma_20 > 0 else 0
            
            # Identify consolidation patterns
            volatility = data['close'].rolling(20).std().iloc[-1] / current_price
            is_consolidating = volatility < 0.01  # Less than 1% volatility
            
            return {
                'swing_points': swing_points,
                'structure_breaks': structure_breaks,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'is_consolidating': is_consolidating,
                'volatility_regime': 'low' if volatility < 0.01 else 'high' if volatility > 0.03 else 'normal'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market structure: {e}")
            return {}
    
    async def _generate_strategy_signals(self, data: pd.DataFrame, market_structure: Dict) -> Dict[str, Dict]:
        """Generate signals from individual strategies"""
        signals = {}
        
        try:
            # RSI Divergence Strategy
            signals['rsi_divergence'] = self._rsi_divergence_signal(data)
            
            # ATR Breakout Strategy
            signals['atr_breakout'] = self._atr_breakout_signal(data)
            
            # Bollinger Band Squeeze Strategy
            signals['bollinger_squeeze'] = self._bollinger_squeeze_signal(data)
            
            # Moving Average Cross Strategy
            signals['ma_cross'] = self._ma_cross_signal(data)
            
            # Market Structure Strategy
            signals['market_structure'] = self._market_structure_signal(data, market_structure)
            
        except Exception as e:
            logger.error(f"Error generating strategy signals: {e}")
        
        return signals
    
    def _rsi_divergence_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect RSI divergence patterns"""
        try:
            if 'rsi' not in data.columns:
                return {'signal': 0, 'confidence': 0, 'reason': 'no_rsi_data'}
            
            rsi = data['rsi'].iloc[-20:]  # Last 20 periods
            price = data['close'].iloc[-20:]
            
            # Look for divergence patterns
            recent_price_high = price.rolling(5).max().iloc[-5:].max()
            recent_price_low = price.rolling(5).min().iloc[-5:].min()
            recent_rsi_high = rsi.rolling(5).max().iloc[-5:].max()
            recent_rsi_low = rsi.rolling(5).min().iloc[-5:].min()
            
            current_price = price.iloc[-1]
            current_rsi = rsi.iloc[-1]
            
            # Bullish divergence: Price makes lower low, RSI makes higher low
            if (current_price <= recent_price_low and current_rsi > recent_rsi_low and 
                current_rsi < 40):  # RSI oversold
                return {
                    'signal': 1,  # Buy
                    'confidence': 0.8,
                    'reason': 'bullish_divergence',
                    'rsi_value': current_rsi
                }
            
            # Bearish divergence: Price makes higher high, RSI makes lower high
            elif (current_price >= recent_price_high and current_rsi < recent_rsi_high and 
                  current_rsi > 60):  # RSI overbought
                return {
                    'signal': -1,  # Sell
                    'confidence': 0.8,
                    'reason': 'bearish_divergence',
                    'rsi_value': current_rsi
                }
            
            return {'signal': 0, 'confidence': 0, 'reason': 'no_divergence'}
            
        except Exception as e:
            logger.error(f"Error in RSI divergence signal: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': 'error'}
    
    def _atr_breakout_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect ATR-based breakout patterns"""
        try:
            if 'atr' not in data.columns:
                return {'signal': 0, 'confidence': 0, 'reason': 'no_atr_data'}
            
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            atr = data['atr'].iloc[-1]
            
            # Calculate breakout threshold
            breakout_threshold = atr * 1.5
            
            # 20-period high/low
            high_20 = data['high'].rolling(20).max().iloc[-2]  # Previous high (avoid current bar)
            low_20 = data['low'].rolling(20).min().iloc[-2]    # Previous low
            
            # Bullish breakout
            if current_price > high_20 and (current_price - prev_price) > breakout_threshold:
                confidence = min(0.9, 0.6 + (current_price - high_20) / atr * 0.1)
                return {
                    'signal': 1,
                    'confidence': confidence,
                    'reason': 'bullish_atr_breakout',
                    'breakout_strength': (current_price - high_20) / atr
                }
            
            # Bearish breakout
            elif current_price < low_20 and (prev_price - current_price) > breakout_threshold:
                confidence = min(0.9, 0.6 + (low_20 - current_price) / atr * 0.1)
                return {
                    'signal': -1,
                    'confidence': confidence,
                    'reason': 'bearish_atr_breakout',
                    'breakout_strength': (low_20 - current_price) / atr
                }
            
            return {'signal': 0, 'confidence': 0, 'reason': 'no_breakout'}
            
        except Exception as e:
            logger.error(f"Error in ATR breakout signal: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': 'error'}
    
    def _bollinger_squeeze_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect Bollinger Band squeeze breakouts"""
        try:
            if not all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                return {'signal': 0, 'confidence': 0, 'reason': 'no_bb_data'}
            
            current_price = data['close'].iloc[-1]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            
            # Calculate band width
            current_width = (bb_upper - bb_lower) / bb_middle
            avg_width = ((data['bb_upper'] - data['bb_lower']) / data['bb_middle']).rolling(20).mean().iloc[-1]
            
            # Squeeze condition: current width < 80% of average width
            is_squeeze = current_width < (avg_width * 0.8)
            
            if is_squeeze:
                # Look for breakout direction
                if current_price > bb_upper:
                    return {
                        'signal': 1,
                        'confidence': 0.85,
                        'reason': 'bullish_squeeze_breakout',
                        'squeeze_intensity': avg_width / current_width
                    }
                elif current_price < bb_lower:
                    return {
                        'signal': -1,
                        'confidence': 0.85,
                        'reason': 'bearish_squeeze_breakout',
                        'squeeze_intensity': avg_width / current_width
                    }
            
            return {'signal': 0, 'confidence': 0, 'reason': 'no_squeeze_breakout'}
            
        except Exception as e:
            logger.error(f"Error in Bollinger squeeze signal: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': 'error'}
    
    def _ma_cross_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect moving average crossover signals"""
        try:
            if not all(col in data.columns for col in ['ema_12', 'ema_26', 'sma_50']):
                return {'signal': 0, 'confidence': 0, 'reason': 'no_ma_data'}
            
            # Current and previous values
            ema_12_curr = data['ema_12'].iloc[-1]
            ema_26_curr = data['ema_26'].iloc[-1]
            sma_50_curr = data['sma_50'].iloc[-1]
            
            ema_12_prev = data['ema_12'].iloc[-2]
            ema_26_prev = data['ema_26'].iloc[-2]
            
            current_price = data['close'].iloc[-1]
            
            # Golden cross with trend confirmation
            if (ema_12_curr > ema_26_curr and ema_12_prev <= ema_26_prev and 
                current_price > sma_50_curr):
                return {
                    'signal': 1,
                    'confidence': 0.75,
                    'reason': 'golden_cross_with_trend',
                    'trend_alignment': True
                }
            
            # Death cross with trend confirmation
            elif (ema_12_curr < ema_26_curr and ema_12_prev >= ema_26_prev and 
                  current_price < sma_50_curr):
                return {
                    'signal': -1,
                    'confidence': 0.75,
                    'reason': 'death_cross_with_trend',
                    'trend_alignment': True
                }
            
            return {'signal': 0, 'confidence': 0, 'reason': 'no_cross_signal'}
            
        except Exception as e:
            logger.error(f"Error in MA cross signal: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': 'error'}
    
    def _market_structure_signal(self, data: pd.DataFrame, market_structure: Dict) -> Dict[str, Any]:
        """Generate signals based on market structure analysis"""
        try:
            structure_breaks = market_structure.get('structure_breaks', [])
            trend_direction = market_structure.get('trend_direction', 'neutral')
            trend_strength = market_structure.get('trend_strength', 0)
            
            if not structure_breaks:
                return {'signal': 0, 'confidence': 0, 'reason': 'no_structure_breaks'}
            
            # Get most recent structure break
            latest_break = structure_breaks[-1]
            
            # High confidence signals on strong structure breaks with trend alignment
            if latest_break['type'] == 'bullish_bos' and trend_direction == 'bullish':
                confidence = min(0.95, 0.7 + trend_strength + latest_break['strength'] * 0.1)
                return {
                    'signal': 1,
                    'confidence': confidence,
                    'reason': 'bullish_structure_break',
                    'break_strength': latest_break['strength']
                }
            
            elif latest_break['type'] == 'bearish_bos' and trend_direction == 'bearish':
                confidence = min(0.95, 0.7 + trend_strength + latest_break['strength'] * 0.1)
                return {
                    'signal': -1,
                    'confidence': confidence,
                    'reason': 'bearish_structure_break',
                    'break_strength': latest_break['strength']
                }
            
            return {'signal': 0, 'confidence': 0, 'reason': 'weak_structure_signal'}
            
        except Exception as e:
            logger.error(f"Error in market structure signal: {e}")
            return {'signal': 0, 'confidence': 0, 'reason': 'error'}
    
    def _calculate_ensemble_signal(self, strategy_signals: Dict[str, Dict]) -> Optional[Dict[str, Any]]:
        """Calculate ensemble signal from individual strategy signals"""
        try:
            weighted_score = 0.0
            total_weight = 0.0
            contributing_strategies = []
            
            for strategy_name, signal_data in strategy_signals.items():
                if signal_data.get('signal', 0) != 0:  # Non-zero signal
                    weight = self.strategy_weights.get(strategy_name, 0.1)
                    confidence = signal_data.get('confidence', 0)
                    signal_value = signal_data.get('signal', 0)
                    
                    weighted_score += signal_value * confidence * weight
                    total_weight += weight
                    contributing_strategies.append({
                        'strategy': strategy_name,
                        'signal': signal_value,
                        'confidence': confidence,
                        'reason': signal_data.get('reason', '')
                    })
            
            if total_weight == 0:
                return None
            
            # Normalize score
            final_score = weighted_score / total_weight
            
            # Determine direction and confidence
            if final_score > 0.3:
                direction = 'buy'
                confidence = min(0.98, abs(final_score))
            elif final_score < -0.3:
                direction = 'sell'
                confidence = min(0.98, abs(final_score))
            else:
                return None  # Signal too weak
            
            # Find primary contributing strategy
            primary_strategy = max(contributing_strategies, 
                                 key=lambda x: x['confidence'] * abs(x['signal']))['strategy']
            
            return {
                'direction': direction,
                'confidence': confidence,
                'primary_strategy': primary_strategy,
                'contributing_strategies': contributing_strategies,
                'ensemble_score': final_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating ensemble signal: {e}")
            return None
    
    async def _perform_additional_analysis(self, data: pd.DataFrame, signal_info: Dict) -> Dict[str, Any]:
        """Perform additional market analysis for signal enhancement"""
        try:
            current_price = signal_info['entry_price']
            
            # ATR analysis
            atr_value = data['atr'].iloc[-1] if 'atr' in data.columns else current_price * 0.01
            
            # Volatility analysis
            volatility = data['close'].rolling(20).std().iloc[-1] / current_price
            
            # Fibonacci levels
            swing_points = self.market_analyzer.find_swing_points(data)
            recent_highs = [sh['price'] for sh in swing_points['swing_highs'][-3:]]
            recent_lows = [sl['price'] for sl in swing_points['swing_lows'][-3:]]
            
            fibonacci_levels = {}
            if recent_highs and recent_lows:
                swing_high = max(recent_highs)
                swing_low = min(recent_lows)
                direction = 'up' if signal_info['direction'] == 'buy' else 'down'
                fibonacci_levels = self.market_analyzer.calculate_fibonacci_levels(
                    swing_high, swing_low, direction
                )
            
            # Swing levels
            swing_levels = {
                'recent_high': max(recent_highs) if recent_highs else current_price,
                'recent_low': min(recent_lows) if recent_lows else current_price,
                'nearest_resistance': min(recent_highs, key=lambda x: abs(x - current_price)) if recent_highs else current_price,
                'nearest_support': max(recent_lows, key=lambda x: abs(x - current_price)) if recent_lows else current_price
            }
            
            # Volume analysis
            volume_analysis = {}
            if 'volume' in data.columns and data['volume'].sum() > 0:
                current_volume = data['volume'].iloc[-1]
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                volume_analysis = {
                    'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1.0,
                    'volume_trend': 'increasing' if current_volume > avg_volume * 1.2 else 'normal',
                    'volume_confirmation': current_volume > avg_volume
                }
            else:
                volume_analysis = {'volume_ratio': 1.0, 'volume_trend': 'unknown', 'volume_confirmation': False}
            
            # Time analysis
            timestamp = signal_info['timestamp']
            time_analysis = {
                'hour': timestamp.hour if hasattr(timestamp, 'hour') else 12,
                'day_of_week': timestamp.weekday() if hasattr(timestamp, 'weekday') else 2,
                'is_market_open': 0 <= timestamp.hour <= 23,  # 24-hour markets
                'session': self._get_trading_session(timestamp)
            }
            
            # Risk metrics
            risk_metrics = {
                'volatility_percentile': self._calculate_volatility_percentile(data),
                'trend_consistency': self._calculate_trend_consistency(data),
                'support_resistance_strength': self._calculate_sr_strength(data, swing_levels)
            }
            
            # Alternative TP/SL levels
            alternative_levels = {
                'conservative_tp': [current_price + atr_value * 1.0, current_price + atr_value * 1.5],
                'aggressive_tp': [current_price + atr_value * 2.5, current_price + atr_value * 3.0],
                'conservative_sl': [current_price - atr_value * 0.5, current_price - atr_value * 0.75],
                'aggressive_sl': [current_price - atr_value * 1.5, current_price - atr_value * 2.0]
            }
            
            if signal_info['direction'] == 'sell':
                # Flip values for sell signals
                alternative_levels = {
                    'conservative_tp': [current_price - atr_value * 1.0, current_price - atr_value * 1.5],
                    'aggressive_tp': [current_price - atr_value * 2.5, current_price - atr_value * 3.0],
                    'conservative_sl': [current_price + atr_value * 0.5, current_price + atr_value * 0.75],
                    'aggressive_sl': [current_price + atr_value * 1.5, current_price + atr_value * 2.0]
                }
            
            return {
                'atr_value': atr_value,
                'volatility': volatility,
                'fibonacci_levels': fibonacci_levels,
                'swing_levels': swing_levels,
                'volume_analysis': volume_analysis,
                'time_analysis': time_analysis,
                'risk_metrics': risk_metrics,
                'alternative_levels': alternative_levels
            }
            
        except Exception as e:
            logger.error(f"Error in additional analysis: {e}")
            return {}
    
    def _get_trading_session(self, timestamp: datetime) -> str:
        """Determine trading session based on timestamp"""
        hour = timestamp.hour
        
        if 22 <= hour or hour < 9:
            return 'asian'
        elif 9 <= hour < 16:
            return 'london'
        elif 16 <= hour < 22:
            return 'new_york'
        else:
            return 'overlap'
    
    def _calculate_volatility_percentile(self, data: pd.DataFrame) -> float:
        """Calculate current volatility percentile"""
        try:
            volatility = data['close'].rolling(20).std() / data['close']
            current_vol = volatility.iloc[-1]
            vol_100 = volatility.rolling(100).quantile(0.0).iloc[-1]
            vol_percentile = (current_vol - vol_100) / (volatility.rolling(100).max().iloc[-1] - vol_100)
            return min(1.0, max(0.0, vol_percentile))
        except:
            return 0.5
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate trend consistency score"""
        try:
            if 'sma_20' not in data.columns:
                return 0.5
            
            price_above_sma = (data['close'] > data['sma_20']).rolling(20).mean().iloc[-1]
            return price_above_sma
        except:
            return 0.5
    
    def _calculate_sr_strength(self, data: pd.DataFrame, swing_levels: Dict) -> float:
        """Calculate support/resistance strength"""
        try:
            current_price = data['close'].iloc[-1]
            
            # Count touches near swing levels
            touches = 0
            for level in swing_levels.values():
                if isinstance(level, (int, float)):
                    tolerance = current_price * 0.002  # 0.2% tolerance
                    near_level = abs(data['close'] - level) <= tolerance
                    touches += near_level.sum()
            
            return min(1.0, touches / 10.0)  # Normalize to 0-1
        except:
            return 0.5
    
    def _generate_signal_id(self, symbol: str, timeframe: str) -> str:
        """Generate unique signal ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{symbol.lower()}_{timeframe}_{timestamp}"
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        if len(self.signal_history) > 0:
            self.performance_metrics['total_signals'] = len(self.signal_history)
            
            # Calculate other metrics (would need trade outcomes for accuracy)
            total_confidence = sum(s.confidence_score for s in self.signal_history)
            self.performance_metrics['avg_confidence'] = total_confidence / len(self.signal_history)
            
            total_rr = sum(s.rr_ratio for s in self.signal_history if s.rr_ratio > 0)
            count_rr = sum(1 for s in self.signal_history if s.rr_ratio > 0)
            self.performance_metrics['avg_rr_ratio'] = total_rr / count_rr if count_rr > 0 else 0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'total_signals_generated': self.performance_metrics['total_signals'],
            'avg_confidence_score': self.performance_metrics.get('avg_confidence', 0),
            'avg_risk_reward_ratio': self.performance_metrics['avg_rr_ratio'],
            'strategy_weights': self.strategy_weights,
            'recent_signals': len([s for s in self.signal_history 
                                 if (datetime.now() - s.timestamp).seconds < 3600])  # Last hour
        }
    
    async def batch_generate_signals(self, symbols: List[str], timeframe: str, 
                                   data_provider) -> List[AdvancedSignal]:
        """Generate signals for multiple symbols concurrently"""
        tasks = []
        
        for symbol in symbols:
            task = asyncio.create_task(self._generate_signal_for_symbol(symbol, timeframe, data_provider))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        signals = [result for result in results 
                  if isinstance(result, AdvancedSignal)]
        
        return signals
    
    async def _generate_signal_for_symbol(self, symbol: str, timeframe: str, 
                                        data_provider) -> Optional[AdvancedSignal]:
        """Generate signal for a specific symbol"""
        try:
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            data = await data_provider.get_historical_data(symbol, timeframe, start_date, end_date)
            
            if data.empty:
                return None
            
            return await self.generate_signal(data, symbol, timeframe)
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None


class SignalValidator:
    """Validates signals before execution"""
    
    def __init__(self):
        self.validation_rules = {
            'min_confidence': 80,  # Minimum confidence score
            'max_spread': 0.0003,  # Maximum spread (0.03%)
            'min_rr_ratio': 1.5,   # Minimum risk-reward ratio
            'max_correlation': 0.8  # Maximum correlation between concurrent signals
        }
    
    def validate_signal(self, signal: AdvancedSignal, market_conditions: Dict = None) -> Dict[str, Any]:
        """Validate signal against multiple criteria"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'confidence_adjustment': 0,
            'validation_score': 100
        }
        
        # Confidence check
        if signal.confidence_score < self.validation_rules['min_confidence']:
            validation_result['is_valid'] = False
            validation_result['warnings'].append(
                f"Low confidence: {signal.confidence_score} < {self.validation_rules['min_confidence']}"
            )
        
        # Risk-reward ratio check
        if signal.rr_ratio < self.validation_rules['min_rr_ratio']:
            validation_result['warnings'].append(
                f"Poor R:R ratio: {signal.rr_ratio:.2f} < {self.validation_rules['min_rr_ratio']}"
            )
            validation_result['confidence_adjustment'] -= 10
        
        # Market conditions check
        if market_conditions:
            spread = market_conditions.get('spread', 0)
            if spread > self.validation_rules['max_spread']:
                validation_result['warnings'].append(
                    f"High spread: {spread:.5f} > {self.validation_rules['max_spread']:.5f}"
                )
                validation_result['confidence_adjustment'] -= 5
        
        # Time-based validation
        if signal.time_analysis.get('session') == 'asian' and signal.volatility < 0.005:
            validation_result['warnings'].append("Low volatility during Asian session")
            validation_result['confidence_adjustment'] -= 5
        
        # Adjust final confidence
        final_confidence = signal.confidence_score + validation_result['confidence_adjustment']
        validation_result['final_confidence'] = max(0, min(100, final_confidence))
        
        # Calculate validation score
        penalty = len(validation_result['warnings']) * 10 + abs(validation_result['confidence_adjustment'])
        validation_result['validation_score'] = max(0, 100 - penalty)
        
        return validation_result