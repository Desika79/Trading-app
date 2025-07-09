"""
Technical Analysis Based Trading Strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

from .base import BaseStrategy, TradeSignal
from ..indicators.technical import TechnicalIndicators, IndicatorSignals, calculate_all_indicators
from ..config.settings import settings


class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy with multiple confirmations"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_confirmation': True,  # Require RSI to move away from extreme before signal
            'volume_confirmation': False,  # Require volume confirmation
            'min_confidence': 0.7
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("RSI Strategy", default_params)
        self.indicators = TechnicalIndicators()
        self.signal_generator = IndicatorSignals()
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate RSI-based trading signals"""
        signals = []
        
        if len(data) < self.parameters['rsi_period'] + 5:
            return signals
        
        try:
            # Calculate RSI
            rsi = self.indicators.rsi(data['close'], self.parameters['rsi_period'])
            
            # Get the latest values
            current_rsi = rsi.iloc[-1]
            prev_rsi = rsi.iloc[-2]
            current_price = data['close'].iloc[-1]
            current_timestamp = data.index[-1]
            
            # RSI oversold condition (potential BUY)
            if (current_rsi < self.parameters['rsi_oversold'] and 
                prev_rsi >= self.parameters['rsi_oversold']):
                
                confidence = self.calculate_confidence(data, rsi, 'BUY')
                
                if confidence >= self.parameters['min_confidence']:
                    signals.append(TradeSignal(
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal_type='BUY',
                        timestamp=current_timestamp,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'rsi_value': current_rsi,
                            'strategy': 'RSI_OVERSOLD'
                        }
                    ))
            
            # RSI overbought condition (potential SELL)
            elif (current_rsi > self.parameters['rsi_overbought'] and 
                  prev_rsi <= self.parameters['rsi_overbought']):
                
                confidence = self.calculate_confidence(data, rsi, 'SELL')
                
                if confidence >= self.parameters['min_confidence']:
                    signals.append(TradeSignal(
                        symbol=data.attrs.get('symbol', 'UNKNOWN'),
                        signal_type='SELL',
                        timestamp=current_timestamp,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'rsi_value': current_rsi,
                            'strategy': 'RSI_OVERBOUGHT'
                        }
                    ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            return signals
    
    def calculate_confidence(self, data: pd.DataFrame, rsi: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence based on multiple factors"""
        confidence = 0.5  # Base confidence
        
        current_rsi = rsi.iloc[-1]
        
        # RSI extreme level confidence
        if signal_type == 'BUY':
            # More oversold = higher confidence
            rsi_confidence = max(0, (self.parameters['rsi_oversold'] - current_rsi) / self.parameters['rsi_oversold'])
        else:  # SELL
            # More overbought = higher confidence
            rsi_confidence = max(0, (current_rsi - self.parameters['rsi_overbought']) / (100 - self.parameters['rsi_overbought']))
        
        confidence += rsi_confidence * 0.3
        
        # Volume confirmation if enabled
        if self.parameters.get('volume_confirmation') and 'volume' in data.columns:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            if current_volume > avg_volume:
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: TradeSignal, account_balance: float) -> float:
        """Calculate position size based on RSI confidence and risk management"""
        base_risk = self.risk_per_trade * account_balance
        adjusted_risk = base_risk * signal.confidence
        
        # Use ATR for position sizing if available
        return adjusted_risk / (signal.price * self.stop_loss_pct)


class MACDStrategy(BaseStrategy):
    """MACD crossover strategy with trend confirmation"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'histogram_confirmation': True,
            'trend_confirmation': True,
            'min_confidence': 0.6
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("MACD Strategy", default_params)
        self.indicators = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate MACD-based trading signals"""
        signals = []
        
        if len(data) < max(self.parameters['macd_slow'], 50):
            return signals
        
        try:
            # Calculate MACD
            macd, macd_signal, macd_histogram = self.indicators.macd(
                data['close'],
                self.parameters['macd_fast'],
                self.parameters['macd_slow'],
                self.parameters['macd_signal']
            )
            
            # Calculate trend confirmation (200 SMA)
            trend_sma = self.indicators.sma(data['close'], 50) if self.parameters['trend_confirmation'] else None
            
            # Get latest values
            current_macd = macd.iloc[-1]
            current_signal = macd_signal.iloc[-1]
            prev_macd = macd.iloc[-2]
            prev_signal = macd_signal.iloc[-2]
            current_price = data['close'].iloc[-1]
            current_timestamp = data.index[-1]
            
            # MACD bullish crossover
            if (current_macd > current_signal and prev_macd <= prev_signal):
                # Trend confirmation
                trend_ok = True
                if trend_sma is not None:
                    trend_ok = current_price > trend_sma.iloc[-1]
                
                if trend_ok:
                    confidence = self.calculate_confidence(data, macd, macd_signal, macd_histogram, 'BUY')
                    
                    if confidence >= self.parameters['min_confidence']:
                        signals.append(TradeSignal(
                            symbol=data.attrs.get('symbol', 'UNKNOWN'),
                            signal_type='BUY',
                            timestamp=current_timestamp,
                            price=current_price,
                            confidence=confidence,
                            metadata={
                                'macd_value': current_macd,
                                'signal_value': current_signal,
                                'strategy': 'MACD_BULLISH_CROSSOVER'
                            }
                        ))
            
            # MACD bearish crossover
            elif (current_macd < current_signal and prev_macd >= prev_signal):
                # Trend confirmation
                trend_ok = True
                if trend_sma is not None:
                    trend_ok = current_price < trend_sma.iloc[-1]
                
                if trend_ok:
                    confidence = self.calculate_confidence(data, macd, macd_signal, macd_histogram, 'SELL')
                    
                    if confidence >= self.parameters['min_confidence']:
                        signals.append(TradeSignal(
                            symbol=data.attrs.get('symbol', 'UNKNOWN'),
                            signal_type='SELL',
                            timestamp=current_timestamp,
                            price=current_price,
                            confidence=confidence,
                            metadata={
                                'macd_value': current_macd,
                                'signal_value': current_signal,
                                'strategy': 'MACD_BEARISH_CROSSOVER'
                            }
                        ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating MACD signals: {e}")
            return signals
    
    def calculate_confidence(self, data: pd.DataFrame, macd: pd.Series, macd_signal: pd.Series, 
                           macd_histogram: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence based on MACD characteristics"""
        confidence = 0.5
        
        # Histogram confirmation
        if self.parameters.get('histogram_confirmation'):
            current_hist = macd_histogram.iloc[-1]
            prev_hist = macd_histogram.iloc[-2]
            
            if signal_type == 'BUY' and current_hist > prev_hist:
                confidence += 0.2
            elif signal_type == 'SELL' and current_hist < prev_hist:
                confidence += 0.2
        
        # MACD line distance from signal line
        macd_diff = abs(macd.iloc[-1] - macd_signal.iloc[-1])
        avg_diff = abs(macd - macd_signal).rolling(20).mean().iloc[-1]
        
        if macd_diff > avg_diff:
            confidence += 0.2
        
        # Zero line position
        if signal_type == 'BUY' and macd.iloc[-1] > 0:
            confidence += 0.1
        elif signal_type == 'SELL' and macd.iloc[-1] < 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: TradeSignal, account_balance: float) -> float:
        """Calculate position size based on signal confidence"""
        base_risk = self.risk_per_trade * account_balance
        adjusted_risk = base_risk * signal.confidence
        
        return adjusted_risk / (signal.price * self.stop_loss_pct)


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_confirmation': True,
            'rsi_period': 14,
            'volume_confirmation': True,
            'min_confidence': 0.65
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Bollinger Bands Strategy", default_params)
        self.indicators = TechnicalIndicators()
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate Bollinger Bands mean reversion signals"""
        signals = []
        
        if len(data) < self.parameters['bb_period'] + 10:
            return signals
        
        try:
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(
                data['close'],
                self.parameters['bb_period'],
                self.parameters['bb_std']
            )
            
            # RSI for confirmation
            rsi = None
            if self.parameters.get('rsi_confirmation'):
                rsi = self.indicators.rsi(data['close'], self.parameters['rsi_period'])
            
            # Get latest values
            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]
            current_upper = bb_upper.iloc[-1]
            current_lower = bb_lower.iloc[-1]
            current_middle = bb_middle.iloc[-1]
            current_timestamp = data.index[-1]
            
            # Lower band bounce (BUY signal)
            if (prev_price <= bb_lower.iloc[-2] and current_price > bb_lower.iloc[-1]):
                rsi_ok = True
                if rsi is not None:
                    rsi_ok = rsi.iloc[-1] < 50  # RSI below 50 for oversold confirmation
                
                if rsi_ok:
                    confidence = self.calculate_confidence(data, bb_upper, bb_middle, bb_lower, rsi, 'BUY')
                    
                    if confidence >= self.parameters['min_confidence']:
                        signals.append(TradeSignal(
                            symbol=data.attrs.get('symbol', 'UNKNOWN'),
                            signal_type='BUY',
                            timestamp=current_timestamp,
                            price=current_price,
                            confidence=confidence,
                            metadata={
                                'bb_position': (current_price - current_lower) / (current_upper - current_lower),
                                'strategy': 'BB_LOWER_BOUNCE'
                            }
                        ))
            
            # Upper band rejection (SELL signal)
            elif (prev_price >= bb_upper.iloc[-2] and current_price < bb_upper.iloc[-1]):
                rsi_ok = True
                if rsi is not None:
                    rsi_ok = rsi.iloc[-1] > 50  # RSI above 50 for overbought confirmation
                
                if rsi_ok:
                    confidence = self.calculate_confidence(data, bb_upper, bb_middle, bb_lower, rsi, 'SELL')
                    
                    if confidence >= self.parameters['min_confidence']:
                        signals.append(TradeSignal(
                            symbol=data.attrs.get('symbol', 'UNKNOWN'),
                            signal_type='SELL',
                            timestamp=current_timestamp,
                            price=current_price,
                            confidence=confidence,
                            metadata={
                                'bb_position': (current_price - current_lower) / (current_upper - current_lower),
                                'strategy': 'BB_UPPER_REJECTION'
                            }
                        ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands signals: {e}")
            return signals
    
    def calculate_confidence(self, data: pd.DataFrame, bb_upper: pd.Series, bb_middle: pd.Series, 
                           bb_lower: pd.Series, rsi: pd.Series, signal_type: str) -> float:
        """Calculate signal confidence"""
        confidence = 0.5
        
        current_price = data['close'].iloc[-1]
        bb_width = bb_upper.iloc[-1] - bb_lower.iloc[-1]
        avg_width = (bb_upper - bb_lower).rolling(20).mean().iloc[-1]
        
        # Wider bands = higher volatility = better signals
        if bb_width > avg_width:
            confidence += 0.2
        
        # Distance from middle line
        distance_from_middle = abs(current_price - bb_middle.iloc[-1]) / bb_middle.iloc[-1]
        confidence += min(0.2, distance_from_middle * 10)
        
        # RSI confirmation
        if rsi is not None:
            current_rsi = rsi.iloc[-1]
            if signal_type == 'BUY' and current_rsi < 40:
                confidence += 0.15
            elif signal_type == 'SELL' and current_rsi > 60:
                confidence += 0.15
        
        # Volume confirmation
        if self.parameters.get('volume_confirmation') and 'volume' in data.columns:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]
            if current_volume > avg_volume * 1.2:
                confidence += 0.15
        
        return min(1.0, confidence)
    
    def calculate_position_size(self, signal: TradeSignal, account_balance: float) -> float:
        """Calculate position size based on Bollinger Bands width"""
        base_risk = self.risk_per_trade * account_balance
        adjusted_risk = base_risk * signal.confidence
        
        return adjusted_risk / (signal.price * self.stop_loss_pct)


class MultiIndicatorStrategy(BaseStrategy):
    """Advanced strategy combining multiple indicators with machine learning scoring"""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        default_params = {
            'rsi_weight': 0.25,
            'macd_weight': 0.25,
            'bb_weight': 0.20,
            'stoch_weight': 0.15,
            'adx_weight': 0.15,
            'min_confidence': 0.75,
            'trend_filter': True,
            'volatility_filter': True
        }
        
        if parameters:
            default_params.update(parameters)
        
        super().__init__("Multi-Indicator Strategy", default_params)
        self.indicators = TechnicalIndicators()
        self.signal_generator = IndicatorSignals()
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate signals using multiple indicators with weighted scoring"""
        signals = []
        
        if len(data) < 60:  # Need sufficient data for all indicators
            return signals
        
        try:
            # Calculate all indicators
            data_with_indicators = calculate_all_indicators(data)
            
            if data_with_indicators.empty:
                return signals
            
            # Generate individual signals
            individual_signals = self.get_individual_signals(data_with_indicators)
            
            # Calculate composite signal
            composite_score = self.calculate_composite_score(individual_signals)
            
            current_price = data['close'].iloc[-1]
            current_timestamp = data.index[-1]
            
            # Apply filters
            if not self.apply_filters(data_with_indicators):
                return signals
            
            # Generate BUY signal
            if composite_score >= self.parameters['min_confidence']:
                signals.append(TradeSignal(
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal_type='BUY',
                    timestamp=current_timestamp,
                    price=current_price,
                    confidence=composite_score,
                    metadata={
                        'composite_score': composite_score,
                        'individual_signals': individual_signals,
                        'strategy': 'MULTI_INDICATOR_BUY'
                    }
                ))
            
            # Generate SELL signal
            elif composite_score <= -self.parameters['min_confidence']:
                signals.append(TradeSignal(
                    symbol=data.attrs.get('symbol', 'UNKNOWN'),
                    signal_type='SELL',
                    timestamp=current_timestamp,
                    price=current_price,
                    confidence=abs(composite_score),
                    metadata={
                        'composite_score': composite_score,
                        'individual_signals': individual_signals,
                        'strategy': 'MULTI_INDICATOR_SELL'
                    }
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating multi-indicator signals: {e}")
            return signals
    
    def get_individual_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get individual indicator signals normalized to -1 to 1"""
        signals = {}
        
        try:
            # RSI signal
            if 'rsi' in data.columns:
                rsi_value = data['rsi'].iloc[-1]
                if rsi_value < 30:
                    signals['rsi'] = (30 - rsi_value) / 30  # Stronger oversold = higher positive signal
                elif rsi_value > 70:
                    signals['rsi'] = -(rsi_value - 70) / 30  # Stronger overbought = higher negative signal
                else:
                    signals['rsi'] = 0
            
            # MACD signal
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd_diff = data['macd'].iloc[-1] - data['macd_signal'].iloc[-1]
                prev_diff = data['macd'].iloc[-2] - data['macd_signal'].iloc[-2]
                
                if macd_diff > 0 and prev_diff <= 0:
                    signals['macd'] = 1.0  # Bullish crossover
                elif macd_diff < 0 and prev_diff >= 0:
                    signals['macd'] = -1.0  # Bearish crossover
                else:
                    signals['macd'] = np.tanh(macd_diff)  # Proportional to difference
            
            # Bollinger Bands signal
            if all(col in data.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
                price = data['close'].iloc[-1]
                bb_upper = data['bb_upper'].iloc[-1]
                bb_lower = data['bb_lower'].iloc[-1]
                bb_middle = data['bb_middle'].iloc[-1]
                
                bb_position = (price - bb_lower) / (bb_upper - bb_lower)
                if bb_position < 0.2:
                    signals['bb'] = 1.0 - bb_position * 5  # Strong buy near lower band
                elif bb_position > 0.8:
                    signals['bb'] = -(bb_position - 0.8) * 5  # Strong sell near upper band
                else:
                    signals['bb'] = 0
            
            # Stochastic signal
            if 'stoch_k' in data.columns and 'stoch_d' in data.columns:
                stoch_k = data['stoch_k'].iloc[-1]
                stoch_d = data['stoch_d'].iloc[-1]
                
                if stoch_k < 20 and stoch_d < 20:
                    signals['stoch'] = (20 - min(stoch_k, stoch_d)) / 20
                elif stoch_k > 80 and stoch_d > 80:
                    signals['stoch'] = -(min(stoch_k, stoch_d) - 80) / 20
                else:
                    signals['stoch'] = 0
            
            # ADX trend strength (not directional, but strength modifier)
            if 'adx' in data.columns:
                adx_value = data['adx'].iloc[-1]
                # ADX above 25 indicates strong trend, use as confidence multiplier
                signals['adx'] = min(1.0, max(0, (adx_value - 25) / 25))
            
        except Exception as e:
            logger.error(f"Error calculating individual signals: {e}")
        
        return signals
    
    def calculate_composite_score(self, individual_signals: Dict[str, float]) -> float:
        """Calculate weighted composite score from individual signals"""
        score = 0.0
        total_weight = 0.0
        
        weights = {
            'rsi': self.parameters['rsi_weight'],
            'macd': self.parameters['macd_weight'],
            'bb': self.parameters['bb_weight'],
            'stoch': self.parameters['stoch_weight'],
            'adx': self.parameters['adx_weight']
        }
        
        # Calculate directional signals
        for indicator, signal in individual_signals.items():
            if indicator in weights and indicator != 'adx':
                weight = weights[indicator]
                score += signal * weight
                total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            score = score / total_weight
        
        # Apply ADX as trend strength multiplier
        if 'adx' in individual_signals:
            adx_multiplier = max(0.5, individual_signals['adx'])  # Minimum 50% confidence
            score *= adx_multiplier
        
        return score
    
    def apply_filters(self, data: pd.DataFrame) -> bool:
        """Apply trend and volatility filters"""
        try:
            # Trend filter
            if self.parameters.get('trend_filter'):
                if 'sma_50' in data.columns:
                    current_price = data['close'].iloc[-1]
                    sma_50 = data['sma_50'].iloc[-1]
                    
                    # Only trade in strong trends
                    trend_strength = abs(current_price - sma_50) / sma_50
                    if trend_strength < 0.02:  # Less than 2% from SMA
                        return False
            
            # Volatility filter
            if self.parameters.get('volatility_filter'):
                if 'atr' in data.columns:
                    current_atr = data['atr'].iloc[-1]
                    avg_atr = data['atr'].rolling(20).mean().iloc[-1]
                    
                    # Avoid trading in extremely low volatility periods
                    if current_atr < avg_atr * 0.5:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return False
    
    def calculate_position_size(self, signal: TradeSignal, account_balance: float) -> float:
        """Calculate position size based on composite confidence"""
        base_risk = self.risk_per_trade * account_balance
        
        # Scale position size with confidence
        confidence_multiplier = signal.confidence
        adjusted_risk = base_risk * confidence_multiplier
        
        return adjusted_risk / (signal.price * self.stop_loss_pct)
    
    def custom_validation(self, signal: TradeSignal) -> bool:
        """Additional validation for multi-indicator signals"""
        # Require higher confidence for multi-indicator strategy
        return signal.confidence >= self.parameters['min_confidence']


# Strategy factory for easy instantiation
class StrategyFactory:
    """Factory class for creating trading strategies"""
    
    @staticmethod
    def create_strategy(strategy_name: str, parameters: Dict[str, Any] = None) -> BaseStrategy:
        """Create a strategy instance by name"""
        strategies = {
            'rsi': RSIStrategy,
            'macd': MACDStrategy,
            'bollinger': BollingerBandsStrategy,
            'multi_indicator': MultiIndicatorStrategy
        }
        
        strategy_class = strategies.get(strategy_name.lower())
        if strategy_class:
            return strategy_class(parameters)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    @staticmethod
    def get_available_strategies() -> List[str]:
        """Get list of available strategies"""
        return ['rsi', 'macd', 'bollinger', 'multi_indicator']
    
    @staticmethod
    def create_all_strategies(parameters: Dict[str, Dict[str, Any]] = None) -> List[BaseStrategy]:
        """Create instances of all available strategies"""
        strategies = []
        strategy_names = StrategyFactory.get_available_strategies()
        
        for name in strategy_names:
            strategy_params = parameters.get(name, {}) if parameters else {}
            try:
                strategy = StrategyFactory.create_strategy(name, strategy_params)
                strategies.append(strategy)
            except Exception as e:
                logger.error(f"Failed to create strategy {name}: {e}")
        
        return strategies