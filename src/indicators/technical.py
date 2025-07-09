"""
Technical Indicators for Trading Analysis
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import talib
from loguru import logger

from ..config.settings import settings


class TechnicalIndicators:
    """Comprehensive technical indicators calculation class"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        try:
            return pd.Series(talib.RSI(data.values, timeperiod=period), index=data.index)
        except Exception:
            # Fallback implementation
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD Indicator"""
        try:
            macd, macd_signal, macd_histogram = talib.MACD(data.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return (
                pd.Series(macd, index=data.index),
                pd.Series(macd_signal, index=data.index),
                pd.Series(macd_histogram, index=data.index)
            )
        except Exception:
            # Fallback implementation
            ema_fast = TechnicalIndicators.ema(data, fast)
            ema_slow = TechnicalIndicators.ema(data, slow)
            macd_line = ema_fast - ema_slow
            signal_line = TechnicalIndicators.ema(macd_line, signal)
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        try:
            upper, middle, lower = talib.BBANDS(data.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return (
                pd.Series(upper, index=data.index),
                pd.Series(middle, index=data.index),
                pd.Series(lower, index=data.index)
            )
        except Exception:
            # Fallback implementation
            middle = TechnicalIndicators.sma(data, period)
            std = data.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper, middle, lower
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        try:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
        except Exception:
            # Fallback implementation
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        try:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values, fastk_period=k_period, slowk_period=3, slowd_period=d_period)
            return (
                pd.Series(slowk, index=close.index),
                pd.Series(slowd, index=close.index)
            )
        except Exception:
            # Fallback implementation
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            return k_percent, d_percent
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        try:
            return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=period), index=close.index)
        except Exception:
            # Fallback implementation
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        try:
            return pd.Series(talib.CCI(high.values, low.values, close.values, timeperiod=period), index=close.index)
        except Exception:
            # Fallback implementation
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            return (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        try:
            return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
        except Exception:
            # Simplified fallback implementation
            tr = TechnicalIndicators.atr(high, low, close, 1)
            plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
            minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
            
            plus_dm_series = pd.Series(plus_dm, index=close.index).rolling(window=period).mean()
            minus_dm_series = pd.Series(minus_dm, index=close.index).rolling(window=period).mean()
            atr_series = tr.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm_series / atr_series)
            minus_di = 100 * (minus_dm_series / atr_series)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            return dx.rolling(window=period).mean()
    
    @staticmethod
    def fibonacci_retracement(high_price: float, low_price: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high_price - low_price
        
        return {
            '0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '78.6%': high_price - 0.786 * diff,
            '100%': low_price
        }
    
    @staticmethod
    def support_resistance(data: pd.Series, window: int = 20, min_touches: int = 2) -> Tuple[list, list]:
        """Identify support and resistance levels"""
        highs = data.rolling(window=window, center=True).max()
        lows = data.rolling(window=window, center=True).min()
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(data) - window):
            if data.iloc[i] == highs.iloc[i]:
                # Check if this level has been tested multiple times
                level = data.iloc[i]
                touches = sum(1 for j in range(max(0, i-50), min(len(data), i+50)) 
                             if abs(data.iloc[j] - level) / level < 0.01)
                
                if touches >= min_touches:
                    resistance_levels.append(level)
            
            if data.iloc[i] == lows.iloc[i]:
                level = data.iloc[i]
                touches = sum(1 for j in range(max(0, i-50), min(len(data), i+50)) 
                             if abs(data.iloc[j] - level) / level < 0.01)
                
                if touches >= min_touches:
                    support_levels.append(level)
        
        return list(set(support_levels)), list(set(resistance_levels))
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period1 = 9
        tenkan_sen = (high.rolling(window=period1).max() + low.rolling(window=period1).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period2 = 26
        kijun_sen = (high.rolling(window=period2).max() + low.rolling(window=period2).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(period2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period3 = 52
        senkou_span_b = ((high.rolling(window=period3).max() + low.rolling(window=period3).min()) / 2).shift(period2)
        
        # Chikou Span (Lagging Span): Close price shifted back 26 periods
        chikou_span = close.shift(-period2)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate pivot points and support/resistance levels"""
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
    
    @staticmethod
    def vwap(close: pd.Series, volume: pd.Series, high: pd.Series = None, low: pd.Series = None) -> pd.Series:
        """Volume Weighted Average Price"""
        if high is not None and low is not None:
            typical_price = (high + low + close) / 3
        else:
            typical_price = close
        
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        try:
            return pd.Series(talib.MFI(high.values, low.values, close.values, volume.values, timeperiod=period), index=close.index)
        except Exception:
            # Fallback implementation
            typical_price = (high + low + close) / 3
            money_flow = typical_price * volume
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
            
            positive_mf = positive_flow.rolling(window=period).sum()
            negative_mf = negative_flow.rolling(window=period).sum()
            
            mfr = positive_mf / negative_mf
            return 100 - (100 / (1 + mfr))


class IndicatorSignals:
    """Generate trading signals from technical indicators"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def rsi_signal(self, rsi: pd.Series, oversold: int = 30, overbought: int = 70) -> pd.Series:
        """Generate RSI buy/sell signals"""
        signals = pd.Series(0, index=rsi.index)
        signals[rsi < oversold] = 1  # Buy signal
        signals[rsi > overbought] = -1  # Sell signal
        return signals
    
    def macd_signal(self, macd: pd.Series, signal: pd.Series) -> pd.Series:
        """Generate MACD crossover signals"""
        signals = pd.Series(0, index=macd.index)
        crossover = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        crossunder = (macd < signal) & (macd.shift(1) >= signal.shift(1))
        
        signals[crossover] = 1  # Buy signal
        signals[crossunder] = -1  # Sell signal
        return signals
    
    def bollinger_signal(self, close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
        """Generate Bollinger Bands signals"""
        signals = pd.Series(0, index=close.index)
        
        # Buy when price touches lower band and bounces
        buy_condition = (close <= lower) & (close.shift(1) > lower.shift(1))
        # Sell when price touches upper band
        sell_condition = (close >= upper) & (close.shift(1) < upper.shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals
    
    def moving_average_signal(self, short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
        """Generate moving average crossover signals"""
        signals = pd.Series(0, index=short_ma.index)
        
        golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        signals[golden_cross] = 1  # Buy signal
        signals[death_cross] = -1  # Sell signal
        return signals
    
    def stochastic_signal(self, k: pd.Series, d: pd.Series, oversold: int = 20, overbought: int = 80) -> pd.Series:
        """Generate Stochastic oscillator signals"""
        signals = pd.Series(0, index=k.index)
        
        # Buy when both K and D are below oversold and K crosses above D
        buy_condition = (k < oversold) & (d < oversold) & (k > d) & (k.shift(1) <= d.shift(1))
        # Sell when both K and D are above overbought and K crosses below D
        sell_condition = (k > overbought) & (d > overbought) & (k < d) & (k.shift(1) >= d.shift(1))
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        return signals
    
    def combined_signal(self, signals_dict: Dict[str, pd.Series], weights: Dict[str, float] = None) -> pd.Series:
        """Combine multiple signals with optional weights"""
        if weights is None:
            weights = {key: 1.0 for key in signals_dict.keys()}
        
        combined = pd.Series(0.0, index=list(signals_dict.values())[0].index)
        
        for name, signal in signals_dict.items():
            weight = weights.get(name, 1.0)
            combined += signal * weight
        
        # Normalize to -1, 0, 1
        threshold = 0.5 * sum(weights.values())
        final_signals = pd.Series(0, index=combined.index)
        final_signals[combined >= threshold] = 1
        final_signals[combined <= -threshold] = -1
        
        return final_signals


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators for a given OHLCV dataframe"""
    
    if df.empty or len(df) < 50:
        logger.warning("Insufficient data for indicator calculation")
        return df
    
    try:
        result_df = df.copy()
        
        # Price-based indicators
        result_df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
        result_df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
        result_df['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
        result_df['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
        
        # Oscillators
        result_df['rsi'] = TechnicalIndicators.rsi(df['close'], settings.RSI_PERIOD)
        result_df['williams_r'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
        result_df['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
        
        # MACD
        macd, macd_signal, macd_hist = TechnicalIndicators.macd(df['close'])
        result_df['macd'] = macd
        result_df['macd_signal'] = macd_signal
        result_df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df['close'])
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle
        result_df['bb_lower'] = bb_lower
        
        # Volatility
        result_df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'], settings.ATR_PERIOD)
        
        # Stochastic
        stoch_k, stoch_d = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
        result_df['stoch_k'] = stoch_k
        result_df['stoch_d'] = stoch_d
        
        # Trend strength
        result_df['adx'] = TechnicalIndicators.adx(df['high'], df['low'], df['close'])
        
        # Volume-based (if volume data is available)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            result_df['mfi'] = TechnicalIndicators.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            result_df['vwap'] = TechnicalIndicators.vwap(df['close'], df['volume'], df['high'], df['low'])
        
        # Ichimoku components
        ichimoku = TechnicalIndicators.ichimoku_cloud(df['high'], df['low'], df['close'])
        for key, value in ichimoku.items():
            result_df[f'ichimoku_{key}'] = value
        
        logger.info(f"Calculated all technical indicators for {len(result_df)} periods")
        return result_df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df