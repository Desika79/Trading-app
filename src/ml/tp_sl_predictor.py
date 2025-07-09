"""
AI-Powered Take Profit & Stop Loss Predictor
Advanced machine learning module for optimal exit level forecasting
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

from ..indicators.technical import TechnicalIndicators
from ..config.settings import settings


class MarketStructureAnalyzer:
    """Advanced market structure analysis for swing points and trend detection"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def find_swing_points(self, data: pd.DataFrame, window: int = 10) -> Dict[str, List[Dict]]:
        """Find swing highs and lows using fractal analysis"""
        highs = data['high'].values
        lows = data['low'].values
        timestamps = data.index.tolist()
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(data) - window):
            # Check for swing high
            is_swing_high = True
            for j in range(window):
                if highs[i] <= highs[i - j - 1] or highs[i] <= highs[i + j + 1]:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing_highs.append({
                    'timestamp': timestamps[i],
                    'price': highs[i],
                    'index': i,
                    'strength': self._calculate_swing_strength(highs, i, window)
                })
            
            # Check for swing low
            is_swing_low = True
            for j in range(window):
                if lows[i] >= lows[i - j - 1] or lows[i] >= lows[i + j + 1]:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing_lows.append({
                    'timestamp': timestamps[i],
                    'price': lows[i],
                    'index': i,
                    'strength': self._calculate_swing_strength(lows, i, window, is_low=True)
                })
        
        return {'swing_highs': swing_highs, 'swing_lows': swing_lows}
    
    def _calculate_swing_strength(self, prices: np.array, index: int, window: int, is_low: bool = False) -> float:
        """Calculate the strength of a swing point"""
        if is_low:
            # For swing lows, strength is how much lower it is compared to surrounding points
            surrounding = np.concatenate([
                prices[index-window:index],
                prices[index+1:index+window+1]
            ])
            strength = np.mean(surrounding) - prices[index]
        else:
            # For swing highs, strength is how much higher it is
            surrounding = np.concatenate([
                prices[index-window:index],
                prices[index+1:index+window+1]
            ])
            strength = prices[index] - np.mean(surrounding)
        
        return max(0, strength)
    
    def detect_market_structure_breaks(self, data: pd.DataFrame, swing_points: Dict) -> List[Dict]:
        """Detect Break of Structure (BOS) and Market Structure Break (MSB) patterns"""
        breaks = []
        swing_highs = swing_points['swing_highs']
        swing_lows = swing_points['swing_lows']
        
        # Analyze recent swing points for structure breaks
        recent_highs = [sh for sh in swing_highs if len(data) - sh['index'] <= 50]
        recent_lows = [sl for sl in swing_lows if len(data) - sl['index'] <= 50]
        
        if len(recent_highs) >= 2:
            last_high = recent_highs[-1]
            prev_high = recent_highs[-2]
            current_price = data['close'].iloc[-1]
            
            # Bullish BOS: Current price breaks above previous high
            if current_price > prev_high['price'] and current_price > last_high['price']:
                breaks.append({
                    'type': 'bullish_bos',
                    'timestamp': data.index[-1],
                    'break_level': max(prev_high['price'], last_high['price']),
                    'strength': min(last_high['strength'], prev_high['strength'])
                })
        
        if len(recent_lows) >= 2:
            last_low = recent_lows[-1]
            prev_low = recent_lows[-2]
            current_price = data['close'].iloc[-1]
            
            # Bearish BOS: Current price breaks below previous low
            if current_price < prev_low['price'] and current_price < last_low['price']:
                breaks.append({
                    'type': 'bearish_bos',
                    'timestamp': data.index[-1],
                    'break_level': min(prev_low['price'], last_low['price']),
                    'strength': min(last_low['strength'], prev_low['strength'])
                })
        
        return breaks
    
    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float, direction: str = 'up') -> Dict[str, float]:
        """Calculate Fibonacci retracement and extension levels"""
        diff = swing_high - swing_low
        
        if direction == 'up':  # Bullish move
            levels = {
                'fib_0': swing_low,
                'fib_236': swing_low + (diff * 0.236),
                'fib_382': swing_low + (diff * 0.382),
                'fib_50': swing_low + (diff * 0.5),
                'fib_618': swing_low + (diff * 0.618),
                'fib_786': swing_low + (diff * 0.786),
                'fib_100': swing_high,
                'fib_1272': swing_high + (diff * 0.272),
                'fib_1618': swing_high + (diff * 0.618),
                'fib_2618': swing_high + (diff * 1.618)
            }
        else:  # Bearish move
            levels = {
                'fib_0': swing_high,
                'fib_236': swing_high - (diff * 0.236),
                'fib_382': swing_high - (diff * 0.382),
                'fib_50': swing_high - (diff * 0.5),
                'fib_618': swing_high - (diff * 0.618),
                'fib_786': swing_high - (diff * 0.786),
                'fib_100': swing_low,
                'fib_1272': swing_low - (diff * 0.272),
                'fib_1618': swing_low - (diff * 0.618),
                'fib_2618': swing_low - (diff * 1.618)
            }
        
        return levels


class TPSLPredictor:
    """AI-powered Take Profit and Stop Loss predictor using machine learning"""
    
    def __init__(self):
        self.tp_model = None
        self.sl_model = None
        self.scaler = StandardScaler()
        self.market_analyzer = MarketStructureAnalyzer()
        self.is_trained = False
        self.feature_columns = []
        
    def extract_features(self, data: pd.DataFrame, signal_info: Dict[str, Any]) -> Dict[str, float]:
        """Extract comprehensive features for TP/SL prediction"""
        try:
            # Basic market features
            current_price = data['close'].iloc[-1]
            atr = self.market_analyzer.indicators.atr(data['high'], data['low'], data['close'], 14).iloc[-1]
            rsi = self.market_analyzer.indicators.rsi(data['close'], 14).iloc[-1]
            
            # Volatility features
            volatility_20 = data['close'].rolling(20).std().iloc[-1]
            price_range_20 = (data['high'].rolling(20).max() - data['low'].rolling(20).min()).iloc[-1]
            
            # Trend features
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1]
            trend_strength = abs(current_price - sma_20) / sma_20
            
            # Market structure features
            swing_points = self.market_analyzer.find_swing_points(data)
            recent_swing_high = max([sh['price'] for sh in swing_points['swing_highs'][-3:]], default=current_price)
            recent_swing_low = min([sl['price'] for sl in swing_points['swing_lows'][-3:]], default=current_price)
            
            # Distance features
            distance_to_swing_high = abs(current_price - recent_swing_high) / current_price
            distance_to_swing_low = abs(current_price - recent_swing_low) / current_price
            
            # Volume features (if available)
            volume_ratio = 1.0
            if 'volume' in data.columns and data['volume'].sum() > 0:
                avg_volume = data['volume'].rolling(20).mean().iloc[-1]
                current_volume = data['volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Time-based features
            hour = data.index[-1].hour if hasattr(data.index[-1], 'hour') else 12
            day_of_week = data.index[-1].weekday() if hasattr(data.index[-1], 'weekday') else 2
            
            # Signal-specific features
            signal_confidence = signal_info.get('confidence', 0.7)
            signal_type = 1 if signal_info.get('direction', 'buy') == 'buy' else -1
            
            features = {
                'atr_normalized': atr / current_price,
                'rsi': rsi,
                'volatility_normalized': volatility_20 / current_price,
                'price_range_normalized': price_range_20 / current_price,
                'trend_strength': trend_strength,
                'distance_to_swing_high': distance_to_swing_high,
                'distance_to_swing_low': distance_to_swing_low,
                'volume_ratio': volume_ratio,
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'day_of_week_sin': np.sin(2 * np.pi * day_of_week / 7),
                'day_of_week_cos': np.cos(2 * np.pi * day_of_week / 7),
                'signal_confidence': signal_confidence,
                'signal_type': signal_type,
                'price_position': (current_price - recent_swing_low) / (recent_swing_high - recent_swing_low) if recent_swing_high != recent_swing_low else 0.5
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def calculate_multiple_tp_sl_levels(self, data: pd.DataFrame, signal_info: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate multiple TP/SL levels using different methods"""
        current_price = signal_info.get('entry_price', data['close'].iloc[-1])
        direction = signal_info.get('direction', 'buy')
        atr = self.market_analyzer.indicators.atr(data['high'], data['low'], data['close'], 14).iloc[-1]
        
        # Get swing points and structure
        swing_points = self.market_analyzer.find_swing_points(data)
        recent_highs = [sh['price'] for sh in swing_points['swing_highs'][-5:]]
        recent_lows = [sl['price'] for sl in swing_points['swing_lows'][-5:]]
        
        tp_levels = {}
        sl_levels = {}
        
        if direction == 'buy':
            # Take Profit levels for BUY
            tp_levels.update({
                'atr_1x': current_price + (atr * 1.0),
                'atr_1_5x': current_price + (atr * 1.5),
                'atr_2x': current_price + (atr * 2.0),
                'atr_3x': current_price + (atr * 3.0),
                'swing_high_nearest': min(recent_highs, default=current_price + atr, key=lambda x: abs(x - current_price)) if recent_highs else current_price + atr,
                'swing_high_strong': max(recent_highs, default=current_price + atr) if recent_highs else current_price + atr,
                'percent_1': current_price * 1.01,
                'percent_2': current_price * 1.02,
                'percent_3': current_price * 1.03
            })
            
            # Stop Loss levels for BUY
            sl_levels.update({
                'atr_0_5x': current_price - (atr * 0.5),
                'atr_1x': current_price - (atr * 1.0),
                'atr_1_5x': current_price - (atr * 1.5),
                'swing_low_nearest': max(recent_lows, default=current_price - atr, key=lambda x: abs(x - current_price)) if recent_lows else current_price - atr,
                'swing_low_strong': min(recent_lows, default=current_price - atr) if recent_lows else current_price - atr,
                'percent_1': current_price * 0.99,
                'percent_2': current_price * 0.98,
                'percent_3': current_price * 0.97
            })
            
            # Fibonacci extensions for BUY
            if recent_lows and recent_highs:
                swing_low = min(recent_lows)
                swing_high = max(recent_highs)
                fib_levels = self.market_analyzer.calculate_fibonacci_levels(swing_high, swing_low, 'up')
                tp_levels.update({
                    'fib_1272': fib_levels['fib_1272'],
                    'fib_1618': fib_levels['fib_1618'],
                    'fib_2618': fib_levels['fib_2618']
                })
                sl_levels.update({
                    'fib_382': fib_levels['fib_382'],
                    'fib_50': fib_levels['fib_50'],
                    'fib_618': fib_levels['fib_618']
                })
        
        else:  # SELL
            # Take Profit levels for SELL
            tp_levels.update({
                'atr_1x': current_price - (atr * 1.0),
                'atr_1_5x': current_price - (atr * 1.5),
                'atr_2x': current_price - (atr * 2.0),
                'atr_3x': current_price - (atr * 3.0),
                'swing_low_nearest': max(recent_lows, default=current_price - atr, key=lambda x: abs(x - current_price)) if recent_lows else current_price - atr,
                'swing_low_strong': min(recent_lows, default=current_price - atr) if recent_lows else current_price - atr,
                'percent_1': current_price * 0.99,
                'percent_2': current_price * 0.98,
                'percent_3': current_price * 0.97
            })
            
            # Stop Loss levels for SELL
            sl_levels.update({
                'atr_0_5x': current_price + (atr * 0.5),
                'atr_1x': current_price + (atr * 1.0),
                'atr_1_5x': current_price + (atr * 1.5),
                'swing_high_nearest': min(recent_highs, default=current_price + atr, key=lambda x: abs(x - current_price)) if recent_highs else current_price + atr,
                'swing_high_strong': max(recent_highs, default=current_price + atr) if recent_highs else current_price + atr,
                'percent_1': current_price * 1.01,
                'percent_2': current_price * 1.02,
                'percent_3': current_price * 1.03
            })
            
            # Fibonacci extensions for SELL
            if recent_lows and recent_highs:
                swing_low = min(recent_lows)
                swing_high = max(recent_highs)
                fib_levels = self.market_analyzer.calculate_fibonacci_levels(swing_high, swing_low, 'down')
                tp_levels.update({
                    'fib_1272': fib_levels['fib_1272'],
                    'fib_1618': fib_levels['fib_1618'],
                    'fib_2618': fib_levels['fib_2618']
                })
                sl_levels.update({
                    'fib_382': fib_levels['fib_382'],
                    'fib_50': fib_levels['fib_50'],
                    'fib_618': fib_levels['fib_618']
                })
        
        return {'take_profit': tp_levels, 'stop_loss': sl_levels}
    
    def train_models(self, historical_data: List[Dict[str, Any]]):
        """Train ML models using historical trade data"""
        if not historical_data:
            logger.warning("No historical data provided for training")
            return
        
        logger.info(f"Training TP/SL models with {len(historical_data)} historical trades")
        
        # Prepare training data
        features_list = []
        tp_targets = []
        sl_targets = []
        
        for trade in historical_data:
            if 'features' in trade and 'actual_tp' in trade and 'actual_sl' in trade:
                features_list.append(list(trade['features'].values()))
                tp_targets.append(trade['actual_tp'])
                sl_targets.append(trade['actual_sl'])
        
        if len(features_list) < 10:
            logger.warning("Insufficient training data, using default models")
            return
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y_tp = np.array(tp_targets)
        y_sl = np.array(sl_targets)
        
        # Store feature columns
        self.feature_columns = list(historical_data[0]['features'].keys())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test = train_test_split(
            X_scaled, y_tp, y_sl, test_size=0.2, random_state=42
        )
        
        # Train Take Profit model
        self.tp_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.tp_model.fit(X_train, y_tp_train)
        
        # Train Stop Loss model
        self.sl_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        self.sl_model.fit(X_train, y_sl_train)
        
        # Evaluate models
        tp_pred = self.tp_model.predict(X_test)
        sl_pred = self.sl_model.predict(X_test)
        
        tp_mae = mean_absolute_error(y_tp_test, tp_pred)
        sl_mae = mean_absolute_error(y_sl_test, sl_pred)
        tp_r2 = r2_score(y_tp_test, tp_pred)
        sl_r2 = r2_score(y_sl_test, sl_pred)
        
        logger.info(f"TP Model - MAE: {tp_mae:.6f}, R²: {tp_r2:.3f}")
        logger.info(f"SL Model - MAE: {sl_mae:.6f}, R²: {sl_r2:.3f}")
        
        self.is_trained = True
    
    def predict_optimal_tp_sl(self, data: pd.DataFrame, signal_info: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal TP/SL levels using trained ML models and rule-based methods"""
        try:
            # Calculate multiple TP/SL levels
            levels = self.calculate_multiple_tp_sl_levels(data, signal_info)
            
            # Extract features for ML prediction
            features = self.extract_features(data, signal_info)
            
            # Use ML models if trained
            ml_tp = None
            ml_sl = None
            
            if self.is_trained and self.tp_model and self.sl_model and features:
                try:
                    feature_vector = np.array([list(features.values())]).reshape(1, -1)
                    feature_vector_scaled = self.scaler.transform(feature_vector)
                    
                    ml_tp = self.tp_model.predict(feature_vector_scaled)[0]
                    ml_sl = self.sl_model.predict(feature_vector_scaled)[0]
                    
                    # Add ML predictions to levels
                    levels['take_profit']['ml_predicted'] = ml_tp
                    levels['stop_loss']['ml_predicted'] = ml_sl
                    
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
            
            # Rank and select optimal levels
            optimal_tp_sl = self._rank_and_select_levels(levels, signal_info, features)
            
            return {
                'optimal_take_profit': optimal_tp_sl['take_profit'],
                'optimal_stop_loss': optimal_tp_sl['stop_loss'],
                'all_levels': levels,
                'ml_predictions': {'tp': ml_tp, 'sl': ml_sl},
                'features': features,
                'confidence': optimal_tp_sl['confidence'],
                'risk_reward_ratio': optimal_tp_sl['rr_ratio']
            }
            
        except Exception as e:
            logger.error(f"Error predicting TP/SL: {e}")
            return self._get_default_tp_sl(data, signal_info)
    
    def _rank_and_select_levels(self, levels: Dict, signal_info: Dict, features: Dict) -> Dict:
        """Rank TP/SL levels and select optimal combination"""
        entry_price = signal_info.get('entry_price', 0)
        direction = signal_info.get('direction', 'buy')
        
        # Scoring criteria weights
        weights = {
            'atr_based': 0.3,      # ATR-based levels are generally reliable
            'structure_based': 0.4, # Swing highs/lows are strong levels
            'fibonacci': 0.2,       # Fibonacci levels have psychological significance
            'ml_predicted': 0.1     # ML predictions (if available)
        }
        
        tp_scores = {}
        sl_scores = {}
        
        # Score Take Profit levels
        for level_name, price in levels['take_profit'].items():
            score = 0
            
            if 'atr' in level_name:
                score += weights['atr_based']
            elif 'swing' in level_name:
                score += weights['structure_based']
            elif 'fib' in level_name:
                score += weights['fibonacci']
            elif 'ml' in level_name:
                score += weights['ml_predicted']
            else:
                score += 0.1  # Base score for other methods
            
            # Bonus for reasonable risk-reward ratios
            if entry_price > 0:
                if direction == 'buy':
                    potential_profit = price - entry_price
                else:
                    potential_profit = entry_price - price
                
                rr_ratio = potential_profit / (entry_price * 0.01)  # Assume 1% risk
                if 1.5 <= rr_ratio <= 4.0:  # Reasonable RR range
                    score += 0.2
            
            tp_scores[level_name] = score
        
        # Score Stop Loss levels
        for level_name, price in levels['stop_loss'].items():
            score = 0
            
            if 'atr' in level_name:
                score += weights['atr_based']
            elif 'swing' in level_name:
                score += weights['structure_based']
            elif 'fib' in level_name:
                score += weights['fibonacci']
            elif 'ml' in level_name:
                score += weights['ml_predicted']
            else:
                score += 0.1
            
            sl_scores[level_name] = score
        
        # Select best levels
        best_tp_name = max(tp_scores.keys(), key=lambda x: tp_scores[x])
        best_sl_name = max(sl_scores.keys(), key=lambda x: sl_scores[x])
        
        optimal_tp = levels['take_profit'][best_tp_name]
        optimal_sl = levels['stop_loss'][best_sl_name]
        
        # Calculate risk-reward ratio
        if entry_price > 0:
            if direction == 'buy':
                profit_potential = optimal_tp - entry_price
                loss_potential = entry_price - optimal_sl
            else:
                profit_potential = entry_price - optimal_tp
                loss_potential = optimal_sl - entry_price
            
            rr_ratio = profit_potential / loss_potential if loss_potential > 0 else 0
        else:
            rr_ratio = 0
        
        # Calculate confidence based on scores and features
        avg_score = (tp_scores[best_tp_name] + sl_scores[best_sl_name]) / 2
        feature_confidence = features.get('signal_confidence', 0.7)
        confidence = (avg_score + feature_confidence) / 2
        
        return {
            'take_profit': optimal_tp,
            'stop_loss': optimal_sl,
            'tp_method': best_tp_name,
            'sl_method': best_sl_name,
            'rr_ratio': rr_ratio,
            'confidence': confidence
        }
    
    def _get_default_tp_sl(self, data: pd.DataFrame, signal_info: Dict) -> Dict:
        """Fallback method for TP/SL calculation"""
        entry_price = signal_info.get('entry_price', data['close'].iloc[-1])
        direction = signal_info.get('direction', 'buy')
        
        # Use simple ATR-based calculation
        atr = self.market_analyzer.indicators.atr(data['high'], data['low'], data['close'], 14).iloc[-1]
        
        if direction == 'buy':
            tp = entry_price + (atr * 2.0)
            sl = entry_price - (atr * 1.0)
        else:
            tp = entry_price - (atr * 2.0)
            sl = entry_price + (atr * 1.0)
        
        return {
            'optimal_take_profit': tp,
            'optimal_stop_loss': sl,
            'all_levels': {},
            'ml_predictions': {'tp': None, 'sl': None},
            'features': {},
            'confidence': 0.6,
            'risk_reward_ratio': 2.0
        }
    
    def save_model(self, filepath: str):
        """Save trained models to disk"""
        if self.is_trained:
            model_data = {
                'tp_model': self.tp_model,
                'sl_model': self.sl_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, filepath)
            logger.info(f"TP/SL models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.tp_model = model_data['tp_model']
            self.sl_model = model_data['sl_model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            logger.info(f"TP/SL models loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


class TPSLOptimizer:
    """Optimizer for finding the best TP/SL combinations through backtesting"""
    
    def __init__(self):
        self.predictor = TPSLPredictor()
        self.optimization_results = {}
    
    def optimize_tp_sl_combinations(self, historical_data: pd.DataFrame, signals: List[Dict], 
                                  tp_range: Tuple[float, float] = (1.0, 4.0),
                                  sl_range: Tuple[float, float] = (0.5, 2.0),
                                  step_size: float = 0.25) -> Dict[str, Any]:
        """Optimize TP/SL combinations using grid search on historical data"""
        logger.info("Starting TP/SL optimization...")
        
        # Generate TP/SL combinations
        tp_multipliers = np.arange(tp_range[0], tp_range[1] + step_size, step_size)
        sl_multipliers = np.arange(sl_range[0], sl_range[1] + step_size, step_size)
        
        results = []
        
        for tp_mult in tp_multipliers:
            for sl_mult in sl_multipliers:
                # Test this combination
                performance = self._test_tp_sl_combination(
                    historical_data, signals, tp_mult, sl_mult
                )
                
                results.append({
                    'tp_multiplier': tp_mult,
                    'sl_multiplier': sl_mult,
                    'win_rate': performance['win_rate'],
                    'profit_factor': performance['profit_factor'],
                    'total_trades': performance['total_trades'],
                    'avg_rr': performance['avg_rr'],
                    'max_drawdown': performance['max_drawdown'],
                    'score': performance['score']
                })
        
        # Find best combination
        best_result = max(results, key=lambda x: x['score'])
        
        # Create heatmap data
        heatmap_data = self._create_heatmap_data(results, tp_multipliers, sl_multipliers)
        
        optimization_result = {
            'best_combination': best_result,
            'all_results': results,
            'heatmap_data': heatmap_data,
            'optimization_date': datetime.now()
        }
        
        self.optimization_results = optimization_result
        logger.info(f"Optimization complete. Best combination: TP {best_result['tp_multiplier']}x, SL {best_result['sl_multiplier']}x")
        
        return optimization_result
    
    def _test_tp_sl_combination(self, data: pd.DataFrame, signals: List[Dict], 
                               tp_mult: float, sl_mult: float) -> Dict[str, float]:
        """Test a specific TP/SL combination"""
        trades = []
        
        for signal in signals:
            entry_price = signal['entry_price']
            direction = signal['direction']
            entry_time = signal['timestamp']
            
            # Calculate TP/SL based on ATR
            atr_value = signal.get('atr', entry_price * 0.01)  # Fallback to 1%
            
            if direction == 'buy':
                tp_price = entry_price + (atr_value * tp_mult)
                sl_price = entry_price - (atr_value * sl_mult)
            else:
                tp_price = entry_price - (atr_value * tp_mult)
                sl_price = entry_price + (atr_value * sl_mult)
            
            # Simulate trade outcome
            outcome = self._simulate_trade_outcome(data, entry_time, entry_price, tp_price, sl_price, direction)
            trades.append(outcome)
        
        # Calculate performance metrics
        return self._calculate_performance_metrics(trades)
    
    def _simulate_trade_outcome(self, data: pd.DataFrame, entry_time: datetime, 
                               entry_price: float, tp_price: float, sl_price: float, 
                               direction: str) -> Dict[str, Any]:
        """Simulate trade outcome based on historical data"""
        # Find entry point in data
        entry_idx = None
        for i, timestamp in enumerate(data.index):
            if timestamp >= entry_time:
                entry_idx = i
                break
        
        if entry_idx is None or entry_idx >= len(data) - 1:
            return {'outcome': 'no_data', 'pnl': 0, 'bars_held': 0}
        
        # Check subsequent bars for TP/SL hits
        for i in range(entry_idx + 1, min(entry_idx + 100, len(data))):  # Max 100 bars
            bar_high = data['high'].iloc[i]
            bar_low = data['low'].iloc[i]
            
            if direction == 'buy':
                if bar_high >= tp_price:
                    return {
                        'outcome': 'tp_hit',
                        'exit_price': tp_price,
                        'pnl': tp_price - entry_price,
                        'bars_held': i - entry_idx
                    }
                elif bar_low <= sl_price:
                    return {
                        'outcome': 'sl_hit',
                        'exit_price': sl_price,
                        'pnl': sl_price - entry_price,
                        'bars_held': i - entry_idx
                    }
            else:  # sell
                if bar_low <= tp_price:
                    return {
                        'outcome': 'tp_hit',
                        'exit_price': tp_price,
                        'pnl': entry_price - tp_price,
                        'bars_held': i - entry_idx
                    }
                elif bar_high >= sl_price:
                    return {
                        'outcome': 'sl_hit',
                        'exit_price': sl_price,
                        'pnl': entry_price - sl_price,
                        'bars_held': i - entry_idx
                    }
        
        # No TP/SL hit within timeframe
        final_price = data['close'].iloc[min(entry_idx + 100, len(data) - 1)]
        if direction == 'buy':
            pnl = final_price - entry_price
        else:
            pnl = entry_price - final_price
        
        return {
            'outcome': 'timeout',
            'exit_price': final_price,
            'pnl': pnl,
            'bars_held': min(100, len(data) - entry_idx - 1)
        }
    
    def _calculate_performance_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics for a set of trades"""
        if not trades:
            return {'win_rate': 0, 'profit_factor': 0, 'total_trades': 0, 'avg_rr': 0, 'max_drawdown': 0, 'score': 0}
        
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(trades) * 100
        
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate average RR
        rr_ratios = []
        for trade in trades:
            if trade['pnl'] < 0:  # Loss
                rr_ratios.append(-trade['pnl'])  # Risk
            else:  # Profit
                rr_ratios.append(trade['pnl'])   # Reward
        
        avg_rr = np.mean(rr_ratios) if rr_ratios else 0
        
        # Calculate max drawdown
        cumulative_pnl = np.cumsum([t['pnl'] for t in trades])
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = (cumulative_pnl - running_max) / np.maximum(running_max, 1)
        max_drawdown = abs(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0
        
        # Calculate composite score
        score = (win_rate * 0.4) + (min(profit_factor, 5) * 10) + (min(avg_rr, 10) * 5) - (max_drawdown * 2)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_rr': avg_rr,
            'max_drawdown': max_drawdown,
            'score': score
        }
    
    def _create_heatmap_data(self, results: List[Dict], tp_multipliers: np.array, 
                           sl_multipliers: np.array) -> Dict[str, Any]:
        """Create heatmap data for visualization"""
        # Create matrices for different metrics
        win_rate_matrix = np.zeros((len(sl_multipliers), len(tp_multipliers)))
        profit_factor_matrix = np.zeros((len(sl_multipliers), len(tp_multipliers)))
        score_matrix = np.zeros((len(sl_multipliers), len(tp_multipliers)))
        
        for result in results:
            tp_idx = np.where(tp_multipliers == result['tp_multiplier'])[0][0]
            sl_idx = np.where(sl_multipliers == result['sl_multiplier'])[0][0]
            
            win_rate_matrix[sl_idx, tp_idx] = result['win_rate']
            profit_factor_matrix[sl_idx, tp_idx] = min(result['profit_factor'], 10)  # Cap for visualization
            score_matrix[sl_idx, tp_idx] = result['score']
        
        return {
            'tp_multipliers': tp_multipliers.tolist(),
            'sl_multipliers': sl_multipliers.tolist(),
            'win_rate_matrix': win_rate_matrix.tolist(),
            'profit_factor_matrix': profit_factor_matrix.tolist(),
            'score_matrix': score_matrix.tolist()
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.optimization_results:
            return {'status': 'not_optimized'}
        
        best = self.optimization_results['best_combination']
        
        return {
            'status': 'optimized',
            'best_tp_multiplier': best['tp_multiplier'],
            'best_sl_multiplier': best['sl_multiplier'],
            'expected_win_rate': best['win_rate'],
            'expected_profit_factor': best['profit_factor'],
            'optimization_date': self.optimization_results['optimization_date'],
            'total_combinations_tested': len(self.optimization_results['all_results'])
        }