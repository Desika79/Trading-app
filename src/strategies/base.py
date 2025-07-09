"""
Base Strategy Class for Trading System
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from ..config.settings import settings


class TradeSignal:
    """Represents a trading signal"""
    
    def __init__(
        self,
        symbol: str,
        signal_type: str,  # 'BUY', 'SELL', 'CLOSE'
        timestamp: datetime,
        price: float,
        confidence: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        position_size: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.symbol = symbol
        self.signal_type = signal_type
        self.timestamp = timestamp
        self.price = price
        self.confidence = confidence
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'timestamp': self.timestamp,
            'price': self.price,
            'confidence': self.confidence,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size,
            'metadata': self.metadata
        }


class StrategyMetrics:
    """Strategy performance metrics"""
    
    def __init__(self):
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.profit_factor = 0.0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.expectancy = 0.0
        self.trades_history: List[Dict[str, Any]] = []
    
    def update_metrics(self, trade_pnl: float, trade_result: str):
        """Update metrics with new trade"""
        self.total_trades += 1
        self.total_pnl += trade_pnl
        
        if trade_result == 'WIN':
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Recalculate metrics
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Store trade
        self.trades_history.append({
            'pnl': trade_pnl,
            'result': trade_result,
            'timestamp': datetime.now()
        })
    
    def calculate_advanced_metrics(self):
        """Calculate advanced performance metrics"""
        if not self.trades_history:
            return
        
        pnls = [trade['pnl'] for trade in self.trades_history]
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [pnl for pnl in pnls if pnl < 0]
        
        # Average win/loss
        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = abs(np.mean(losses)) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        self.expectancy = (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)
        
        # Max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = (cumulative_pnl - running_max) / running_max * 100
        self.max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns_std = np.std(pnls)
            self.sharpe_ratio = np.mean(pnls) / returns_std if returns_std > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        self.calculate_advanced_metrics()
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate * 100, 2),
            'total_pnl': round(self.total_pnl, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'expectancy': round(self.expectancy, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2)
        }


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}
        self.metrics = StrategyMetrics()
        self.is_active = True
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.signals_history: List[TradeSignal] = []
        
        # Risk management parameters
        self.risk_per_trade = self.parameters.get('risk_per_trade', settings.DEFAULT_RISK_PER_TRADE)
        self.max_positions = self.parameters.get('max_positions', settings.MAX_POSITIONS)
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', settings.DEFAULT_STOP_LOSS)
        self.take_profit_pct = self.parameters.get('take_profit_pct', settings.DEFAULT_TAKE_PROFIT)
        
        logger.info(f"Initialized strategy: {self.name}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradeSignal]:
        """Generate trading signals based on market data"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradeSignal, account_balance: float) -> float:
        """Calculate position size based on risk management rules"""
        pass
    
    def validate_signal(self, signal: TradeSignal) -> bool:
        """Validate if a signal meets strategy criteria"""
        try:
            # Basic validation
            if not signal.symbol or signal.confidence < 0:
                return False
            
            # Check if we already have a position
            if signal.symbol in self.positions and signal.signal_type in ['BUY', 'SELL']:
                return False
            
            # Check max positions limit
            if len(self.positions) >= self.max_positions and signal.signal_type in ['BUY', 'SELL']:
                return False
            
            # Strategy-specific validation
            return self.custom_validation(signal)
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return False
    
    def custom_validation(self, signal: TradeSignal) -> bool:
        """Override this method for custom signal validation"""
        return True
    
    def calculate_stop_loss(self, entry_price: float, signal_type: str) -> float:
        """Calculate stop loss price"""
        if signal_type == 'BUY':
            return entry_price * (1 - self.stop_loss_pct)
        else:  # SELL
            return entry_price * (1 + self.stop_loss_pct)
    
    def calculate_take_profit(self, entry_price: float, signal_type: str) -> float:
        """Calculate take profit price"""
        if signal_type == 'BUY':
            return entry_price * (1 + self.take_profit_pct)
        else:  # SELL
            return entry_price * (1 - self.take_profit_pct)
    
    def update_position(self, symbol: str, signal: TradeSignal):
        """Update position based on signal"""
        if signal.signal_type in ['BUY', 'SELL']:
            # Open new position
            self.positions[symbol] = {
                'signal_type': signal.signal_type,
                'entry_price': signal.price,
                'entry_time': signal.timestamp,
                'position_size': signal.position_size,
                'stop_loss': signal.stop_loss or self.calculate_stop_loss(signal.price, signal.signal_type),
                'take_profit': signal.take_profit or self.calculate_take_profit(signal.price, signal.signal_type),
                'unrealized_pnl': 0.0
            }
            logger.info(f"Opened {signal.signal_type} position for {symbol} at {signal.price}")
            
        elif signal.signal_type == 'CLOSE' and symbol in self.positions:
            # Close position
            position = self.positions[symbol]
            pnl = self.calculate_pnl(position, signal.price)
            
            # Update metrics
            result = 'WIN' if pnl > 0 else 'LOSS'
            self.metrics.update_metrics(pnl, result)
            
            # Remove position
            del self.positions[symbol]
            logger.info(f"Closed position for {symbol} at {signal.price}, PnL: {pnl}")
    
    def calculate_pnl(self, position: Dict[str, Any], current_price: float) -> float:
        """Calculate P&L for a position"""
        entry_price = position['entry_price']
        position_size = position['position_size']
        signal_type = position['signal_type']
        
        if signal_type == 'BUY':
            return (current_price - entry_price) / entry_price * position_size
        else:  # SELL
            return (entry_price - current_price) / entry_price * position_size
    
    def check_exit_conditions(self, symbol: str, current_price: float) -> Optional[TradeSignal]:
        """Check if position should be closed based on stop loss or take profit"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        signal_type = position['signal_type']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        
        # Check stop loss
        if signal_type == 'BUY' and current_price <= stop_loss:
            return TradeSignal(
                symbol=symbol,
                signal_type='CLOSE',
                timestamp=datetime.now(),
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'stop_loss'}
            )
        elif signal_type == 'SELL' and current_price >= stop_loss:
            return TradeSignal(
                symbol=symbol,
                signal_type='CLOSE',
                timestamp=datetime.now(),
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'stop_loss'}
            )
        
        # Check take profit
        if signal_type == 'BUY' and current_price >= take_profit:
            return TradeSignal(
                symbol=symbol,
                signal_type='CLOSE',
                timestamp=datetime.now(),
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'take_profit'}
            )
        elif signal_type == 'SELL' and current_price <= take_profit:
            return TradeSignal(
                symbol=symbol,
                signal_type='CLOSE',
                timestamp=datetime.now(),
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'take_profit'}
            )
        
        return None
    
    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """Update unrealized P&L for open positions"""
        if symbol in self.positions:
            position = self.positions[symbol]
            position['unrealized_pnl'] = self.calculate_pnl(position, current_price)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = self.metrics.to_dict()
        summary.update({
            'strategy_name': self.name,
            'parameters': self.parameters,
            'open_positions': len(self.positions),
            'total_signals': len(self.signals_history)
        })
        return summary
    
    def reset_metrics(self):
        """Reset strategy metrics"""
        self.metrics = StrategyMetrics()
        self.positions.clear()
        self.signals_history.clear()
        logger.info(f"Reset metrics for strategy: {self.name}")
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.parameters.update(parameters)
        
        # Update risk management parameters
        self.risk_per_trade = parameters.get('risk_per_trade', self.risk_per_trade)
        self.max_positions = parameters.get('max_positions', self.max_positions)
        self.stop_loss_pct = parameters.get('stop_loss_pct', self.stop_loss_pct)
        self.take_profit_pct = parameters.get('take_profit_pct', self.take_profit_pct)
        
        logger.info(f"Updated parameters for strategy: {self.name}")
    
    def process_market_data(self, data: pd.DataFrame, current_prices: Dict[str, float] = None) -> List[TradeSignal]:
        """Main method to process market data and generate signals"""
        try:
            if not self.is_active:
                return []
            
            # Generate new signals
            new_signals = self.generate_signals(data)
            
            # Validate signals
            valid_signals = [signal for signal in new_signals if self.validate_signal(signal)]
            
            # Check exit conditions for existing positions
            exit_signals = []
            if current_prices:
                for symbol, price in current_prices.items():
                    exit_signal = self.check_exit_conditions(symbol, price)
                    if exit_signal:
                        exit_signals.append(exit_signal)
                    
                    # Update unrealized P&L
                    self.update_unrealized_pnl(symbol, price)
            
            # Combine new signals and exit signals
            all_signals = valid_signals + exit_signals
            
            # Store signals history
            self.signals_history.extend(all_signals)
            
            # Update positions
            for signal in all_signals:
                self.update_position(signal.symbol, signal)
            
            if all_signals:
                logger.info(f"Strategy {self.name} generated {len(all_signals)} signals")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"Error processing market data in strategy {self.name}: {e}")
            return []