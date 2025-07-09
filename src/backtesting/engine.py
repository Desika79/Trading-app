"""
Backtesting Engine for Trading Strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import asyncio

from ..strategies.base import BaseStrategy, TradeSignal
from ..data.providers import DataProviderManager
from ..config.settings import settings


class BacktestResult:
    """Container for backtesting results"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.initial_capital: float = 0
        self.final_capital: float = 0
    
    def add_trade(self, trade: Dict[str, Any]):
        """Add a completed trade to results"""
        self.trades.append(trade)
    
    def add_equity_point(self, timestamp: datetime, equity: float, drawdown: float):
        """Add a point to the equity curve"""
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'drawdown': drawdown
        })
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        
        # P&L metrics
        total_pnl = sum(t['pnl'] for t in self.trades)
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = sum(t['pnl'] for t in self.trades if t['pnl'] < 0)
        
        # Win rate and averages
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = abs(gross_loss) / losing_trades if losing_trades > 0 else 0
        
        # Risk metrics
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Drawdown
        max_drawdown = max([point['drawdown'] for point in self.equity_curve], default=0)
        
        # Returns
        total_return = (self.final_capital - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe ratio (simplified)
        returns = [t['pnl'] / self.initial_capital for t in self.trades]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate * 100, 2),
            'total_return': round(total_return, 2),
            'total_pnl': round(total_pnl, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'expectancy': round(expectancy, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'metrics': self.metrics,
            'total_trades': len(self.trades),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital
        }


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.data_manager = DataProviderManager()
        self.commission_rate = 0.0001  # 0.01% commission
        self.slippage = 0.0001  # 0.01% slippage
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Run backtest for a single strategy"""
        
        logger.info(f"Starting backtest for {strategy.name} on {symbol}")
        
        # Initialize result
        result = BacktestResult(strategy.name)
        result.start_date = start_date
        result.end_date = end_date
        result.initial_capital = self.initial_capital
        
        # Connect to data providers
        await self.data_manager.connect_all()
        
        try:
            # Get historical data
            data = await self.data_manager.get_historical_data(
                symbol, timeframe, start_date, end_date
            )
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return result
            
            # Add symbol attribute to data
            data.attrs['symbol'] = symbol
            
            # Initialize tracking variables
            current_capital = self.initial_capital
            peak_capital = self.initial_capital
            open_trades: Dict[str, Dict[str, Any]] = {}
            
            # Process each bar
            for i in range(len(data)):
                current_bar = data.iloc[i:i+1]
                historical_data = data.iloc[:i+1]
                
                if len(historical_data) < 50:  # Need enough data for indicators
                    continue
                
                current_price = current_bar['close'].iloc[0]
                current_timestamp = current_bar.index[0]
                
                # Check for exit conditions first
                exit_signals = []
                for trade_id, trade in list(open_trades.items()):
                    exit_signal = self.check_exit_conditions(trade, current_price, current_timestamp)
                    if exit_signal:
                        exit_signals.append((trade_id, exit_signal))
                
                # Process exit signals
                for trade_id, exit_signal in exit_signals:
                    trade = open_trades[trade_id]
                    pnl = self.calculate_trade_pnl(trade, exit_signal.price)
                    
                    # Apply commission and slippage
                    pnl -= self.calculate_costs(trade['entry_price'], exit_signal.price, trade['position_size'])
                    
                    # Update capital
                    current_capital += pnl
                    
                    # Record trade
                    completed_trade = {
                        'entry_time': trade['entry_time'],
                        'exit_time': exit_signal.timestamp,
                        'symbol': symbol,
                        'signal_type': trade['signal_type'],
                        'entry_price': trade['entry_price'],
                        'exit_price': exit_signal.price,
                        'position_size': trade['position_size'],
                        'pnl': pnl,
                        'exit_reason': exit_signal.metadata.get('reason', 'signal')
                    }
                    
                    result.add_trade(completed_trade)
                    del open_trades[trade_id]
                    
                    logger.debug(f"Closed trade: {completed_trade}")
                
                # Generate new signals
                current_prices = {symbol: current_price}
                signals = strategy.process_market_data(historical_data, current_prices)
                
                # Process entry signals
                for signal in signals:
                    if signal.signal_type in ['BUY', 'SELL'] and len(open_trades) < strategy.max_positions:
                        position_size = strategy.calculate_position_size(signal, current_capital)
                        
                        # Create trade record
                        trade_id = f"{symbol}_{signal.signal_type}_{current_timestamp}"
                        trade = {
                            'id': trade_id,
                            'entry_time': signal.timestamp,
                            'signal_type': signal.signal_type,
                            'entry_price': signal.price,
                            'position_size': position_size,
                            'stop_loss': signal.stop_loss or strategy.calculate_stop_loss(signal.price, signal.signal_type),
                            'take_profit': signal.take_profit or strategy.calculate_take_profit(signal.price, signal.signal_type)
                        }
                        
                        open_trades[trade_id] = trade
                        logger.debug(f"Opened trade: {trade}")
                
                # Update equity curve
                peak_capital = max(peak_capital, current_capital)
                drawdown = (peak_capital - current_capital) / peak_capital * 100
                result.add_equity_point(current_timestamp, current_capital, drawdown)
            
            # Close any remaining open trades
            final_price = data['close'].iloc[-1]
            final_timestamp = data.index[-1]
            
            for trade in open_trades.values():
                pnl = self.calculate_trade_pnl(trade, final_price)
                pnl -= self.calculate_costs(trade['entry_price'], final_price, trade['position_size'])
                current_capital += pnl
                
                completed_trade = {
                    'entry_time': trade['entry_time'],
                    'exit_time': final_timestamp,
                    'symbol': symbol,
                    'signal_type': trade['signal_type'],
                    'entry_price': trade['entry_price'],
                    'exit_price': final_price,
                    'position_size': trade['position_size'],
                    'pnl': pnl,
                    'exit_reason': 'end_of_data'
                }
                
                result.add_trade(completed_trade)
            
            result.final_capital = current_capital
            result.calculate_metrics()
            
            logger.info(f"Backtest completed for {strategy.name}: {result.metrics}")
            
        except Exception as e:
            logger.error(f"Error during backtest: {e}")
        
        finally:
            await self.data_manager.disconnect_all()
        
        return result
    
    def check_exit_conditions(self, trade: Dict[str, Any], current_price: float, timestamp: datetime) -> Optional[TradeSignal]:
        """Check if trade should be closed"""
        signal_type = trade['signal_type']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        
        # Check stop loss
        if signal_type == 'BUY' and current_price <= stop_loss:
            return TradeSignal(
                symbol=trade.get('symbol', ''),
                signal_type='CLOSE',
                timestamp=timestamp,
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'stop_loss'}
            )
        elif signal_type == 'SELL' and current_price >= stop_loss:
            return TradeSignal(
                symbol=trade.get('symbol', ''),
                signal_type='CLOSE',
                timestamp=timestamp,
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'stop_loss'}
            )
        
        # Check take profit
        if signal_type == 'BUY' and current_price >= take_profit:
            return TradeSignal(
                symbol=trade.get('symbol', ''),
                signal_type='CLOSE',
                timestamp=timestamp,
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'take_profit'}
            )
        elif signal_type == 'SELL' and current_price <= take_profit:
            return TradeSignal(
                symbol=trade.get('symbol', ''),
                signal_type='CLOSE',
                timestamp=timestamp,
                price=current_price,
                confidence=1.0,
                metadata={'reason': 'take_profit'}
            )
        
        return None
    
    def calculate_trade_pnl(self, trade: Dict[str, Any], exit_price: float) -> float:
        """Calculate P&L for a trade"""
        entry_price = trade['entry_price']
        position_size = trade['position_size']
        signal_type = trade['signal_type']
        
        if signal_type == 'BUY':
            return (exit_price - entry_price) / entry_price * position_size
        else:  # SELL
            return (entry_price - exit_price) / entry_price * position_size
    
    def calculate_costs(self, entry_price: float, exit_price: float, position_size: float) -> float:
        """Calculate trading costs (commission + slippage)"""
        entry_cost = entry_price * position_size * self.commission_rate
        exit_cost = exit_price * position_size * self.commission_rate
        slippage_cost = (entry_price + exit_price) * position_size * self.slippage
        
        return entry_cost + exit_cost + slippage_cost
    
    async def run_multi_strategy_backtest(
        self,
        strategies: List[BaseStrategy],
        symbols: List[str],
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[BacktestResult]:
        """Run backtest for multiple strategies and symbols"""
        
        results = []
        
        for strategy in strategies:
            for symbol in symbols:
                try:
                    result = await self.run_backtest(strategy, symbol, timeframe, start_date, end_date)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed backtest for {strategy.name} on {symbol}: {e}")
        
        return results
    
    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """Compare multiple strategy results"""
        
        comparison_data = []
        
        for result in results:
            if result.metrics:
                row = {
                    'Strategy': result.strategy_name,
                    'Total Trades': result.metrics.get('total_trades', 0),
                    'Win Rate (%)': result.metrics.get('win_rate', 0),
                    'Total Return (%)': result.metrics.get('total_return', 0),
                    'Profit Factor': result.metrics.get('profit_factor', 0),
                    'Max Drawdown (%)': result.metrics.get('max_drawdown', 0),
                    'Sharpe Ratio': result.metrics.get('sharpe_ratio', 0),
                    'Expectancy': result.metrics.get('expectancy', 0)
                }
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values('Total Return (%)', ascending=False)


class StrategyOptimizer:
    """Optimize strategy parameters using grid search"""
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.backtest_engine = backtest_engine
    
    async def optimize_strategy(
        self,
        strategy_class,
        parameter_grid: Dict[str, List[Any]],
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        optimization_metric: str = 'total_return'
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """Optimize strategy parameters using grid search"""
        
        logger.info(f"Starting optimization for {strategy_class.__name__}")
        
        best_params = None
        best_result = None
        best_score = float('-inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(parameter_grid)
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing parameter combination {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Create strategy with parameters
                strategy = strategy_class(params)
                
                # Run backtest
                result = await self.backtest_engine.run_backtest(
                    strategy, symbol, timeframe, start_date, end_date
                )
                
                # Check if this is the best result
                if result.metrics:
                    score = result.metrics.get(optimization_metric, float('-inf'))
                    
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_result = result
                        
                        logger.info(f"New best parameters found: {params}, Score: {score}")
                
            except Exception as e:
                logger.error(f"Error testing parameters {params}: {e}")
        
        return best_params, best_result
    
    def _generate_param_combinations(self, parameter_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters"""
        import itertools
        
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations