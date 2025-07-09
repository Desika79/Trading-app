"""
Data Providers for Forex and Synthetic Indices
"""
import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import ccxt
import yfinance as yf
from loguru import logger

from ..config.settings import settings


class BaseDataProvider(ABC):
    """Abstract base class for data providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    async def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get live price data"""
        pass


class BinanceDataProvider(BaseDataProvider):
    """Binance data provider for Forex and crypto pairs"""
    
    def __init__(self):
        super().__init__("Binance")
        self.exchange = None
        self.websocket = None
    
    async def connect(self) -> bool:
        """Connect to Binance API"""
        try:
            self.exchange = ccxt.binance({
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_SECRET_KEY,
                'sandbox': False,  # Set to True for testing
                'enableRateLimit': True,
            })
            
            # Test connection
            await self.exchange.load_markets()
            self.is_connected = True
            logger.info(f"Connected to {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close Binance connection"""
        if self.exchange:
            await self.exchange.close()
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical OHLCV data from Binance"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Convert timeframe to Binance format
            tf_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            
            binance_tf = tf_map.get(timeframe, '1h')
            
            # Fetch data
            since = int(start_date.timestamp() * 1000)
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol, binance_tf, since=since, limit=1000
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Filter by end date
            df = df[df.index <= end_date]
            
            logger.info(f"Fetched {len(df)} records for {symbol} from {self.name}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data from {self.name}: {e}")
            return pd.DataFrame()
    
    async def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get live price data from Binance"""
        try:
            if not self.is_connected:
                await self.connect()
            
            ticker = await self.exchange.fetch_ticker(symbol)
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'timestamp': datetime.now(),
                'source': self.name
            }
            
        except Exception as e:
            logger.error(f"Error fetching live data from {self.name}: {e}")
            return {}


class DerivDataProvider(BaseDataProvider):
    """Deriv data provider for synthetic indices like V75"""
    
    def __init__(self):
        super().__init__("Deriv")
        self.api_token = settings.DERIV_API_TOKEN
        self.websocket = None
        self.websocket_url = "wss://ws.binaryws.com/websockets/v3"
    
    async def connect(self) -> bool:
        """Connect to Deriv WebSocket API"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            
            # Authorize connection
            auth_request = {
                "authorize": self.api_token
            }
            await self.websocket.send(json.dumps(auth_request))
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            if 'authorize' in auth_response:
                self.is_connected = True
                logger.info(f"Connected to {self.name}")
                return True
            else:
                logger.error(f"Authorization failed for {self.name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close Deriv connection"""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical data for synthetic indices"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # Map symbols to Deriv market names
            symbol_map = {
                'V75': 'R_75',
                'V100': 'R_100',
                'V25': 'R_25',
                'V10': 'R_10'
            }
            
            deriv_symbol = symbol_map.get(symbol, symbol)
            
            # Convert timeframe to granularity
            granularity_map = {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '4h': 14400, '1d': 86400
            }
            
            granularity = granularity_map.get(timeframe, 3600)
            
            # Request historical ticks
            request = {
                "ticks_history": deriv_symbol,
                "adjust_start_time": 1,
                "count": 5000,
                "end": "latest",
                "start": int(start_date.timestamp()),
                "style": "candles",
                "granularity": granularity
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if 'candles' in data:
                candles = data['candles']
                df_data = []
                
                for candle in candles:
                    df_data.append({
                        'timestamp': pd.to_datetime(candle['epoch'], unit='s'),
                        'open': candle['open'],
                        'high': candle['high'],
                        'low': candle['low'],
                        'close': candle['close'],
                        'volume': 1  # Synthetic indices don't have volume
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                
                # Filter by end date
                df = df[df.index <= end_date]
                
                logger.info(f"Fetched {len(df)} records for {symbol} from {self.name}")
                return df
            else:
                logger.error(f"No candles data received for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data from {self.name}: {e}")
            return pd.DataFrame()
    
    async def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get live price data for synthetic indices"""
        try:
            if not self.is_connected:
                await self.connect()
            
            symbol_map = {
                'V75': 'R_75',
                'V100': 'R_100',
                'V25': 'R_25',
                'V10': 'R_10'
            }
            
            deriv_symbol = symbol_map.get(symbol, symbol)
            
            request = {
                "ticks": deriv_symbol,
                "subscribe": 1
            }
            
            await self.websocket.send(json.dumps(request))
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if 'tick' in data:
                tick = data['tick']
                return {
                    'symbol': symbol,
                    'price': tick['quote'],
                    'bid': tick['quote'],
                    'ask': tick['quote'],
                    'volume': 1,
                    'timestamp': pd.to_datetime(tick['epoch'], unit='s'),
                    'source': self.name
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching live data from {self.name}: {e}")
            return {}


class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance provider for Forex data"""
    
    def __init__(self):
        super().__init__("Yahoo Finance")
        self.is_connected = True  # Yahoo Finance doesn't require explicit connection
    
    async def connect(self) -> bool:
        """Yahoo Finance doesn't require explicit connection"""
        self.is_connected = True
        return True
    
    async def disconnect(self) -> None:
        """Yahoo Finance doesn't require explicit disconnection"""
        pass
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical Forex data from Yahoo Finance"""
        try:
            # Convert symbol to Yahoo Finance format (e.g., EURUSD -> EURUSD=X)
            if symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']:
                yahoo_symbol = f"{symbol}=X"
            else:
                yahoo_symbol = symbol
            
            # Map timeframe to Yahoo Finance interval
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '1h', '1d': '1d'
            }
            
            interval = interval_map.get(timeframe, '1h')
            
            # Download data
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if not df.empty:
                # Rename columns to match our standard format
                df.columns = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.index.name = 'timestamp'
                
                logger.info(f"Fetched {len(df)} records for {symbol} from {self.name}")
                return df
            else:
                logger.warning(f"No data available for {symbol} from {self.name}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching historical data from {self.name}: {e}")
            return pd.DataFrame()
    
    async def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get live price data (limited for Yahoo Finance)"""
        try:
            if symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']:
                yahoo_symbol = f"{symbol}=X"
            else:
                yahoo_symbol = symbol
            
            ticker = yf.Ticker(yahoo_symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('regularMarketPrice', 0),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'volume': info.get('volume', 0),
                'timestamp': datetime.now(),
                'source': self.name
            }
            
        except Exception as e:
            logger.error(f"Error fetching live data from {self.name}: {e}")
            return {}


class DataProviderManager:
    """Manager class to coordinate multiple data providers"""
    
    def __init__(self):
        self.providers = {
            'binance': BinanceDataProvider(),
            'deriv': DerivDataProvider(),
            'yahoo': YahooFinanceProvider()
        }
        self.symbol_routing = {
            # Forex pairs - prioritize Binance, fallback to Yahoo
            'EURUSD': ['binance', 'yahoo'],
            'GBPUSD': ['binance', 'yahoo'],
            'USDJPY': ['binance', 'yahoo'],
            'AUDUSD': ['binance', 'yahoo'],
            'USDCAD': ['binance', 'yahoo'],
            
            # Synthetic indices - Deriv only
            'V75': ['deriv'],
            'V100': ['deriv'],
            'V25': ['deriv'],
            'V10': ['deriv']
        }
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all data providers"""
        results = {}
        for name, provider in self.providers.items():
            results[name] = await provider.connect()
        return results
    
    async def disconnect_all(self) -> None:
        """Disconnect from all data providers"""
        for provider in self.providers.values():
            await provider.disconnect()
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical data using the best available provider"""
        
        provider_names = self.symbol_routing.get(symbol, ['yahoo'])
        
        for provider_name in provider_names:
            provider = self.providers.get(provider_name)
            if provider and provider.is_connected:
                try:
                    df = await provider.get_historical_data(symbol, timeframe, start_date, end_date)
                    if not df.empty:
                        return df
                except Exception as e:
                    logger.warning(f"Provider {provider_name} failed for {symbol}: {e}")
                    continue
        
        logger.error(f"No data available for {symbol} from any provider")
        return pd.DataFrame()
    
    async def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """Get live data using the best available provider"""
        
        provider_names = self.symbol_routing.get(symbol, ['yahoo'])
        
        for provider_name in provider_names:
            provider = self.providers.get(provider_name)
            if provider and provider.is_connected:
                try:
                    data = await provider.get_live_data(symbol)
                    if data:
                        return data
                except Exception as e:
                    logger.warning(f"Provider {provider_name} failed for live data {symbol}: {e}")
                    continue
        
        logger.error(f"No live data available for {symbol}")
        return {}
    
    async def get_multiple_symbols_data(
        self, 
        symbols: List[str], 
        timeframe: str, 
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols concurrently"""
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(
                self.get_historical_data(symbol, timeframe, start_date, end_date)
            )
            tasks.append((symbol, task))
        
        results = {}
        for symbol, task in tasks:
            try:
                df = await task
                results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results