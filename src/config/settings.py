"""
Trading System Configuration Settings
"""
import os
from typing import List, Dict, Any
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Main configuration class for the trading system"""
    
    # API Configuration
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=1, env="API_WORKERS")
    
    # Database Configuration
    DATABASE_URL: str = Field(env="DATABASE_URL", default="postgresql://user:password@localhost:5432/trading_db")
    REDIS_URL: str = Field(env="REDIS_URL", default="redis://localhost:6379/0")
    INFLUXDB_URL: str = Field(env="INFLUXDB_URL", default="http://localhost:8086")
    INFLUXDB_TOKEN: str = Field(env="INFLUXDB_TOKEN", default="")
    INFLUXDB_ORG: str = Field(env="INFLUXDB_ORG", default="trading_org")
    INFLUXDB_BUCKET: str = Field(env="INFLUXDB_BUCKET", default="market_data")
    
    # External API Keys
    BINANCE_API_KEY: str = Field(env="BINANCE_API_KEY", default="")
    BINANCE_SECRET_KEY: str = Field(env="BINANCE_SECRET_KEY", default="")
    DERIV_API_TOKEN: str = Field(env="DERIV_API_TOKEN", default="")
    MT5_LOGIN: str = Field(env="MT5_LOGIN", default="")
    MT5_PASSWORD: str = Field(env="MT5_PASSWORD", default="")
    MT5_SERVER: str = Field(env="MT5_SERVER", default="")
    
    # Trading Configuration
    DEFAULT_SYMBOLS: List[str] = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
        "V75", "V100", "V25", "V10"  # Volatility Indices
    ]
    
    DEFAULT_TIMEFRAMES: List[str] = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    
    # Backtesting Configuration
    BACKTEST_START_DATE: str = "2023-01-01"
    BACKTEST_END_DATE: str = "2024-01-01"
    INITIAL_CAPITAL: float = 10000.0
    MAX_POSITIONS: int = 5
    
    # Signal Generation
    MIN_WIN_RATE: float = 0.70  # 70% minimum win rate
    TARGET_WIN_RATE: float = 0.90  # 90% target win rate
    MIN_PROFIT_FACTOR: float = 1.5
    MAX_DRAWDOWN: float = 0.15  # 15% max drawdown
    
    # Risk Management
    DEFAULT_RISK_PER_TRADE: float = 0.02  # 2% per trade
    MAX_DAILY_RISK: float = 0.06  # 6% per day
    DEFAULT_STOP_LOSS: float = 0.02  # 2% stop loss
    DEFAULT_TAKE_PROFIT: float = 0.04  # 4% take profit
    
    # Technical Indicators Parameters
    RSI_PERIOD: int = 14
    RSI_OVERSOLD: int = 30
    RSI_OVERBOUGHT: int = 70
    
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    
    ATR_PERIOD: int = 14
    
    MA_SHORT: int = 50
    MA_LONG: int = 200
    
    # Machine Learning
    ML_FEATURES_WINDOW: int = 60  # 60 periods for feature engineering
    ML_PREDICTION_HORIZON: int = 5  # 5 periods ahead prediction
    ML_RETRAIN_INTERVAL: int = 24  # Retrain every 24 hours
    
    # Logging
    LOG_LEVEL: str = Field(env="LOG_LEVEL", default="INFO")
    LOG_FILE: str = "logs/trading_system.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()