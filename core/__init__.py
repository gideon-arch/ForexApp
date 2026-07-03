# core/__init__.py
# AI Crypto Signal Bot v4.0 — Core Modules (No LSTM)

from .data_feed import DataFeedManager
from .technical_analysis import TechnicalAnalyzer
from .filters import FilterManager
from .notifications import NotificationManager, SignalAlert
from .candlestick_patterns import CandlestickPatternRecognizer
from .fakeout_detector import FakeoutDetector

__all__ = [
    'DataFeedManager',
    'TechnicalAnalyzer',
    'FilterManager',
    'NotificationManager',
    'SignalAlert',
    'CandlestickPatternRecognizer',
    'FakeoutDetector',
]

