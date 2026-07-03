# core/__init__.py
# AI Crypto Signal Bot v4.1 — Core Modules (No LSTM)
# Added: Fibonacci zones, improved trend filter, symmetrical triangle alignment

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
