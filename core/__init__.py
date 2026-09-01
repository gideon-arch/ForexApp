# core/__init__.py
# AI Crypto Signal Bot v5.0 — Multi-Timeframe + Dual Asset
# Added: Deriv support, MultiTimeframeEngine, duplicate setup prevention

from .data_feed import DataFeedManager
from .technical_analysis import TechnicalAnalyzer
from .filters import FilterManager
from .notifications import NotificationManager, SignalAlert
from .candlestick_patterns import CandlestickPatternRecognizer
from .fakeout_detector import FakeoutDetector
from .multi_timeframe import MultiTimeframeEngine

__all__ = [
    'DataFeedManager',
    'TechnicalAnalyzer',
    'FilterManager',
    'NotificationManager',
    'SignalAlert',
    'CandlestickPatternRecognizer',
    'FakeoutDetector',
    'MultiTimeframeEngine',]