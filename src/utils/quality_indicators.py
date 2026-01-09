"""
Forecast quality indicators for non-technical users.
Provides confidence scores, traffic light status, and plain-language explanations.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class QualityLevel(Enum):
    """Traffic light quality levels."""
    HIGH = "high"       # Green - 80-100%
    MEDIUM = "medium"   # Yellow - 60-79%
    LOW = "low"         # Red - below 60%


@dataclass
class QualityIndicator:
    """Container for forecast quality information."""
    confidence_score: float  # 0-100
    level: QualityLevel
    color: str  # hex color
    emoji: str
    label: str
    explanation: str
    details: List[str]


@dataclass
class ForecastComparison:
    """Container for forecast vs actuals comparison."""
    forecast_id: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    
    # Overall metrics
    overall_accuracy: float  # 0-100%
    overall_mape: float
    
    # Per-target metrics
    target_metrics: Dict[str, Dict[str, float]]
    
    # Plain language summary
    summary: str
    highlights: List[str]
    areas_for_improvement: List[str]
    
    # Detailed comparison data
    comparison_df: Optional[pd.DataFrame] = None


def calculate_confidence_score(
    training_metrics: Dict[str, Dict[str, float]],
    data_quality_score: float = 100.0,
    forecast_horizon_days: int = 7
) -> float:
    """
    Calculate overall confidence score (0-100) based on model performance.
    
    Args:
        training_metrics: Dict of {target: {rmse, mae, r2, mape}}
        data_quality_score: Score for data completeness/quality (0-100)
        forecast_horizon_days: How far ahead we're forecasting
        
    Returns:
        Confidence score from 0-100
    """
    if not training_metrics:
        return 50.0  # Default when no metrics available
    
    # Base score from MAPE (lower is better)
    mape_scores = []
    r2_scores = []
    
    for target, metrics in training_metrics.items():
        mape = metrics.get('mape', 20)  # Default 20% if missing
        r2 = metrics.get('r2', 0.5)
        
        # Convert MAPE to score (0% MAPE = 100 score, 50% MAPE = 0 score)
        mape_score = max(0, 100 - (mape * 2))
        mape_scores.append(mape_score)
        
        # R² directly maps to 0-100 (capped)
        r2_score = max(0, min(100, r2 * 100))
        r2_scores.append(r2_score)
    
    # Average scores
    avg_mape_score = np.mean(mape_scores) if mape_scores else 50
    avg_r2_score = np.mean(r2_scores) if r2_scores else 50
    
    # Horizon penalty (longer forecasts are less certain)
    horizon_penalty = min(20, forecast_horizon_days * 0.5)
    
    # Weighted combination
    base_score = (avg_mape_score * 0.5) + (avg_r2_score * 0.3) + (data_quality_score * 0.2)
    
    # Apply horizon penalty
    final_score = max(0, min(100, base_score - horizon_penalty))
    
    return round(final_score, 1)


def get_traffic_light_status(confidence_score: float) -> Tuple[QualityLevel, str, str, str]:
    """
    Get traffic light status based on confidence score.
    
    Returns:
        Tuple of (level, color, emoji, label)
    """
    if confidence_score >= 80:
        return QualityLevel.HIGH, "#10b981", "🟢", "Sehr zuverlässig"
    elif confidence_score >= 60:
        return QualityLevel.MEDIUM, "#f59e0b", "🟡", "Mäßig zuverlässig"
    else:
        return QualityLevel.LOW, "#ef4444", "🔴", "Mit Vorsicht verwenden"


def generate_plain_explanation(
    confidence_score: float,
    level: QualityLevel,
    training_metrics: Dict[str, Dict[str, float]],
    forecast_horizon_days: int = 7
) -> Tuple[str, List[str]]:
    """
    Generate plain-language explanation for the forecast quality (German).
    
    Returns:
        Tuple of (main_explanation, detail_points)
    """
    details = []
    
    # Main explanation based on level
    if level == QualityLevel.HIGH:
        main = "Diese Prognose basiert auf starken, konsistenten Mustern in Ihren historischen Daten."
        details.append("Das Modell hat Ihre wöchentlichen und täglichen Muster präzise erfasst")
        details.append("Historische Vorhersagen stimmten sehr gut mit den tatsächlichen Werten überein")
    elif level == QualityLevel.MEDIUM:
        main = "Diese Prognose zeigt akzeptable Genauigkeit, hat aber gewisse Unsicherheit."
        details.append("Empfehlung: 10-15% Pufferkapazität für unerwartete Spitzen einplanen")
        details.append("Das Modell hat Muster erkannt, jedoch mit gewisser Variabilität")
    else:
        main = "Diese Prognose sollte nur als grobe Orientierung verwendet werden."
        details.append("Begrenzte historische Daten oder ungewöhnliche Muster erkannt")
        details.append("Empfehlung: Manuelle Überprüfung und Anpassung der Personalplanung")
    
    # Add horizon-specific note
    if forecast_horizon_days > 14:
        details.append(f"Prognose über {forecast_horizon_days} Tage erhöht die Unsicherheit")
    
    # Add metric-specific insights
    if training_metrics:
        avg_mape = np.mean([m.get('mape', 0) for m in training_metrics.values()])
        if avg_mape < 10:
            details.append("Vorhersagen liegen typischerweise innerhalb von 10% der tatsächlichen Werte")
        elif avg_mape < 20:
            details.append("Vorhersagen liegen typischerweise innerhalb von 20% der tatsächlichen Werte")
        else:
            details.append(f"Vorhersagen können um bis zu {avg_mape:.0f}% von den tatsächlichen Werten abweichen")
    
    return main, details


def get_quality_indicator(
    training_metrics: Dict[str, Dict[str, float]],
    forecast_horizon_days: int = 7,
    data_quality_score: float = 100.0
) -> QualityIndicator:
    """
    Get complete quality indicator for display.
    
    Args:
        training_metrics: Model training metrics
        forecast_horizon_days: How far ahead we're forecasting
        data_quality_score: Data completeness score (0-100)
        
    Returns:
        QualityIndicator with all display information
    """
    # Calculate confidence
    confidence = calculate_confidence_score(
        training_metrics, 
        data_quality_score, 
        forecast_horizon_days
    )
    
    # Get traffic light
    level, color, emoji, label = get_traffic_light_status(confidence)
    
    # Get explanation
    explanation, details = generate_plain_explanation(
        confidence, level, training_metrics, forecast_horizon_days
    )
    
    return QualityIndicator(
        confidence_score=confidence,
        level=level,
        color=color,
        emoji=emoji,
        label=label,
        explanation=explanation,
        details=details
    )


def compare_forecast_to_actuals(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    forecast_id: str = "unknown"
) -> ForecastComparison:
    """
    Compare forecast predictions to actual values.
    
    Args:
        forecast_df: DataFrame with timestamp and predicted values
        actuals_df: DataFrame with timestamp and actual values
        forecast_id: Identifier for this forecast
        
    Returns:
        ForecastComparison with metrics and insights
    """
    # Ensure timestamp columns are datetime
    forecast_df = forecast_df.copy()
    actuals_df = actuals_df.copy()
    
    if 'timestamp' in forecast_df.columns:
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    if 'timestamp' in actuals_df.columns:
        actuals_df['timestamp'] = pd.to_datetime(actuals_df['timestamp'])
    
    # Merge on timestamp
    merged = pd.merge(
        forecast_df, 
        actuals_df, 
        on='timestamp', 
        suffixes=('_pred', '_actual'),
        how='inner'
    )
    
    if merged.empty:
        return ForecastComparison(
            forecast_id=forecast_id,
            period_start=forecast_df['timestamp'].min(),
            period_end=forecast_df['timestamp'].max(),
            overall_accuracy=0,
            overall_mape=100,
            target_metrics={},
            summary="Vergleich nicht möglich: Keine übereinstimmenden Zeitstempel zwischen Prognose und Ist-Daten.",
            highlights=[],
            areas_for_improvement=["Stellen Sie sicher, dass die Ist-Daten den gleichen Zeitraum wie die Prognose abdecken"],
            comparison_df=None
        )
    
    # Find target columns (those that appear in both with _pred and _actual)
    pred_cols = [c.replace('_pred', '') for c in merged.columns if c.endswith('_pred')]
    actual_cols = [c.replace('_actual', '') for c in merged.columns if c.endswith('_actual')]
    target_cols = list(set(pred_cols) & set(actual_cols))
    
    if not target_cols:
        # Try to match column names more flexibly
        target_cols = [c for c in forecast_df.columns if c != 'timestamp' and c in actuals_df.columns]
    
    # Calculate metrics per target
    target_metrics = {}
    all_mapes = []
    all_accuracies = []
    
    for target in target_cols:
        pred_col = f"{target}_pred" if f"{target}_pred" in merged.columns else target
        actual_col = f"{target}_actual" if f"{target}_actual" in merged.columns else target
        
        if pred_col not in merged.columns or actual_col not in merged.columns:
            continue
            
        y_pred = merged[pred_col].values
        y_actual = merged[actual_col].values
        
        # Avoid division by zero
        mask = y_actual != 0
        if mask.sum() == 0:
            continue
            
        # MAPE
        mape = np.mean(np.abs((y_actual[mask] - y_pred[mask]) / y_actual[mask])) * 100
        
        # Accuracy (100 - MAPE, capped at 0)
        accuracy = max(0, 100 - mape)
        
        # MAE
        mae = np.mean(np.abs(y_actual - y_pred))
        
        # RMSE
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        
        # Correlation
        corr = np.corrcoef(y_pred, y_actual)[0, 1] if len(y_pred) > 1 else 0
        
        target_metrics[target] = {
            'mape': round(mape, 1),
            'accuracy': round(accuracy, 1),
            'mae': round(mae, 1),
            'rmse': round(rmse, 1),
            'correlation': round(corr, 3),
            'total_predicted': round(y_pred.sum(), 0),
            'total_actual': round(y_actual.sum(), 0)
        }
        
        all_mapes.append(mape)
        all_accuracies.append(accuracy)
    
    # Overall metrics
    overall_mape = np.mean(all_mapes) if all_mapes else 100
    overall_accuracy = np.mean(all_accuracies) if all_accuracies else 0
    
    # Generate plain language summary
    summary, highlights, improvements = _generate_comparison_summary(
        overall_accuracy, overall_mape, target_metrics
    )
    
    # Build comparison dataframe
    comparison_df = merged[['timestamp'] + [c for c in merged.columns if c != 'timestamp']]
    
    return ForecastComparison(
        forecast_id=forecast_id,
        period_start=merged['timestamp'].min(),
        period_end=merged['timestamp'].max(),
        overall_accuracy=round(overall_accuracy, 1),
        overall_mape=round(overall_mape, 1),
        target_metrics=target_metrics,
        summary=summary,
        highlights=highlights,
        areas_for_improvement=improvements,
        comparison_df=comparison_df
    )


def _generate_comparison_summary(
    accuracy: float,
    mape: float,
    target_metrics: Dict[str, Dict[str, float]]
) -> Tuple[str, List[str], List[str]]:
    """Generate plain-language summary of forecast vs actuals comparison (German)."""
    
    highlights = []
    improvements = []
    
    # Main summary
    if accuracy >= 90:
        summary = "Ausgezeichnet! Die Prognose war sehr genau."
    elif accuracy >= 80:
        summary = "Gute Leistung. Die Prognose stimmte weitgehend mit den tatsächlichen Werten überein."
    elif accuracy >= 70:
        summary = "Akzeptable Genauigkeit. Es gab einige Abweichungen zwischen Prognose und Ist-Werten."
    elif accuracy >= 60:
        summary = "Mäßige Genauigkeit. Zukünftige Prognosen sollten angepasst werden."
    else:
        summary = "Erhebliche Abweichungen festgestellt. Bitte die Details unten prüfen."
    
    # Per-target highlights
    for target, metrics in target_metrics.items():
        target_name = _translate_target_name(target)
        target_acc = metrics.get('accuracy', 0)
        
        if target_acc >= 90:
            highlights.append(f"✓ {target_name}: {target_acc:.0f}% genau")
        elif target_acc < 70:
            diff = metrics['total_actual'] - metrics['total_predicted']
            if diff > 0:
                improvements.append(f"{target_name}: Um {abs(diff):.0f} zu niedrig prognostiziert")
            else:
                improvements.append(f"{target_name}: Um {abs(diff):.0f} zu hoch prognostiziert")
    
    # Add general insights
    if accuracy >= 80:
        highlights.append("Das Modell hat Stoß- und Ruhezeiten korrekt erkannt")
    
    if not improvements:
        improvements.append("Keine größeren Probleme erkannt – Ansatz beibehalten")
    
    return summary, highlights, improvements


def _translate_target_name(target: str) -> str:
    """Translate target column names to German display names."""
    translations = {
        "call_volume": "Anrufvolumen",
        "email_count": "E-Mail-Volumen",
        "outbound_ook": "Outbound OOK",
        "outbound_omk": "Outbound OMK",
        "outbound_nb": "Outbound NB",
        "calls": "Anrufe",
        "emails": "E-Mails",
        "outbound": "Outbound"
    }
    return translations.get(target, target.replace('_', ' ').title())


def calculate_peak_detection_accuracy(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    threshold_percentile: float = 75
) -> Dict[str, float]:
    """
    Calculate how well the forecast identified peak hours.
    
    Returns:
        Dict with peak detection metrics
    """
    # Merge data
    merged = pd.merge(
        forecast_df, actuals_df,
        on='timestamp', suffixes=('_pred', '_actual'),
        how='inner'
    )
    
    if merged.empty:
        return {'detection_rate': 0, 'false_positive_rate': 0}
    
    # Find numeric columns
    pred_cols = [c for c in merged.columns if c.endswith('_pred')]
    actual_cols = [c for c in merged.columns if c.endswith('_actual')]
    
    if not pred_cols or not actual_cols:
        return {'detection_rate': 0, 'false_positive_rate': 0}
    
    # Use first target for peak detection
    pred_col = pred_cols[0]
    actual_col = actual_cols[0]
    
    # Define peaks based on threshold
    pred_threshold = merged[pred_col].quantile(threshold_percentile / 100)
    actual_threshold = merged[actual_col].quantile(threshold_percentile / 100)
    
    pred_peaks = merged[pred_col] >= pred_threshold
    actual_peaks = merged[actual_col] >= actual_threshold
    
    # True positives: predicted peak and was actually peak
    true_positives = (pred_peaks & actual_peaks).sum()
    
    # False positives: predicted peak but wasn't
    false_positives = (pred_peaks & ~actual_peaks).sum()
    
    # False negatives: didn't predict but was peak
    false_negatives = (~pred_peaks & actual_peaks).sum()
    
    # Detection rate (recall)
    actual_peak_count = actual_peaks.sum()
    detection_rate = (true_positives / actual_peak_count * 100) if actual_peak_count > 0 else 0
    
    # False positive rate
    pred_peak_count = pred_peaks.sum()
    false_positive_rate = (false_positives / pred_peak_count * 100) if pred_peak_count > 0 else 0
    
    return {
        'detection_rate': round(detection_rate, 1),
        'false_positive_rate': round(false_positive_rate, 1),
        'peaks_detected': int(true_positives),
        'peaks_missed': int(false_negatives),
        'false_alarms': int(false_positives)
    }

