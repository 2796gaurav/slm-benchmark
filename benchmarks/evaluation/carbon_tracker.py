import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Make codecarbon import optional to avoid dependency conflicts
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    EmissionsTracker = None  # type: ignore

class CarbonTrackerWrapper:
    """
    Wrapper around CodeCarbon EmissionsTracker
    """
    def __init__(self, output_dir: str, enable: bool = True):
        self.enabled = enable
        self.output_dir = output_dir
        self.tracker: Optional[Any] = None
        
        if self.enabled:
            if not CODECARBON_AVAILABLE:
                logger.warning("codecarbon is not installed. Carbon tracking will be disabled. Install with: pip install codecarbon")
                self.enabled = False
                return
                
            try:
                # Create output directory for carbon logs
                os.makedirs(output_dir, exist_ok=True)
                
                self.tracker = EmissionsTracker(
                    project_name="slm-benchmark",
                    measure_power_secs=15,
                    output_dir=output_dir,
                    save_to_file=True,
                )
                logger.info("Carbon tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize carbon tracking: {e}")
                self.tracker = None
                self.enabled = False

    def start(self):
        if self.enabled and self.tracker:
            try:
                self.tracker.start()
                logger.info("Carbon tracking started")
            except Exception as e:
                logger.error(f"Error starting carbon tracker: {e}")

    def stop(self) -> Dict[str, float]:
        """
        Stop tracking and return metrics
        """
        if not self.enabled or not self.tracker:
            return {}
            
        try:
            emissions = self.tracker.stop()
            # emissions is the total CO2 in kg
            
            # codecarbon saves details to emissions.csv, but `stop()` returns the co2 float
            # We can also access other metrics via the tracker's internal state or flush/final data
            
            # For simplicity, we return the core metrics if available from the object
            # Note: emissions is a float (kg CO2)
            
            data = {
                'co2_emissions_kg': float(emissions) if emissions is not None else 0.0,
                'energy_consumed_kwh': self.tracker.final_emissions_data.energy_consumed if hasattr(self.tracker, 'final_emissions_data') else 0.0,
                'duration_seconds': self.tracker.final_emissions_data.duration if hasattr(self.tracker, 'final_emissions_data') else 0.0,
                'cpu_power_w': self.tracker.final_emissions_data.cpu_power if hasattr(self.tracker, 'final_emissions_data') else 0.0,
                'gpu_power_w': self.tracker.final_emissions_data.gpu_power if hasattr(self.tracker, 'final_emissions_data') else 0.0,
                'ram_power_w': self.tracker.final_emissions_data.ram_power if hasattr(self.tracker, 'final_emissions_data') else 0.0,
            }
            
            logger.info(f"Carbon metrics: {data}")
            return data
            
        except Exception as e:
            logger.error(f"Error stopping carbon tracker: {e}")
            return {}
