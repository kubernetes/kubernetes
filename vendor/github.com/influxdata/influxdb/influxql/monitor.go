package influxql

import "time"

// PointLimitMonitor is a query monitor that exits when the number of points
// emitted exceeds a threshold.
func PointLimitMonitor(itrs Iterators, interval time.Duration, limit int) QueryMonitorFunc {
	return func(closing <-chan struct{}) error {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				stats := itrs.Stats()
				if stats.PointN >= limit {
					return ErrMaxSelectPointsLimitExceeded(stats.PointN, limit)
				}
			case <-closing:
				return nil
			}
		}
	}
}
