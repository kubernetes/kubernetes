// +build windows

package stats

// platformNewStatsCollector performs platform specific initialisation of the
// Collector structure. This is a no-op on Windows.
func platformNewStatsCollector(s *Collector) {
}

// getSystemCPUUsage returns the host system's cpu usage in
// nanoseconds. An error is returned if the format of the underlying
// file does not match. This is a no-op on Windows.
func (s *Collector) getSystemCPUUsage() (uint64, error) {
	return 0, nil
}

func (s *Collector) getNumberOnlineCPUs() (uint32, error) {
	return 0, nil
}
