package daemon

import (
	"runtime"
	"time"

	"github.com/docker/docker/daemon/stats"
	"github.com/docker/docker/pkg/system"
)

// newStatsCollector returns a new statsCollector that collections
// stats for a registered container at the specified interval.
// The collector allows non-running containers to be added
// and will start processing stats when they are started.
func (daemon *Daemon) newStatsCollector(interval time.Duration) *stats.Collector {
	// FIXME(vdemeester) move this elsewhere
	if runtime.GOOS == "linux" {
		meminfo, err := system.ReadMemInfo()
		if err == nil && meminfo.MemTotal > 0 {
			daemon.machineMemory = uint64(meminfo.MemTotal)
		}
	}
	s := stats.NewCollector(daemon, interval)
	go s.Run()
	return s
}
