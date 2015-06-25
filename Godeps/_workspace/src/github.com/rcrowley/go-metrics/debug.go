package metrics

import (
	"runtime/debug"
	"time"
)

var (
	debugMetrics struct {
		GCStats struct {
			LastGC Gauge
			NumGC  Gauge
			Pause  Histogram
			//PauseQuantiles Histogram
			PauseTotal Gauge
		}
		ReadGCStats Timer
	}
	gcStats debug.GCStats
)

// Capture new values for the Go garbage collector statistics exported in
// debug.GCStats.  This is designed to be called as a goroutine.
func CaptureDebugGCStats(r Registry, d time.Duration) {
	for _ = range time.Tick(d) {
		CaptureDebugGCStatsOnce(r)
	}
}

// Capture new values for the Go garbage collector statistics exported in
// debug.GCStats.  This is designed to be called in a background goroutine.
// Giving a registry which has not been given to RegisterDebugGCStats will
// panic.
//
// Be careful (but much less so) with this because debug.ReadGCStats calls
// the C function runtime·lock(runtime·mheap) which, while not a stop-the-world
// operation, isn't something you want to be doing all the time.
func CaptureDebugGCStatsOnce(r Registry) {
	lastGC := gcStats.LastGC
	t := time.Now()
	debug.ReadGCStats(&gcStats)
	debugMetrics.ReadGCStats.UpdateSince(t)

	debugMetrics.GCStats.LastGC.Update(int64(gcStats.LastGC.UnixNano()))
	debugMetrics.GCStats.NumGC.Update(int64(gcStats.NumGC))
	if lastGC != gcStats.LastGC && 0 < len(gcStats.Pause) {
		debugMetrics.GCStats.Pause.Update(int64(gcStats.Pause[0]))
	}
	//debugMetrics.GCStats.PauseQuantiles.Update(gcStats.PauseQuantiles)
	debugMetrics.GCStats.PauseTotal.Update(int64(gcStats.PauseTotal))
}

// Register metrics for the Go garbage collector statistics exported in
// debug.GCStats.  The metrics are named by their fully-qualified Go symbols,
// i.e. debug.GCStats.PauseTotal.
func RegisterDebugGCStats(r Registry) {
	debugMetrics.GCStats.LastGC = NewGauge()
	debugMetrics.GCStats.NumGC = NewGauge()
	debugMetrics.GCStats.Pause = NewHistogram(NewExpDecaySample(1028, 0.015))
	//debugMetrics.GCStats.PauseQuantiles = NewHistogram(NewExpDecaySample(1028, 0.015))
	debugMetrics.GCStats.PauseTotal = NewGauge()
	debugMetrics.ReadGCStats = NewTimer()

	r.Register("debug.GCStats.LastGC", debugMetrics.GCStats.LastGC)
	r.Register("debug.GCStats.NumGC", debugMetrics.GCStats.NumGC)
	r.Register("debug.GCStats.Pause", debugMetrics.GCStats.Pause)
	//r.Register("debug.GCStats.PauseQuantiles", debugMetrics.GCStats.PauseQuantiles)
	r.Register("debug.GCStats.PauseTotal", debugMetrics.GCStats.PauseTotal)
	r.Register("debug.ReadGCStats", debugMetrics.ReadGCStats)
}

// Allocate an initial slice for gcStats.Pause to avoid allocations during
// normal operation.
func init() {
	gcStats.Pause = make([]time.Duration, 11)
}
