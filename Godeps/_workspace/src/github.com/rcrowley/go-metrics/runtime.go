package metrics

import (
	"runtime"
	"time"
)

var (
	memStats       runtime.MemStats
	runtimeMetrics struct {
		MemStats struct {
			Alloc        Gauge
			BuckHashSys  Gauge
			DebugGC      Gauge
			EnableGC     Gauge
			Frees        Gauge
			HeapAlloc    Gauge
			HeapIdle     Gauge
			HeapInuse    Gauge
			HeapObjects  Gauge
			HeapReleased Gauge
			HeapSys      Gauge
			LastGC       Gauge
			Lookups      Gauge
			Mallocs      Gauge
			MCacheInuse  Gauge
			MCacheSys    Gauge
			MSpanInuse   Gauge
			MSpanSys     Gauge
			NextGC       Gauge
			NumGC        Gauge
			PauseNs      Histogram
			PauseTotalNs Gauge
			StackInuse   Gauge
			StackSys     Gauge
			Sys          Gauge
			TotalAlloc   Gauge
		}
		NumCgoCall   Gauge
		NumGoroutine Gauge
		ReadMemStats Timer
	}
	frees       uint64
	lookups     uint64
	mallocs     uint64
	numGC       uint32
	numCgoCalls int64
)

// Capture new values for the Go runtime statistics exported in
// runtime.MemStats.  This is designed to be called as a goroutine.
func CaptureRuntimeMemStats(r Registry, d time.Duration) {
	for _ = range time.Tick(d) {
		CaptureRuntimeMemStatsOnce(r)
	}
}

// Capture new values for the Go runtime statistics exported in
// runtime.MemStats.  This is designed to be called in a background
// goroutine.  Giving a registry which has not been given to
// RegisterRuntimeMemStats will panic.
//
// Be very careful with this because runtime.ReadMemStats calls the C
// functions runtime·semacquire(&runtime·worldsema) and runtime·stoptheworld()
// and that last one does what it says on the tin.
func CaptureRuntimeMemStatsOnce(r Registry) {
	t := time.Now()
	runtime.ReadMemStats(&memStats) // This takes 50-200us.
	runtimeMetrics.ReadMemStats.UpdateSince(t)

	runtimeMetrics.MemStats.Alloc.Update(int64(memStats.Alloc))
	runtimeMetrics.MemStats.BuckHashSys.Update(int64(memStats.BuckHashSys))
	if memStats.DebugGC {
		runtimeMetrics.MemStats.DebugGC.Update(1)
	} else {
		runtimeMetrics.MemStats.DebugGC.Update(0)
	}
	if memStats.EnableGC {
		runtimeMetrics.MemStats.EnableGC.Update(1)
	} else {
		runtimeMetrics.MemStats.EnableGC.Update(0)
	}

	runtimeMetrics.MemStats.Frees.Update(int64(memStats.Frees - frees))
	runtimeMetrics.MemStats.HeapAlloc.Update(int64(memStats.HeapAlloc))
	runtimeMetrics.MemStats.HeapIdle.Update(int64(memStats.HeapIdle))
	runtimeMetrics.MemStats.HeapInuse.Update(int64(memStats.HeapInuse))
	runtimeMetrics.MemStats.HeapObjects.Update(int64(memStats.HeapObjects))
	runtimeMetrics.MemStats.HeapReleased.Update(int64(memStats.HeapReleased))
	runtimeMetrics.MemStats.HeapSys.Update(int64(memStats.HeapSys))
	runtimeMetrics.MemStats.LastGC.Update(int64(memStats.LastGC))
	runtimeMetrics.MemStats.Lookups.Update(int64(memStats.Lookups - lookups))
	runtimeMetrics.MemStats.Mallocs.Update(int64(memStats.Mallocs - mallocs))
	runtimeMetrics.MemStats.MCacheInuse.Update(int64(memStats.MCacheInuse))
	runtimeMetrics.MemStats.MCacheSys.Update(int64(memStats.MCacheSys))
	runtimeMetrics.MemStats.MSpanInuse.Update(int64(memStats.MSpanInuse))
	runtimeMetrics.MemStats.MSpanSys.Update(int64(memStats.MSpanSys))
	runtimeMetrics.MemStats.NextGC.Update(int64(memStats.NextGC))
	runtimeMetrics.MemStats.NumGC.Update(int64(memStats.NumGC - numGC))

	// <https://code.google.com/p/go/source/browse/src/pkg/runtime/mgc0.c>
	i := numGC % uint32(len(memStats.PauseNs))
	ii := memStats.NumGC % uint32(len(memStats.PauseNs))
	if memStats.NumGC-numGC >= uint32(len(memStats.PauseNs)) {
		for i = 0; i < uint32(len(memStats.PauseNs)); i++ {
			runtimeMetrics.MemStats.PauseNs.Update(int64(memStats.PauseNs[i]))
		}
	} else {
		if i > ii {
			for ; i < uint32(len(memStats.PauseNs)); i++ {
				runtimeMetrics.MemStats.PauseNs.Update(int64(memStats.PauseNs[i]))
			}
			i = 0
		}
		for ; i < ii; i++ {
			runtimeMetrics.MemStats.PauseNs.Update(int64(memStats.PauseNs[i]))
		}
	}
	frees = memStats.Frees
	lookups = memStats.Lookups
	mallocs = memStats.Mallocs
	numGC = memStats.NumGC

	runtimeMetrics.MemStats.PauseTotalNs.Update(int64(memStats.PauseTotalNs))
	runtimeMetrics.MemStats.StackInuse.Update(int64(memStats.StackInuse))
	runtimeMetrics.MemStats.StackSys.Update(int64(memStats.StackSys))
	runtimeMetrics.MemStats.Sys.Update(int64(memStats.Sys))
	runtimeMetrics.MemStats.TotalAlloc.Update(int64(memStats.TotalAlloc))

	currentNumCgoCalls := numCgoCall()
	runtimeMetrics.NumCgoCall.Update(currentNumCgoCalls - numCgoCalls)
	numCgoCalls = currentNumCgoCalls

	runtimeMetrics.NumGoroutine.Update(int64(runtime.NumGoroutine()))
}

// Register runtimeMetrics for the Go runtime statistics exported in runtime and
// specifically runtime.MemStats.  The runtimeMetrics are named by their
// fully-qualified Go symbols, i.e. runtime.MemStats.Alloc.
func RegisterRuntimeMemStats(r Registry) {
	runtimeMetrics.MemStats.Alloc = NewGauge()
	runtimeMetrics.MemStats.BuckHashSys = NewGauge()
	runtimeMetrics.MemStats.DebugGC = NewGauge()
	runtimeMetrics.MemStats.EnableGC = NewGauge()
	runtimeMetrics.MemStats.Frees = NewGauge()
	runtimeMetrics.MemStats.HeapAlloc = NewGauge()
	runtimeMetrics.MemStats.HeapIdle = NewGauge()
	runtimeMetrics.MemStats.HeapInuse = NewGauge()
	runtimeMetrics.MemStats.HeapObjects = NewGauge()
	runtimeMetrics.MemStats.HeapReleased = NewGauge()
	runtimeMetrics.MemStats.HeapSys = NewGauge()
	runtimeMetrics.MemStats.LastGC = NewGauge()
	runtimeMetrics.MemStats.Lookups = NewGauge()
	runtimeMetrics.MemStats.Mallocs = NewGauge()
	runtimeMetrics.MemStats.MCacheInuse = NewGauge()
	runtimeMetrics.MemStats.MCacheSys = NewGauge()
	runtimeMetrics.MemStats.MSpanInuse = NewGauge()
	runtimeMetrics.MemStats.MSpanSys = NewGauge()
	runtimeMetrics.MemStats.NextGC = NewGauge()
	runtimeMetrics.MemStats.NumGC = NewGauge()
	runtimeMetrics.MemStats.PauseNs = NewHistogram(NewExpDecaySample(1028, 0.015))
	runtimeMetrics.MemStats.PauseTotalNs = NewGauge()
	runtimeMetrics.MemStats.StackInuse = NewGauge()
	runtimeMetrics.MemStats.StackSys = NewGauge()
	runtimeMetrics.MemStats.Sys = NewGauge()
	runtimeMetrics.MemStats.TotalAlloc = NewGauge()
	runtimeMetrics.NumCgoCall = NewGauge()
	runtimeMetrics.NumGoroutine = NewGauge()
	runtimeMetrics.ReadMemStats = NewTimer()

	r.Register("runtime.MemStats.Alloc", runtimeMetrics.MemStats.Alloc)
	r.Register("runtime.MemStats.BuckHashSys", runtimeMetrics.MemStats.BuckHashSys)
	r.Register("runtime.MemStats.DebugGC", runtimeMetrics.MemStats.DebugGC)
	r.Register("runtime.MemStats.EnableGC", runtimeMetrics.MemStats.EnableGC)
	r.Register("runtime.MemStats.Frees", runtimeMetrics.MemStats.Frees)
	r.Register("runtime.MemStats.HeapAlloc", runtimeMetrics.MemStats.HeapAlloc)
	r.Register("runtime.MemStats.HeapIdle", runtimeMetrics.MemStats.HeapIdle)
	r.Register("runtime.MemStats.HeapInuse", runtimeMetrics.MemStats.HeapInuse)
	r.Register("runtime.MemStats.HeapObjects", runtimeMetrics.MemStats.HeapObjects)
	r.Register("runtime.MemStats.HeapReleased", runtimeMetrics.MemStats.HeapReleased)
	r.Register("runtime.MemStats.HeapSys", runtimeMetrics.MemStats.HeapSys)
	r.Register("runtime.MemStats.LastGC", runtimeMetrics.MemStats.LastGC)
	r.Register("runtime.MemStats.Lookups", runtimeMetrics.MemStats.Lookups)
	r.Register("runtime.MemStats.Mallocs", runtimeMetrics.MemStats.Mallocs)
	r.Register("runtime.MemStats.MCacheInuse", runtimeMetrics.MemStats.MCacheInuse)
	r.Register("runtime.MemStats.MCacheSys", runtimeMetrics.MemStats.MCacheSys)
	r.Register("runtime.MemStats.MSpanInuse", runtimeMetrics.MemStats.MSpanInuse)
	r.Register("runtime.MemStats.MSpanSys", runtimeMetrics.MemStats.MSpanSys)
	r.Register("runtime.MemStats.NextGC", runtimeMetrics.MemStats.NextGC)
	r.Register("runtime.MemStats.NumGC", runtimeMetrics.MemStats.NumGC)
	r.Register("runtime.MemStats.PauseNs", runtimeMetrics.MemStats.PauseNs)
	r.Register("runtime.MemStats.PauseTotalNs", runtimeMetrics.MemStats.PauseTotalNs)
	r.Register("runtime.MemStats.StackInuse", runtimeMetrics.MemStats.StackInuse)
	r.Register("runtime.MemStats.StackSys", runtimeMetrics.MemStats.StackSys)
	r.Register("runtime.MemStats.Sys", runtimeMetrics.MemStats.Sys)
	r.Register("runtime.MemStats.TotalAlloc", runtimeMetrics.MemStats.TotalAlloc)
	r.Register("runtime.NumCgoCall", runtimeMetrics.NumCgoCall)
	r.Register("runtime.NumGoroutine", runtimeMetrics.NumGoroutine)
	r.Register("runtime.ReadMemStats", runtimeMetrics.ReadMemStats)
}
