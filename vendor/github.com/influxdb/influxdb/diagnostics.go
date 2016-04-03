package influxdb

import (
	"os"
	"runtime"
	"time"

	"github.com/influxdb/influxdb/influxql"
)

// GoDiagnostics captures basic information about the runtime.
type GoDiagnostics struct {
	GoMaxProcs   int
	NumGoroutine int
	Version      string
}

// NewGoDiagnostics returns a GoDiagnostics object.
func NewGoDiagnostics() *GoDiagnostics {
	return &GoDiagnostics{
		GoMaxProcs:   runtime.GOMAXPROCS(0),
		NumGoroutine: runtime.NumGoroutine(),
		Version:      runtime.Version(),
	}
}

// AsRow returns the GoDiagnostic object as an InfluxQL row.
func (g *GoDiagnostics) AsRow(measurement string, tags map[string]string) *influxql.Row {
	return &influxql.Row{
		Name:    measurement,
		Columns: []string{"time", "goMaxProcs", "numGoRoutine", "version"},
		Tags:    tags,
		Values: [][]interface{}{[]interface{}{time.Now().UTC(),
			g.GoMaxProcs, g.NumGoroutine, g.Version}},
	}
}

// SystemDiagnostics captures basic machine data.
type SystemDiagnostics struct {
	Hostname string
	PID      int
	OS       string
	Arch     string
	NumCPU   int
}

// NewSystemDiagnostics returns a SystemDiagnostics object.
func NewSystemDiagnostics() *SystemDiagnostics {
	hostname, err := os.Hostname()
	if err != nil {
		hostname = "unknown"
	}

	return &SystemDiagnostics{
		Hostname: hostname,
		PID:      os.Getpid(),
		OS:       runtime.GOOS,
		Arch:     runtime.GOARCH,
		NumCPU:   runtime.NumCPU(),
	}
}

// AsRow returns the GoDiagnostic object as an InfluxQL row.
func (s *SystemDiagnostics) AsRow(measurement string, tags map[string]string) *influxql.Row {
	return &influxql.Row{
		Name:    measurement,
		Columns: []string{"time", "hostname", "pid", "os", "arch", "numCPU"},
		Tags:    tags,
		Values: [][]interface{}{[]interface{}{time.Now().UTC(),
			s.Hostname, s.PID, s.OS, s.Arch, s.NumCPU}},
	}
}

// MemoryDiagnostics captures Go memory stats.
type MemoryDiagnostics struct {
	Alloc        int64
	TotalAlloc   int64
	Sys          int64
	Lookups      int64
	Mallocs      int64
	Frees        int64
	HeapAlloc    int64
	HeapSys      int64
	HeapIdle     int64
	HeapInUse    int64
	HeapReleased int64
	HeapObjects  int64
	PauseTotalNs int64
	NumGC        int64
}

// NewMemoryDiagnostics returns a MemoryDiagnostics object.
func NewMemoryDiagnostics() *MemoryDiagnostics {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return &MemoryDiagnostics{
		Alloc:        int64(m.Alloc),
		TotalAlloc:   int64(m.TotalAlloc),
		Sys:          int64(m.Sys),
		Lookups:      int64(m.Lookups),
		Mallocs:      int64(m.Mallocs),
		Frees:        int64(m.Frees),
		HeapAlloc:    int64(m.HeapAlloc),
		HeapSys:      int64(m.HeapSys),
		HeapIdle:     int64(m.HeapIdle),
		HeapInUse:    int64(m.HeapInuse),
		HeapReleased: int64(m.HeapReleased),
		HeapObjects:  int64(m.HeapObjects),
		PauseTotalNs: int64(m.PauseTotalNs),
		NumGC:        int64(m.NumGC),
	}
}

// AsRow returns the MemoryDiagnostics object as an InfluxQL row.
func (m *MemoryDiagnostics) AsRow(measurement string, tags map[string]string) *influxql.Row {
	return &influxql.Row{
		Name: measurement,
		Columns: []string{"time", "alloc", "totalAlloc", "sys", "lookups", "mallocs", "frees", "heapAlloc",
			"heapSys", "heapIdle", "heapInUse", "heapReleased", "heapObjects", "pauseTotalNs", "numGG"},
		Tags: tags,
		Values: [][]interface{}{[]interface{}{time.Now().UTC(),
			m.Alloc, m.TotalAlloc, m.Sys, m.Lookups, m.Mallocs, m.Frees, m.HeapAlloc,
			m.HeapSys, m.HeapIdle, m.HeapInUse, m.HeapReleased, m.HeapObjects, m.PauseTotalNs, m.NumGC}},
	}
}

// BuildDiagnostics capture basic build version information.
type BuildDiagnostics struct {
	Version    string
	CommitHash string
}

// AsRow returns the BuildDiagnostics object as an InfluxQL row.
func (b *BuildDiagnostics) AsRow(measurement string, tags map[string]string) *influxql.Row {
	return &influxql.Row{
		Name:    measurement,
		Columns: []string{"time", "version", "commitHash"},
		Tags:    tags,
		Values: [][]interface{}{[]interface{}{time.Now().UTC(),
			b.Version, b.CommitHash}},
	}
}
