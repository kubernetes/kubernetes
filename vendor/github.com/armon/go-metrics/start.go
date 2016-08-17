package metrics

import (
	"os"
	"time"
)

// Config is used to configure metrics settings
type Config struct {
	ServiceName          string        // Prefixed with keys to seperate services
	HostName             string        // Hostname to use. If not provided and EnableHostname, it will be os.Hostname
	EnableHostname       bool          // Enable prefixing gauge values with hostname
	EnableRuntimeMetrics bool          // Enables profiling of runtime metrics (GC, Goroutines, Memory)
	EnableTypePrefix     bool          // Prefixes key with a type ("counter", "gauge", "timer")
	TimerGranularity     time.Duration // Granularity of timers.
	ProfileInterval      time.Duration // Interval to profile runtime metrics
}

// Metrics represents an instance of a metrics sink that can
// be used to emit
type Metrics struct {
	Config
	lastNumGC uint32
	sink      MetricSink
}

// Shared global metrics instance
var globalMetrics *Metrics

func init() {
	// Initialize to a blackhole sink to avoid errors
	globalMetrics = &Metrics{sink: &BlackholeSink{}}
}

// DefaultConfig provides a sane default configuration
func DefaultConfig(serviceName string) *Config {
	c := &Config{
		ServiceName:          serviceName, // Use client provided service
		HostName:             "",
		EnableHostname:       true,             // Enable hostname prefix
		EnableRuntimeMetrics: true,             // Enable runtime profiling
		EnableTypePrefix:     false,            // Disable type prefix
		TimerGranularity:     time.Millisecond, // Timers are in milliseconds
		ProfileInterval:      time.Second,      // Poll runtime every second
	}

	// Try to get the hostname
	name, _ := os.Hostname()
	c.HostName = name
	return c
}

// New is used to create a new instance of Metrics
func New(conf *Config, sink MetricSink) (*Metrics, error) {
	met := &Metrics{}
	met.Config = *conf
	met.sink = sink

	// Start the runtime collector
	if conf.EnableRuntimeMetrics {
		go met.collectStats()
	}
	return met, nil
}

// NewGlobal is the same as New, but it assigns the metrics object to be
// used globally as well as returning it.
func NewGlobal(conf *Config, sink MetricSink) (*Metrics, error) {
	metrics, err := New(conf, sink)
	if err == nil {
		globalMetrics = metrics
	}
	return metrics, err
}

// Proxy all the methods to the globalMetrics instance
func SetGauge(key []string, val float32) {
	globalMetrics.SetGauge(key, val)
}

func EmitKey(key []string, val float32) {
	globalMetrics.EmitKey(key, val)
}

func IncrCounter(key []string, val float32) {
	globalMetrics.IncrCounter(key, val)
}

func AddSample(key []string, val float32) {
	globalMetrics.AddSample(key, val)
}

func MeasureSince(key []string, start time.Time) {
	globalMetrics.MeasureSince(key, start)
}
