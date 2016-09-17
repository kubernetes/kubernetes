package metrics

// The MetricSink interface is used to transmit metrics information
// to an external system
type MetricSink interface {
	// A Gauge should retain the last value it is set to
	SetGauge(key []string, val float32)

	// Should emit a Key/Value pair for each call
	EmitKey(key []string, val float32)

	// Counters should accumulate values
	IncrCounter(key []string, val float32)

	// Samples are for timing information, where quantiles are used
	AddSample(key []string, val float32)
}

// BlackholeSink is used to just blackhole messages
type BlackholeSink struct{}

func (*BlackholeSink) SetGauge(key []string, val float32)    {}
func (*BlackholeSink) EmitKey(key []string, val float32)     {}
func (*BlackholeSink) IncrCounter(key []string, val float32) {}
func (*BlackholeSink) AddSample(key []string, val float32)   {}

// FanoutSink is used to sink to fanout values to multiple sinks
type FanoutSink []MetricSink

func (fh FanoutSink) SetGauge(key []string, val float32) {
	for _, s := range fh {
		s.SetGauge(key, val)
	}
}

func (fh FanoutSink) EmitKey(key []string, val float32) {
	for _, s := range fh {
		s.EmitKey(key, val)
	}
}

func (fh FanoutSink) IncrCounter(key []string, val float32) {
	for _, s := range fh {
		s.IncrCounter(key, val)
	}
}

func (fh FanoutSink) AddSample(key []string, val float32) {
	for _, s := range fh {
		s.AddSample(key, val)
	}
}
