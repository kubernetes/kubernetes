// Circonus Metrics Sink

package circonus

import (
	"strings"

	cgm "github.com/circonus-labs/circonus-gometrics"
)

// CirconusSink provides an interface to forward metrics to Circonus with
// automatic check creation and metric management
type CirconusSink struct {
	metrics *cgm.CirconusMetrics
}

// Config options for CirconusSink
// See https://github.com/circonus-labs/circonus-gometrics for configuration options
type Config cgm.Config

// NewCirconusSink - create new metric sink for circonus
//
// one of the following must be supplied:
//    - API Token - search for an existing check or create a new check
//    - API Token + Check Id - the check identified by check id will be used
//    - API Token + Check Submission URL - the check identified by the submission url will be used
//    - Check Submission URL - the check identified by the submission url will be used
//      metric management will be *disabled*
//
// Note: If submission url is supplied w/o an api token, the public circonus ca cert will be used
// to verify the broker for metrics submission.
func NewCirconusSink(cc *Config) (*CirconusSink, error) {
	cfg := cgm.Config{}
	if cc != nil {
		cfg = cgm.Config(*cc)
	}

	metrics, err := cgm.NewCirconusMetrics(&cfg)
	if err != nil {
		return nil, err
	}

	return &CirconusSink{
		metrics: metrics,
	}, nil
}

// Start submitting metrics to Circonus (flush every SubmitInterval)
func (s *CirconusSink) Start() {
	s.metrics.Start()
}

// Flush manually triggers metric submission to Circonus
func (s *CirconusSink) Flush() {
	s.metrics.Flush()
}

// SetGauge sets value for a gauge metric
func (s *CirconusSink) SetGauge(key []string, val float32) {
	flatKey := s.flattenKey(key)
	s.metrics.SetGauge(flatKey, int64(val))
}

// EmitKey is not implemented in circonus
func (s *CirconusSink) EmitKey(key []string, val float32) {
	// NOP
}

// IncrCounter increments a counter metric
func (s *CirconusSink) IncrCounter(key []string, val float32) {
	flatKey := s.flattenKey(key)
	s.metrics.IncrementByValue(flatKey, uint64(val))
}

// AddSample adds a sample to a histogram metric
func (s *CirconusSink) AddSample(key []string, val float32) {
	flatKey := s.flattenKey(key)
	s.metrics.RecordValue(flatKey, float64(val))
}

// Flattens key to Circonus metric name
func (s *CirconusSink) flattenKey(parts []string) string {
	joined := strings.Join(parts, "`")
	return strings.Map(func(r rune) rune {
		switch r {
		case ' ':
			return '_'
		default:
			return r
		}
	}, joined)
}
