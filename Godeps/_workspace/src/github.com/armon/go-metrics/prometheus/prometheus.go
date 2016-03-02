// +build go1.3
package prometheus

import (
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

type PrometheusSink struct {
	mu        sync.Mutex
	gauges    map[string]prometheus.Gauge
	summaries map[string]prometheus.Summary
	counters  map[string]prometheus.Counter
}

func NewPrometheusSink() (*PrometheusSink, error) {
	return &PrometheusSink{
		gauges:    make(map[string]prometheus.Gauge),
		summaries: make(map[string]prometheus.Summary),
		counters:  make(map[string]prometheus.Counter),
	}, nil
}

func (p *PrometheusSink) flattenKey(parts []string) string {
	joined := strings.Join(parts, "_")
	joined = strings.Replace(joined, " ", "_", -1)
	joined = strings.Replace(joined, ".", "_", -1)
	joined = strings.Replace(joined, "-", "_", -1)
	return joined
}

func (p *PrometheusSink) SetGauge(parts []string, val float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	key := p.flattenKey(parts)
	g, ok := p.gauges[key]
	if !ok {
		g = prometheus.NewGauge(prometheus.GaugeOpts{
			Name: key,
			Help: key,
		})
		prometheus.MustRegister(g)
		p.gauges[key] = g
	}
	g.Set(float64(val))
}

func (p *PrometheusSink) AddSample(parts []string, val float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	key := p.flattenKey(parts)
	g, ok := p.summaries[key]
	if !ok {
		g = prometheus.NewSummary(prometheus.SummaryOpts{
			Name:   key,
			Help:   key,
			MaxAge: 10 * time.Second,
		})
		prometheus.MustRegister(g)
		p.summaries[key] = g
	}
	g.Observe(float64(val))
}

// EmitKey is not implemented. Prometheus doesn’t offer a type for which an
// arbitrary number of values is retained, as Prometheus works with a pull
// model, rather than a push model.
func (p *PrometheusSink) EmitKey(key []string, val float32) {
}

func (p *PrometheusSink) IncrCounter(parts []string, val float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	key := p.flattenKey(parts)
	g, ok := p.counters[key]
	if !ok {
		g = prometheus.NewCounter(prometheus.CounterOpts{
			Name: key,
			Help: key,
		})
		prometheus.MustRegister(g)
		p.counters[key] = g
	}
	g.Add(float64(val))
}
