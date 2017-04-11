package proxy

import (
	"expvar"
	"sync/atomic"
)

// Metrics is used to hold metric counters
// related to the proxy
type Metrics struct {
	Requests    uint64
	Hits        uint64
	Misses      uint64
	BytesPulled uint64
	BytesPushed uint64
}

type proxyMetricsCollector struct {
	blobMetrics     Metrics
	manifestMetrics Metrics
}

// BlobPull tracks metrics about blobs pulled into the cache
func (pmc *proxyMetricsCollector) BlobPull(bytesPulled uint64) {
	atomic.AddUint64(&pmc.blobMetrics.Misses, 1)
	atomic.AddUint64(&pmc.blobMetrics.BytesPulled, bytesPulled)
}

// BlobPush tracks metrics about blobs pushed to clients
func (pmc *proxyMetricsCollector) BlobPush(bytesPushed uint64) {
	atomic.AddUint64(&pmc.blobMetrics.Requests, 1)
	atomic.AddUint64(&pmc.blobMetrics.Hits, 1)
	atomic.AddUint64(&pmc.blobMetrics.BytesPushed, bytesPushed)
}

// ManifestPull tracks metrics related to Manifests pulled into the cache
func (pmc *proxyMetricsCollector) ManifestPull(bytesPulled uint64) {
	atomic.AddUint64(&pmc.manifestMetrics.Misses, 1)
	atomic.AddUint64(&pmc.manifestMetrics.BytesPulled, bytesPulled)
}

// ManifestPush tracks metrics about manifests pushed to clients
func (pmc *proxyMetricsCollector) ManifestPush(bytesPushed uint64) {
	atomic.AddUint64(&pmc.manifestMetrics.Requests, 1)
	atomic.AddUint64(&pmc.manifestMetrics.Hits, 1)
	atomic.AddUint64(&pmc.manifestMetrics.BytesPushed, bytesPushed)
}

// proxyMetrics tracks metrics about the proxy cache.  This is
// kept globally and made available via expvar.
var proxyMetrics = &proxyMetricsCollector{}

func init() {
	registry := expvar.Get("registry")
	if registry == nil {
		registry = expvar.NewMap("registry")
	}

	pm := registry.(*expvar.Map).Get("proxy")
	if pm == nil {
		pm = &expvar.Map{}
		pm.(*expvar.Map).Init()
		registry.(*expvar.Map).Set("proxy", pm)
	}

	pm.(*expvar.Map).Set("blobs", expvar.Func(func() interface{} {
		return proxyMetrics.blobMetrics
	}))

	pm.(*expvar.Map).Set("manifests", expvar.Func(func() interface{} {
		return proxyMetrics.manifestMetrics
	}))

}
