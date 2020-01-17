// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package grpcproxy

import (
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.etcd.io/etcd/etcdserver/api/etcdhttp"
)

var (
	watchersCoalescing = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "grpc_proxy",
		Name:      "watchers_coalescing_total",
		Help:      "Total number of current watchers coalescing",
	})
	eventsCoalescing = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "etcd",
		Subsystem: "grpc_proxy",
		Name:      "events_coalescing_total",
		Help:      "Total number of events coalescing",
	})
	cacheKeys = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "grpc_proxy",
		Name:      "cache_keys_total",
		Help:      "Total number of keys/ranges cached",
	})
	cacheHits = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "grpc_proxy",
		Name:      "cache_hits_total",
		Help:      "Total number of cache hits",
	})
	cachedMisses = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "etcd",
		Subsystem: "grpc_proxy",
		Name:      "cache_misses_total",
		Help:      "Total number of cache misses",
	})
)

func init() {
	prometheus.MustRegister(watchersCoalescing)
	prometheus.MustRegister(eventsCoalescing)
	prometheus.MustRegister(cacheKeys)
	prometheus.MustRegister(cacheHits)
	prometheus.MustRegister(cachedMisses)
}

// HandleMetrics performs a GET request against etcd endpoint and returns '/metrics'.
func HandleMetrics(mux *http.ServeMux, c *http.Client, eps []string) {
	// random shuffle endpoints
	r := rand.New(rand.NewSource(int64(time.Now().Nanosecond())))
	if len(eps) > 1 {
		eps = shuffleEndpoints(r, eps)
	}

	pathMetrics := etcdhttp.PathMetrics
	mux.HandleFunc(pathMetrics, func(w http.ResponseWriter, r *http.Request) {
		target := fmt.Sprintf("%s%s", eps[0], pathMetrics)
		if !strings.HasPrefix(target, "http") {
			scheme := "http"
			if r.TLS != nil {
				scheme = "https"
			}
			target = fmt.Sprintf("%s://%s", scheme, target)
		}

		resp, err := c.Get(target)
		if err != nil {
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		body, _ := ioutil.ReadAll(resp.Body)
		fmt.Fprintf(w, "%s", body)
	})
}

func shuffleEndpoints(r *rand.Rand, eps []string) []string {
	// copied from Go 1.9<= rand.Rand.Perm
	n := len(eps)
	p := make([]int, n)
	for i := 0; i < n; i++ {
		j := r.Intn(i + 1)
		p[i] = p[j]
		p[j] = i
	}
	neps := make([]string, n)
	for i, k := range p {
		neps[i] = eps[k]
	}
	return neps
}
