// Copyright 2020 Google Inc. All Rights Reserved.
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

package metrics

import (
	"sync"
	"time"

	info "github.com/google/cadvisor/info/v1"
	v2 "github.com/google/cadvisor/info/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/cache"
	dto "github.com/prometheus/client_model/go"
)

// metricValue describes a single metric value for a given set of label values
// within a parent containerMetric.
type metricValue struct {
	value     float64
	labels    []string
	timestamp time.Time
}

type metricValues []metricValue

// infoProvider will usually be manager.Manager, but can be swapped out for testing.
type infoProvider interface {
	// GetRequestedContainersInfo gets info for all requested containers based on the request options.
	GetRequestedContainersInfo(containerName string, options v2.RequestOptions) (map[string]*info.ContainerInfo, error)
	// GetVersionInfo provides information about the version.
	GetVersionInfo() (*info.VersionInfo, error)
	// GetMachineInfo provides information about the machine.
	GetMachineInfo() (*info.MachineInfo, error)
}

type CollectFn func(opts v2.RequestOptions, inserts []cache.Insert) []cache.Insert

var _ prometheus.TransactionalGatherer = &CachedGatherer{}

// CachedGatherer is an TransactionalGatherer that is able to cache and update in place metrics from defined Cadvisor collectors.
// Caller has responsibility to use `UpdateOnMaxAge` whenever cache has to be updated.
type CachedGatherer struct {
	*cache.CachedTGatherer
	buf []cache.Insert

	mu         sync.Mutex
	collectFns []CollectFn
	lastUpdate time.Time
}

func NewCachedGatherer(cfs ...CollectFn) *CachedGatherer {
	return &CachedGatherer{
		CachedTGatherer: cache.NewCachedTGatherer(),
		collectFns:      cfs,
	}
}

// UpdateOnMaxAge updates cache using provided collectorFns whenever cache is older than provided `MaxAge`. If `MaxAge` is nil, we always update cache.
// UpdateOnMaxAge is goroutine safe.
func (c *CachedGatherer) UpdateOnMaxAge(opts v2.RequestOptions) {
	c.mu.Lock() // CachedTGatherer.Update is goroutine safe, but collectFns and lastUpdate are not, so lock it.
	defer c.mu.Unlock()

	if opts.MaxAge == nil || time.Since(c.lastUpdate) > *opts.MaxAge {
		c.lastUpdate = time.Now()

		c.buf = c.buf[:0]
		for _, collect := range c.collectFns {
			c.buf = collect(opts, c.buf)
		}
		if err := c.CachedTGatherer.Update(true, c.buf, nil); err != nil {
			panic(err) // Programmatic error.
		}
	}
}

// UpdateOnMaxAgeGatherer makes `TransactionalGatherer` that updates cache on every `Gather` call according to provided `MaxAge`.
func UpdateOnMaxAgeGatherer(opts v2.RequestOptions, g *CachedGatherer) prometheus.TransactionalGatherer {
	return &callUpdateCG{opts: opts, CachedGatherer: g}
}

type callUpdateCG struct {
	*CachedGatherer

	opts v2.RequestOptions
}

func (c *callUpdateCG) Gather() (_ []*dto.MetricFamily, done func(), err error) {
	c.UpdateOnMaxAge(c.opts)
	return c.CachedTGatherer.Gather()
}
