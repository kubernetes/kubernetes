/*
 *
 * Copyright 2025 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package lrsclient

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/internal/xds/clients"
	lrsclientinternal "google.golang.org/grpc/internal/xds/clients/lrsclient/internal"
)

// A LoadStore aggregates loads for multiple clusters and services that are
// intended to be reported via LRS.
//
// LoadStore stores loads reported to a single LRS server. Use multiple stores
// for multiple servers.
//
// It is safe for concurrent use.
type LoadStore struct {
	// stop is the function to call to Stop the LoadStore reporting.
	stop func(ctx context.Context)

	// mu only protects the map (2 layers). The read/write to
	// *PerClusterReporter doesn't need to hold the mu.
	mu sync.Mutex
	// clusters is a map with cluster name as the key. The second layer is a
	// map with service name as the key. Each value (PerClusterReporter)
	// contains data for a (cluster, service) pair.
	//
	// Note that new entries are added to this map, but never removed. This is
	// potentially a memory leak. But the memory is allocated for each new
	// (cluster,service) pair, and the memory allocated is just pointers and
	// maps. So this shouldn't get too bad.
	clusters map[string]map[string]*PerClusterReporter
}

func init() {
	lrsclientinternal.TimeNow = time.Now
}

// newLoadStore creates a LoadStore.
func newLoadStore() *LoadStore {
	return &LoadStore{
		clusters: make(map[string]map[string]*PerClusterReporter),
	}
}

// Stop signals the LoadStore to stop reporting.
//
// Before closing the underlying LRS stream, this method may block until a
// final load report send attempt completes or the provided context `ctx`
// expires.
//
// The provided context must have a deadline or timeout set to prevent Stop
// from blocking indefinitely if the final send attempt fails to complete.
//
// Calling Stop on an already stopped LoadStore is a no-op.
func (ls *LoadStore) Stop(ctx context.Context) {
	ls.stop(ctx)
}

// ReporterForCluster returns the PerClusterReporter for the given cluster and
// service.
func (ls *LoadStore) ReporterForCluster(clusterName, serviceName string) *PerClusterReporter {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	c, ok := ls.clusters[clusterName]
	if !ok {
		c = make(map[string]*PerClusterReporter)
		ls.clusters[clusterName] = c
	}

	if p, ok := c[serviceName]; ok {
		return p
	}
	p := &PerClusterReporter{
		cluster:          clusterName,
		service:          serviceName,
		lastLoadReportAt: lrsclientinternal.TimeNow(),
	}
	c[serviceName] = p
	return p
}

// stats returns the load data for the given cluster names. Data is returned in
// a slice with no specific order.
//
// If no clusterName is given (an empty slice), all data for all known clusters
// is returned.
//
// If a cluster's loadData is empty (no load to report), it's not appended to
// the returned slice.
func (ls *LoadStore) stats(clusterNames []string) []*loadData {
	ls.mu.Lock()
	defer ls.mu.Unlock()

	var ret []*loadData
	if len(clusterNames) == 0 {
		for _, c := range ls.clusters {
			ret = appendClusterStats(ret, c)
		}
		return ret
	}
	for _, n := range clusterNames {
		if c, ok := ls.clusters[n]; ok {
			ret = appendClusterStats(ret, c)
		}
	}

	return ret
}

// PerClusterReporter records load data pertaining to a single cluster. It
// provides methods to record call starts, finishes, server-reported loads,
// and dropped calls.
//
// It is safe for concurrent use.
//
// TODO(purnesh42h): Use regular maps with mutexes instead of sync.Map here.
// The latter is optimized for two common use cases: (1) when the entry for a
// given key is only ever written once but read many times, as in caches that
// only grow, or (2) when multiple goroutines read, write, and overwrite
// entries for disjoint sets of keys. In these two cases, use of a Map may
// significantly reduce lock contention compared to a Go map paired with a
// separate Mutex or RWMutex.
// Neither of these conditions are met here, and we should transition to a
// regular map with a mutex for better type safety.
type PerClusterReporter struct {
	cluster, service string
	drops            sync.Map // map[string]*uint64
	localityRPCCount sync.Map // map[clients.Locality]*rpcCountData

	mu               sync.Mutex
	lastLoadReportAt time.Time
}

// CallStarted records a call started in the LoadStore.
func (p *PerClusterReporter) CallStarted(locality clients.Locality) {
	s, ok := p.localityRPCCount.Load(locality)
	if !ok {
		tp := newRPCCountData()
		s, _ = p.localityRPCCount.LoadOrStore(locality, tp)
	}
	s.(*rpcCountData).incrInProgress()
	s.(*rpcCountData).incrIssued()
}

// CallFinished records a call finished in the LoadStore.
func (p *PerClusterReporter) CallFinished(locality clients.Locality, err error) {
	f, ok := p.localityRPCCount.Load(locality)
	if !ok {
		// The map is never cleared, only values in the map are reset. So the
		// case where entry for call-finish is not found should never happen.
		return
	}
	f.(*rpcCountData).decrInProgress()
	if err == nil {
		f.(*rpcCountData).incrSucceeded()
	} else {
		f.(*rpcCountData).incrErrored()
	}
}

// CallServerLoad records the server load in the LoadStore.
func (p *PerClusterReporter) CallServerLoad(locality clients.Locality, name string, val float64) {
	s, ok := p.localityRPCCount.Load(locality)
	if !ok {
		// The map is never cleared, only values in the map are reset. So the
		// case where entry for callServerLoad is not found should never happen.
		return
	}
	s.(*rpcCountData).addServerLoad(name, val)
}

// CallDropped records a call dropped in the LoadStore.
func (p *PerClusterReporter) CallDropped(category string) {
	d, ok := p.drops.Load(category)
	if !ok {
		tp := new(uint64)
		d, _ = p.drops.LoadOrStore(category, tp)
	}
	atomic.AddUint64(d.(*uint64), 1)
}

// stats returns and resets all loads reported to the store, except inProgress
// rpc counts.
//
// It returns nil if the store doesn't contain any (new) data.
func (p *PerClusterReporter) stats() *loadData {
	sd := newLoadData(p.cluster, p.service)
	p.drops.Range(func(key, val any) bool {
		d := atomic.SwapUint64(val.(*uint64), 0)
		if d == 0 {
			return true
		}
		sd.totalDrops += d
		keyStr := key.(string)
		if keyStr != "" {
			// Skip drops without category. They are counted in total_drops, but
			// not in per category. One example is drops by circuit breaking.
			sd.drops[keyStr] = d
		}
		return true
	})
	p.localityRPCCount.Range(func(key, val any) bool {
		countData := val.(*rpcCountData)
		succeeded := countData.loadAndClearSucceeded()
		inProgress := countData.loadInProgress()
		errored := countData.loadAndClearErrored()
		issued := countData.loadAndClearIssued()
		if succeeded == 0 && inProgress == 0 && errored == 0 && issued == 0 {
			return true
		}

		ld := localityData{
			requestStats: requestData{
				succeeded:  succeeded,
				errored:    errored,
				inProgress: inProgress,
				issued:     issued,
			},
			loadStats: make(map[string]serverLoadData),
		}
		countData.serverLoads.Range(func(key, val any) bool {
			sum, count := val.(*rpcLoadData).loadAndClear()
			if count == 0 {
				return true
			}
			ld.loadStats[key.(string)] = serverLoadData{
				count: count,
				sum:   sum,
			}
			return true
		})
		sd.localityStats[key.(clients.Locality)] = ld
		return true
	})

	p.mu.Lock()
	sd.reportInterval = lrsclientinternal.TimeNow().Sub(p.lastLoadReportAt)
	p.lastLoadReportAt = lrsclientinternal.TimeNow()
	p.mu.Unlock()

	if sd.totalDrops == 0 && len(sd.drops) == 0 && len(sd.localityStats) == 0 {
		return nil
	}
	return sd
}

// loadData contains all load data reported to the LoadStore since the most recent
// call to stats().
type loadData struct {
	// cluster is the name of the cluster this data is for.
	cluster string
	// service is the name of the EDS service this data is for.
	service string
	// totalDrops is the total number of dropped requests.
	totalDrops uint64
	// drops is the number of dropped requests per category.
	drops map[string]uint64
	// localityStats contains load reports per locality.
	localityStats map[clients.Locality]localityData
	// reportInternal is the duration since last time load was reported (stats()
	// was called).
	reportInterval time.Duration
}

// localityData contains load data for a single locality.
type localityData struct {
	// requestStats contains counts of requests made to the locality.
	requestStats requestData
	// loadStats contains server load data for requests made to the locality,
	// indexed by the load type.
	loadStats map[string]serverLoadData
}

// requestData contains request counts.
type requestData struct {
	// succeeded is the number of succeeded requests.
	succeeded uint64
	// errored is the number of requests which ran into errors.
	errored uint64
	// inProgress is the number of requests in flight.
	inProgress uint64
	// issued is the total number requests that were sent.
	issued uint64
}

// serverLoadData contains server load data.
type serverLoadData struct {
	// count is the number of load reports.
	count uint64
	// sum is the total value of all load reports.
	sum float64
}

// appendClusterStats gets the Data for all the given clusters, append to ret,
// and return the new slice.
//
// Data is only appended to ret if it's not empty.
func appendClusterStats(ret []*loadData, clusters map[string]*PerClusterReporter) []*loadData {
	for _, d := range clusters {
		data := d.stats()
		if data == nil {
			// Skip this data if it doesn't contain any information.
			continue
		}
		ret = append(ret, data)
	}
	return ret
}

func newLoadData(cluster, service string) *loadData {
	return &loadData{
		cluster:       cluster,
		service:       service,
		drops:         make(map[string]uint64),
		localityStats: make(map[clients.Locality]localityData),
	}
}

type rpcCountData struct {
	// Only atomic accesses are allowed for the fields.
	succeeded  *uint64
	errored    *uint64
	inProgress *uint64
	issued     *uint64

	// Map from load desc to load data (sum+count). Loading data from map is
	// atomic, but updating data takes a lock, which could cause contention when
	// multiple RPCs try to report loads for the same desc.
	//
	// To fix the contention, shard this map.
	serverLoads sync.Map // map[string]*rpcLoadData
}

func newRPCCountData() *rpcCountData {
	return &rpcCountData{
		succeeded:  new(uint64),
		errored:    new(uint64),
		inProgress: new(uint64),
		issued:     new(uint64),
	}
}

func (rcd *rpcCountData) incrSucceeded() {
	atomic.AddUint64(rcd.succeeded, 1)
}

func (rcd *rpcCountData) loadAndClearSucceeded() uint64 {
	return atomic.SwapUint64(rcd.succeeded, 0)
}

func (rcd *rpcCountData) incrErrored() {
	atomic.AddUint64(rcd.errored, 1)
}

func (rcd *rpcCountData) loadAndClearErrored() uint64 {
	return atomic.SwapUint64(rcd.errored, 0)
}

func (rcd *rpcCountData) incrInProgress() {
	atomic.AddUint64(rcd.inProgress, 1)
}

func (rcd *rpcCountData) decrInProgress() {
	atomic.AddUint64(rcd.inProgress, ^uint64(0)) // atomic.Add(x, -1)
}

func (rcd *rpcCountData) loadInProgress() uint64 {
	return atomic.LoadUint64(rcd.inProgress) // InProgress count is not clear when reading.
}

func (rcd *rpcCountData) incrIssued() {
	atomic.AddUint64(rcd.issued, 1)
}

func (rcd *rpcCountData) loadAndClearIssued() uint64 {
	return atomic.SwapUint64(rcd.issued, 0)
}

func (rcd *rpcCountData) addServerLoad(name string, d float64) {
	loads, ok := rcd.serverLoads.Load(name)
	if !ok {
		tl := newRPCLoadData()
		loads, _ = rcd.serverLoads.LoadOrStore(name, tl)
	}
	loads.(*rpcLoadData).add(d)
}

// rpcLoadData is data for server loads (from trailers or oob). Fields in this
// struct must be updated consistently.
//
// The current solution is to hold a lock, which could cause contention. To fix,
// shard serverLoads map in rpcCountData.
type rpcLoadData struct {
	mu    sync.Mutex
	sum   float64
	count uint64
}

func newRPCLoadData() *rpcLoadData {
	return &rpcLoadData{}
}

func (rld *rpcLoadData) add(v float64) {
	rld.mu.Lock()
	rld.sum += v
	rld.count++
	rld.mu.Unlock()
}

func (rld *rpcLoadData) loadAndClear() (s float64, c uint64) {
	rld.mu.Lock()
	s, rld.sum = rld.sum, 0
	c, rld.count = rld.count, 0
	rld.mu.Unlock()
	return s, c
}
