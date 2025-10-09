/*
 *
 * Copyright 2020 gRPC authors.
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

package clusterimpl

import (
	"context"

	v3orcapb "github.com/cncf/xds/go/xds/data/orca/v3"
	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/internal/stats"
	"google.golang.org/grpc/internal/wrr"
	xdsinternal "google.golang.org/grpc/internal/xds"
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/grpc/internal/xds/xdsclient"
	"google.golang.org/grpc/status"
)

// NewRandomWRR is used when calculating drops. It's exported so that tests can
// override it.
var NewRandomWRR = wrr.NewRandom

const million = 1000000

type dropper struct {
	category string
	w        wrr.WRR
}

// greatest common divisor (GCD) via Euclidean algorithm
func gcd(a, b uint32) uint32 {
	for b != 0 {
		t := b
		b = a % b
		a = t
	}
	return a
}

func newDropper(c DropConfig) *dropper {
	w := NewRandomWRR()
	gcdv := gcd(c.RequestsPerMillion, million)
	// Return true for RequestPerMillion, false for the rest.
	w.Add(true, int64(c.RequestsPerMillion/gcdv))
	w.Add(false, int64((million-c.RequestsPerMillion)/gcdv))

	return &dropper{
		category: c.Category,
		w:        w,
	}
}

func (d *dropper) drop() (ret bool) {
	return d.w.Next().(bool)
}

// loadReporter wraps the methods from the loadStore that are used here.
type loadReporter interface {
	CallStarted(locality clients.Locality)
	CallFinished(locality clients.Locality, err error)
	CallServerLoad(locality clients.Locality, name string, val float64)
	CallDropped(category string)
}

// Picker implements RPC drop, circuit breaking drop and load reporting.
type picker struct {
	drops           []*dropper
	s               balancer.State
	loadStore       loadReporter
	counter         *xdsclient.ClusterRequestsCounter
	countMax        uint32
	telemetryLabels map[string]string
}

func telemetryLabels(ctx context.Context) map[string]string {
	if ctx == nil {
		return nil
	}
	labels := stats.GetLabels(ctx)
	if labels == nil {
		return nil
	}
	return labels.TelemetryLabels
}

func (d *picker) Pick(info balancer.PickInfo) (balancer.PickResult, error) {
	// Unconditionally set labels if present, even dropped or queued RPC's can
	// use these labels.
	if labels := telemetryLabels(info.Ctx); labels != nil {
		for key, value := range d.telemetryLabels {
			labels[key] = value
		}
	}

	// Don't drop unless the inner picker is READY. Similar to
	// https://github.com/grpc/grpc-go/issues/2622.
	if d.s.ConnectivityState == connectivity.Ready {
		// Check if this RPC should be dropped by category.
		for _, dp := range d.drops {
			if dp.drop() {
				if d.loadStore != nil {
					d.loadStore.CallDropped(dp.category)
				}
				return balancer.PickResult{}, status.Errorf(codes.Unavailable, "RPC is dropped")
			}
		}
	}

	// Check if this RPC should be dropped by circuit breaking.
	if d.counter != nil {
		if err := d.counter.StartRequest(d.countMax); err != nil {
			// Drops by circuit breaking are reported with empty category. They
			// will be reported only in total drops, but not in per category.
			if d.loadStore != nil {
				d.loadStore.CallDropped("")
			}
			return balancer.PickResult{}, status.Error(codes.Unavailable, err.Error())
		}
	}

	var lID clients.Locality
	pr, err := d.s.Picker.Pick(info)
	if scw, ok := pr.SubConn.(*scWrapper); ok {
		// This OK check also covers the case err!=nil, because SubConn will be
		// nil.
		pr.SubConn = scw.SubConn
		// If locality ID isn't found in the wrapper, an empty locality ID will
		// be used.
		lID = scw.localityID()
	}

	if err != nil {
		if d.counter != nil {
			// Release one request count if this pick fails.
			d.counter.EndRequest()
		}
		return pr, err
	}

	if labels := telemetryLabels(info.Ctx); labels != nil {
		labels["grpc.lb.locality"] = xdsinternal.LocalityString(lID)
	}

	if d.loadStore != nil {
		locality := clients.Locality{Region: lID.Region, Zone: lID.Zone, SubZone: lID.SubZone}
		d.loadStore.CallStarted(locality)
		oldDone := pr.Done
		pr.Done = func(info balancer.DoneInfo) {
			if oldDone != nil {
				oldDone(info)
			}
			d.loadStore.CallFinished(locality, info.Err)

			load, ok := info.ServerLoad.(*v3orcapb.OrcaLoadReport)
			if !ok || load == nil {
				return
			}
			for n, c := range load.NamedMetrics {
				d.loadStore.CallServerLoad(locality, n, c)
			}
		}
	}

	if d.counter != nil {
		// Update Done() so that when the RPC finishes, the request count will
		// be released.
		oldDone := pr.Done
		pr.Done = func(doneInfo balancer.DoneInfo) {
			d.counter.EndRequest()
			if oldDone != nil {
				oldDone(doneInfo)
			}
		}
	}

	return pr, err
}
