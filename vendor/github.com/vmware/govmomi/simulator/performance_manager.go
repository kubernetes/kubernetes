/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package simulator

import (
	"math/rand"
	"strconv"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/simulator/esx"
	"github.com/vmware/govmomi/simulator/vpx"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

var realtimeProviderSummary = types.PerfProviderSummary{
	CurrentSupported: true,
	SummarySupported: true,
	RefreshRate:      20,
}

var historicProviderSummary = types.PerfProviderSummary{
	CurrentSupported: false,
	SummarySupported: true,
	RefreshRate:      -1,
}

type PerformanceManager struct {
	mo.PerformanceManager
	vmMetrics         []types.PerfMetricId
	hostMetrics       []types.PerfMetricId
	rpMetrics         []types.PerfMetricId
	clusterMetrics    []types.PerfMetricId
	datastoreMetrics  []types.PerfMetricId
	datacenterMetrics []types.PerfMetricId
	perfCounterIndex  map[int32]types.PerfCounterInfo
	metricData        map[string]map[int32][]int64
}

func NewPerformanceManager(ref types.ManagedObjectReference) object.Reference {
	m := &PerformanceManager{}
	m.Self = ref
	if Map.IsESX() {
		m.PerfCounter = esx.PerfCounter[:]
		m.hostMetrics = esx.HostMetrics[:]
		m.vmMetrics = esx.VmMetrics[:]
		m.rpMetrics = esx.ResourcePoolMetrics[:]
		m.metricData = esx.MetricData
	} else {
		m.PerfCounter = vpx.PerfCounter[:]
		m.hostMetrics = vpx.HostMetrics[:]
		m.vmMetrics = vpx.VmMetrics[:]
		m.rpMetrics = vpx.ResourcePoolMetrics[:]
		m.clusterMetrics = vpx.ClusterMetrics[:]
		m.datastoreMetrics = vpx.DatastoreMetrics[:]
		m.datacenterMetrics = vpx.DatacenterMetrics[:]
		m.metricData = vpx.MetricData
	}
	m.perfCounterIndex = make(map[int32]types.PerfCounterInfo, len(m.PerfCounter))
	for _, p := range m.PerfCounter {
		m.perfCounterIndex[p.Key] = p
	}
	return m
}

func (p *PerformanceManager) QueryPerfCounter(ctx *Context, req *types.QueryPerfCounter) soap.HasFault {
	body := new(methods.QueryPerfCounterBody)
	body.Req = req
	body.Res.Returnval = make([]types.PerfCounterInfo, len(req.CounterId))
	for i, id := range req.CounterId {
		if info, ok := p.perfCounterIndex[id]; !ok {
			body.Fault_ = Fault("", &types.InvalidArgument{
				InvalidProperty: "CounterId",
			})
			return body
		} else {
			body.Res.Returnval[i] = info
		}
	}
	return body
}

func (p *PerformanceManager) QueryPerfProviderSummary(ctx *Context, req *types.QueryPerfProviderSummary) soap.HasFault {
	body := new(methods.QueryPerfProviderSummaryBody)
	body.Req = req
	body.Res = new(types.QueryPerfProviderSummaryResponse)

	// The entity must exist
	if Map.Get(req.Entity) == nil {
		body.Fault_ = Fault("", &types.InvalidArgument{
			InvalidProperty: "Entity",
		})
		return body
	}

	switch req.Entity.Type {
	case "VirtualMachine", "HostSystem", "ResourcePool":
		body.Res.Returnval = realtimeProviderSummary
	default:
		body.Res.Returnval = historicProviderSummary
	}
	body.Res.Returnval.Entity = req.Entity
	return body
}

func (p *PerformanceManager) buildAvailablePerfMetricsQueryResponse(ids []types.PerfMetricId, numCPU int, datastoreURL string) *types.QueryAvailablePerfMetricResponse {
	r := new(types.QueryAvailablePerfMetricResponse)
	r.Returnval = make([]types.PerfMetricId, 0, len(ids))
	for _, id := range ids {
		switch id.Instance {
		case "$cpu":
			for i := 0; i < numCPU; i++ {
				r.Returnval = append(r.Returnval, types.PerfMetricId{CounterId: id.CounterId, Instance: strconv.Itoa(i)})
			}
		case "$physDisk":
			r.Returnval = append(r.Returnval, types.PerfMetricId{CounterId: id.CounterId, Instance: datastoreURL})
		case "$file":
			r.Returnval = append(r.Returnval, types.PerfMetricId{CounterId: id.CounterId, Instance: "DISKFILE"})
			r.Returnval = append(r.Returnval, types.PerfMetricId{CounterId: id.CounterId, Instance: "DELTAFILE"})
			r.Returnval = append(r.Returnval, types.PerfMetricId{CounterId: id.CounterId, Instance: "SWAPFILE"})
			r.Returnval = append(r.Returnval, types.PerfMetricId{CounterId: id.CounterId, Instance: "OTHERFILE"})
		default:
			r.Returnval = append(r.Returnval, types.PerfMetricId{CounterId: id.CounterId, Instance: id.Instance})
		}
	}
	return r
}

func (p *PerformanceManager) queryAvailablePerfMetric(entity types.ManagedObjectReference, interval int32) *types.QueryAvailablePerfMetricResponse {
	switch entity.Type {
	case "VirtualMachine":
		vm := Map.Get(entity).(*VirtualMachine)
		return p.buildAvailablePerfMetricsQueryResponse(p.vmMetrics, int(vm.Summary.Config.NumCpu), vm.Datastore[0].Value)
	case "HostSystem":
		host := Map.Get(entity).(*HostSystem)
		return p.buildAvailablePerfMetricsQueryResponse(p.hostMetrics, int(host.Hardware.CpuInfo.NumCpuThreads), host.Datastore[0].Value)
	case "ResourcePool":
		return p.buildAvailablePerfMetricsQueryResponse(p.rpMetrics, 0, "")
	case "ClusterComputeResource":
		if interval != 20 {
			return p.buildAvailablePerfMetricsQueryResponse(p.clusterMetrics, 0, "")
		}
	case "Datastore":
		if interval != 20 {
			return p.buildAvailablePerfMetricsQueryResponse(p.datastoreMetrics, 0, "")
		}
	case "Datacenter":
		if interval != 20 {
			return p.buildAvailablePerfMetricsQueryResponse(p.datacenterMetrics, 0, "")
		}
	}

	// Don't know how to handle this. Return empty response.
	return new(types.QueryAvailablePerfMetricResponse)
}

func (p *PerformanceManager) QueryAvailablePerfMetric(ctx *Context, req *types.QueryAvailablePerfMetric) soap.HasFault {
	body := new(methods.QueryAvailablePerfMetricBody)
	body.Req = req
	body.Res = p.queryAvailablePerfMetric(req.Entity, req.IntervalId)

	return body
}

func (p *PerformanceManager) QueryPerf(ctx *Context, req *types.QueryPerf) soap.HasFault {
	body := new(methods.QueryPerfBody)
	body.Req = req
	body.Res = new(types.QueryPerfResponse)
	body.Res.Returnval = make([]types.BasePerfEntityMetricBase, len(req.QuerySpec))

	for i, qs := range req.QuerySpec {
		metrics := new(types.PerfEntityMetric)
		metrics.Entity = qs.Entity

		// Get metric data for this entity type
		metricData, ok := p.metricData[qs.Entity.Type]
		if !ok {
			body.Fault_ = Fault("", &types.InvalidArgument{
				InvalidProperty: "Entity",
			})
		}
		var start, end time.Time
		if qs.StartTime == nil {
			start = time.Now().Add(time.Duration(-365*24) * time.Hour) // Assume we have data for a year
		} else {
			start = *qs.StartTime
		}
		if qs.EndTime == nil {
			end = time.Now()
		} else {
			end = *qs.EndTime
		}

		// Generate metric series. Divide into n buckets of interval seconds
		interval := qs.IntervalId
		if interval == -1 || interval == 0 {
			interval = 20 // TODO: Determine from entity type
		}
		n := 1 + int32(end.Sub(start).Seconds())/interval
		if n > qs.MaxSample {
			n = qs.MaxSample
		}

		// Loop through each interval "tick"
		metrics.SampleInfo = make([]types.PerfSampleInfo, n)
		metrics.Value = make([]types.BasePerfMetricSeries, len(qs.MetricId))
		for tick := int32(0); tick < n; tick++ {
			metrics.SampleInfo[tick] = types.PerfSampleInfo{Timestamp: end.Add(time.Duration(-interval*tick) * time.Second), Interval: interval}
		}

		for j, mid := range qs.MetricId {
			// Create list of metrics for this tick
			series := &types.PerfMetricIntSeries{Value: make([]int64, n)}
			series.Id = mid
			points := metricData[mid.CounterId]
			offset := int64(start.Unix()) / int64(interval)

			for tick := int32(0); tick < n; tick++ {
				var p int64

				// Use sample data if we have it. Otherwise, just send 0.
				if len(points) > 0 {
					p = points[(offset+int64(tick))%int64(len(points))]
					scale := p / 5
					if scale > 0 {
						// Add some gaussian noise to make the data look more "real"
						p += int64(rand.NormFloat64() * float64(scale))
						if p < 0 {
							p = 0
						}
					}
				} else {
					p = 0
				}
				series.Value[tick] = p
			}
			metrics.Value[j] = series
		}
		body.Res.Returnval[i] = metrics
	}
	return body
}
