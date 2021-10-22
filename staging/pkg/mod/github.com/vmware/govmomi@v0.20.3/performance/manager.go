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

package performance

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

// Manager wraps mo.PerformanceManager.
type Manager struct {
	object.Common

	Sort bool

	pm struct {
		sync.Mutex
		*mo.PerformanceManager
	}

	infoByName struct {
		sync.Mutex
		m map[string]*types.PerfCounterInfo
	}

	infoByKey struct {
		sync.Mutex
		m map[int32]*types.PerfCounterInfo
	}
}

// NewManager creates a new Manager instance.
func NewManager(client *vim25.Client) *Manager {
	m := Manager{
		Common: object.NewCommon(client, *client.ServiceContent.PerfManager),
	}

	m.pm.PerformanceManager = new(mo.PerformanceManager)

	return &m
}

// IntervalList wraps []types.PerfInterval.
type IntervalList []types.PerfInterval

// Enabled returns a map with Level as the key and enabled PerfInterval.Name(s) as the value.
func (l IntervalList) Enabled() map[int32][]string {
	enabled := make(map[int32][]string)

	for level := int32(0); level <= 4; level++ {
		var names []string

		for _, interval := range l {
			if interval.Enabled && interval.Level >= level {
				names = append(names, interval.Name)
			}
		}

		enabled[level] = names
	}

	return enabled
}

// HistoricalInterval gets the PerformanceManager.HistoricalInterval property and wraps as an IntervalList.
func (m *Manager) HistoricalInterval(ctx context.Context) (IntervalList, error) {
	var pm mo.PerformanceManager

	err := m.Properties(ctx, m.Reference(), []string{"historicalInterval"}, &pm)
	if err != nil {
		return nil, err
	}

	return IntervalList(pm.HistoricalInterval), nil
}

// CounterInfo gets the PerformanceManager.PerfCounter property.
// The property value is only collected once, subsequent calls return the cached value.
func (m *Manager) CounterInfo(ctx context.Context) ([]types.PerfCounterInfo, error) {
	m.pm.Lock()
	defer m.pm.Unlock()

	if len(m.pm.PerfCounter) == 0 {
		err := m.Properties(ctx, m.Reference(), []string{"perfCounter"}, m.pm.PerformanceManager)
		if err != nil {
			return nil, err
		}
	}

	return m.pm.PerfCounter, nil
}

// CounterInfoByName converts the PerformanceManager.PerfCounter property to a map,
// where key is types.PerfCounterInfo.Name().
func (m *Manager) CounterInfoByName(ctx context.Context) (map[string]*types.PerfCounterInfo, error) {
	m.infoByName.Lock()
	defer m.infoByName.Unlock()

	if m.infoByName.m != nil {
		return m.infoByName.m, nil
	}

	info, err := m.CounterInfo(ctx)
	if err != nil {
		return nil, err
	}

	m.infoByName.m = make(map[string]*types.PerfCounterInfo)

	for i := range info {
		c := &info[i]

		m.infoByName.m[c.Name()] = c
	}

	return m.infoByName.m, nil
}

// CounterInfoByKey converts the PerformanceManager.PerfCounter property to a map,
// where key is types.PerfCounterInfo.Key.
func (m *Manager) CounterInfoByKey(ctx context.Context) (map[int32]*types.PerfCounterInfo, error) {
	m.infoByKey.Lock()
	defer m.infoByKey.Unlock()

	if m.infoByKey.m != nil {
		return m.infoByKey.m, nil
	}

	info, err := m.CounterInfo(ctx)
	if err != nil {
		return nil, err
	}

	m.infoByKey.m = make(map[int32]*types.PerfCounterInfo)

	for i := range info {
		c := &info[i]

		m.infoByKey.m[c.Key] = c
	}

	return m.infoByKey.m, nil
}

// ProviderSummary wraps the QueryPerfProviderSummary method, caching the value based on entity.Type.
func (m *Manager) ProviderSummary(ctx context.Context, entity types.ManagedObjectReference) (*types.PerfProviderSummary, error) {
	req := types.QueryPerfProviderSummary{
		This:   m.Reference(),
		Entity: entity,
	}

	res, err := methods.QueryPerfProviderSummary(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return &res.Returnval, nil
}

type groupPerfCounterInfo struct {
	info map[int32]*types.PerfCounterInfo
	ids  []types.PerfMetricId
}

func (d groupPerfCounterInfo) Len() int {
	return len(d.ids)
}

func (d groupPerfCounterInfo) Less(i, j int) bool {
	ci := d.ids[i].CounterId
	cj := d.ids[j].CounterId
	gi := d.info[ci].GroupInfo.GetElementDescription()
	gj := d.info[cj].GroupInfo.GetElementDescription()

	return gi.Key < gj.Key
}

func (d groupPerfCounterInfo) Swap(i, j int) {
	d.ids[i], d.ids[j] = d.ids[j], d.ids[i]
}

// MetricList wraps []types.PerfMetricId
type MetricList []types.PerfMetricId

// ByKey converts MetricList to map, where key is types.PerfMetricId.CounterId / types.PerfCounterInfo.Key
func (l MetricList) ByKey() map[int32][]*types.PerfMetricId {
	ids := make(map[int32][]*types.PerfMetricId, len(l))

	for i := range l {
		id := &l[i]
		ids[id.CounterId] = append(ids[id.CounterId], id)
	}

	return ids
}

// AvailableMetric wraps the QueryAvailablePerfMetric method.
// The MetricList is sorted by PerfCounterInfo.GroupInfo.Key if Manager.Sort == true.
func (m *Manager) AvailableMetric(ctx context.Context, entity types.ManagedObjectReference, interval int32) (MetricList, error) {
	req := types.QueryAvailablePerfMetric{
		This:       m.Reference(),
		Entity:     entity.Reference(),
		IntervalId: interval,
	}

	res, err := methods.QueryAvailablePerfMetric(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	if m.Sort {
		info, err := m.CounterInfoByKey(ctx)
		if err != nil {
			return nil, err
		}

		sort.Sort(groupPerfCounterInfo{info, res.Returnval})
	}

	return MetricList(res.Returnval), nil
}

// Query wraps the QueryPerf method.
func (m *Manager) Query(ctx context.Context, spec []types.PerfQuerySpec) ([]types.BasePerfEntityMetricBase, error) {
	req := types.QueryPerf{
		This:      m.Reference(),
		QuerySpec: spec,
	}

	res, err := methods.QueryPerf(ctx, m.Client(), &req)
	if err != nil {
		return nil, err
	}

	return res.Returnval, nil
}

// SampleByName uses the spec param as a template, constructing a []types.PerfQuerySpec for the given metrics and entities
// and invoking the Query method.
// The spec template can specify instances using the MetricId.Instance field, by default all instances are collected.
// The spec template MaxSample defaults to 1.
// If the spec template IntervalId is a historical interval and StartTime is not specified,
// the StartTime is set to the current time - (IntervalId * MaxSample).
func (m *Manager) SampleByName(ctx context.Context, spec types.PerfQuerySpec, metrics []string, entity []types.ManagedObjectReference) ([]types.BasePerfEntityMetricBase, error) {
	info, err := m.CounterInfoByName(ctx)
	if err != nil {
		return nil, err
	}

	var ids []types.PerfMetricId

	instances := spec.MetricId
	if len(instances) == 0 {
		// Default to all instances
		instances = []types.PerfMetricId{{Instance: "*"}}
	}

	for _, name := range metrics {
		counter, ok := info[name]
		if !ok {
			return nil, fmt.Errorf("counter %q not found", name)
		}

		for _, i := range instances {
			ids = append(ids, types.PerfMetricId{CounterId: counter.Key, Instance: i.Instance})
		}
	}

	spec.MetricId = ids

	if spec.MaxSample == 0 {
		spec.MaxSample = 1
	}

	if spec.IntervalId >= 60 && spec.StartTime == nil {
		// Need a StartTime to make use of history
		now, err := methods.GetCurrentTime(ctx, m.Client())
		if err != nil {
			return nil, err
		}

		// Go back in time
		x := spec.IntervalId * -1 * spec.MaxSample
		t := now.Add(time.Duration(x) * time.Second)
		spec.StartTime = &t
	}

	var query []types.PerfQuerySpec

	for _, e := range entity {
		spec.Entity = e.Reference()
		query = append(query, spec)
	}

	return m.Query(ctx, query)
}

// MetricSeries contains the same data as types.PerfMetricIntSeries, but with the CounterId converted to Name.
type MetricSeries struct {
	Name     string
	unit     string
	Instance string
	Value    []int64
}

func (s *MetricSeries) Format(val int64) string {
	switch types.PerformanceManagerUnit(s.unit) {
	case types.PerformanceManagerUnitPercent:
		return strconv.FormatFloat(float64(val)/100.0, 'f', 2, 64)
	default:
		return strconv.FormatInt(val, 10)
	}
}

// ValueCSV converts the Value field to a CSV string
func (s *MetricSeries) ValueCSV() string {
	vals := make([]string, len(s.Value))

	for i := range s.Value {
		vals[i] = s.Format(s.Value[i])
	}

	return strings.Join(vals, ",")
}

// EntityMetric contains the same data as types.PerfEntityMetric, but with MetricSeries type for the Value field.
type EntityMetric struct {
	Entity types.ManagedObjectReference

	SampleInfo []types.PerfSampleInfo
	Value      []MetricSeries
}

// SampleInfoCSV converts the SampleInfo field to a CSV string
func (m *EntityMetric) SampleInfoCSV() string {
	vals := make([]string, len(m.SampleInfo)*2)

	i := 0

	for _, s := range m.SampleInfo {
		vals[i] = s.Timestamp.Format(time.RFC3339)
		i++
		vals[i] = strconv.Itoa(int(s.Interval))
		i++
	}

	return strings.Join(vals, ",")
}

// ToMetricSeries converts []BasePerfEntityMetricBase to []EntityMetric
func (m *Manager) ToMetricSeries(ctx context.Context, series []types.BasePerfEntityMetricBase) ([]EntityMetric, error) {
	counters, err := m.CounterInfoByKey(ctx)
	if err != nil {
		return nil, err
	}

	var result []EntityMetric

	for i := range series {
		var values []MetricSeries
		s, ok := series[i].(*types.PerfEntityMetric)
		if !ok {
			panic(fmt.Errorf("expected type %T, got: %T", s, series[i]))
		}

		for j := range s.Value {
			v := s.Value[j].(*types.PerfMetricIntSeries)

			values = append(values, MetricSeries{
				Name:     counters[v.Id.CounterId].Name(),
				unit:     counters[v.Id.CounterId].UnitInfo.GetElementDescription().Key,
				Instance: v.Id.Instance,
				Value:    v.Value,
			})
		}

		result = append(result, EntityMetric{
			Entity:     s.Entity,
			SampleInfo: s.SampleInfo,
			Value:      values,
		})
	}

	return result, nil
}
