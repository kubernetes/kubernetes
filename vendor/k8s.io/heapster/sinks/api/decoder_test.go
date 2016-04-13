// Copyright 2015 Google Inc. All Rights Reserved.
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

package v1

import (
	"testing"
	"time"

	cadvisor_api "github.com/google/cadvisor/info/v1"
	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/heapster/sinks/cache"
	source_api "k8s.io/heapster/sources/api"
)

const (
	fakeContainerCreationTime = 12345
	fakeCurrentTime           = 12350
)

func TestEmptyInput(t *testing.T) {
	timeseries, err := NewDecoder().TimeseriesFromPods([]*cache.PodElement{})
	assert.NoError(t, err)
	assert.Empty(t, timeseries)
	timeseries, err = NewDecoder().TimeseriesFromContainers([]*cache.ContainerElement{})
	assert.NoError(t, err)
	assert.Empty(t, timeseries)
}

func TestFuzzInput(t *testing.T) {
	var pods []*cache.PodElement
	f := fuzz.New().NumElements(2, 10)
	f.Fuzz(&pods)
	_, err := NewDecoder().TimeseriesFromPods(pods)
	assert.NoError(t, err)
}

func getContainerElement(name string) *cache.ContainerElement {
	f := fuzz.New().NumElements(1, 1).NilChance(0)
	containerSpec := &source_api.ContainerSpec{
		ContainerSpec: cadvisor_api.ContainerSpec{
			CreationTime:  time.Unix(fakeContainerCreationTime, 0),
			HasCpu:        true,
			HasMemory:     true,
			HasNetwork:    true,
			HasFilesystem: true,
			HasDiskIo:     true,
			Cpu: cadvisor_api.CpuSpec{
				Limit: 100,
			},
			Memory: cadvisor_api.MemorySpec{
				Limit: 100,
			},
		},
		CpuRequest:    200,
		MemoryRequest: 200,
	}
	containerStats := make([]*source_api.ContainerStats, 1)
	f.Fuzz(&containerStats)
	return &cache.ContainerElement{
		Metadata: cache.Metadata{
			Name: name,
		},
		Metrics: []*cache.ContainerMetricElement{
			{
				Spec:  containerSpec,
				Stats: containerStats[0],
			},
		},
	}
}

type fsStats struct {
	limit int64
	usage int64
}

func getFsStatsFromContainerElement(input []*cache.ContainerElement) map[string]fsStats {
	expectedFsStats := map[string]fsStats{}
	for _, cont := range input {
		for _, cme := range cont.Metrics {
			for _, fs := range cme.Stats.Filesystem {
				expectedFsStats[fs.Device] = fsStats{int64(fs.Limit), int64(fs.Usage)}
			}
		}
	}
	return expectedFsStats
}

func TestRealInput(t *testing.T) {
	timeSince = func(t time.Time) time.Duration {
		return time.Unix(fakeCurrentTime, 0).Sub(t)
	}
	defer func() { timeSince = time.Since }()

	containers := []*cache.ContainerElement{
		getContainerElement("container1"),
	}
	pods := []*cache.PodElement{
		{
			Metadata: cache.Metadata{
				Name:      "pod1",
				UID:       "123",
				Namespace: "test",
				Hostname:  "1.2.3.4",
			},
			Containers: containers,
		},
		{
			Metadata: cache.Metadata{
				Name:      "pod2",
				UID:       "123",
				Namespace: "test",
				Hostname:  "1.2.3.5",
			},
			Containers: containers,
		},
	}
	timeseries, err := NewDecoder().TimeseriesFromPods(pods)
	assert.NoError(t, err)
	assert.NotEmpty(t, timeseries)
	expectedFsStats := getFsStatsFromContainerElement(containers)
	metrics := make(map[string][]Timeseries)
	for index := range timeseries {
		series, ok := metrics[timeseries[index].Point.Name]
		if !ok {
			series = make([]Timeseries, 0)
		}
		series = append(series, timeseries[index])
		metrics[timeseries[index].Point.Name] = series
	}
	for index := range statMetrics {
		series, ok := metrics[statMetrics[index].MetricDescriptor.Name]
		require.True(t, ok)
		for innerIndex, entry := range series {
			assert.Equal(t, statMetrics[index].MetricDescriptor, *series[innerIndex].MetricDescriptor)
			spec := containers[0].Metrics[0].Spec
			stats := containers[0].Metrics[0].Stats
			switch entry.Point.Name {
			case "uptime":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				expected := timeSince(spec.CreationTime).Nanoseconds() / time.Millisecond.Nanoseconds()
				assert.Equal(t, expected, value)
			case "cpu/usage":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Cpu.Usage.Total, value)
			case "memory/usage":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Memory.Usage, value)
			case "memory/working_set":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Memory.WorkingSet, value)
			case "memory/page_faults":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Memory.ContainerData.Pgfault, value)
			case "memory/major_page_faults":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Memory.ContainerData.Pgmajfault, value)
			case "network/rx":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Network.RxBytes, value)
			case "network/rx_errors":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Network.RxErrors, value)
			case "network/tx":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Network.TxBytes, value)
			case "network/tx_errors":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, stats.Network.TxErrors, value)
			case "filesystem/usage":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				name, ok := entry.Point.Labels[LabelResourceID.Key]
				require.True(t, ok)
				assert.Equal(t, expectedFsStats[name].usage, value)
			case "cpu/limit":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				expected := (spec.Cpu.Limit * 1000) / 1024
				assert.Equal(t, expected, value)
			case "cpu/request":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, spec.CpuRequest, value)
			case "memory/limit":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, spec.Memory.Limit, value)
			case "memory/request":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				assert.Equal(t, spec.MemoryRequest, value)
			case "filesystem/limit":
				value, ok := entry.Point.Value.(int64)
				require.True(t, ok)
				name, ok := entry.Point.Labels[LabelResourceID.Key]
				require.True(t, ok)
				assert.Equal(t, expectedFsStats[name].limit, value)
			default:
				t.Errorf("unexpected metric type")
			}
		}
	}
}

func copyMetric(el *cache.ContainerMetricElement) *cache.ContainerMetricElement {
	res := *el
	s := *el.Stats
	res.Stats = &s
	return &res
}

func TestNoDuplicates(t *testing.T) {
	decoder := NewDecoder()
	c := getContainerElement("container1")
	now := time.Now()

	m0 := c.Metrics[0]
	m0.Stats.Timestamp = now.Add(-1 * time.Minute)
	m1 := copyMetric(m0)
	m1.Stats.Timestamp = now.Add(-2 * time.Minute)
	m2 := copyMetric(m0)
	m2.Stats.Timestamp = now.Add(-3 * time.Minute)
	m3 := copyMetric(m0)
	m3.Stats.Timestamp = now.Add(-4 * time.Minute)

	c.Metrics = []*cache.ContainerMetricElement{m2, m3}
	ts, err := decoder.TimeseriesFromContainers([]*cache.ContainerElement{c})
	assert.NoError(t, err)
	assert.Equal(t, 2*len(statMetrics), len(ts))

	c.Metrics = []*cache.ContainerMetricElement{m0, m1, m2, m3}
	ts, err = decoder.TimeseriesFromContainers([]*cache.ContainerElement{c})
	assert.NoError(t, err)
	assert.Equal(t, 2*len(statMetrics), len(ts))
}
