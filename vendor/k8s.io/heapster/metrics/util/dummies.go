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

package util

import (
	"sync"
	"time"

	"k8s.io/heapster/metrics/core"
)

type DummySink struct {
	name        string
	mutex       sync.Mutex
	exportCount int
	stopped     bool
	latency     time.Duration
}

func (this *DummySink) Name() string {
	return this.name
}
func (this *DummySink) ExportData(*core.DataBatch) {
	this.mutex.Lock()
	this.exportCount++
	this.mutex.Unlock()

	time.Sleep(this.latency)
}

func (this *DummySink) Stop() {
	this.mutex.Lock()
	this.stopped = true
	this.mutex.Unlock()

	time.Sleep(this.latency)
}

func (this *DummySink) IsStopped() bool {
	this.mutex.Lock()
	defer this.mutex.Unlock()
	return this.stopped
}

func (this *DummySink) GetExportCount() int {
	this.mutex.Lock()
	defer this.mutex.Unlock()
	return this.exportCount
}

func NewDummySink(name string, latency time.Duration) *DummySink {
	return &DummySink{
		name:        name,
		latency:     latency,
		exportCount: 0,
		stopped:     false,
	}
}

type DummyMetricsSource struct {
	latency   time.Duration
	metricSet core.MetricSet
}

func (this *DummyMetricsSource) Name() string {
	return "dummy"
}

func (this *DummyMetricsSource) ScrapeMetrics(start, end time.Time) *core.DataBatch {
	time.Sleep(this.latency)
	return &core.DataBatch{
		Timestamp: end,
		MetricSets: map[string]*core.MetricSet{
			this.metricSet.Labels["name"]: &this.metricSet,
		},
	}
}

func newDummyMetricSet(name string) core.MetricSet {
	return core.MetricSet{
		MetricValues: map[string]core.MetricValue{},
		Labels: map[string]string{
			"name": name,
		},
	}
}

func NewDummyMetricsSource(name string, latency time.Duration) *DummyMetricsSource {
	return &DummyMetricsSource{
		latency:   latency,
		metricSet: newDummyMetricSet(name),
	}
}

type DummyMetricsSourceProvider struct {
	sources []core.MetricsSource
}

func (this *DummyMetricsSourceProvider) GetMetricsSources() []core.MetricsSource {
	return this.sources
}

func NewDummyMetricsSourceProvider(sources ...core.MetricsSource) *DummyMetricsSourceProvider {
	return &DummyMetricsSourceProvider{
		sources: sources,
	}
}

type DummyDataProcessor struct {
	latency time.Duration
}

func (this *DummyDataProcessor) Name() string {
	return "dummy"
}

func (this *DummyDataProcessor) Process(data *core.DataBatch) (*core.DataBatch, error) {
	time.Sleep(this.latency)
	return data, nil
}

func NewDummyDataProcessor(latency time.Duration) *DummyDataProcessor {
	return &DummyDataProcessor{
		latency: latency,
	}
}
