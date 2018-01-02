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

package integration

/*
Commented out until the new data flow supports model api

import (
	"fmt"
	"testing"
	"time"

	cadvisor "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"

	"k8s.io/heapster/metrics/manager"
	model_api "k8s.io/heapster/metrics/model"
	"k8s.io/heapster/metrics/sinks"
	sink_api "k8s.io/heapster/metrics/sinks/api"
	"k8s.io/heapster/metrics/sinks/cache"
	source_api "k8s.io/heapster/metrics/sources/api"
)

type testSource struct {
	createTimestamp time.Time
}

const (
	podCount         = 10
	testNamespace    = "testnamespace"
	loadAverageMilli = 300
)

func (t testSource) buildPods(start time.Time) []source_api.Pod {

	timeElapsed := start.Sub(t.createTimestamp)

	result := []source_api.Pod{}
	for i := 0; i < podCount; i++ {
		stat := source_api.ContainerStats{
			ContainerStats: cadvisor.ContainerStats{
				Timestamp: start.Add(time.Millisecond * 500),
				Cpu: cadvisor.CpuStats{
					Usage: cadvisor.CpuUsage{
						Total:  uint64(loadAverageMilli * 1000000 * timeElapsed.Seconds()),
						User:   uint64(loadAverageMilli * 1000000 * timeElapsed.Seconds()),
						System: 0,
					},
					LoadAverage: loadAverageMilli,
				},
			},
		}

		pod := source_api.Pod{
			PodMetadata: source_api.PodMetadata{
				Name:         fmt.Sprintf("pod-%d", i),
				Namespace:    testNamespace,
				NamespaceUID: testNamespace + "UID",
				ID:           fmt.Sprintf("pid-%d", i),
				Hostname:     fmt.Sprintf("node-%d", i),
				Status:       "Running",
				PodIP:        fmt.Sprintf("10.0.0.%d", i),
			},
			Containers: []source_api.Container{
				{
					Hostname:   fmt.Sprintf("node-%d", i),
					ExternalID: fmt.Sprintf("cont-%d", i),
					Name:       "cont",
					Spec: source_api.ContainerSpec{
						ContainerSpec: cadvisor.ContainerSpec{
							HasCpu: true,
							Cpu: cadvisor.CpuSpec{
								Limit:    500,
								MaxLimit: 600,
							},
						},
					},
					Stats: []*source_api.ContainerStats{&stat},
				},
			},
		}
		result = append(result, pod)
	}
	return result
}

func (t testSource) GetInfo(start, end time.Time) (source_api.AggregateData, error) {
	return source_api.AggregateData{
		Pods: t.buildPods(start),
	}, nil
}

func (t testSource) DebugInfo() string {
	return "test-debug-info"
}

func (t testSource) Name() string {
	return "test-source"
}

func newTestSource() source_api.Source {
	return &testSource{
		createTimestamp: time.Now().Add(-10 * time.Second),
	}
}

func TestModelMetricPassing(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping heapster model integration test.")
	}
	assert := assert.New(t)
	resolution := 2 * time.Second

	sources := []source_api.Source{newTestSource()}
	cache := cache.NewCache(time.Hour, time.Hour)
	assert.NotNil(cache)
	sinkManager, err := sinks.NewExternalSinkManager([]sink_api.ExternalSink{}, cache, resolution)
	assert.NoError(err)

	manager, err := manager.NewManager(sources, sinkManager, resolution, time.Hour, cache, true, resolution, resolution)
	assert.NoError(err)
	start := time.Now()

	manager.Start()
	defer manager.Stop()
	time.Sleep(10 * time.Second)

	model := manager.GetModel()
	pods := model.GetPods(testNamespace)
	assert.Equal(podCount, len(pods))

	metrics, _, err := model.GetPodMetric(model_api.PodMetricRequest{
		NamespaceName: testNamespace,
		PodName:       "pod-0",
		MetricRequest: model_api.MetricRequest{
			Start:      start,
			End:        time.Time{},
			MetricName: "cpu-usage",
		},
	})
	assert.NoError(err)
	//TODO: Expect more than 1 metric once #551 is fixed
	assert.NotEmpty(metrics)
	assert.InEpsilon(loadAverageMilli, metrics[0].Value, 50)
}
*/
