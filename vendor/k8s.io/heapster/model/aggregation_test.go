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

package model

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/heapster/store/daystore"
	"k8s.io/heapster/store/statstore"
)

// TestAggregateNodeMetricsEmpty tests the empty flow of aggregateNodeMetrics.
// The normal flow is tested through TestUpdate.
func TestAggregateNodeMetricsEmpty(t *testing.T) {
	var (
		model  = newRealModel(time.Minute)
		c      = make(chan error)
		assert = assert.New(t)
	)

	// Invocation with empty model
	go model.aggregateNodeMetrics(c, time.Now())
	assert.NoError(<-c)
	assert.Empty(model.Nodes)
	assert.Empty(model.Metrics)
}

// TestAggregateKubeMetricsError tests the error flows of aggregateKubeMetrics.
// The normal flow is tested through TestUpdate.
func TestAggregateKubeMetricsEmpty(t *testing.T) {
	var (
		model  = newRealModel(time.Minute)
		c      = make(chan error)
		assert = assert.New(t)
	)

	// Invocation with empty model
	go model.aggregateKubeMetrics(c, time.Now())
	assert.NoError(<-c)
	assert.Empty(model.Namespaces)

	// Error propagation from aggregateNamespaceMetrics
	model.Namespaces["default"] = nil
	go model.aggregateKubeMetrics(c, time.Now())
	assert.Error(<-c)
	assert.NotEmpty(model.Namespaces)
}

// TestAggregateNamespaceMetrics tests the error flows of aggregateNamespaceMetrics.
// The normal flow is tested through TestUpdate.
func TestAggregateNamespaceMetricsError(t *testing.T) {
	var (
		model  = newRealModel(time.Minute)
		c      = make(chan error)
		assert = assert.New(t)
	)

	// Invocation with nil namespace
	go model.aggregateNamespaceMetrics(nil, c, time.Now())
	assert.Error(<-c)
	assert.Empty(model.Namespaces)

	// Invocation for a namespace with no pods
	ns := model.addNamespace("default")
	go model.aggregateNamespaceMetrics(ns, c, time.Now())
	assert.NoError(<-c)
	assert.Len(ns.Pods, 0)

	// Error propagation from aggregatePodMetrics
	ns.Pods["pod1"] = nil
	go model.aggregateNamespaceMetrics(ns, c, time.Now())
	assert.Error(<-c)
}

// TestAggregatePodMetricsError tests the error flows of aggregatePodMetrics.
// The normal flow is tested through TestUpdate.
func TestAggregatePodMetricsError(t *testing.T) {
	var (
		model  = newRealModel(time.Minute)
		c      = make(chan error)
		assert = assert.New(t)
	)
	ns := model.addNamespace("default")
	node := model.addNode("newnode")
	pod := model.addPod("pod1", "uid111", ns, node)

	// Invocation with nil pod
	go model.aggregatePodMetrics(nil, c, time.Now())
	assert.Error(<-c)

	// Invocation with empty pod
	go model.aggregatePodMetrics(pod, c, time.Now())
	assert.NoError(<-c)

	// Invocation with a normal pod
	addContainerToMap("new_container", pod.Containers)
	addContainerToMap("new_container2", pod.Containers)
	go model.aggregatePodMetrics(pod, c, time.Now())
	assert.NoError(<-c)
}

// TestAggregateMetricsError tests the error flows of aggregateMetrics.
func TestAggregateMetricsError(t *testing.T) {
	var (
		model      = newRealModel(time.Minute)
		targetInfo = InfoType{
			Metrics: make(map[string]*daystore.DayStore),
			Labels:  make(map[string]string),
		}
		srcInfo = InfoType{
			Metrics: make(map[string]*daystore.DayStore),
			Labels:  make(map[string]string),
		}
		assert = assert.New(t)
	)

	// Invocation with nil first argument
	sources := []*InfoType{&srcInfo}
	assert.Error(model.aggregateMetrics(nil, sources, time.Now()))

	// Invocation with empty second argument
	sources = []*InfoType{}
	assert.Error(model.aggregateMetrics(&targetInfo, sources, time.Now()))

	// Invocation with a nil element in the second argument
	sources = []*InfoType{&srcInfo, nil}
	assert.Error(model.aggregateMetrics(&targetInfo, sources, time.Now()))

	// Invocation with the target being also part of sources
	sources = []*InfoType{&srcInfo, &targetInfo}
	assert.Error(model.aggregateMetrics(&targetInfo, sources, time.Now()))

	// Normal Invocation with latestTime being zero
	sources = []*InfoType{&srcInfo}
	assert.Error(model.aggregateMetrics(&targetInfo, sources, time.Time{}))
}

// TestAggregateMetricsNormal tests the normal flows of aggregateMetrics.
func TestAggregateMetricsNormal(t *testing.T) {
	var (
		model      = newRealModel(time.Minute)
		targetInfo = InfoType{
			Metrics: make(map[string]*daystore.DayStore),
			Labels:  make(map[string]string),
		}
		srcInfo1 = InfoType{
			Metrics: make(map[string]*daystore.DayStore),
			Labels:  make(map[string]string),
		}
		srcInfo2 = InfoType{
			Metrics: make(map[string]*daystore.DayStore),
			Labels:  make(map[string]string),
		}
		now     = time.Now().Round(time.Minute)
		assert  = assert.New(t)
		require = require.New(t)
	)

	newTS := newDayStore()
	newTS.Put(statstore.TimePoint{
		Timestamp: now,
		Value:     uint64(5000),
	})
	newTS.Put(statstore.TimePoint{
		Timestamp: now.Add(20 * time.Minute),
		Value:     uint64(3000),
	})
	newTS.Put(statstore.TimePoint{
		Timestamp: now.Add(50 * time.Minute),
		Value:     uint64(9000),
	})
	newTS2 := newDayStore()
	newTS2.Put(statstore.TimePoint{
		Timestamp: now,
		Value:     uint64(2000),
	})
	newTS2.Put(statstore.TimePoint{
		Timestamp: now.Add(20 * time.Minute),
		Value:     uint64(3500),
	})
	newTS2.Put(statstore.TimePoint{
		Timestamp: now.Add(40 * time.Minute),
		Value:     uint64(9000),
	})
	newTS2.Put(statstore.TimePoint{
		Timestamp: now.Add(50 * time.Minute),
		Value:     uint64(9000),
	})

	// Use a dummy metric to round values to the defaultEpsilon
	srcInfo1.Metrics["dummy"] = newTS
	srcInfo2.Metrics["dummy"] = newTS2
	sources := []*InfoType{&srcInfo1, &srcInfo2}

	// Normal Invocation
	model.aggregateMetrics(&targetInfo, sources, now.Add(50*time.Minute))

	assert.NotNil(targetInfo.Metrics["dummy"])
	targetMemTS := targetInfo.Metrics["dummy"]
	res := targetMemTS.Hour.Get(time.Time{}, time.Time{})

	require.Len(res, 51)
	assert.Equal(res[0].Value, uint64(18000))
	assert.Equal(res[1].Value, uint64(12000))
	assert.Equal(res[10].Value, uint64(12000))
	assert.Equal(res[11].Value, uint64(6500))
	assert.Equal(res[30].Value, uint64(6500))
	assert.Equal(res[31].Value, uint64(7000))
	assert.Equal(res[50].Value, uint64(7000))
}
