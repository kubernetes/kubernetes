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
)

// TestGetClusterMetric tests all flows of GetClusterMetric.
func TestGetClusterMetric(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
		zeroTime     = time.Time{}
		dummyRequest = MetricRequest{
			MetricName: cpuUsage,
			Start:      zeroTime,
			End:        zeroTime,
		}
	)
	// Invocation with no cluster metrics
	res, stamp, err := model.GetClusterMetric(ClusterMetricRequest{
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with non-existant metric
	dummyRequest.MetricName = "doesnotexist"
	res, stamp, err = model.GetClusterMetric(ClusterMetricRequest{
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Normal Invocation - memoryUsage
	dummyRequest.MetricName = memUsage
	res, stamp, err = model.GetClusterMetric(ClusterMetricRequest{
		MetricRequest: dummyRequest,
	})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
}

// TestGetNodeMetric tests all flows of GetNodeMetric.
func TestGetNodeMetric(t *testing.T) {
	var (
		zeroTime     = time.Time{}
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
		hostname     = "hostname3"
		dummyRequest = MetricRequest{
			MetricName: cpuUsage,
			Start:      zeroTime,
			End:        zeroTime,
		}
	)
	// Invocation with no nodes in model
	res, stamp, err := model.GetNodeMetric(NodeMetricRequest{
		NodeName:      hostname,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with non-existant node
	res, stamp, err = model.GetNodeMetric(NodeMetricRequest{
		NodeName:      "doesnotexist",
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with new node - no metrics
	model.addNode("newnode")
	res, stamp, err = model.GetNodeMetric(NodeMetricRequest{
		NodeName:      "newnode",
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant metric
	dummyRequest.MetricName = "doesnotexist"
	res, stamp, err = model.GetNodeMetric(NodeMetricRequest{
		NodeName:      hostname,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Normal Invocation - memoryUsage
	dummyRequest.MetricName = memUsage
	res, stamp, err = model.GetNodeMetric(NodeMetricRequest{
		NodeName:      hostname,
		MetricRequest: dummyRequest,
	})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
}

// TestGetNamespaceMetric tests all flows of GetNamespaceMetric.
func TestGetNamespaceMetric(t *testing.T) {
	var (
		zeroTime     = time.Time{}
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
		namespace    = "test"
		dummyRequest = MetricRequest{
			MetricName: cpuUsage,
			Start:      zeroTime,
			End:        zeroTime,
		}
	)
	// Invocation with no namespaces in model
	res, stamp, err := model.GetNamespaceMetric(NamespaceMetricRequest{
		NamespaceName: namespace,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with non-existant namespace
	res, stamp, err = model.GetNamespaceMetric(NamespaceMetricRequest{
		NamespaceName: "doesnotexist",
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with new namespace - no metrics
	model.addNamespace("newns")
	res, stamp, err = model.GetNamespaceMetric(NamespaceMetricRequest{
		NamespaceName: "newns",
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant metric
	dummyRequest.MetricName = "doesnotexist"
	res, stamp, err = model.GetNamespaceMetric(NamespaceMetricRequest{
		NamespaceName: namespace,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Normal Invocation - memoryUsage
	dummyRequest.MetricName = memUsage
	res, stamp, err = model.GetNamespaceMetric(NamespaceMetricRequest{
		NamespaceName: namespace,
		MetricRequest: dummyRequest,
	})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
}

// TestGetPodMetric tests all flows of GetPodMetric.
func TestGetPodMetric(t *testing.T) {
	var (
		zeroTime     = time.Time{}
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
		namespace    = "test"
		pod          = "pod1"
		dummyRequest = MetricRequest{
			MetricName: cpuUsage,
			Start:      zeroTime,
			End:        zeroTime,
		}
	)
	// Invocation with no namespaces in model
	res, stamp, err := model.GetPodMetric(PodMetricRequest{
		NamespaceName: namespace,
		PodName:       pod,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with non-existant namespace
	res, stamp, err = model.GetPodMetric(PodMetricRequest{
		NamespaceName: "doesnotexist",
		PodName:       pod,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant pod
	res, stamp, err = model.GetPodMetric(PodMetricRequest{
		NamespaceName: namespace,
		PodName:       "otherpod",
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant metric
	dummyRequest.MetricName = "doesnotexist"
	res, stamp, err = model.GetPodMetric(PodMetricRequest{
		NamespaceName: namespace,
		PodName:       pod,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Normal Invocation - memoryUsage
	dummyRequest.MetricName = memUsage
	res, stamp, err = model.GetPodMetric(PodMetricRequest{
		NamespaceName: namespace,
		PodName:       pod,
		MetricRequest: dummyRequest,
	})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
}

// TestGetBatchPodMetric tests all flows of GetBatchPodMetric.
func TestGetBatchPodMetric(t *testing.T) {
	var (
		zeroTime     = time.Time{}
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
		namespace    = "test"
		pod          = "pod1"
	)

	// Invocation with no namespaces in model
	res, stamp, err := model.GetBatchPodMetric(BatchPodRequest{namespace, []string{pod}, cpuUsage, zeroTime, zeroTime})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with non-existant namespace
	res, stamp, err = model.GetBatchPodMetric(BatchPodRequest{"doesnotexist", []string{pod}, cpuUsage, zeroTime, zeroTime})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant pod
	res, stamp, err = model.GetBatchPodMetric(BatchPodRequest{namespace, []string{"otherpod"}, cpuUsage, zeroTime, zeroTime})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
	assert.Equal(len(res[0]), 0)

	// Invocation with non-existant metric
	res, stamp, err = model.GetBatchPodMetric(BatchPodRequest{namespace, []string{pod}, "doesnotexist", zeroTime, zeroTime})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
	assert.Equal(len(res[0]), 0)

	// Normal Invocation - memoryUsage
	res, stamp, err = model.GetBatchPodMetric(BatchPodRequest{namespace, []string{pod}, memUsage, zeroTime, zeroTime})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
	assert.Equal(len(res[0]), 8)
}

// TestGetPodContainerMetric tests all flows of GetPodContainerMetric.
func TestGetPodContainerMetric(t *testing.T) {
	var (
		zeroTime     = time.Time{}
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
		namespace    = "test"
		pod          = "pod1"
		container    = "container1"
		dummyRequest = MetricRequest{
			MetricName: cpuUsage,
			Start:      zeroTime,
			End:        zeroTime,
		}
	)
	// Invocation with no namespaces in model
	res, stamp, err := model.GetPodContainerMetric(PodContainerMetricRequest{
		NamespaceName: namespace,
		PodName:       pod,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with non-existant namespace
	res, stamp, err = model.GetPodContainerMetric(PodContainerMetricRequest{
		NamespaceName: "doesnotexist",
		PodName:       pod,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with new namespace - no metrics
	model.addNamespace("newns")
	res, stamp, err = model.GetPodContainerMetric(PodContainerMetricRequest{
		NamespaceName: "newns",
		PodName:       pod,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant pod
	res, stamp, err = model.GetPodContainerMetric(PodContainerMetricRequest{
		NamespaceName: namespace,
		PodName:       "otherpod",
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant container
	res, stamp, err = model.GetPodContainerMetric(PodContainerMetricRequest{
		NamespaceName: namespace,
		PodName:       pod,
		ContainerName: "doesnotexist",
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant metric
	dummyRequest.MetricName = "doesnotexist"
	res, stamp, err = model.GetPodContainerMetric(PodContainerMetricRequest{
		NamespaceName: namespace,
		PodName:       pod,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Normal Invocation - memoryUsage
	dummyRequest.MetricName = memUsage
	res, stamp, err = model.GetPodContainerMetric(PodContainerMetricRequest{
		NamespaceName: namespace,
		PodName:       pod,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
}

// TestGetFreeContainerMetric tests all flows of GetFreeContainerMetric.
func TestGetFreeContainerMetric(t *testing.T) {
	var (
		zeroTime     = time.Time{}
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
		node         = "hostname2"
		container    = "free_container1"
		dummyRequest = MetricRequest{
			MetricName: cpuUsage,
			Start:      zeroTime,
			End:        zeroTime,
		}
	)
	// Invocation with no nodes in model
	res, stamp, err := model.GetFreeContainerMetric(FreeContainerMetricRequest{
		NodeName:      node,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with non-existant node
	res, stamp, err = model.GetFreeContainerMetric(FreeContainerMetricRequest{
		NodeName:      "doesnotexist",
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with new node - no metrics
	model.addNode("newnode")
	res, stamp, err = model.GetFreeContainerMetric(FreeContainerMetricRequest{
		NodeName:      "newnode",
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant container
	res, stamp, err = model.GetFreeContainerMetric(FreeContainerMetricRequest{
		NodeName:      node,
		ContainerName: "not_actual_ctr",
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Invocation with non-existant metric
	dummyRequest.MetricName = "doesnotexist"
	res, stamp, err = model.GetFreeContainerMetric(FreeContainerMetricRequest{
		NodeName:      node,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.Error(err)
	assert.Equal(stamp, zeroTime)
	assert.Nil(res)

	// Normal Invocation - memoryUsage
	dummyRequest.MetricName = memUsage
	res, stamp, err = model.GetFreeContainerMetric(FreeContainerMetricRequest{
		NodeName:      node,
		ContainerName: container,
		MetricRequest: dummyRequest,
	})
	assert.NoError(err)
	assert.NotEqual(stamp, zeroTime)
	assert.NotNil(res)
}

//TestGetNodes tests the flow of GetNodes.
func TestGetNodes(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	// Invocation with empty model
	res := model.GetNodes()
	assert.Len(res, 0)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Normal Invocation
	res = model.GetNodes()
	assert.Len(res, 2)
}

//TestGetNodePods tests the flow of GetNodePods.
func TestGetNodePods(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	// Populate model
	assert.NoError(model.Update(source_cache))

	// Invocation with nonexistant node
	res := model.GetNodePods("nonexistant")
	assert.Len(res, 0)

	// Normal Invocation
	res = model.GetNodePods("hostname2")
	assert.Len(res, 1)
}

//TestGetNamespaces tests the flow of GetNamespaces.
func TestGetNamespaces(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	// Invocation with empty model
	res := model.GetNamespaces()
	assert.Len(res, 0)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Normal Invocation
	res = model.GetNamespaces()
	assert.Len(res, 1)
}

//TestGetPods tests the flow of GetPods.
func TestGetPods(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	// Invocation with empty model
	res := model.GetPods("test")
	assert.Len(res, 0)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Normal Invocation
	res = model.GetPods("test")
	assert.Len(res, 2)

	// Invocation with non-existant namespace
	res = model.GetPods("fakenamespace")
	assert.Len(res, 0)
}

//TestGetPodContainers tests the flow of GetPodContainers.
func TestGetPodContainers(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	// Invocation with empty model
	res := model.GetPodContainers("test", "pod1")
	assert.Len(res, 0)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Normal Invocation
	res = model.GetPodContainers("test", "pod1")
	assert.Len(res, 2)

	// Invocation with non-existant namespace
	res = model.GetPodContainers("fail", "pod1")
	assert.Len(res, 0)

	// Invocation with non-existant pod
	res = model.GetPodContainers("test", "pod5")
	assert.Len(res, 0)
}

//TestGetFreeContainers tests the flow of GetFreeContainers.
func TestGetFreeContainers(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	// Invocation with empty model
	res := model.GetFreeContainers("hostname2")
	assert.Len(res, 0)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Normal Invocation
	res = model.GetFreeContainers("hostname2")
	assert.Len(res, 1)

	// Invocation with non-existant node
	res = model.GetFreeContainers("hostname9")
	assert.Len(res, 0)
}

// TestGetAvailableMetrics tests the flow of GetAvailableMetrics.
func TestGetAvailableMetrics(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	// Invocation with no model metrics
	res := model.GetAvailableMetrics()
	assert.Len(res, 0)

	// Populate model
	assert.NoError(model.Update(source_cache))

	// Normal Invocation
	res = model.GetAvailableMetrics()
	assert.Len(res, 7)
}

// TestGetClusterStats tests all flows of GetClusterStats.
func TestGetClusterStats(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)

	// Invocation with no cluster stats
	res, err := model.GetClusterStats()
	assert.Len(res.ByName, 0)
	assert.Equal(res.Uptime, time.Duration(0))
	assert.NoError(err)

	// Invocation with cluster stats
	model.Update(source_cache)
	res, err = model.GetClusterStats()
	assert.True(len(res.ByName) >= 6)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.Max)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.NinetyFifth)
	assert.Equal(res.ByName[memUsage].Minute.Average, uint64(10000))
	assert.Equal(res.ByName[memUsage].Hour.Max, 2*memUsageEpsilon)
	assert.Equal(res.ByName[memUsage].Hour.Average, 2*memUsageEpsilon)
	assert.Equal(res.ByName[memUsage].Hour.NinetyFifth, 2*memUsageEpsilon)
	assert.NotEqual(res.Uptime, time.Duration(0))
	assert.NoError(err)
}

// TestGetNodeStats tests all flows of GetNodeStats.
func TestGetNodeStats(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	model.Update(source_cache)

	// Invocation with nonexistant node
	res, err := model.GetNodeStats(NodeRequest{
		NodeName: "nonexistant",
	})
	assert.Nil(res)
	assert.Error(err)

	// Invocation with normal node
	res, err = model.GetNodeStats(NodeRequest{
		NodeName: "hostname2",
	})
	assert.True(len(res.ByName) >= 6)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.Max)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.NinetyFifth)
	assert.NotEqual(res.ByName[memUsage].Minute.Average, uint64(0))
	assert.NoError(err)
}

// TestGetNamespaceStats tests all flows of GetNamespaceStats.
func TestGetNamespaceStats(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	model.Update(source_cache)

	// Invocation with nonexistant namespace
	res, err := model.GetNamespaceStats(NamespaceRequest{
		NamespaceName: "nonexistant",
	})
	assert.Nil(res)
	assert.Error(err)

	// Invocation with normal namespace
	res, err = model.GetNamespaceStats(NamespaceRequest{
		NamespaceName: "test",
	})
	assert.True(len(res.ByName) >= 6)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.Max)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.NinetyFifth)
	assert.NotEqual(res.ByName[memUsage].Minute.Average, uint64(0))
	assert.NotEqual(res.ByName[memUsage].Hour.Average, uint64(0))
	assert.NoError(err)
}

// TestGetPodStats tests all flows of GetPodStats.
func TestGetPodStats(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	model.Update(source_cache)

	// Invocation with nonexistant namespace
	res, err := model.GetPodStats(PodRequest{
		NamespaceName: "nonexistant",
		PodName:       "pod1",
	})
	assert.Nil(res)
	assert.Error(err)

	// Invocation with normal namespace and nonexistant pod
	res, err = model.GetPodStats(PodRequest{
		NamespaceName: "test",
		PodName:       "nonexistant",
	})
	assert.Nil(res)
	assert.Error(err)

	// Normal Invocation
	res, err = model.GetPodStats(PodRequest{
		NamespaceName: "test",
		PodName:       "pod1",
	})
	assert.True(len(res.ByName) >= 6)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.Max)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.NinetyFifth)
	assert.NotEqual(res.ByName[memUsage].Minute.Average, uint64(0))
	assert.NotEqual(res.ByName[memUsage].Hour.Average, uint64(0))

	assert.Equal(res.ByName[cpuLimit].Minute.Average, res.ByName[cpuLimit].Minute.Max)
	assert.Equal(res.ByName[cpuLimit].Minute.Average, res.ByName[cpuLimit].Minute.NinetyFifth)
	assert.NotEqual(res.ByName[cpuLimit].Minute.Average, uint64(0))
	assert.NotEqual(res.ByName[cpuLimit].Hour.Average, uint64(0))
	assert.NoError(err)
}

// TestGetPodContainerStats tests all flows of GetPodContainerStats.
func TestGetPodContainerStats(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	model.Update(source_cache)

	// Invocation with nonexistant namespace
	res, err := model.GetPodContainerStats(PodContainerRequest{
		NamespaceName: "nonexistant",
		PodName:       "pod1",
		ContainerName: "container1",
	})
	assert.Nil(res)
	assert.Error(err)

	// Invocation with normal namespace and nonexistant pod
	res, err = model.GetPodContainerStats(PodContainerRequest{
		NamespaceName: "test",
		PodName:       "nonexistant",
		ContainerName: "container1",
	})
	assert.Nil(res)
	assert.Error(err)

	// Invocation with normal namespace and pod, but nonexistant container
	res, err = model.GetPodContainerStats(PodContainerRequest{
		NamespaceName: "test",
		PodName:       "pod1",
		ContainerName: "nonexistant",
	})
	assert.Nil(res)
	assert.Error(err)

	// Normal Invocation
	res, err = model.GetPodContainerStats(PodContainerRequest{
		NamespaceName: "test",
		PodName:       "pod1",
		ContainerName: "container1",
	})
	assert.True(len(res.ByName) >= 6)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.Max)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.NinetyFifth)
	assert.NotEqual(res.ByName[memUsage].Minute.Average, uint64(0))
	assert.NotEqual(res.ByName[memUsage].Minute.Max, uint64(0))
	assert.NotEqual(res.ByName[memUsage].Minute.NinetyFifth, uint64(0))
	assert.NoError(err)
}

// TestGetFreeContainerStats tests all flows of GetFreeContainerStats.
func TestGetFreeContainerStats(t *testing.T) {
	var (
		model        = newRealModel(time.Minute)
		source_cache = cacheFactory()
		assert       = assert.New(t)
	)
	model.Update(source_cache)

	// Invocation with nonexistant node
	res, err := model.GetFreeContainerStats(FreeContainerRequest{
		NodeName:      "nonexistant",
		ContainerName: "free_container1",
	})
	assert.Nil(res)
	assert.Error(err)

	// Invocation with normal node and nonexistant container
	res, err = model.GetFreeContainerStats(FreeContainerRequest{
		NodeName:      "hostname2",
		ContainerName: "nonexistant",
	})
	assert.Nil(res)
	assert.Error(err)

	// Normal Invocation
	res, err = model.GetFreeContainerStats(FreeContainerRequest{
		NodeName:      "hostname2",
		ContainerName: "free_container1",
	})
	assert.True(len(res.ByName) >= 6)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.Max)
	assert.Equal(res.ByName[memUsage].Minute.Average, res.ByName[memUsage].Minute.NinetyFifth)
	assert.NotEqual(res.ByName[memUsage].Minute.Average, uint64(0))
	assert.NotEqual(res.ByName[memUsage].Hour.Average, uint64(0))
	assert.NoError(err)
}
