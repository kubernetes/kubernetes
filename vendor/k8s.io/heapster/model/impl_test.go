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
	"math/rand"
	"strings"
	"testing"
	"time"

	cadvisor "github.com/google/cadvisor/info/v1"
	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/heapster/sinks/cache"
	source_api "k8s.io/heapster/sources/api"

	"k8s.io/heapster/store/daystore"
	"k8s.io/heapster/store/statstore"
)

func newDayStore() *daystore.DayStore {
	return daystore.NewDayStore(defaultEpsilon, time.Minute)
}

// TestNewCluster tests the sanity of NewCluster
func TestNewCluster(t *testing.T) {
	cluster := NewModel(time.Minute)
	assert.NotNil(t, cluster)
}

// TestAddNamespace tests all flows of addNamespace.
func TestAddNamespace(t *testing.T) {
	var (
		cluster        = newRealModel(time.Minute)
		namespace_name = "default"
		assert         = assert.New(t)
	)

	// First call : namespace does not exist
	namespace := cluster.addNamespace(namespace_name)

	assert.NotNil(namespace)
	assert.Equal(cluster.Namespaces[namespace_name], namespace)
	assert.NotNil(namespace.Metrics)
	assert.NotNil(namespace.Labels)
	assert.NotNil(namespace.Pods)

	// Second call : namespace already exists
	new_namespace := cluster.addNamespace(namespace_name)
	assert.Equal(new_namespace, namespace)
}

// TestDeleteNamespace tests all flows of deleteNode.
func TestDeleteNamespace(t *testing.T) {
	var (
		cluster   = newRealModel(time.Minute)
		hostname  = "kubernetes-minion-xkhz"
		namespace = "testns"
		podName   = "mypod"
		assert    = assert.New(t)
	)

	// First call : node does not exist
	cluster.addNode("other")
	otherNs := cluster.addNamespace("other")
	nodeInfo := cluster.addNode(hostname)
	namespaceInfo := cluster.addNamespace(namespace)
	cluster.addPod(podName, "123", namespaceInfo, nodeInfo)
	cluster.addPod(podName, "123", otherNs, nodeInfo)

	cluster.deleteNamespace(namespace)

	assert.Equal(2, len(cluster.Nodes))
	assert.Equal(1, len(cluster.Namespaces))
	assert.NotNil(cluster.Nodes[hostname])
	assert.Equal(1, len(cluster.Nodes[hostname].Pods))
	assert.Nil(cluster.Namespaces[namespace])

	// Second call: already deleted
	cluster.deleteNamespace(namespace)
	// No panic etc.
}

// TestAddNode tests all flows of addNode.
func TestAddNode(t *testing.T) {
	var (
		cluster  = newRealModel(time.Minute)
		hostname = "kubernetes-minion-xkhz"
		assert   = assert.New(t)
	)

	// First call : node does not exist
	node := cluster.addNode(hostname)

	assert.NotNil(node)
	assert.Equal(cluster.Nodes[hostname], node)
	assert.NotNil(node.Metrics)
	assert.NotNil(node.Labels)
	assert.NotNil(node.FreeContainers)
	assert.NotNil(node.Pods)

	// Second call : node already exists
	new_node := cluster.addNode(hostname)
	assert.Equal(new_node, node)
}

// TestDeleteNode tests all flows of deleteNode.
func TestDeleteNode(t *testing.T) {
	var (
		cluster   = newRealModel(time.Minute)
		hostname  = "kubernetes-minion-xkhz"
		namespace = "testns"
		podName   = "mypod"
		assert    = assert.New(t)
	)

	// First call : node does not exist
	cluster.addNode("other")
	cluster.addNamespace("other")

	nodeInfo := cluster.addNode(hostname)
	namespaceInfo := cluster.addNamespace(namespace)
	cluster.addPod(podName, "123", namespaceInfo, nodeInfo)

	cluster.deleteNode(hostname)

	assert.Equal(0, len(namespaceInfo.Pods))
	assert.Equal(1, len(cluster.Nodes))
	assert.Equal(2, len(cluster.Namespaces))
	assert.Nil(cluster.Nodes[hostname])
	assert.NotNil(cluster.Namespaces[namespace])

	// Second call: already deleted
	cluster.deleteNode(hostname)
	// No panic etc.
}

// TestAddPod tests all flows of addPod.
func TestAddPod(t *testing.T) {
	var (
		cluster   = newRealModel(time.Minute)
		pod_name  = "podname-xkhz"
		pod_uid   = "123124-124124-124124124124"
		namespace = cluster.addNamespace("default")
		node      = cluster.addNode("kubernetes-minion-xkhz")
		assert    = assert.New(t)
	)

	// First call : pod does not exist
	pod := cluster.addPod(pod_name, pod_uid, namespace, node)

	assert.NotNil(pod)
	assert.Equal(node.Pods[getPodKey(namespace.Name, pod_name)], pod)
	assert.Equal(namespace.Pods[pod_name], pod)
	assert.Equal(pod.UID, pod_uid)
	assert.NotNil(pod.Metrics)
	assert.NotNil(pod.Labels)
	assert.NotNil(pod.Containers)

	// Second call : pod already exists
	new_pod := cluster.addPod(pod_name, pod_uid, namespace, node)
	assert.NotNil(new_pod)
	assert.Equal(new_pod, pod)

	// Third call : Nil namespace
	new_pod = cluster.addPod(pod_name, pod_uid, nil, node)
	assert.Nil(new_pod)

	// Fourth call : Nil node
	new_pod = cluster.addPod(pod_name, pod_uid, namespace, nil)
	assert.Nil(new_pod)
}

// TestDeleteNode tests all flows of deleteNode.
func TestDeletePod(t *testing.T) {
	var (
		cluster   = newRealModel(time.Minute)
		hostname  = "kubernetes-minion-xkhz"
		namespace = "testns"
		podName   = "mypod"
		assert    = assert.New(t)
	)

	// First call : node does not exist
	cluster.addNode("other")
	cluster.addNamespace("other")

	nodeInfo := cluster.addNode(hostname)
	namespaceInfo := cluster.addNamespace(namespace)
	cluster.addPod(podName, "123", namespaceInfo, nodeInfo)
	cluster.addPod("other", "123", namespaceInfo, nodeInfo)

	cluster.deletePod(namespace, podName)

	assert.Equal(1, len(namespaceInfo.Pods))
	assert.Equal(2, len(cluster.Nodes))
	assert.Equal(2, len(cluster.Namespaces))
	assert.Nil(namespaceInfo.Pods[podName])
	assert.Nil(nodeInfo.Pods[getPodKey(namespace, podName)])
	assert.NotNil(cluster.Namespaces[namespace])

	// Second call: already deleted
	cluster.deletePod(namespace, podName)
	// No panic etc.
}

// TestUpdateTime tests the sanity of updateTime.
func TestUpdateTime(t *testing.T) {
	var (
		cluster = newRealModel(time.Minute)
		stamp   = time.Now()
	)

	// First call: update with non-zero time
	cluster.updateTime(stamp)
	assert.Equal(t, cluster.timestamp, stamp)

	// Second call: update with zero time
	cluster.updateTime(time.Time{})
	assert.Equal(t, cluster.timestamp, stamp)
}

// Tests the flow of AddMetricToMap where the metric name is present in the map
func TestAddMetricToMapExistingKey(t *testing.T) {
	var (
		cluster         = newRealModel(time.Minute)
		metrics         = make(map[string]*daystore.DayStore)
		new_metric_name = "name_already_in_map"
		value           = uint64(1234567890)
		zeroTime        = time.Time{}
		stamp           = time.Now().Round(time.Minute)
		assert          = assert.New(t)
		require         = require.New(t)
	)

	// Fist Call: addMetricToMap for a new metric
	assert.NoError(cluster.addMetricToMap(new_metric_name, stamp, value, metrics))

	ts := *metrics[new_metric_name]
	results := ts.Hour.Get(zeroTime, zeroTime)
	require.Len(results, 1)

	// Second Call: addMetricToMap for an existing key, same time
	new_value := uint64(1234567890)
	assert.NoError(cluster.addMetricToMap(new_metric_name, stamp, new_value, metrics))

	require.Len(results, 1)

	// Second Call: addMetricToMap for an existing key, new time
	new_value = uint64(617283996)
	later_stamp := stamp.Add(20 * time.Minute)
	assert.NoError(cluster.addMetricToMap(new_metric_name, later_stamp, new_value, metrics))

	ts = *metrics[new_metric_name]
	results = ts.Hour.Get(zeroTime, zeroTime)
	require.Len(results, 21)
	assert.Equal(results[0].Timestamp, stamp.Add(20*time.Minute))
	assert.Equal(roundToEpsilon(results[0].Value, defaultEpsilon), roundToEpsilon(new_value, defaultEpsilon))
	assert.Equal(results[20].Timestamp, stamp)
	assert.Equal(roundToEpsilon(results[20].Value, defaultEpsilon), roundToEpsilon(1234567890, defaultEpsilon))

	// Second Call: addMetricToMap for an existing key, same time
	assert.NoError(cluster.addMetricToMap(new_metric_name, later_stamp, new_value, metrics))

	ts = *metrics[new_metric_name]
	results = ts.Hour.Get(zeroTime, zeroTime)
	require.Len(results, 21)
	assert.Equal(results[0].Timestamp, stamp.Add(20*time.Minute))
	resVal := roundToEpsilon(results[0].Value, defaultEpsilon)
	assert.Equal(resVal, roundToEpsilon(new_value, defaultEpsilon))

	assert.Equal(results[1].Timestamp, stamp.Add(19*time.Minute))
	resVal = roundToEpsilon(results[1].Value, defaultEpsilon)
	assert.Equal(resVal, roundToEpsilon(1234567890, defaultEpsilon))

	assert.Equal(results[20].Timestamp, stamp)
	resVal = roundToEpsilon(results[20].Value, defaultEpsilon)
	assert.Equal(resVal, roundToEpsilon(1234567890, defaultEpsilon))

	// Third Call: addMetricToMap for an existing key in the distant future
	stamp = later_stamp
	later_stamp = stamp.Add(14 * time.Hour)
	new_value = uint64(1234567890)
	assert.NoError(cluster.addMetricToMap(new_metric_name, later_stamp, new_value, metrics))

	ts = *metrics[new_metric_name]
	results = ts.Hour.Get(zeroTime, zeroTime)
	require.Len(results, 61) // One full hour of data
	assert.Equal(results[0].Timestamp, stamp.Add(14*time.Hour))
	resVal = roundToEpsilon(results[0].Value, defaultEpsilon)
	assert.Equal(resVal, roundToEpsilon(1234567890, defaultEpsilon))

	assert.Equal(results[1].Timestamp, stamp.Add(13*time.Hour).Add(59*time.Minute))
	resVal = roundToEpsilon(results[1].Value, defaultEpsilon)
	assert.Equal(resVal, roundToEpsilon(617283996, defaultEpsilon))

	assert.Equal(results[60].Timestamp, stamp.Add(13*time.Hour))
	assert.Equal(results[60].Value, uint64(roundToEpsilon(617283996, defaultEpsilon)))

	// Fourth Call: addMetricToMap for an existing key, cause TimeStore failure
	assert.Error(cluster.addMetricToMap(new_metric_name, zeroTime, new_value, metrics))
}

// Tests the flow of AddMetricToMap where the metric name is not present in the map
func TestAddMetricToMapNewKey(t *testing.T) {
	var (
		cluster         = newRealModel(time.Minute)
		metrics         = make(map[string]*daystore.DayStore)
		new_metric_name = "name_not_in_map"
		stamp           = time.Now().Round(cluster.resolution)
		zeroTime        = time.Time{}
		value           = uint64(1234567890)
		assert          = assert.New(t)
		require         = require.New(t)
	)

	// First Call: Add a new metric to the map
	assert.NoError(cluster.addMetricToMap(new_metric_name, stamp, value, metrics))

	// Second Call: Add a second metric to the map on a later time. Flushes previous metric.
	assert.NoError(cluster.addMetricToMap(new_metric_name, stamp.Add(time.Minute), value, metrics))
	new_ts := *metrics[new_metric_name]
	results := new_ts.Hour.Get(zeroTime, zeroTime)
	require.Len(results, 2)
	assert.Equal(results[0].Timestamp, stamp.Add(time.Minute))
	assert.Equal(roundToEpsilon(results[0].Value, defaultEpsilon), roundToEpsilon(value, defaultEpsilon))
	assert.Equal(results[1].Timestamp, stamp)
	assert.Equal(roundToEpsilon(results[1].Value, defaultEpsilon), roundToEpsilon(value, defaultEpsilon))

	// Second Call: addMetricToMap for a new key, cause TimeStore failure
	assert.NoError(cluster.addMetricToMap("other_metric", stamp, value, metrics))
	assert.Error(cluster.addMetricToMap("other_metric", stamp.Add(-time.Minute), value, metrics))
}

// TestParseMetricError tests the error flows of ParseMetric
func TestParseMetricError(t *testing.T) {
	var (
		cluster = newRealModel(time.Minute)
		context = make(map[string]*statstore.TimePoint)
		dict    = make(map[string]*daystore.DayStore)
		cme     = cmeFactory()
		assert  = assert.New(t)
	)

	// Invoke parseMetric with a nil cme argument
	stamp, err := cluster.parseMetric(nil, dict, context)
	assert.Error(err)
	assert.Equal(stamp, time.Time{})

	// Invoke parseMetric with a nil dict argument
	stamp, err = cluster.parseMetric(cme, nil, context)
	assert.Error(err)
	assert.Equal(stamp, time.Time{})

	// Invoke parseMetric with a nil context argument
	stamp, err = cluster.parseMetric(cme, dict, nil)
	assert.Error(err)
	assert.Equal(stamp, time.Time{})
}

// roundToEpsilon is a helper function that rounds a value based on a given epsilon.
func roundToEpsilon(value, epsilon uint64) uint64 {
	res := (value / epsilon) * epsilon
	if value%epsilon != 0 {
		res += epsilon
	}
	return res
}

// TestParseMetricNormal tests the normal flow of ParseMetric
func TestParseMetricNormal(t *testing.T) {
	var (
		cluster    = newRealModel(time.Minute)
		zeroTime   = time.Time{}
		metrics    = make(map[string]*daystore.DayStore)
		context    = make(map[string]*statstore.TimePoint)
		normal_cme = cmeFactory()
		other_cme  = cmeFactory()
		flush_cme  = cmeFactory()
		assert     = assert.New(t)
		require    = require.New(t)
	)
	normal_stamp := normal_cme.Stats.Timestamp.Truncate(time.Minute)
	normal_cme.Stats.Cpu.Usage.Total = uint64(1000)

	other_cme.Stats.Timestamp = normal_stamp.Add(3 * time.Minute)
	other_cme.Stats.Cpu.Usage.Total = uint64(360000001000) // 2 stable over 3 minutes in NS
	other_stamp := other_cme.Stats.Timestamp.Truncate(time.Minute)

	// Create a CME that will flush other_cme to the store
	flush_cme.Stats.Timestamp = normal_stamp.Add(4 * time.Minute)
	flush_cme.Stats.Cpu.Usage.Total = uint64(760000001000)
	flush_stamp := flush_cme.Stats.Timestamp.Truncate(time.Minute)

	// Three Normal Invocations
	stamp, err := cluster.parseMetric(normal_cme, metrics, context)
	assert.NoError(err)
	assert.Equal(stamp, normal_stamp)
	stamp, err = cluster.parseMetric(other_cme, metrics, context)
	assert.NoError(err)
	assert.Equal(stamp, other_stamp)
	stamp, err = cluster.parseMetric(flush_cme, metrics, context)
	assert.NoError(err)
	assert.Equal(stamp, flush_stamp)
	for key, ts := range metrics {
		actual_ts := *ts
		pointSlice := actual_ts.Hour.Get(zeroTime, zeroTime)
		require.True(len(pointSlice) >= 1)
		switch key {
		case cpuLimit:
			require.Len(pointSlice, 5)
			assert.Equal(pointSlice[1].Timestamp, other_stamp)
			assert.Equal(pointSlice[0].Value, other_cme.Spec.Cpu.Limit*1000/1024)
			for i := 1; i <= 3; i++ {
				assert.Equal(pointSlice[i+1].Timestamp, normal_stamp.Add(time.Duration(3-i)*time.Minute))
				assert.Equal(pointSlice[i+1].Value, normal_cme.Spec.Cpu.Limit*1000/1024)
			}
		case memLimit:
			require.Len(pointSlice, 5)
			assert.Equal(pointSlice[1].Timestamp, other_stamp)
			actualML := roundToEpsilon(pointSlice[0].Value, memLimitEpsilon)
			expectedML := roundToEpsilon(other_cme.Spec.Memory.Limit, memLimitEpsilon)
			assert.Equal(actualML, expectedML)
			for i := 1; i <= 3; i++ {
				assert.Equal(pointSlice[i+1].Timestamp, normal_stamp.Add(time.Duration(3-i)*time.Minute))
				actualML = roundToEpsilon(pointSlice[i+1].Value, memLimitEpsilon)
				expectedML = roundToEpsilon(normal_cme.Spec.Memory.Limit, memLimitEpsilon)
				assert.Equal(actualML, expectedML)
			}
		case cpuUsage:
			require.Len(pointSlice, 2)
			assert.Equal(pointSlice[1].Timestamp, other_stamp)
			assert.Equal(pointSlice[1].Value, 2000) //Two full cores
		case memUsage:
			require.Len(pointSlice, 5)
			assert.Equal(pointSlice[1].Timestamp, other_stamp)
			actualMU := roundToEpsilon(pointSlice[0].Value, memUsageEpsilon)
			expectedMU := roundToEpsilon(other_cme.Stats.Memory.Usage, memUsageEpsilon)
			assert.Equal(actualMU, expectedMU)
			for i := 1; i <= 3; i++ {
				assert.Equal(pointSlice[i+1].Timestamp, normal_stamp.Add(time.Duration(3-i)*time.Minute))
				actualMU := roundToEpsilon(pointSlice[i+1].Value, memUsageEpsilon)
				expectedMU := roundToEpsilon(normal_cme.Stats.Memory.Usage, memUsageEpsilon)
				assert.Equal(actualMU, expectedMU)
			}
		case memWorking:
			require.Len(pointSlice, 5)
			assert.Equal(pointSlice[1].Timestamp, other_stamp)
			actualMWS := roundToEpsilon(pointSlice[0].Value, memWorkingEpsilon)
			expectedMWS := roundToEpsilon(other_cme.Stats.Memory.WorkingSet, memWorkingEpsilon)
			assert.Equal(actualMWS, expectedMWS)
			for i := 1; i <= 3; i++ {
				assert.Equal(pointSlice[i+1].Timestamp, normal_stamp.Add(time.Duration(3-i)*time.Minute))
				actualMWS = roundToEpsilon(pointSlice[i+1].Value, memWorkingEpsilon)
				expectedMWS = roundToEpsilon(normal_cme.Stats.Memory.WorkingSet, memWorkingEpsilon)
				assert.Equal(actualMWS, expectedMWS)
			}
		default:
			// Filesystem or error
			require.Len(pointSlice, 5)
			if strings.Contains(key, "limit") {
				actualFSL := roundToEpsilon(pointSlice[0].Value, fsLimitEpsilon)
				expectedFSL := roundToEpsilon(other_cme.Stats.Filesystem[0].Limit, fsLimitEpsilon)
				assert.Equal(actualFSL, expectedFSL)
			} else if strings.Contains(key, "usage") {
				actualFSU := roundToEpsilon(pointSlice[0].Value, fsUsageEpsilon)
				expectedFSU := roundToEpsilon(other_cme.Stats.Filesystem[0].Usage, fsUsageEpsilon)
				assert.Equal(actualFSU, expectedFSU)
			} else {
				assert.True(false, "Unknown key in resulting metrics slice")
			}
		}
	}
}

// TestUpdateInfoTypeError Tests the error flows of updateInfoType.
func TestUpdateInfoTypeError(t *testing.T) {
	var (
		cluster      = newRealModel(time.Minute)
		new_infotype = newInfoType(nil, nil, nil)
		full_ce      = containerElementFactory(nil)
		assert       = assert.New(t)
	)

	// Invocation with a nil InfoType argument
	stamp, err := cluster.updateInfoType(nil, full_ce)
	assert.Error(err)
	assert.Equal(stamp, time.Time{})

	// Invocation with a nil ContainerElement argument
	stamp, err = cluster.updateInfoType(&new_infotype, nil)
	assert.Error(err)
	assert.Empty(new_infotype.Metrics)
	assert.Equal(stamp, time.Time{})
}

// TestUpdateInfoTypeNormal tests the normal flows of UpdateInfoType.
func TestUpdateInfoTypeNormal(t *testing.T) {
	var (
		cluster      = newRealModel(time.Minute)
		new_cme      = cmeFactory()
		empty_ce     = containerElementFactory([]*cache.ContainerMetricElement{})
		nil_ce       = containerElementFactory([]*cache.ContainerMetricElement{new_cme, nil})
		new_infotype = newInfoType(nil, nil, nil)
		zeroTime     = time.Time{}
		assert       = assert.New(t)
		require      = require.New(t)
	)

	// Invocation with a ContainerElement argument with no CMEs
	stamp, err := cluster.updateInfoType(&new_infotype, empty_ce)
	assert.NoError(err)
	assert.Empty(new_infotype.Metrics)
	assert.Equal(stamp, time.Time{})

	// Invocation with a ContainerElement argument with a nil CME
	stamp, err = cluster.updateInfoType(&new_infotype, nil_ce)
	assert.NoError(err)
	assert.NotEmpty(new_infotype.Metrics)
	assert.NotEqual(stamp, time.Time{})
	require.Len(new_infotype.Metrics, 6) // 6 stats in total - no CPU Usage yet
	for _, metricStore := range new_infotype.Metrics {
		metricSlice := (*metricStore).Hour.Get(zeroTime, zeroTime)
		assert.Len(metricSlice, 1) // One value in the store
	}

	// Invocation with an empty InfoType argument
	// The new ContainerElement adds one TimePoint to each of 7 Metrics
	newer_cme := cmeFactory()
	newer_cme.Stats.Timestamp = new_cme.Stats.Timestamp.Add(10 * time.Minute)
	newer_cme.Stats.Cpu.Usage.Total = new_cme.Stats.Cpu.Usage.Total + uint64(600000000000)
	new_ce := containerElementFactory([]*cache.ContainerMetricElement{newer_cme})
	stamp, err = cluster.updateInfoType(&new_infotype, new_ce)
	assert.NoError(err)
	assert.Empty(new_infotype.Labels)
	assert.NotEqual(stamp, time.Time{})
	require.Len(new_infotype.Metrics, 7) // 7 stats in total
	for key, metricStore := range new_infotype.Metrics {
		metricSlice := (*metricStore).Hour.Get(zeroTime, zeroTime)
		if key == cpuUsage {
			// cpuUsage has 1 value at the newer_cme timestamp
			// That value has not been flushed yet into the DayStore
			assert.Len(metricSlice, 1)
		} else {
			assert.Len(metricSlice, 11) // All other metrics have 11 values, one per elapsed minute
		}
	}

	// Invocation with an existing infotype as argument
	// The new ContainerElement adds two TimePoints to each Metric
	newer_cme2 := cmeFactory()
	newer_cme2.Stats.Timestamp = newer_cme.Stats.Timestamp.Add(10 * time.Minute)
	newer_cme2.Stats.Cpu.Usage.Total = newer_cme.Stats.Cpu.Usage.Total + uint64(3600000000000)
	newer_cme3 := cmeFactory()
	newer_cme3.Stats.Timestamp = newer_cme2.Stats.Timestamp.Add(10 * time.Minute)
	newer_cme3.Stats.Cpu.Usage.Total = newer_cme2.Stats.Cpu.Usage.Total + uint64(600000000000)
	new_ce = containerElementFactory([]*cache.ContainerMetricElement{newer_cme3, newer_cme2})
	stamp, err = cluster.updateInfoType(&new_infotype, new_ce)
	assert.NoError(err)
	assert.Empty(new_infotype.Labels)
	assert.NotEqual(stamp, time.Time{})
	require.Len(new_infotype.Metrics, 7) // 7 stats total
	for key, metricStore := range new_infotype.Metrics {
		metricSlice := (*metricStore).Hour.Get(zeroTime, zeroTime)
		if key == cpuUsage {
			require.Len(metricSlice, 21) // cpuUsage consists of 1+10+10 values.
			assert.Equal(metricSlice[0].Value, 1000)
			assert.Equal(metricSlice[1].Value, 6000)
			assert.Equal(metricSlice[10].Value, 6000)
			assert.Equal(metricSlice[11].Value, 1000)
			assert.Equal(metricSlice[20].Value, 1000)
		} else {
			assert.Len(metricSlice, 31) // All other metrics have 31 values, one per minute
		}
	}
}

// TestUpdateFreeContainer tests the flow of updateFreeContainer
func TestUpdateFreeContainer(t *testing.T) {
	var (
		cluster = newRealModel(time.Minute)
		ce      = containerElementFactory(nil)
		assert  = assert.New(t)
	)

	// Invocation with regular parameters
	stamp, err := cluster.updateFreeContainer(ce)
	assert.NoError(err)
	assert.NotEqual(stamp, time.Time{})
	assert.NotNil(cluster.Nodes[ce.Hostname])
	node := cluster.Nodes[ce.Hostname]
	assert.NotNil(node.FreeContainers[ce.Name])
	container := node.FreeContainers[ce.Name]
	assert.Empty(container.Labels)
	assert.NotEmpty(container.Metrics)
}

// TestUpdatePodContainer tests the flow of updatePodContainer
func TestUpdatePodContainer(t *testing.T) {
	var (
		cluster   = newRealModel(time.Minute)
		namespace = cluster.addNamespace("default")
		node      = cluster.addNode("new_node_xyz")
		pod_ptr   = cluster.addPod("new_pod", "1234-1245-235235", namespace, node)
		ce        = containerElementFactory(nil)
		assert    = assert.New(t)
	)

	// Invocation with regular parameters
	stamp, err := cluster.updatePodContainer(pod_ptr, ce)
	assert.NoError(err)
	assert.NotEqual(stamp, time.Time{})
	assert.NotNil(pod_ptr.Containers[ce.Name])
}

// TestUpdatePodNormal tests the normal flow of updatePod.
func TestUpdatePodNormal(t *testing.T) {
	var (
		cluster  = newRealModel(time.Minute)
		pod_elem = podElementFactory()
		assert   = assert.New(t)
	)
	// Invocation with a regular parameter
	stamp, err := cluster.updatePod(pod_elem)
	assert.NoError(err)
	assert.NotEqual(stamp, time.Time{})
	assert.NotNil(cluster.Nodes[pod_elem.Hostname])
	pod_ptr := cluster.Nodes[pod_elem.Hostname].Pods[getPodKey(pod_elem.Namespace, pod_elem.Name)]
	assert.NotNil(cluster.Namespaces[pod_elem.Namespace])
	assert.Equal(pod_ptr, cluster.Namespaces[pod_elem.Namespace].Pods[pod_elem.Name])
	assert.Equal(pod_ptr.Labels, pod_elem.Labels)
}

// TestUpdatePodError tests the error flow of updatePod.
func TestUpdatePodError(t *testing.T) {
	var (
		cluster = newRealModel(time.Minute)
		assert  = assert.New(t)
	)
	// Invocation with a nil parameter
	stamp, err := cluster.updatePod(nil)
	assert.Error(err)
	assert.Equal(stamp, time.Time{})
}

// TestUpdateNodeInvalid tests the error flow of updateNode.
func TestUpdateNodeInvalid(t *testing.T) {
	var (
		cluster = newRealModel(time.Minute)
		ce      = containerElementFactory(nil)
		assert  = assert.New(t)
	)

	// Invocation with a ContainerElement that is not "machine"-tagged
	stamp, err := cluster.updateNode(ce)
	assert.Error(err)
	assert.Equal(stamp, time.Time{})
}

// TestUpdateNodeNormal tests the normal flow of updateNode.
func TestUpdateNodeNormal(t *testing.T) {
	var (
		cluster = newRealModel(time.Minute)
		ce      = containerElementFactory(nil)
		assert  = assert.New(t)
	)
	ce.Name = "machine"
	ce.Hostname = "dummy-minion-xkz"

	// Invocation with regular parameters
	stamp, err := cluster.updateNode(ce)
	assert.NoError(err)
	assert.NotEqual(stamp, time.Time{})
}

// TestUpdate tests the normal flows of Update.
// TestUpdate performs consecutive calls to Update with both empty and non-empty caches
func TestUpdate(t *testing.T) {
	var (
		cluster      = newRealModel(time.Minute)
		source_cache = cacheFactory()
		empty_cache  = cache.NewCache(24*time.Hour, time.Hour)
		zeroTime     = time.Time{}
		assert       = assert.New(t)
		require      = require.New(t)
	)

	// Invocation with empty cache
	assert.NoError(cluster.Update(empty_cache))
	assert.Empty(cluster.Nodes)
	assert.Empty(cluster.Namespaces)
	assert.Empty(cluster.Metrics)

	// Invocation with regular parameters
	assert.NoError(cluster.Update(source_cache))
	verifyCacheFactoryCluster(&cluster.ClusterInfo, t)

	// Assert Node Metric aggregation
	require.NotEmpty(cluster.Nodes)
	require.NotEmpty(cluster.Metrics)
	require.NotNil(cluster.Metrics[memWorking])
	mem_work_ts := *(cluster.Metrics[memWorking])
	actual := mem_work_ts.Hour.Get(zeroTime, zeroTime)
	require.Len(actual, 6)
	// Datapoint present in both nodes,

	assert.Equal(actual[0].Value, uint64(602+602))
	assert.Equal(actual[1].Value, 2*memWorkingEpsilon)
	assert.Equal(actual[5].Value, 2*memWorkingEpsilon)

	require.NotNil(cluster.Metrics[memUsage])
	mem_usage_ts := *(cluster.Metrics[memUsage])
	actual = mem_usage_ts.Hour.Get(zeroTime, zeroTime)
	require.Len(actual, 6)
	// Datapoint present in only one node, second node's metric is extended
	assert.Equal(actual[0].Value, uint64(10000))
	// Datapoint present in both nodes, added up to 10000
	assert.Equal(actual[1].Value, 2*memWorkingEpsilon)

	// Assert Kubernetes Metric aggregation up to namespaces
	ns := cluster.Namespaces["test"]
	mem_work_ts = *(ns.Metrics[memWorking])
	actual = mem_work_ts.Hour.Get(zeroTime, zeroTime)
	require.Len(actual, 8)
	assert.Equal(actual[0].Value, uint64(2408))

	// Invocation with no fresh data - expect no change in cluster
	assert.NoError(cluster.Update(source_cache))
	verifyCacheFactoryCluster(&cluster.ClusterInfo, t)

	// Invocation with empty cache - expect no change in cluster
	assert.NoError(cluster.Update(empty_cache))
	verifyCacheFactoryCluster(&cluster.ClusterInfo, t)
}

// verifyCacheFactoryCluster performs assertions over a ClusterInfo structure,
// based on the values and structure generated by cacheFactory.
func verifyCacheFactoryCluster(clinfo *ClusterInfo, t *testing.T) {
	assert := assert.New(t)
	assert.NotNil(clinfo.Nodes["hostname2"])
	node2 := clinfo.Nodes["hostname2"]
	assert.NotEmpty(node2.Metrics)
	assert.Len(node2.FreeContainers, 1)
	assert.NotNil(node2.FreeContainers["free_container1"])

	assert.NotNil(clinfo.Nodes["hostname3"])
	node3 := clinfo.Nodes["hostname3"]
	assert.NotEmpty(node3.Metrics)

	assert.NotNil(clinfo.Namespaces["test"])
	namespace := clinfo.Namespaces["test"]

	assert.NotNil(namespace.Pods)
	pod1_ptr := namespace.Pods["pod1"]
	require.NotNil(t, pod1_ptr)
	assert.Equal(pod1_ptr, node2.Pods[getPodKey(namespace.Name, "pod1")])
	assert.Len(pod1_ptr.Containers, 2)
	pod2_ptr := namespace.Pods["pod2"]
	require.NotNil(t, pod2_ptr)
	assert.Equal(pod2_ptr, node3.Pods[getPodKey(namespace.Name, "pod2")])
	assert.Len(pod2_ptr.Containers, 2)
}

// Factory Functions

// cmeFactory generates a complete ContainerMetricElement with fuzzed data.
// CMEs created by cmeFactory contain partially fuzzed stats, aside from hardcoded values for Memory usage.
// The timestamp of the CME is rouded to the current minute and offset by a random number of hours.
func cmeFactory() *cache.ContainerMetricElement {
	f := fuzz.New().NilChance(0).NumElements(1, 1)
	containerSpec := source_api.ContainerSpec{
		ContainerSpec: cadvisor.ContainerSpec{
			CreationTime:  time.Now(),
			HasCpu:        true,
			HasMemory:     true,
			HasNetwork:    true,
			HasFilesystem: true,
			HasDiskIo:     true,
		},
	}
	containerSpec.Cpu.Limit = 1024
	containerSpec.Memory.Limit = 10000000

	// Create a fuzzed ContainerStats struct
	var containerStats source_api.ContainerStats
	f.Fuzz(&containerStats)

	// Standardize timestamp to the current minute plus a random number of hours ([1, 10])
	now_time := time.Now().Round(time.Minute)
	new_time := now_time
	for new_time == now_time {
		new_time = now_time.Add(time.Duration(rand.Intn(10)) * 5 * time.Minute)
	}
	containerStats.Timestamp = new_time
	containerSpec.CreationTime = new_time.Add(-time.Hour)

	// Standardize memory usage and limit to test aggregation
	containerStats.Memory.Usage = uint64(5000)
	containerStats.Memory.WorkingSet = uint64(602)

	// Standardize the device name, usage and limit
	new_fs := cadvisor.FsStats{}
	f.Fuzz(&new_fs)
	new_fs.Device = "/dev/device1"
	new_fs.Usage = 50000
	new_fs.Limit = 100000
	containerStats.Filesystem = []cadvisor.FsStats{new_fs}

	return &cache.ContainerMetricElement{
		Spec:  &containerSpec,
		Stats: &containerStats,
	}
}

// emptyCMEFactory generates an empty ContainerMetricElement.
func emptyCMEFactory() *cache.ContainerMetricElement {
	f := fuzz.New().NilChance(0).NumElements(1, 1)
	containerSpec := source_api.ContainerSpec{
		ContainerSpec: cadvisor.ContainerSpec{
			CreationTime:  time.Now(),
			HasCpu:        false,
			HasMemory:     false,
			HasNetwork:    false,
			HasFilesystem: false,
			HasDiskIo:     false,
		},
	}
	var containerStats source_api.ContainerStats
	f.Fuzz(&containerStats)
	containerStats.Timestamp = time.Now()

	return &cache.ContainerMetricElement{
		Spec:  &containerSpec,
		Stats: &containerStats,
	}
}

// containerElementFactory generates a new ContainerElement.
// The `cmes` argument represents a []*ContainerMetricElement, used for the Metrics field.
// If the `cmes` argument is nil, two ContainerMetricElements are automatically generated.
func containerElementFactory(cmes []*cache.ContainerMetricElement) *cache.ContainerElement {
	var metrics []*cache.ContainerMetricElement
	if cmes == nil {
		// If the argument is nil, generate two CMEs
		cme_1 := cmeFactory()
		cme_2 := cmeFactory()
		if !cme_1.Stats.Timestamp.After(cme_2.Stats.Timestamp) {
			// Ensure random but different timestamps
			cme_2.Stats.Timestamp = cme_1.Stats.Timestamp.Add(-2 * time.Minute)
		}
		metrics = []*cache.ContainerMetricElement{cme_1, cme_2}
	} else {
		metrics = cmes
	}
	metadata := cache.Metadata{
		Name:      "test",
		Namespace: "default",
		UID:       "123123123",
		Hostname:  "testhost",
		Labels:    make(map[string]string),
	}
	new_ce := cache.ContainerElement{
		Metadata: metadata,
		Metrics:  metrics,
	}
	return &new_ce
}

// podElementFactory creates a new PodElement with predetermined structure and fuzzed CME values.
// The resulting PodElement contains one ContainerElement with two ContainerMetricElements
func podElementFactory() *cache.PodElement {
	container_metadata := cache.Metadata{
		Name:      "test",
		Namespace: "default",
		UID:       "123123123",
		Hostname:  "testhost",
		Labels:    make(map[string]string),
	}
	cme_1 := cmeFactory()
	cme_2 := cmeFactory()
	for cme_1.Stats.Timestamp == cme_2.Stats.Timestamp {
		// Ensure random but different timestamps
		cme_2 = cmeFactory()
	}
	metrics := []*cache.ContainerMetricElement{cme_1, cme_2}

	new_ce := &cache.ContainerElement{
		Metadata: container_metadata,
		Metrics:  metrics,
	}
	pod_metadata := cache.Metadata{
		Name:      "pod-xyz",
		Namespace: "default",
		UID:       "12312-124125-135135",
		Hostname:  "testhost",
		Labels:    make(map[string]string),
	}
	pod_ele := cache.PodElement{
		Metadata:   pod_metadata,
		Containers: []*cache.ContainerElement{new_ce},
	}
	return &pod_ele
}

// cacheFactory generates a cache with a predetermined structure.
// The cache contains 2 pods, one with two containers and one without any containers.
// The cache also contains a free container and a "machine"-tagged container.
func cacheFactory() cache.Cache {
	source_cache := cache.NewCache(10*time.Minute, time.Minute)

	// Generate Container CMEs - same timestamp for aggregation
	cme_1 := cmeFactory()
	cme_2 := cmeFactory()
	cme_2.Stats.Timestamp = cme_1.Stats.Timestamp
	cme_2.Stats.Cpu.Usage.Total = cme_1.Stats.Cpu.Usage.Total

	// Generate a flush CME for cme_1 and cme_2
	cme_2flush := cmeFactory()
	cme_2flush.Stats.Timestamp = cme_1.Stats.Timestamp.Add(time.Minute)

	// Genete Machine CMEs - same timestamp for aggregation
	cme_3 := cmeFactory()
	cme_4 := cmeFactory()
	cme_3.Stats.Timestamp = cme_1.Stats.Timestamp.Add(2 * time.Minute)
	cme_4.Stats.Timestamp = cme_3.Stats.Timestamp
	cme_3.Stats.Memory.WorkingSet = 602
	cme_4.Stats.Memory.WorkingSet = 1062

	// Generate a flush CME for cme_3 and cme_4
	cme_4flush := cmeFactory()
	cme_4flush.Stats.Timestamp = cme_4.Stats.Timestamp.Add(time.Minute)
	cme_4flush.Stats.Cpu.Usage.Total = cme_4.Stats.Cpu.Usage.Total + uint64(360000000000)

	// Genete a generic container further than one resolution in the future
	cme_5 := cmeFactory()
	cme_5.Stats.Timestamp = cme_4.Stats.Timestamp.Add(4 * time.Minute)
	cme_5.Stats.Cpu.Usage.Total = cme_4.Stats.Cpu.Usage.Total + uint64(4*360000000000)

	// Generate a flush CME for cme_5 and cme_4
	cme_5flush := cmeFactory()
	cme_5flush.Stats.Timestamp = cme_5.Stats.Timestamp.Add(time.Minute)
	cme_5flush.Stats.Cpu.Usage.Total = cme_5.Stats.Cpu.Usage.Total + uint64(360000000000)

	// Generate a pod with two containers, and a pod without any containers
	container1 := source_api.Container{
		Name:     "container1",
		Hostname: "hostname2",
		Spec:     *cme_1.Spec,
		Stats:    []*source_api.ContainerStats{cme_2flush.Stats, cme_1.Stats},
	}
	container2 := source_api.Container{
		Name:     "container2",
		Hostname: "hostname3",
		Spec:     *cme_2.Spec,
		Stats:    []*source_api.ContainerStats{cme_2flush.Stats, cme_2.Stats},
	}

	containers := []source_api.Container{container1, container2}
	pods := []source_api.Pod{
		{
			PodMetadata: source_api.PodMetadata{
				Name:      "pod1",
				ID:        "123",
				Namespace: "test",
				Hostname:  "hostname2",
				Status:    "Running",
			},
			Containers: containers,
		},
		{
			PodMetadata: source_api.PodMetadata{
				Name:      "pod2",
				ID:        "1234",
				Namespace: "test",
				Hostname:  "hostname3",
				Status:    "Running",
			},
			Containers: containers,
		},
	}

	// Generate two machine containers
	machine_container := source_api.Container{
		Name:     "/",
		Hostname: "hostname2",
		Spec:     *cme_3.Spec,
		Stats:    []*source_api.ContainerStats{cme_4flush.Stats, cme_3.Stats},
	}
	machine_container2 := source_api.Container{
		Name:     "/",
		Hostname: "hostname3",
		Spec:     *cme_4.Spec,
		Stats:    []*source_api.ContainerStats{cme_5flush.Stats, cme_5.Stats, cme_4.Stats},
	}
	// Generate a free container
	free_container := source_api.Container{
		Name:     "free_container1",
		Hostname: "hostname2",
		Spec:     *cme_5.Spec,
		Stats:    []*source_api.ContainerStats{cme_5flush.Stats, cme_5.Stats},
	}

	other_containers := []source_api.Container{
		machine_container,
		machine_container2,
		free_container,
	}

	// Enter everything in the cache
	source_cache.StorePods(pods)
	source_cache.StoreContainers(other_containers)

	return source_cache
}

// TestDeleteFreeContainer tests all flows of deleteFreeContainer.
func TestDeleteFreeContainer(t *testing.T) {
	var (
		cluster           = newRealModel(time.Minute)
		ce                = containerElementFactory(nil)
		assert            = assert.New(t)
		freeContainerName = "testFreeContainerName"
		hostName          = "testhost"
	)

	// Add a free container with the specific container name and host name
	ce.Name = freeContainerName
	ce.Hostname = hostName
	_, err := cluster.updateFreeContainer(ce)
	assert.NoError(err)
	nodeInfo := cluster.Nodes[hostName]
	assert.NotNil(nodeInfo)
	assert.Equal(1, len(nodeInfo.FreeContainers))

	// First call : try to delete newly added free container
	cluster.deleteFreeContainer(hostName, freeContainerName)
	nodeInfo = cluster.Nodes[hostName]
	assert.NotNil(nodeInfo)
	assert.Equal(0, len(nodeInfo.FreeContainers))
	assert.Nil(nodeInfo.FreeContainers[freeContainerName])

	// Second call: already deleted
	cluster.deleteFreeContainer(hostName, freeContainerName)
	// No panic etc.
}

// TestDeletePodContainer tests all flows of deletePodContainer.
func TestDeletePodContainer(t *testing.T) {
	var (
		cluster = newRealModel(time.Minute)
		podElem = podElementFactory()
		assert  = assert.New(t)
	)

	// Initialize the cluster model with a pod with fake pod container
	timestamp, err := cluster.updatePod(podElem)
	assert.NoError(err)
	assert.NotEqual(timestamp, time.Time{})
	assert.NotNil(cluster.Nodes[podElem.Hostname])
	podInfo := cluster.Nodes[podElem.Hostname].Pods[getPodKey(podElem.Namespace, podElem.Name)]
	assert.NotNil(cluster.Namespaces[podElem.Namespace])
	assert.Equal(podInfo, cluster.Namespaces[podElem.Namespace].Pods[podElem.Name])
	assert.Equal(podInfo.Labels, podElem.Labels)

	// Record the number of pod containers after initialization
	containersNumAfterAdded := len(podInfo.Containers)
	assert.NotEqual(0, containersNumAfterAdded)

	// Record the names to be used to invoke deletePodContainer
	namespace := podInfo.Namespace
	podName := podInfo.Name
	var containerName string
	for name := range podInfo.Containers {
		containerName = name
		break
	}

	// First call : try to delete newly added pod container
	cluster.deletePodContainer(namespace, podName, containerName)
	assert.Equal(containersNumAfterAdded, len(podInfo.Containers)+1)

	// Second call: already deleted
	cluster.deletePodContainer(namespace, podName, containerName)
	// No panic etc.
}
