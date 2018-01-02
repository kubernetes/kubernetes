// Copyright 2014 Google Inc. All Rights Reserved.
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
package kubelet

import (
	"encoding/json"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	cadvisor_api "github.com/google/cadvisor/info/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/heapster/metrics/core"
	kube_api "k8s.io/kubernetes/pkg/api"
	util "k8s.io/kubernetes/pkg/util/testing"
)

func TestDecodeMetrics1(t *testing.T) {
	kMS := kubeletMetricsSource{
		nodename: "test",
		hostname: "test-hostname",
	}
	c1 := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "/",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime: time.Now(),
			HasCpu:       true,
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
				Cpu: cadvisor_api.CpuStats{
					Usage: cadvisor_api.CpuUsage{
						Total:  100,
						PerCpu: []uint64{5, 10},
						User:   1,
						System: 1,
					},
					LoadAverage: 20,
				},
			},
		},
	}
	metricSetKey, metricSet := kMS.decodeMetrics(&c1)
	assert.Equal(t, metricSetKey, "node:test")
	assert.Equal(t, metricSet.Labels[core.LabelMetricSetType.Key], core.MetricSetTypeNode)
}

func TestDecodeMetrics2(t *testing.T) {
	kMS := kubeletMetricsSource{
		nodename: "test",
		hostname: "test-hostname",
	}
	c1 := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "/",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime: time.Now(),
			HasCpu:       true,
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
				Cpu: cadvisor_api.CpuStats{
					Usage: cadvisor_api.CpuUsage{
						Total:  100,
						PerCpu: []uint64{5, 10},
						User:   1,
						System: 1,
					},
					LoadAverage: 20,
				},
			},
		},
	}
	metricSetKey, metricSet := kMS.decodeMetrics(&c1)
	assert.Equal(t, metricSetKey, "node:test")
	assert.Equal(t, metricSet.Labels[core.LabelMetricSetType.Key], core.MetricSetTypeNode)
}

func TestDecodeMetrics3(t *testing.T) {
	kMS := kubeletMetricsSource{
		nodename: "test",
		hostname: "test-hostname",
	}
	c1 := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "/docker-daemon",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime: time.Now(),
			HasCpu:       true,
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
				Cpu: cadvisor_api.CpuStats{
					Usage: cadvisor_api.CpuUsage{
						Total:  100,
						PerCpu: []uint64{5, 10},
						User:   1,
						System: 1,
					},
					LoadAverage: 20,
				},
			},
		},
	}
	metricSetKey, _ := kMS.decodeMetrics(&c1)
	assert.Equal(t, metricSetKey, "node:test/container:docker-daemon")
}

func TestDecodeMetrics4(t *testing.T) {
	kMS := kubeletMetricsSource{
		nodename: "test",
		hostname: "test-hostname",
	}
	c1 := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "testKubelet",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime: time.Now(),
			HasCpu:       true,
			Labels:       make(map[string]string),
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
				Cpu: cadvisor_api.CpuStats{
					Usage: cadvisor_api.CpuUsage{
						Total:  100,
						PerCpu: []uint64{5, 10},
						User:   1,
						System: 1,
					},
					LoadAverage: 20,
				},
			},
		},
	}

	c1.Spec.Labels[kubernetesContainerLabel] = "testContainer"
	c1.Spec.Labels[kubernetesPodNamespaceLabel] = "testPodNS"
	c1.Spec.Labels[kubernetesPodNameLabel] = "testPodName"
	metricSetKey, metricSet := kMS.decodeMetrics(&c1)
	assert.Equal(t, metricSetKey, "namespace:testPodNS/pod:testPodName/container:testContainer")
	assert.Equal(t, metricSet.Labels[core.LabelMetricSetType.Key], core.MetricSetTypePodContainer)
}

func TestDecodeMetrics5(t *testing.T) {
	kMS := kubeletMetricsSource{
		nodename: "test",
		hostname: "test-hostname",
	}
	c1 := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "k8s_test.testkubelet",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime: time.Now(),
			HasCpu:       true,
			Labels:       make(map[string]string),
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
				Cpu: cadvisor_api.CpuStats{
					Usage: cadvisor_api.CpuUsage{
						Total:  100,
						PerCpu: []uint64{5, 10},
						User:   1,
						System: 1,
					},
					LoadAverage: 20,
				},
			},
		},
	}
	c1.Spec.Labels[kubernetesContainerLabel] = "POD"
	c1.Spec.Labels[kubernetesPodNameLabel] = "testnamespace/testPodName"
	metricSetKey, metricSet := kMS.decodeMetrics(&c1)
	assert.Equal(t, metricSetKey, "namespace:testnamespace/pod:testPodName")
	assert.Equal(t, metricSet.Labels[core.LabelMetricSetType.Key], core.MetricSetTypePod)

	c1.Spec.Labels[kubernetesContainerLabel] = ""
	c1.Spec.Labels[kubernetesPodNameLabel] = "testnamespace/testPodName"
	metricSetKey, metricSet = kMS.decodeMetrics(&c1)
	assert.Equal(t, metricSetKey, "namespace:testnamespace/pod:testPodName/container:test")
	assert.Equal(t, metricSet.Labels[core.LabelMetricSetType.Key], core.MetricSetTypePodContainer)
}

func TestDecodeMetrics6(t *testing.T) {
	kMS := kubeletMetricsSource{
		nodename: "test",
		hostname: "test-hostname",
	}
	c1 := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "/",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime:     time.Now(),
			HasCustomMetrics: true,
			CustomMetrics: []cadvisor_api.MetricSpec{
				{
					Name:   "test1",
					Type:   cadvisor_api.MetricGauge,
					Format: cadvisor_api.IntType,
				},
				{
					Name:   "test2",
					Type:   cadvisor_api.MetricCumulative,
					Format: cadvisor_api.IntType,
				},
				{
					Name:   "test3",
					Type:   cadvisor_api.MetricGauge,
					Format: cadvisor_api.FloatType,
				},
				{
					Name:   "test4",
					Type:   cadvisor_api.MetricCumulative,
					Format: cadvisor_api.FloatType,
				},
			},
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
				Cpu: cadvisor_api.CpuStats{
					Usage: cadvisor_api.CpuUsage{
						Total:  100,
						PerCpu: []uint64{5, 10},
						User:   1,
						System: 1,
					},
					LoadAverage: 20,
				},
				CustomMetrics: map[string][]cadvisor_api.MetricVal{
					"test1": []cadvisor_api.MetricVal{
						{
							Label:      "test1",
							Timestamp:  time.Now(),
							IntValue:   1,
							FloatValue: 1.0,
						},
					},
					"test2": []cadvisor_api.MetricVal{
						{
							Label:      "test2",
							Timestamp:  time.Now(),
							IntValue:   1,
							FloatValue: 1.0,
						},
					},
					"test3": []cadvisor_api.MetricVal{
						{
							Label:      "test3",
							Timestamp:  time.Now(),
							IntValue:   1,
							FloatValue: 1.0,
						},
					},
					"test4": []cadvisor_api.MetricVal{
						{
							Label:      "test4",
							Timestamp:  time.Now(),
							IntValue:   1,
							FloatValue: 1.0,
						},
					},
				},
			},
		},
	}
	metricSetKey, metricSet := kMS.decodeMetrics(&c1)
	assert.Equal(t, metricSetKey, "node:test")
	assert.Equal(t, metricSet.Labels[core.LabelMetricSetType.Key], core.MetricSetTypeNode)
}

var nodes = []kube_api.Node{
	kube_api.Node{
		ObjectMeta: kube_api.ObjectMeta{
			Name: "testNode",
		},
		Status: kube_api.NodeStatus{
			Conditions: []kube_api.NodeCondition{
				{
					Type:   "NotReady",
					Status: kube_api.ConditionTrue,
				},
			},
			Addresses: []kube_api.NodeAddress{
				{
					Type:    kube_api.NodeHostName,
					Address: "testNode",
				},
				{
					Type:    kube_api.NodeInternalIP,
					Address: "127.0.0.1",
				},
			},
		},
	},
	kube_api.Node{
		ObjectMeta: kube_api.ObjectMeta{
			Name: "testNode",
		},
		Status: kube_api.NodeStatus{
			Conditions: []kube_api.NodeCondition{
				{
					Type:   "NotReady",
					Status: kube_api.ConditionTrue,
				},
			},
			Addresses: []kube_api.NodeAddress{
				{
					Type:    kube_api.NodeHostName,
					Address: "testNode",
				},
				{
					Type:    kube_api.NodeLegacyHostIP,
					Address: "127.0.0.1",
				},
			},
		},
	},
	kube_api.Node{
		ObjectMeta: kube_api.ObjectMeta{
			Name: "testNode",
		},
		Status: kube_api.NodeStatus{
			Conditions: []kube_api.NodeCondition{
				{
					Type:   "NotReady",
					Status: kube_api.ConditionTrue,
				},
			},
			Addresses: []kube_api.NodeAddress{
				{
					Type:    kube_api.NodeHostName,
					Address: "testNode",
				},
				{
					Type:    kube_api.NodeLegacyHostIP,
					Address: "127.0.0.2",
				},
				{
					Type:    kube_api.NodeInternalIP,
					Address: "127.0.0.1",
				},
			},
		},
	},
}

func TestGetNodeHostnameAndIP(t *testing.T) {
	for _, node := range nodes {
		hostname, ip, err := getNodeHostnameAndIP(&node)
		assert.NoError(t, err)
		assert.Equal(t, hostname, "testNode")
		assert.Equal(t, ip, "127.0.0.1")
	}
}

func TestScrapeMetrics(t *testing.T) {
	rootContainer := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "/",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime: time.Now(),
			HasCpu:       true,
			HasMemory:    true,
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
			},
		},
	}

	subcontainer := cadvisor_api.ContainerInfo{
		ContainerReference: cadvisor_api.ContainerReference{
			Name: "/docker-daemon",
		},
		Spec: cadvisor_api.ContainerSpec{
			CreationTime: time.Now(),
			HasCpu:       true,
			HasMemory:    true,
		},
		Stats: []*cadvisor_api.ContainerStats{
			{
				Timestamp: time.Now(),
			},
		},
	}
	response := map[string]cadvisor_api.ContainerInfo{
		rootContainer.Name: {
			ContainerReference: cadvisor_api.ContainerReference{
				Name: rootContainer.Name,
			},
			Spec: rootContainer.Spec,
			Stats: []*cadvisor_api.ContainerStats{
				rootContainer.Stats[0],
			},
		},
		subcontainer.Name: {
			ContainerReference: cadvisor_api.ContainerReference{
				Name: subcontainer.Name,
			},
			Spec: subcontainer.Spec,
			Stats: []*cadvisor_api.ContainerStats{
				subcontainer.Stats[0],
			},
		},
	}
	data, err := json.Marshal(&response)
	require.NoError(t, err)
	handler := util.FakeHandler{
		StatusCode:   200,
		RequestBody:  "",
		ResponseBody: string(data),
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()

	var client KubeletClient

	mtrcSrc := kubeletMetricsSource{
		kubeletClient: &client,
	}

	split := strings.SplitN(strings.Replace(server.URL, "http://", "", 1), ":", 2)
	mtrcSrc.host.IP = split[0]
	mtrcSrc.host.Port, err = strconv.Atoi(split[1])

	start := time.Now()
	end := start.Add(5 * time.Second)
	res := mtrcSrc.ScrapeMetrics(start, end)
	assert.Equal(t, res.MetricSets["node:/container:docker-daemon"].Labels["type"], "sys_container")
	assert.Equal(t, res.MetricSets["node:/container:docker-daemon"].Labels["container_name"], "docker-daemon")

}
