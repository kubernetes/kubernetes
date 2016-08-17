/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"encoding/json"
	"io"
	"io/ioutil"
	"time"

	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	api "k8s.io/kubernetes/pkg/api/v1"
	"testing"
)

const (
	baseHeapsterServiceAddress = "/api/v1/namespaces/kube-system/services/http:heapster:"
	baseMetricsAddress         = baseHeapsterServiceAddress + "/proxy/apis/metrics"
	metricsApiVersion          = "v1alpha1"
)

func TestTopSubcommandsExist(t *testing.T) {
	initTestErrorHandler(t)

	f, _, _, _ := NewAPIFactory()
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdTop(f, buf)
	if !cmd.HasSubCommands() {
		t.Error("top command should have subcommands")
	}
}

func marshallBody(metrics interface{}) (io.ReadCloser, error) {
	result, err := json.Marshal(metrics)
	if err != nil {
		return nil, err
	}
	return ioutil.NopCloser(bytes.NewReader(result)), nil
}

func testNodeMetricsData() []metrics_api.NodeMetrics {
	return []metrics_api.NodeMetrics{
		{
			ObjectMeta: api.ObjectMeta{Name: "node1", ResourceVersion: "10"},
			Window:     unversioned.Duration{Duration: time.Minute},
			Usage: api.ResourceList{
				api.ResourceCPU:     *resource.NewMilliQuantity(1, resource.DecimalSI),
				api.ResourceMemory:  *resource.NewQuantity(2*(1024*1024), resource.DecimalSI),
				api.ResourceStorage: *resource.NewQuantity(3*(1024*1024), resource.DecimalSI),
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "node2", ResourceVersion: "11"},
			Window:     unversioned.Duration{Duration: time.Minute},
			Usage: api.ResourceList{
				api.ResourceCPU:     *resource.NewMilliQuantity(5, resource.DecimalSI),
				api.ResourceMemory:  *resource.NewQuantity(6*(1024*1024), resource.DecimalSI),
				api.ResourceStorage: *resource.NewQuantity(7*(1024*1024), resource.DecimalSI),
			},
		},
	}
}

func testPodMetricsData() []metrics_api.PodMetrics {
	return []metrics_api.PodMetrics{
		{
			ObjectMeta: api.ObjectMeta{Name: "pod1", Namespace: "test", ResourceVersion: "10"},
			Window:     unversioned.Duration{Duration: time.Minute},
			Containers: []metrics_api.ContainerMetrics{
				{
					Name: "container1-1",
					Usage: api.ResourceList{
						api.ResourceCPU:     *resource.NewMilliQuantity(1, resource.DecimalSI),
						api.ResourceMemory:  *resource.NewQuantity(2*(1024*1024), resource.DecimalSI),
						api.ResourceStorage: *resource.NewQuantity(3*(1024*1024), resource.DecimalSI),
					},
				},
				{
					Name: "container1-2",
					Usage: api.ResourceList{
						api.ResourceCPU:     *resource.NewMilliQuantity(4, resource.DecimalSI),
						api.ResourceMemory:  *resource.NewQuantity(5*(1024*1024), resource.DecimalSI),
						api.ResourceStorage: *resource.NewQuantity(6*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod2", Namespace: "test", ResourceVersion: "11"},
			Window:     unversioned.Duration{Duration: time.Minute},
			Containers: []metrics_api.ContainerMetrics{
				{
					Name: "container2-1",
					Usage: api.ResourceList{
						api.ResourceCPU:     *resource.NewMilliQuantity(7, resource.DecimalSI),
						api.ResourceMemory:  *resource.NewQuantity(8*(1024*1024), resource.DecimalSI),
						api.ResourceStorage: *resource.NewQuantity(9*(1024*1024), resource.DecimalSI),
					},
				},
				{
					Name: "container2-2",
					Usage: api.ResourceList{
						api.ResourceCPU:     *resource.NewMilliQuantity(10, resource.DecimalSI),
						api.ResourceMemory:  *resource.NewQuantity(11*(1024*1024), resource.DecimalSI),
						api.ResourceStorage: *resource.NewQuantity(12*(1024*1024), resource.DecimalSI),
					},
				},
				{
					Name: "container2-3",
					Usage: api.ResourceList{
						api.ResourceCPU:     *resource.NewMilliQuantity(13, resource.DecimalSI),
						api.ResourceMemory:  *resource.NewQuantity(14*(1024*1024), resource.DecimalSI),
						api.ResourceStorage: *resource.NewQuantity(15*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pod3", Namespace: "test", ResourceVersion: "12"},
			Window:     unversioned.Duration{Duration: time.Minute},
			Containers: []metrics_api.ContainerMetrics{
				{
					Name: "container3-1",
					Usage: api.ResourceList{
						api.ResourceCPU:     *resource.NewMilliQuantity(7, resource.DecimalSI),
						api.ResourceMemory:  *resource.NewQuantity(8*(1024*1024), resource.DecimalSI),
						api.ResourceStorage: *resource.NewQuantity(9*(1024*1024), resource.DecimalSI),
					},
				},
			},
		},
	}
}
