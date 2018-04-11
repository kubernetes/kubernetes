/*
Copyright 2017 The Kubernetes Authors.

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

package priorities

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/features"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
	metricsapi "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsfake "k8s.io/metrics/pkg/client/clientset_generated/clientset/fake"
	resourceclient "k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1beta1"
)

type nodeUsage struct {
	nodeName    string
	cpuUsage    resource.Quantity
	memoryUsage resource.Quantity
}

func getNewNodeMetrics() []nodeUsage {
	nodeMetricsList := make([]nodeUsage, 0)
	for i := 0; i < 3; i++ {
		nodeMetric := nodeUsage{}
		nodeMetric.nodeName = fmt.Sprintf("node-%d", i)
		// This ensures that 3rd node has highest utilization.
		nodeMetric.cpuUsage = resource.MustParse(fmt.Sprintf("%dm", 1000+(i*100)))
		nodeMetric.memoryUsage = resource.MustParse("0")
		nodeMetricsList = append(nodeMetricsList, nodeMetric)
	}
	return nodeMetricsList
}

func TestNewLeastUsedResourceBasedOnMetrics(t *testing.T) {
	utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("%s=true", features.UsageBasedScheduling))
	cpuOnly := v1.PodSpec{
		NodeName: "machine1",
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2000m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
		},
	}
	tests := []struct {
		pod               *v1.Pod
		nodes             []*v1.Node
		expectedNodeScore schedulerapi.HostPriorityList
		metricsClient     *resourceclient.MetricsV1beta1Client
		test              string
	}{
		{
			pod:               &v1.Pod{Spec: cpuOnly},
			nodes:             []*v1.Node{makeNode("node-0", 4000, 10000), makeNode("node-1", 4000, 0), makeNode("node-2", 4000, 0)},
			expectedNodeScore: []schedulerapi.HostPriority{{Host: "node-0", Score: 10}, {Host: "node-1", Score: 0}, {Host: "node-2", Score: 0}},
			test:              "pod with cpu only request. The 1st node has least utilization, so it should have highest score.",
		},
	}
	fakeMetricsClient := &metricsfake.Clientset{}
	fakeMetricsClient.AddReactor("list", "nodes", func(action core.Action) (handled bool, ret runtime.Object, err error) {

		nodeMetrics := &metricsapi.NodeMetricsList{}
		for _, nm := range getNewNodeMetrics() {
			nodeMetric := metricsapi.NodeMetrics{
				ObjectMeta: metav1.ObjectMeta{
					Name: nm.nodeName,
				},
				Timestamp: metav1.Time{Time: time.Now()},
				Usage: v1.ResourceList{
					v1.ResourceCPU:    nm.cpuUsage,
					v1.ResourceMemory: nm.memoryUsage,
				},
			}
			nodeMetrics.Items = append(nodeMetrics.Items, nodeMetric)
		}
		return true, nodeMetrics, nil
	})

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(nil, test.nodes)
		currentUsage := &usageDataOnNode{
			metricsNodeClient: priorityutil.NewRESTMetricsClient(fakeMetricsClient.MetricsV1beta1()),
		}
		list, err := priorityFunction(currentUsage.leastUsagePriorityMap, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedNodeScore, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedNodeScore, list)
		}
	}
	utilfeature.DefaultFeatureGate.Set(fmt.Sprintf("%s=false", features.UsageBasedScheduling))
}
