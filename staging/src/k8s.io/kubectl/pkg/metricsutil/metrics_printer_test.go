/*
Copyright 2024 The Kubernetes Authors.

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

package metricsutil

import (
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

func TestPrintNodeMetrics(t *testing.T) {

	tests := []struct {
		name           string
		nodeMetric     []metricsapi.NodeMetrics
		nodes          []*v1.Node
		noHeader       bool
		sortBy         string
		expectedErr    error
		expectedOutput string
	}{
		{
			name:  "Single node with default header",
			nodes: []*v1.Node{newNode("node1")},
			nodeMetric: []metricsapi.NodeMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node1", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
					Window:     metav1.Duration{Duration: time.Minute},
					Usage: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
					},
				},
			},
			expectedOutput: `NAME    CPU(cores)   CPU(%)   MEMORY(bytes)   MEMORY(%)   
node1   1000m        10%      1024Mi          10%         
`,
		},
		{
			name:  "Single node without header",
			nodes: []*v1.Node{newNode("node1")},
			nodeMetric: []metricsapi.NodeMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node1", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
					Window:     metav1.Duration{Duration: time.Minute},
					Usage: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
					},
				},
			},
			noHeader: true,
			expectedOutput: `node1   1000m   10%   1024Mi   10%   
`,
		},
		{
			name:  "Multiple nodes with one missing metrics",
			nodes: []*v1.Node{newNode("node1"), newNode("node2")},
			nodeMetric: []metricsapi.NodeMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node1", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
					Window:     metav1.Duration{Duration: time.Minute},
					Usage: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
					},
				},
			},
			expectedOutput: `NAME    CPU(cores)   CPU(%)      MEMORY(bytes)   MEMORY(%)   
node1   1000m        10%         1024Mi          10%         
node2   <unknown>    <unknown>   <unknown>       <unknown>   
`,
		},
		{
			name:  "Multiple nodes with metrics sorted by memory",
			nodes: []*v1.Node{newNode("node1"), newNode("node2")},
			nodeMetric: []metricsapi.NodeMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node1", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
					Window:     metav1.Duration{Duration: time.Minute},
					Usage: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "node2", ResourceVersion: "10", Labels: map[string]string{"key": "value"}},
					Window:     metav1.Duration{Duration: time.Minute},
					Usage: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2"),
						v1.ResourceMemory: resource.MustParse("5Gi"),
					},
				},
			},
			sortBy: "memory",
			expectedOutput: `NAME    CPU(cores)   CPU(%)   MEMORY(bytes)   MEMORY(%)   
node2   2000m        20%      5120Mi          50%         
node1   1000m        10%      1024Mi          10%         
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create a new TopCmdPrinter with a test writer.
			_, _, out, _ := genericiooptions.NewTestIOStreams()

			availableResources := make(map[string]v1.ResourceList)
			for _, n := range test.nodes {
				availableResources[n.Name] = n.Status.Capacity
			}
			top := NewTopCmdPrinter(out)
			err := top.PrintNodeMetrics(test.nodeMetric, availableResources, test.noHeader, test.sortBy)
			assert.Equal(t, test.expectedErr, err)
			assert.Equal(t, test.expectedOutput, out.String())
		})
	}
}

func TestPrintPodMetrics(t *testing.T) {

	tests := []struct {
		name            string
		podMetric       []metricsapi.PodMetrics
		printContainers bool
		withNamespace   bool
		noHeader        bool
		sortBy          string
		sum             bool
		expectedErr     error
		expectedOutput  string
	}{
		{
			name: "Single Pod with Multiple Containers",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
test   400m         2048Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers - Print Containers",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			printContainers: true,
			expectedOutput: `POD    NAME         CPU(cores)   MEMORY(bytes)   
test   container1   200m         1024Mi          
test   container2   200m         1024Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers - Calculate Resource Usage Without Header",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			noHeader: true,
			expectedOutput: `test   400m   2048Mi   
`,
		},
		{
			name: "Multiple Pods with Multiple Containers - Sort by Memory Usage",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-1",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
					},
				},
			},
			sortBy: "memory",
			expectedOutput: `NAME     CPU(cores)   MEMORY(bytes)   
test-1   400m         5120Mi          
test     400m         2048Mi          
`,
		},
		{
			name: "Multiple Pods with Multiple Containers - Calculate Total Resource Usage",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-1",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
					},
				},
			},
			sum: true,
			expectedOutput: `NAME     CPU(cores)   MEMORY(bytes)   
test     400m         2048Mi          
test-1   400m         5120Mi          
         ________     ________        
         800m         7168Mi          
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create a new TopCmdPrinter with a test writer.
			_, _, out, _ := genericiooptions.NewTestIOStreams()

			top := NewTopCmdPrinter(out)
			err := top.PrintPodMetrics(test.podMetric, test.printContainers,
				test.withNamespace, test.noHeader, test.sortBy, test.sum)
			assert.Equal(t, test.expectedErr, err)
			assert.Equal(t, test.expectedOutput, out.String())
		})
	}
}

func TestPrintObjOutputSameAsPrintPodMetrics(t *testing.T) {

	tests := []struct {
		name            string
		podMetric       []metricsapi.PodMetrics
		printContainers bool
		withNamespace   bool
		noHeader        bool
		sortBy          string
		sum             bool
		expectedErr     error
		expectedOutput  string
	}{
		{
			name: "Single Pod with Multiple Containers",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
test   400m         2048Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers - Print Containers",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			printContainers: true,
			expectedOutput: `POD    NAME         CPU(cores)   MEMORY(bytes)   
test   container1   200m         1024Mi          
test   container2   200m         1024Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers - Calculate Resource Usage Without Header",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			noHeader: true,
			expectedOutput: `test   400m   2048Mi   
`,
		},
		{
			name: "Multiple Pods with Multiple Containers - Sort by Memory Usage",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-1",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
					},
				},
			},
			sortBy: "memory",
			expectedOutput: `NAME     CPU(cores)   MEMORY(bytes)   
test-1   400m         5120Mi          
test     400m         2048Mi          
`,
		},
		{
			name: "Multiple Pods with Multiple Containers - Calculate Total Resource Usage",
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-1",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
					},
				},
			},
			sum: true,
			expectedOutput: `NAME     CPU(cores)   MEMORY(bytes)   
test     400m         2048Mi          
test-1   400m         5120Mi          
         ________     ________        
         800m         7168Mi          
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create a new TopCmdPrinter with a test writer.
			_, _, out, _ := genericiooptions.NewTestIOStreams()

			options := &PodPrintOptions{
				PrintContainers: test.printContainers,
				PrintNamespaces: test.withNamespace,
				NoHeaders:       test.noHeader,
				SortBy:          test.sortBy,
				Sum:             test.sum,
			}
			top := &TopCmdPrinter{out: out, PodPrintOptions: options}
			metrics := &metricsapi.PodMetricsList{Items: test.podMetric}
			err := top.PrintObj(metrics, out)
			assert.Equal(t, test.expectedErr, err)
			assert.Equal(t, test.expectedOutput, out.String())
		})
	}
}

func TestPrintObj(t *testing.T) {
	t.Run("print node metrics", func(t *testing.T) {
		// Create a new TopCmdPrinter with a test writer.
		_, _, out, _ := genericiooptions.NewTestIOStreams()

		top := &TopCmdPrinter{out: out}
		metrics := &metricsapi.NodeMetricsList{}
		err := top.PrintObj(metrics, out)
		assert.Equal(t, errors.New("printing node metrics is not yet supported"), err)
		assert.Equal(t, "", out.String())
	})

	t.Run("print pod metrics without options", func(t *testing.T) {
		// Create a new TopCmdPrinter with a test writer.
		_, _, out, _ := genericiooptions.NewTestIOStreams()

		top := &TopCmdPrinter{out: out}
		metrics := &metricsapi.PodMetricsList{}
		err := top.PrintObj(metrics, out)
		assert.Equal(t, nil, err)
		assert.Equal(t, "", out.String())
	})

}

func TestPrintObjPrintPodMetrics(t *testing.T) {

	tests := []struct {
		name            string
		podMetric       []metricsapi.PodMetrics
		printContainers bool
		withNamespace   bool
		noHeader        bool
		sortBy          string
		sum             bool
		expectedErr     error
		expectedOutput  string

		wide     bool
		podsInfo *[]v1.Pod
	}{
		{
			name: "Single Pod with Multiple Containers and Wide",
			wide: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			expectedOutput: `NAME   NODE    CPU(cores)   MEMORY(bytes)   
test   node1   400m         2048Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers and Wide but no pods info",
			wide: true,
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			expectedOutput: `NAME   NODE     CPU(cores)   MEMORY(bytes)   
test   <none>   400m         2048Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers and Wide but no spec pods info",
			wide: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "testother", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			expectedOutput: `NAME   NODE     CPU(cores)   MEMORY(bytes)   
test   <none>   400m         2048Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers - Print Containers and Wide",
			wide: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			printContainers: true,
			expectedOutput: `POD    NAME         NODE    CPU(cores)   MEMORY(bytes)   
test   container1   node1   200m         1024Mi          
test   container2   node1   200m         1024Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers - Print Containers and Wide but no spec pods info",
			wide: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "testother", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			printContainers: true,
			expectedOutput: `POD    NAME         NODE     CPU(cores)   MEMORY(bytes)   
test   container1   <none>   200m         1024Mi          
test   container2   <none>   200m         1024Mi          
`,
		},
		{
			name: "Single Pod with Multiple Containers - Wide and Calculate Resource Usage Without Header",
			wide: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			noHeader: true,
			expectedOutput: `test   node1   400m   2048Mi   
`,
		},
		{
			name: "Single Pod with Multiple Containers - Wide and no podInfo and Calculate Resource Usage Without Header",
			wide: true,
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
			},
			noHeader: true,
			expectedOutput: `test   <none>   400m   2048Mi   
`,
		},
		{
			name: "Multiple Pods with Multiple Containers - Wide and Sort by Memory Usage",
			wide: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "test-1", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node3"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-1",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
					},
				},
			},
			sortBy: "memory",
			expectedOutput: `NAME     NODE    CPU(cores)   MEMORY(bytes)   
test-1   node3   400m         5120Mi          
test     node1   400m         2048Mi          
`,
		},
		{
			name: "Multiple Pods with Multiple Containers - Wide and Calculate Total Resource Usage",
			wide: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "test-1", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node3"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-1",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
					},
				},
			},
			sum: true,
			expectedOutput: `NAME     NODE    CPU(cores)   MEMORY(bytes)   
test     node1   400m         2048Mi          
test-1   node3   400m         5120Mi          
                 ________     ________        
                 800m         7168Mi          
`,
		},
		{
			name:          "Multiple Pods with Multiple Containers with difference namespace and sort by cpu - Wide and Calculate Total Resource Usage",
			wide:          true,
			sortBy:        "cpu",
			withNamespace: true,
			podsInfo: &[]v1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "test2", Namespace: "ns2"}, Spec: v1.PodSpec{NodeName: "node2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "test3", Namespace: "otherns"}, Spec: v1.PodSpec{NodeName: "node3"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "test-1", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node-1"}},
			},
			podMetric: []metricsapi.PodMetrics{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test2",
						Namespace: "ns2",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.5"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.2"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test3",
						Namespace: "ns2",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.3"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.3"),
								v1.ResourceMemory: resource.MustParse("1Gi"),
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-1",
						Namespace: "default",
					},
					Timestamp: metav1.Time{Time: time.Now()},
					Window:    metav1.Duration{Duration: time.Minute},
					Containers: []metricsapi.ContainerMetrics{
						{
							Name: "container1",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.6"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
							},
						},
						{
							Name: "container2",
							Usage: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("0.3"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
							},
						},
					},
				},
			},
			sum: true,
			expectedOutput: `NAMESPACE   NAME     NODE     CPU(cores)   MEMORY(bytes)   
default     test-1   node-1   900m         5120Mi          
ns2         test2    node2    700m         2048Mi          
ns2         test3    <none>   600m         2048Mi          
default     test     node1    400m         2048Mi          
                              ________     ________        
                              2600m        11264Mi         
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create a new TopCmdPrinter with a test writer.
			_, _, out, _ := genericiooptions.NewTestIOStreams()

			options := &PodPrintOptions{
				PrintContainers: test.printContainers,
				PrintNamespaces: test.withNamespace,
				NoHeaders:       test.noHeader,
				SortBy:          test.sortBy,
				Sum:             test.sum,

				Wide:     test.wide,
				PodsInfo: test.podsInfo,
			}
			top := &TopCmdPrinter{out: out, PodPrintOptions: options}
			metrics := &metricsapi.PodMetricsList{Items: test.podMetric}
			err := top.PrintObj(metrics, out)
			assert.Equal(t, test.expectedErr, err)
			assert.Equal(t, test.expectedOutput, out.String())
		})
	}
}

func newNode(name string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10"),
				v1.ResourceName(v1.ResourceMemory): resource.MustParse("10Gi"),
			},
		},
	}
}
