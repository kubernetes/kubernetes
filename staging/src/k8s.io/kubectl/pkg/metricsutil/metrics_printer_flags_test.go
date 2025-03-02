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

package metricsutil

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
)

func TestTopCmdPrinterSupportsExpectedPodPrintOptions(t *testing.T) {
	podsInfo := &[]v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "default"}, Spec: v1.PodSpec{NodeName: "node1"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: "ns2"}, Spec: v1.PodSpec{NodeName: "node2"}},
	}
	metrics := metricsapi.PodMetricsList{Items: []metricsapi.PodMetrics{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "default"},
			Timestamp:  metav1.Time{Time: time.Now()},
			Window:     metav1.Duration{Duration: time.Minute},
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
						v1.ResourceCPU:    resource.MustParse("0.3"),
						v1.ResourceMemory: resource.MustParse("3Gi"),
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: "ns2"},
			Timestamp:  metav1.Time{Time: time.Now()},
			Window:     metav1.Duration{Duration: time.Minute},
			Containers: []metricsapi.ContainerMetrics{
				{
					Name: "container1",
					Usage: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
					},
				},
			},
		},
	}}

	testCases := []struct {
		name       string
		testObject runtime.Object

		sortBy          string
		PrintContainers bool
		PrintNamespaces bool
		noHeaders       bool
		Sum             bool

		PodsInfo *[]v1.Pod

		outputFormat string

		expectedError  string
		expectedOutput string
		expectNoMatch  bool
	}{
		{
			name:       "empty output format matches a TopCmdPrinter printer",
			testObject: metrics.DeepCopy(),
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod1   500m         4096Mi          
pod2   1000m        1024Mi          
`,
		},
		{
			name:         "wide output format prints without pods info",
			testObject:   metrics.DeepCopy(),
			outputFormat: "wide",
			expectedOutput: `NAME   NODE     CPU(cores)   MEMORY(bytes)   
pod1   <none>   500m         4096Mi          
pod2   <none>   1000m        1024Mi          
`,
		},
		{
			name:         "wide output format prints with pods info",
			PodsInfo:     podsInfo,
			testObject:   metrics.DeepCopy(),
			outputFormat: "wide",
			expectedOutput: `NAME   NODE    CPU(cores)   MEMORY(bytes)   
pod1   node1   500m         4096Mi          
pod2   node2   1000m        1024Mi          
`,
		},
		{
			name:       "no-headers prints output with no headers",
			testObject: metrics.DeepCopy(),
			noHeaders:  true,
			expectedOutput: `pod1   500m    4096Mi   
pod2   1000m   1024Mi   
`,
		},
		{
			name:         "no-headers prints output with no headers and wide",
			outputFormat: "wide",
			testObject:   metrics.DeepCopy(),
			noHeaders:    true,
			expectedOutput: `pod1   <none>   500m    4096Mi   
pod2   <none>   1000m   1024Mi   
`,
		},
		{
			name:         "no-headers prints output with no headers and pods info and wide",
			PodsInfo:     podsInfo,
			outputFormat: "wide",
			testObject:   metrics.DeepCopy(),
			noHeaders:    true,
			expectedOutput: `pod1   node1   500m    4096Mi   
pod2   node2   1000m   1024Mi   
`,
		},
		{
			name:            "containers displays containers list",
			testObject:      metrics.DeepCopy(),
			PrintContainers: true,
			expectedOutput: `POD    NAME         CPU(cores)   MEMORY(bytes)   
pod1   container1   200m         1024Mi          
pod1   container2   300m         3072Mi          
pod2   container1   1000m        1024Mi          
`,
		},
		{
			name:            "containers displays containers list and wide without pods info",
			testObject:      metrics.DeepCopy(),
			outputFormat:    "wide",
			PrintContainers: true,
			expectedOutput: `POD    NAME         NODE     CPU(cores)   MEMORY(bytes)   
pod1   container1   <none>   200m         1024Mi          
pod1   container2   <none>   300m         3072Mi          
pod2   container1   <none>   1000m        1024Mi          
`},
		{
			name:            "containers displays containers list and wide with pods info",
			testObject:      metrics.DeepCopy(),
			PodsInfo:        podsInfo,
			outputFormat:    "wide",
			PrintContainers: true,
			expectedOutput: `POD    NAME         NODE    CPU(cores)   MEMORY(bytes)   
pod1   container1   node1   200m         1024Mi          
pod1   container2   node1   300m         3072Mi          
pod2   container1   node2   1000m        1024Mi          
`,
		},
		{
			name:            "all-namespaces displays all namespaces metrics",
			testObject:      metrics.DeepCopy(),
			PrintNamespaces: true,
			expectedOutput: `NAMESPACE   NAME   CPU(cores)   MEMORY(bytes)   
default     pod1   500m         4096Mi          
ns2         pod2   1000m        1024Mi          
`,
		},
		{
			name:            "all-namespaces displays all namespaces metrics and wide without pods info",
			testObject:      metrics.DeepCopy(),
			outputFormat:    "wide",
			PrintNamespaces: true,
			expectedOutput: `NAMESPACE   NAME   NODE     CPU(cores)   MEMORY(bytes)   
default     pod1   <none>   500m         4096Mi          
ns2         pod2   <none>   1000m        1024Mi          
`,
		},
		{
			name:            "all-namespaces displays all namespaces metrics and wide and pods info",
			testObject:      metrics.DeepCopy(),
			outputFormat:    "wide",
			PodsInfo:        podsInfo,
			PrintNamespaces: true,
			expectedOutput: `NAMESPACE   NAME   NODE    CPU(cores)   MEMORY(bytes)   
default     pod1   node1   500m         4096Mi          
ns2         pod2   node2   1000m        1024Mi          
`,
		},
		{
			name:       "sort-by cpu sorted",
			sortBy:     "cpu",
			testObject: metrics.DeepCopy(),
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod2   1000m        1024Mi          
pod1   500m         4096Mi          
`,
		},
		{
			name:            "sort-by cpu sorted with containers",
			sortBy:          "cpu",
			PrintContainers: true,
			testObject:      metrics.DeepCopy(),
			expectedOutput: `POD    NAME         CPU(cores)   MEMORY(bytes)   
pod2   container1   1000m        1024Mi          
pod1   container2   300m         3072Mi          
pod1   container1   200m         1024Mi          
`,
		},
		{
			name:            "sort-by cpu sorted with containers and wide and pods info",
			sortBy:          "cpu",
			PrintContainers: true,
			PodsInfo:        podsInfo,
			outputFormat:    "wide",
			testObject:      metrics.DeepCopy(),
			expectedOutput: `POD    NAME         NODE    CPU(cores)   MEMORY(bytes)   
pod2   container1   node2   1000m        1024Mi          
pod1   container2   node1   300m         3072Mi          
pod1   container1   node1   200m         1024Mi          
`,
		},
		{
			name:       "sort-by memory sorted",
			sortBy:     "memory",
			testObject: metrics.DeepCopy(),
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod1   500m         4096Mi          
pod2   1000m        1024Mi          
`,
		},
		{
			name:            "sort-by memory sorted with containers",
			sortBy:          "memory",
			testObject:      metrics.DeepCopy(),
			PrintContainers: true,
			expectedOutput: `POD    NAME         CPU(cores)   MEMORY(bytes)   
pod1   container2   300m         3072Mi          
pod1   container1   200m         1024Mi          
pod2   container1   1000m        1024Mi          
`,
		},
		{
			name:       "sum display total resource usage",
			testObject: metrics.DeepCopy(),
			Sum:        true,
			expectedOutput: `NAME   CPU(cores)   MEMORY(bytes)   
pod1   500m         4096Mi          
pod2   1000m        1024Mi          
       ________     ________        
       1500m        5120Mi          
`,
		},
		{
			name:            "all options",
			testObject:      metrics.DeepCopy(),
			sortBy:          "cpu",
			PrintContainers: true,
			PrintNamespaces: true,
			noHeaders:       true,
			Sum:             true,
			outputFormat:    "wide",
			PodsInfo:        podsInfo,
			expectedOutput: `ns2       pod2   container1   node2   1000m      1024Mi     
default   pod1   container2   node1   300m       3072Mi     
default   pod1   container1   node1   200m       1024Mi     
                                      ________   ________   
                                      1500m      5120Mi     
`,
		},
		{
			name:            "all options without no-headers",
			testObject:      metrics.DeepCopy(),
			sortBy:          "cpu",
			PrintContainers: true,
			PrintNamespaces: true,
			Sum:             true,
			outputFormat:    "wide",
			PodsInfo:        podsInfo,
			expectedOutput: `NAMESPACE   POD    NAME         NODE    CPU(cores)   MEMORY(bytes)   
ns2         pod2   container1   node2   1000m        1024Mi          
default     pod1   container2   node1   300m         3072Mi          
default     pod1   container1   node1   200m         1024Mi          
                                        ________     ________        
                                        1500m        5120Mi          
`,
		},
		{
			name:          "no printer is matched on an invalid outputFormat",
			testObject:    metrics.DeepCopy(),
			outputFormat:  "invalid",
			expectNoMatch: true,
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s %T", tc.name, tc.testObject), func(t *testing.T) {
			printFlags := PodPrintFlags{
				SortBy:          tc.sortBy,
				PrintContainers: tc.PrintContainers,
				PrintNamespaces: tc.PrintNamespaces,
				NoHeaders:       tc.noHeaders,
				Sum:             tc.Sum,
				PodsInfo:        tc.PodsInfo,
			}

			p, err := printFlags.ToPrinter(tc.outputFormat)
			if tc.expectNoMatch {
				if !genericclioptions.IsNoCompatiblePrinterError(err) {
					t.Fatalf("expected no printer matches for output format %q", tc.outputFormat)
				}
				return
			}
			if genericclioptions.IsNoCompatiblePrinterError(err) {
				t.Fatalf("expected to match template printer for output format %q", tc.outputFormat)
			}

			if len(tc.expectedError) > 0 {
				if err == nil || !strings.Contains(err.Error(), tc.expectedError) {
					t.Errorf("expecting error %q, got %v", tc.expectedError, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			_, _, out, _ := genericiooptions.NewTestIOStreams()

			err = p.PrintObj(tc.testObject, out)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			assert.Equal(t, tc.expectedOutput, out.String())
		})
	}
}
