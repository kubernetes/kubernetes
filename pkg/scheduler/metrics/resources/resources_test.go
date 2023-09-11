/*
Copyright 2020 The Kubernetes Authors.

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

// Package resources provides a metrics collector that reports the
// resource consumption (requests and limits) of the pods in the cluster
// as the scheduler and kubelet would interpret it.
package resources

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/utils/ptr"
)

type fakePodLister struct {
	pods []*v1.Pod
}

func (l *fakePodLister) List(selector labels.Selector) (ret []*v1.Pod, err error) {
	return l.pods, nil
}

func (l *fakePodLister) Pods(namespace string) corelisters.PodNamespaceLister {
	panic("not implemented")
}

func Test_podResourceCollector_Handler(t *testing.T) {
	h := Handler(&fakePodLister{pods: []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
			Spec: v1.PodSpec{
				NodeName: "node-one",
				InitContainers: []v1.Container{
					{Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							"cpu":    resource.MustParse("2"),
							"custom": resource.MustParse("3"),
						},
						Limits: v1.ResourceList{
							"memory": resource.MustParse("1G"),
							"custom": resource.MustParse("5"),
						},
					}},
				},
				Containers: []v1.Container{
					{Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							"cpu":    resource.MustParse("1"),
							"custom": resource.MustParse("0"),
						},
						Limits: v1.ResourceList{
							"memory": resource.MustParse("2.5Gi"),
							"custom": resource.MustParse("6"),
						},
					}},
				},
			},
			Status: v1.PodStatus{
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue},
				},
			},
		},
	}})

	r := httptest.NewRecorder()
	req, err := http.NewRequest("GET", "/metrics/resources", nil)
	if err != nil {
		t.Fatal(err)
	}
	h.ServeHTTP(r, req)

	expected := `# HELP kube_pod_resource_limit [STABLE] Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
# TYPE kube_pod_resource_limit gauge
kube_pod_resource_limit{namespace="test",node="node-one",pod="foo",priority="",resource="custom",scheduler="",unit=""} 6
kube_pod_resource_limit{namespace="test",node="node-one",pod="foo",priority="",resource="memory",scheduler="",unit="bytes"} 2.68435456e+09
# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
# TYPE kube_pod_resource_request gauge
kube_pod_resource_request{namespace="test",node="node-one",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 2
kube_pod_resource_request{namespace="test",node="node-one",pod="foo",priority="",resource="custom",scheduler="",unit=""} 3
`
	out := r.Body.String()
	if expected != out {
		t.Fatal(out)
	}
}

func Test_podResourceCollector_CollectWithStability(t *testing.T) {
	tests := []struct {
		name string

		pods     []*v1.Pod
		expected string
	}{
		{},
		{
			name: "no containers",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
				},
			},
		},
		{
			name: "no resources",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{},
						Containers:     []v1.Container{},
					},
				},
			},
		},
		{
			name: "request only",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
				},
			},
			expected: `
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 1
				`,
		},
		{
			name: "limits only",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Limits: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
				},
			},
			expected: `      
				# HELP kube_pod_resource_limit [STABLE] Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_limit gauge
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 1
				`,
		},
		{
			name: "terminal pods are excluded",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo-unscheduled-succeeded"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
					// until node name is set, phase is ignored
					Status: v1.PodStatus{Phase: v1.PodSucceeded},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo-succeeded"},
					Spec: v1.PodSpec{
						NodeName: "node-one",
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
					Status: v1.PodStatus{Phase: v1.PodSucceeded},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo-failed"},
					Spec: v1.PodSpec{
						NodeName: "node-one",
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
					Status: v1.PodStatus{Phase: v1.PodFailed},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo-unknown"},
					Spec: v1.PodSpec{
						NodeName: "node-one",
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
					Status: v1.PodStatus{Phase: v1.PodUnknown},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo-pending"},
					Spec: v1.PodSpec{
						NodeName: "node-one",
						InitContainers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
					Status: v1.PodStatus{
						Phase: v1.PodPending,
						Conditions: []v1.PodCondition{
							{Type: "ArbitraryCondition", Status: v1.ConditionTrue},
						},
					},
				},
			},
			expected: `
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="",pod="foo-unscheduled-succeeded",priority="",resource="cpu",scheduler="",unit="cores"} 1
				kube_pod_resource_request{namespace="test",node="node-one",pod="foo-pending",priority="",resource="cpu",scheduler="",unit="cores"} 1
				kube_pod_resource_request{namespace="test",node="node-one",pod="foo-unknown",priority="",resource="cpu",scheduler="",unit="cores"} 1
				`,
		},
		{
			name: "zero resource should be excluded",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":                    resource.MustParse("0"),
									"custom":                 resource.MustParse("0"),
									"test.com/custom-metric": resource.MustParse("0"),
								},
								Limits: v1.ResourceList{
									"cpu":                    resource.MustParse("0"),
									"custom":                 resource.MustParse("0"),
									"test.com/custom-metric": resource.MustParse("0"),
								},
							}},
						},
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":                    resource.MustParse("0"),
									"custom":                 resource.MustParse("0"),
									"test.com/custom-metric": resource.MustParse("0"),
								},
								Limits: v1.ResourceList{
									"cpu":                    resource.MustParse("0"),
									"custom":                 resource.MustParse("0"),
									"test.com/custom-metric": resource.MustParse("0"),
								},
							}},
						},
					},
				},
			},
			expected: ``,
		},
		{
			name: "optional field labels",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						SchedulerName: "default-scheduler",
						Priority:      ptr.To[int32](0),
						NodeName:      "node-one",
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{"cpu": resource.MustParse("1")}}},
						},
					},
				},
			},
			expected: `
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="node-one",pod="foo",priority="0",resource="cpu",scheduler="default-scheduler",unit="cores"} 1
				`,
		},
		{
			name: "init containers and regular containers when initialized",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						NodeName: "node-one",
						InitContainers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":    resource.MustParse("2"),
									"custom": resource.MustParse("3"),
								},
								Limits: v1.ResourceList{
									"memory": resource.MustParse("1G"),
									"custom": resource.MustParse("5"),
								},
							}},
						},
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":    resource.MustParse("1"),
									"custom": resource.MustParse("0"),
								},
								Limits: v1.ResourceList{
									"memory": resource.MustParse("2G"),
									"custom": resource.MustParse("6"),
								},
							}},
						},
					},
					Status: v1.PodStatus{
						Conditions: []v1.PodCondition{
							{Type: v1.PodInitialized, Status: v1.ConditionTrue},
						},
					},
				},
			},
			expected: `
				# HELP kube_pod_resource_limit [STABLE] Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_limit gauge
				kube_pod_resource_limit{namespace="test",node="node-one",pod="foo",priority="",resource="custom",scheduler="",unit=""} 6
				kube_pod_resource_limit{namespace="test",node="node-one",pod="foo",priority="",resource="memory",scheduler="",unit="bytes"} 2e+09
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="node-one",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 2
				kube_pod_resource_request{namespace="test",node="node-one",pod="foo",priority="",resource="custom",scheduler="",unit=""} 3
				`,
		},
		{
			name: "init containers and regular containers when initializing",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						NodeName: "node-one",
						InitContainers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":    resource.MustParse("2"),
									"custom": resource.MustParse("3"),
								},
								Limits: v1.ResourceList{
									"memory": resource.MustParse("1G"),
									"custom": resource.MustParse("5"),
								},
							}},
						},
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":    resource.MustParse("1"),
									"custom": resource.MustParse("0"),
								},
								Limits: v1.ResourceList{
									"memory": resource.MustParse("2G"),
									"custom": resource.MustParse("6"),
								},
							}},
						},
					},
					Status: v1.PodStatus{
						Conditions: []v1.PodCondition{
							{Type: "AnotherCondition", Status: v1.ConditionUnknown},
							{Type: v1.PodInitialized, Status: v1.ConditionFalse},
						},
					},
				},
			},
			expected: `
				# HELP kube_pod_resource_limit [STABLE] Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_limit gauge
				kube_pod_resource_limit{namespace="test",node="node-one",pod="foo",priority="",resource="custom",scheduler="",unit=""} 6
				kube_pod_resource_limit{namespace="test",node="node-one",pod="foo",priority="",resource="memory",scheduler="",unit="bytes"} 2e+09
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="node-one",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 2
				kube_pod_resource_request{namespace="test",node="node-one",pod="foo",priority="",resource="custom",scheduler="",unit=""} 3
				`,
		},
		{
			name: "aggregate container requests and limits",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{"cpu": resource.MustParse("1")},
								Limits:   v1.ResourceList{"cpu": resource.MustParse("2")},
							}},
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{"memory": resource.MustParse("1G")},
								Limits:   v1.ResourceList{"memory": resource.MustParse("2G")},
							}},
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{"cpu": resource.MustParse("0.5")},
								Limits:   v1.ResourceList{"cpu": resource.MustParse("1.25")},
							}},
							{Resources: v1.ResourceRequirements{
								Limits: v1.ResourceList{"memory": resource.MustParse("2G")},
							}},
						},
					},
				},
			},
			expected: `            
				# HELP kube_pod_resource_limit [STABLE] Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_limit gauge
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 3.25
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="memory",scheduler="",unit="bytes"} 4e+09
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 1.5
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="memory",scheduler="",unit="bytes"} 1e+09
				`,
		},
		{
			name: "overhead added to requests and limits",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						Overhead: v1.ResourceList{
							"cpu":    resource.MustParse("0.25"),
							"memory": resource.MustParse("0.75G"),
							"custom": resource.MustParse("0.5"),
						},
						InitContainers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":    resource.MustParse("2"),
									"custom": resource.MustParse("3"),
								},
								Limits: v1.ResourceList{
									"memory": resource.MustParse("1G"),
									"custom": resource.MustParse("5"),
								},
							}},
						},
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"cpu":    resource.MustParse("1"),
									"custom": resource.MustParse("0"),
								},
								Limits: v1.ResourceList{
									"memory": resource.MustParse("2G"),
									"custom": resource.MustParse("6"),
								},
							}},
						},
					},
				},
			},
			expected: `
				# HELP kube_pod_resource_limit [STABLE] Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_limit gauge
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="custom",scheduler="",unit=""} 6.5
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="memory",scheduler="",unit="bytes"} 2.75e+09
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="cpu",scheduler="",unit="cores"} 2.25
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="custom",scheduler="",unit=""} 3.5
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="memory",scheduler="",unit="bytes"} 7.5e+08
				`,
		},
		{
			name: "units for standard resources",
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									"storage":           resource.MustParse("5"),
									"ephemeral-storage": resource.MustParse("6"),
								},
								Limits: v1.ResourceList{
									"hugepages-x":            resource.MustParse("1"),
									"hugepages-":             resource.MustParse("2"),
									"attachable-volumes-aws": resource.MustParse("3"),
									"attachable-volumes-":    resource.MustParse("4"),
								},
							}},
						},
					},
				},
			},
			expected: `
				# HELP kube_pod_resource_limit [STABLE] Resources limit for workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_limit gauge
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="attachable-volumes-",scheduler="",unit="integer"} 4
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="attachable-volumes-aws",scheduler="",unit="integer"} 3
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="hugepages-",scheduler="",unit="bytes"} 2
				kube_pod_resource_limit{namespace="test",node="",pod="foo",priority="",resource="hugepages-x",scheduler="",unit="bytes"} 1
				# HELP kube_pod_resource_request [STABLE] Resources requested by workloads on the cluster, broken down by pod. This shows the resource usage the scheduler and kubelet expect per pod for resources along with the unit for the resource if any.
				# TYPE kube_pod_resource_request gauge
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="ephemeral-storage",scheduler="",unit="bytes"} 6
				kube_pod_resource_request{namespace="test",node="",pod="foo",priority="",resource="storage",scheduler="",unit="bytes"} 5
				`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewPodResourcesMetricsCollector(&fakePodLister{pods: tt.pods})
			registry := metrics.NewKubeRegistry()
			registry.CustomMustRegister(c)
			err := testutil.GatherAndCompare(registry, strings.NewReader(tt.expected))
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
