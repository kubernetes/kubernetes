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

package core

import (
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
	"k8s.io/apiserver/pkg/quota/v1/generic"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/util/node"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/ptr"
)

func TestPodConstraintsFunc(t *testing.T) {
	testCases := map[string]struct {
		pod                      *api.Pod
		required                 []corev1.ResourceName
		err                      string
		podLevelResourcesEnabled bool
	}{
		"init container resource missing": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Name: "dummy",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []corev1.ResourceName{corev1.ResourceMemory},
			err:      `must specify memory for: dummy`,
		},
		"multiple init container resource missing": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Name: "foo",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}, {
						Name: "bar",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []corev1.ResourceName{corev1.ResourceMemory},
			err:      `must specify memory for: bar,foo`,
		},
		"container resource missing": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name: "dummy",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []corev1.ResourceName{corev1.ResourceMemory},
			err:      `must specify memory for: dummy`,
		},
		"multiple container resource missing": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name: "foo",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}, {
						Name: "bar",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			required: []corev1.ResourceName{corev1.ResourceMemory},
			err:      `must specify memory for: bar,foo`,
		},
		"container resource missing multiple": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Name:      "foo",
						Resources: api.ResourceRequirements{},
					}, {
						Name:      "bar",
						Resources: api.ResourceRequirements{},
					}},
				},
			},
			required: []corev1.ResourceName{corev1.ResourceMemory, corev1.ResourceCPU},
			err:      `must specify cpu for: bar,foo; memory for: bar,foo`,
		},
		"pod-level resource set, container-level required resources missing": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Resources: &api.ResourceRequirements{
						Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
					},
					Containers: []api.Container{{
						Name:      "foo",
						Resources: api.ResourceRequirements{},
					}, {
						Name:      "bar",
						Resources: api.ResourceRequirements{},
					}},
				},
			},
			required:                 []corev1.ResourceName{corev1.ResourceMemory, corev1.ResourceCPU},
			podLevelResourcesEnabled: true,
			err:                      ``,
		},
	}
	evaluator := NewPodEvaluator(nil, clock.RealClock{})
	for testName, test := range testCases {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, test.podLevelResourcesEnabled)

		err := evaluator.Constraints(test.required, test.pod)
		switch {
		case err != nil && len(test.err) == 0,
			err == nil && len(test.err) != 0,
			err != nil && test.err != err.Error():
			t.Errorf("%s want: %v,got: %v", testName, test.err, err)
		}
	}
}

func TestPodEvaluatorUsage(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	evaluator := NewPodEvaluator(nil, fakeClock)

	// fields use to simulate a pod undergoing termination
	// note: we set the deletion time in the past
	now := fakeClock.Now()
	terminationGracePeriodSeconds := int64(30)
	deletionTimestampPastGracePeriod := metav1.NewTime(now.Add(time.Duration(terminationGracePeriodSeconds) * time.Second * time.Duration(-2)))
	deletionTimestampNotPastGracePeriod := metav1.NewTime(fakeClock.Now())

	testCases := map[string]struct {
		pod                      *api.Pod
		usage                    corev1.ResourceList
		podLevelResourcesEnabled bool
	}{
		"init container CPU": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("1m"),
				corev1.ResourceLimitsCPU:   resource.MustParse("2m"),
				corev1.ResourcePods:        resource.MustParse("1"),
				corev1.ResourceCPU:         resource.MustParse("1m"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"init container MEM": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceMemory: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceMemory: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceRequestsMemory: resource.MustParse("1m"),
				corev1.ResourceLimitsMemory:   resource.MustParse("2m"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceMemory:         resource.MustParse("1m"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"init container local ephemeral storage": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceEphemeralStorage: resource.MustParse("32Mi")},
							Limits:   api.ResourceList{api.ResourceEphemeralStorage: resource.MustParse("64Mi")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceEphemeralStorage:         resource.MustParse("32Mi"),
				corev1.ResourceRequestsEphemeralStorage: resource.MustParse("32Mi"),
				corev1.ResourceLimitsEphemeralStorage:   resource.MustParse("64Mi"),
				corev1.ResourcePods:                     resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"init container hugepages": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceName(api.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("100Mi")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceName(corev1.ResourceHugePagesPrefix + "2Mi"):         resource.MustParse("100Mi"),
				corev1.ResourceName(corev1.ResourceRequestsHugePagesPrefix + "2Mi"): resource.MustParse("100Mi"),
				corev1.ResourcePods: resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"init container extended resources": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceName("example.com/dongle"): resource.MustParse("3")},
							Limits:   api.ResourceList{api.ResourceName("example.com/dongle"): resource.MustParse("3")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceName("requests.example.com/dongle"): resource.MustParse("3"),
				corev1.ResourcePods: resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"container CPU": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("1m"),
				corev1.ResourceLimitsCPU:   resource.MustParse("2m"),
				corev1.ResourcePods:        resource.MustParse("1"),
				corev1.ResourceCPU:         resource.MustParse("1m"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"container MEM": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceMemory: resource.MustParse("1m")},
							Limits:   api.ResourceList{api.ResourceMemory: resource.MustParse("2m")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceRequestsMemory: resource.MustParse("1m"),
				corev1.ResourceLimitsMemory:   resource.MustParse("2m"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceMemory:         resource.MustParse("1m"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"container local ephemeral storage": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceEphemeralStorage: resource.MustParse("32Mi")},
							Limits:   api.ResourceList{api.ResourceEphemeralStorage: resource.MustParse("64Mi")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceEphemeralStorage:         resource.MustParse("32Mi"),
				corev1.ResourceRequestsEphemeralStorage: resource.MustParse("32Mi"),
				corev1.ResourceLimitsEphemeralStorage:   resource.MustParse("64Mi"),
				corev1.ResourcePods:                     resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"container hugepages": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceName(api.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("100Mi")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceName(api.ResourceHugePagesPrefix + "2Mi"):         resource.MustParse("100Mi"),
				corev1.ResourceName(api.ResourceRequestsHugePagesPrefix + "2Mi"): resource.MustParse("100Mi"),
				corev1.ResourcePods: resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"container extended resources": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceName("example.com/dongle"): resource.MustParse("3")},
							Limits:   api.ResourceList{api.ResourceName("example.com/dongle"): resource.MustParse("3")},
						},
					}},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceName("requests.example.com/dongle"): resource.MustParse("3"),
				corev1.ResourcePods: resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"terminated generic count still appears": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceName("example.com/dongle"): resource.MustParse("3")},
							Limits:   api.ResourceList{api.ResourceName("example.com/dongle"): resource.MustParse("3")},
						},
					}},
				},
				Status: api.PodStatus{
					Phase: api.PodSucceeded,
				},
			},
			usage: corev1.ResourceList{
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"init container maximums override sum of containers": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					InitContainers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("4"),
									api.ResourceMemory:                     resource.MustParse("100M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("4"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("8"),
									api.ResourceMemory:                     resource.MustParse("200M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("4"),
								},
							},
						},
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("1"),
									api.ResourceMemory:                     resource.MustParse("50M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("2"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("2"),
									api.ResourceMemory:                     resource.MustParse("100M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("2"),
								},
							},
						},
					},
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("1"),
									api.ResourceMemory:                     resource.MustParse("50M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("1"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("2"),
									api.ResourceMemory:                     resource.MustParse("100M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("1"),
								},
							},
						},
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("2"),
									api.ResourceMemory:                     resource.MustParse("25M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("2"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:                        resource.MustParse("5"),
									api.ResourceMemory:                     resource.MustParse("50M"),
									api.ResourceName("example.com/dongle"): resource.MustParse("2"),
								},
							},
						},
					},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceRequestsCPU:                         resource.MustParse("4"),
				corev1.ResourceRequestsMemory:                      resource.MustParse("100M"),
				corev1.ResourceLimitsCPU:                           resource.MustParse("8"),
				corev1.ResourceLimitsMemory:                        resource.MustParse("200M"),
				corev1.ResourcePods:                                resource.MustParse("1"),
				corev1.ResourceCPU:                                 resource.MustParse("4"),
				corev1.ResourceMemory:                              resource.MustParse("100M"),
				corev1.ResourceName("requests.example.com/dongle"): resource.MustParse("4"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"pod deletion timestamp exceeded": {
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp:          &deletionTimestampPastGracePeriod,
					DeletionGracePeriodSeconds: &terminationGracePeriodSeconds,
				},
				Status: api.PodStatus{
					Reason: node.NodeUnreachablePodReason,
				},
				Spec: api.PodSpec{
					TerminationGracePeriodSeconds: &terminationGracePeriodSeconds,
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("1"),
									api.ResourceMemory: resource.MustParse("50M"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("2"),
									api.ResourceMemory: resource.MustParse("100M"),
								},
							},
						},
					},
				},
			},
			usage: corev1.ResourceList{
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"pod deletion timestamp not exceeded": {
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp:          &deletionTimestampNotPastGracePeriod,
					DeletionGracePeriodSeconds: &terminationGracePeriodSeconds,
				},
				Status: api.PodStatus{
					Reason: node.NodeUnreachablePodReason,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU: resource.MustParse("1"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU: resource.MustParse("2"),
								},
							},
						},
					},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("1"),
				corev1.ResourceLimitsCPU:   resource.MustParse("2"),
				corev1.ResourcePods:        resource.MustParse("1"),
				corev1.ResourceCPU:         resource.MustParse("1"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"count pod overhead as usage": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Overhead: api.ResourceList{
						api.ResourceCPU: resource.MustParse("1"),
					},
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU: resource.MustParse("1"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU: resource.MustParse("2"),
								},
							},
						},
					},
				},
			},
			usage: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("2"),
				corev1.ResourceLimitsCPU:   resource.MustParse("3"),
				corev1.ResourcePods:        resource.MustParse("1"),
				corev1.ResourceCPU:         resource.MustParse("2"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"pod-level CPU": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Resources: &api.ResourceRequirements{
						Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("1m")},
						Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("2m")},
					},
				},
			},
			podLevelResourcesEnabled: true,
			usage: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("1m"),
				corev1.ResourceLimitsCPU:   resource.MustParse("2m"),
				corev1.ResourcePods:        resource.MustParse("1"),
				corev1.ResourceCPU:         resource.MustParse("1m"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"pod-level Memory": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Resources: &api.ResourceRequirements{
						Requests: api.ResourceList{api.ResourceMemory: resource.MustParse("1Mi")},
						Limits:   api.ResourceList{api.ResourceMemory: resource.MustParse("2Mi")},
					},
				},
			},
			podLevelResourcesEnabled: true,
			usage: corev1.ResourceList{
				corev1.ResourceRequestsMemory: resource.MustParse("1Mi"),
				corev1.ResourceLimitsMemory:   resource.MustParse("2Mi"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceMemory:         resource.MustParse("1Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"pod-level memory with container-level ephemeral storage": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Resources: &api.ResourceRequirements{
						Requests: api.ResourceList{api.ResourceMemory: resource.MustParse("1Mi")},
						Limits:   api.ResourceList{api.ResourceMemory: resource.MustParse("2Mi")},
					},
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceEphemeralStorage: resource.MustParse("32Mi")},
							Limits:   api.ResourceList{api.ResourceEphemeralStorage: resource.MustParse("64Mi")},
						},
					}},
				},
			},
			podLevelResourcesEnabled: true,
			usage: corev1.ResourceList{
				corev1.ResourceEphemeralStorage:         resource.MustParse("32Mi"),
				corev1.ResourceRequestsEphemeralStorage: resource.MustParse("32Mi"),
				corev1.ResourceLimitsEphemeralStorage:   resource.MustParse("64Mi"),
				corev1.ResourcePods:                     resource.MustParse("1"),
				corev1.ResourceRequestsMemory:           resource.MustParse("1Mi"),
				corev1.ResourceLimitsMemory:             resource.MustParse("2Mi"),
				corev1.ResourceMemory:                   resource.MustParse("1Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
	}
	t.Parallel()
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, testCase.podLevelResourcesEnabled)
			actual, err := evaluator.Usage(testCase.pod)
			if err != nil {
				t.Error(err)
			}
			if !quota.Equals(testCase.usage, actual) {
				t.Errorf("expected: %v, actual: %v", testCase.usage, actual)
			}
		})
	}
}

func TestPodEvaluatorUsageStats(t *testing.T) {
	cpu1 := api.ResourceList{api.ResourceCPU: resource.MustParse("1")}
	tests := []struct {
		name               string
		objs               []runtime.Object
		quotaScopes        []corev1.ResourceQuotaScope
		quotaScopeSelector *corev1.ScopeSelector
		want               corev1.ResourceList
	}{
		{
			name: "nil case",
		},
		{
			name: "all pods in running state",
			objs: []runtime.Object{
				makePod("p1", "", cpu1, api.PodRunning),
				makePod("p2", "", cpu1, api.PodRunning),
			},
			want: corev1.ResourceList{
				corev1.ResourcePods:               resource.MustParse("2"),
				corev1.ResourceName("count/pods"): resource.MustParse("2"),
				corev1.ResourceCPU:                resource.MustParse("2"),
				corev1.ResourceRequestsCPU:        resource.MustParse("2"),
				corev1.ResourceLimitsCPU:          resource.MustParse("2"),
			},
		},
		{
			name: "one pods in terminal state",
			objs: []runtime.Object{
				makePod("p1", "", cpu1, api.PodRunning),
				makePod("p2", "", cpu1, api.PodSucceeded),
			},
			want: corev1.ResourceList{
				corev1.ResourcePods:               resource.MustParse("1"),
				corev1.ResourceName("count/pods"): resource.MustParse("2"),
				corev1.ResourceCPU:                resource.MustParse("1"),
				corev1.ResourceRequestsCPU:        resource.MustParse("1"),
				corev1.ResourceLimitsCPU:          resource.MustParse("1"),
			},
		},
		{
			name: "partial pods matching quotaScopeSelector",
			objs: []runtime.Object{
				makePod("p1", "high-priority", cpu1, api.PodRunning),
				makePod("p2", "high-priority", cpu1, api.PodSucceeded),
				makePod("p3", "low-priority", cpu1, api.PodRunning),
			},
			quotaScopeSelector: &corev1.ScopeSelector{
				MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
					{
						ScopeName: corev1.ResourceQuotaScopePriorityClass,
						Operator:  corev1.ScopeSelectorOpIn,
						Values:    []string{"high-priority"},
					},
				},
			},
			want: corev1.ResourceList{
				corev1.ResourcePods:               resource.MustParse("1"),
				corev1.ResourceName("count/pods"): resource.MustParse("2"),
				corev1.ResourceCPU:                resource.MustParse("1"),
				corev1.ResourceRequestsCPU:        resource.MustParse("1"),
				corev1.ResourceLimitsCPU:          resource.MustParse("1"),
			},
		},
		{
			name: "partial pods matching quotaScopeSelector - w/ scopeName specified",
			objs: []runtime.Object{
				makePod("p1", "high-priority", cpu1, api.PodRunning),
				makePod("p2", "high-priority", cpu1, api.PodSucceeded),
				makePod("p3", "low-priority", cpu1, api.PodRunning),
			},
			quotaScopes: []corev1.ResourceQuotaScope{
				corev1.ResourceQuotaScopePriorityClass,
			},
			quotaScopeSelector: &corev1.ScopeSelector{
				MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
					{
						ScopeName: corev1.ResourceQuotaScopePriorityClass,
						Operator:  corev1.ScopeSelectorOpIn,
						Values:    []string{"high-priority"},
					},
				},
			},
			want: corev1.ResourceList{
				corev1.ResourcePods:               resource.MustParse("1"),
				corev1.ResourceName("count/pods"): resource.MustParse("2"),
				corev1.ResourceCPU:                resource.MustParse("1"),
				corev1.ResourceRequestsCPU:        resource.MustParse("1"),
				corev1.ResourceLimitsCPU:          resource.MustParse("1"),
			},
		},
		{
			name: "partial pods matching quotaScopeSelector - w/ multiple scopeNames specified",
			objs: []runtime.Object{
				makePod("p1", "high-priority", cpu1, api.PodRunning),
				makePod("p2", "high-priority", cpu1, api.PodSucceeded),
				makePod("p3", "low-priority", cpu1, api.PodRunning),
				makePod("p4", "high-priority", nil, api.PodFailed),
			},
			quotaScopes: []corev1.ResourceQuotaScope{
				corev1.ResourceQuotaScopePriorityClass,
				corev1.ResourceQuotaScopeBestEffort,
			},
			quotaScopeSelector: &corev1.ScopeSelector{
				MatchExpressions: []corev1.ScopedResourceSelectorRequirement{
					{
						ScopeName: corev1.ResourceQuotaScopePriorityClass,
						Operator:  corev1.ScopeSelectorOpIn,
						Values:    []string{"high-priority"},
					},
				},
			},
			want: corev1.ResourceList{
				corev1.ResourceName("count/pods"): resource.MustParse("1"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gvr := corev1.SchemeGroupVersion.WithResource("pods")
			listerForPod := map[schema.GroupVersionResource]cache.GenericLister{
				gvr: newGenericLister(gvr.GroupResource(), tt.objs),
			}
			evaluator := NewPodEvaluator(mockListerForResourceFunc(listerForPod), testingclock.NewFakeClock(time.Now()))
			usageStatsOption := quota.UsageStatsOptions{
				Scopes:        tt.quotaScopes,
				ScopeSelector: tt.quotaScopeSelector,
			}
			actual, err := evaluator.UsageStats(usageStatsOption)
			if err != nil {
				t.Error(err)
			}
			if !quota.Equals(tt.want, actual.Used) {
				t.Errorf("expected: %v, actual: %v", tt.want, actual.Used)
			}
		})
	}
}

func TestPodEvaluatorMatchingScopes(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	evaluator := NewPodEvaluator(nil, fakeClock)
	activeDeadlineSeconds := int64(30)
	testCases := map[string]struct {
		pod           *api.Pod
		selectors     []corev1.ScopedResourceSelectorRequirement
		wantSelectors []corev1.ScopedResourceSelectorRequirement
	}{
		"EmptyPod": {
			pod: &api.Pod{},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeNotTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
			},
		},
		"PriorityClass": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					PriorityClassName: "class1",
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeNotTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
				{ScopeName: corev1.ResourceQuotaScopePriorityClass, Operator: corev1.ScopeSelectorOpIn, Values: []string{"class1"}},
			},
		},
		"NotBestEffort": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{{
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{
								api.ResourceCPU:                        resource.MustParse("1"),
								api.ResourceMemory:                     resource.MustParse("50M"),
								api.ResourceName("example.com/dongle"): resource.MustParse("1"),
							},
							Limits: api.ResourceList{
								api.ResourceCPU:                        resource.MustParse("2"),
								api.ResourceMemory:                     resource.MustParse("100M"),
								api.ResourceName("example.com/dongle"): resource.MustParse("1"),
							},
						},
					}},
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeNotTerminating},
				{ScopeName: corev1.ResourceQuotaScopeNotBestEffort},
			},
		},
		"Terminating": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
			},
		},
		"OnlyTerminating": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
				},
			},
			selectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
			},
		},
		"CrossNamespaceRequiredAffinity": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
					Affinity: &api.Affinity{
						PodAffinity: &api.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
								{LabelSelector: &metav1.LabelSelector{}, Namespaces: []string{"ns1"}, NamespaceSelector: &metav1.LabelSelector{}},
							},
						},
					},
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
				{ScopeName: corev1.ResourceQuotaScopeCrossNamespacePodAffinity},
			},
		},
		"CrossNamespaceRequiredAffinityWithSlice": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
					Affinity: &api.Affinity{
						PodAffinity: &api.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
								{LabelSelector: &metav1.LabelSelector{}, Namespaces: []string{"ns1"}},
							},
						},
					},
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
				{ScopeName: corev1.ResourceQuotaScopeCrossNamespacePodAffinity},
			},
		},
		"CrossNamespacePreferredAffinity": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
					Affinity: &api.Affinity{
						PodAffinity: &api.PodAffinity{
							PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
								{PodAffinityTerm: api.PodAffinityTerm{LabelSelector: &metav1.LabelSelector{}, Namespaces: []string{"ns2"}, NamespaceSelector: &metav1.LabelSelector{}}},
							},
						},
					},
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
				{ScopeName: corev1.ResourceQuotaScopeCrossNamespacePodAffinity},
			},
		},
		"CrossNamespacePreferredAffinityWithSelector": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
					Affinity: &api.Affinity{
						PodAffinity: &api.PodAffinity{
							PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
								{PodAffinityTerm: api.PodAffinityTerm{LabelSelector: &metav1.LabelSelector{}, NamespaceSelector: &metav1.LabelSelector{}}},
							},
						},
					},
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
				{ScopeName: corev1.ResourceQuotaScopeCrossNamespacePodAffinity},
			},
		},
		"CrossNamespacePreferredAntiAffinity": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
					Affinity: &api.Affinity{
						PodAntiAffinity: &api.PodAntiAffinity{
							PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
								{PodAffinityTerm: api.PodAffinityTerm{LabelSelector: &metav1.LabelSelector{}, NamespaceSelector: &metav1.LabelSelector{}}},
							},
						},
					},
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
				{ScopeName: corev1.ResourceQuotaScopeCrossNamespacePodAffinity},
			},
		},
		"CrossNamespaceRequiredAntiAffinity": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					ActiveDeadlineSeconds: &activeDeadlineSeconds,
					Affinity: &api.Affinity{
						PodAntiAffinity: &api.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
								{LabelSelector: &metav1.LabelSelector{}, Namespaces: []string{"ns3"}},
							},
						},
					},
				},
			},
			wantSelectors: []corev1.ScopedResourceSelectorRequirement{
				{ScopeName: corev1.ResourceQuotaScopeTerminating},
				{ScopeName: corev1.ResourceQuotaScopeBestEffort},
				{ScopeName: corev1.ResourceQuotaScopeCrossNamespacePodAffinity},
			},
		},
	}
	for testName, testCase := range testCases {
		t.Run(testName, func(t *testing.T) {
			if testCase.selectors == nil {
				testCase.selectors = []corev1.ScopedResourceSelectorRequirement{
					{ScopeName: corev1.ResourceQuotaScopeTerminating},
					{ScopeName: corev1.ResourceQuotaScopeNotTerminating},
					{ScopeName: corev1.ResourceQuotaScopeBestEffort},
					{ScopeName: corev1.ResourceQuotaScopeNotBestEffort},
					{ScopeName: corev1.ResourceQuotaScopePriorityClass, Operator: corev1.ScopeSelectorOpIn, Values: []string{"class1"}},
					{ScopeName: corev1.ResourceQuotaScopeCrossNamespacePodAffinity},
				}
			}
			gotSelectors, err := evaluator.MatchingScopes(testCase.pod, testCase.selectors)
			if err != nil {
				t.Error(err)
			}
			if diff := cmp.Diff(testCase.wantSelectors, gotSelectors); diff != "" {
				t.Errorf("%v: unexpected diff (-want, +got):\n%s", testName, diff)
			}
		})
	}
}

func TestPodEvaluatorUsageResourceResize(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	evaluator := NewPodEvaluator(nil, fakeClock)

	testCases := map[string]struct {
		pod             *api.Pod
		usageFgEnabled  corev1.ResourceList
		usageFgDisabled corev1.ResourceList
	}{
		"verify Max(Container.Spec.Requests, ContainerStatus.Resources) for memory resource": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceMemory: resource.MustParse("200Mi"),
								},
								Limits: api.ResourceList{
									api.ResourceMemory: resource.MustParse("400Mi"),
								},
							},
						},
					},
				},
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{
							Resources: &api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceMemory: resource.MustParse("150Mi"),
								},
							},
						},
					},
				},
			},
			usageFgEnabled: corev1.ResourceList{
				corev1.ResourceRequestsMemory: resource.MustParse("200Mi"),
				corev1.ResourceLimitsMemory:   resource.MustParse("400Mi"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceMemory:         resource.MustParse("200Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
			usageFgDisabled: corev1.ResourceList{
				corev1.ResourceRequestsMemory: resource.MustParse("200Mi"),
				corev1.ResourceLimitsMemory:   resource.MustParse("400Mi"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceMemory:         resource.MustParse("200Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"verify Max(Container.Spec.Requests, ContainerStatus.Resources) for CPU resource": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU: resource.MustParse("100m"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU: resource.MustParse("200m"),
								},
							},
						},
					},
				},
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{
							Resources: &api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU: resource.MustParse("150m"),
								},
							},
						},
					},
				},
			},
			usageFgEnabled: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("150m"),
				corev1.ResourceLimitsCPU:   resource.MustParse("200m"),
				corev1.ResourcePods:        resource.MustParse("1"),
				corev1.ResourceCPU:         resource.MustParse("150m"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
			usageFgDisabled: corev1.ResourceList{
				corev1.ResourceRequestsCPU: resource.MustParse("100m"),
				corev1.ResourceLimitsCPU:   resource.MustParse("200m"),
				corev1.ResourcePods:        resource.MustParse("1"),
				corev1.ResourceCPU:         resource.MustParse("100m"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"verify Max(Container.Spec.Requests, ContainerStatus.Resources) for CPU and memory resource": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("100m"),
									api.ResourceMemory: resource.MustParse("200Mi"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("200m"),
									api.ResourceMemory: resource.MustParse("400Mi"),
								},
							},
						},
					},
				},
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{
							Resources: &api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("150m"),
									api.ResourceMemory: resource.MustParse("250Mi"),
								},
							},
						},
					},
				},
			},
			usageFgEnabled: corev1.ResourceList{
				corev1.ResourceRequestsCPU:    resource.MustParse("150m"),
				corev1.ResourceLimitsCPU:      resource.MustParse("200m"),
				corev1.ResourceRequestsMemory: resource.MustParse("250Mi"),
				corev1.ResourceLimitsMemory:   resource.MustParse("400Mi"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceCPU:            resource.MustParse("150m"),
				corev1.ResourceMemory:         resource.MustParse("250Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
			usageFgDisabled: corev1.ResourceList{
				corev1.ResourceRequestsCPU:    resource.MustParse("100m"),
				corev1.ResourceLimitsCPU:      resource.MustParse("200m"),
				corev1.ResourceRequestsMemory: resource.MustParse("200Mi"),
				corev1.ResourceLimitsMemory:   resource.MustParse("400Mi"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceCPU:            resource.MustParse("100m"),
				corev1.ResourceMemory:         resource.MustParse("200Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
		"verify Max(Container.Spec.Requests, ContainerStatus.Resources==nil) for CPU and memory resource": {
			pod: &api.Pod{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Resources: api.ResourceRequirements{
								Requests: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("100m"),
									api.ResourceMemory: resource.MustParse("200Mi"),
								},
								Limits: api.ResourceList{
									api.ResourceCPU:    resource.MustParse("200m"),
									api.ResourceMemory: resource.MustParse("400Mi"),
								},
							},
						},
					},
				},
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						{},
					},
				},
			},
			usageFgEnabled: corev1.ResourceList{
				corev1.ResourceRequestsCPU:    resource.MustParse("100m"),
				corev1.ResourceLimitsCPU:      resource.MustParse("200m"),
				corev1.ResourceRequestsMemory: resource.MustParse("200Mi"),
				corev1.ResourceLimitsMemory:   resource.MustParse("400Mi"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceCPU:            resource.MustParse("100m"),
				corev1.ResourceMemory:         resource.MustParse("200Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
			usageFgDisabled: corev1.ResourceList{
				corev1.ResourceRequestsCPU:    resource.MustParse("100m"),
				corev1.ResourceLimitsCPU:      resource.MustParse("200m"),
				corev1.ResourceRequestsMemory: resource.MustParse("200Mi"),
				corev1.ResourceLimitsMemory:   resource.MustParse("400Mi"),
				corev1.ResourcePods:           resource.MustParse("1"),
				corev1.ResourceCPU:            resource.MustParse("100m"),
				corev1.ResourceMemory:         resource.MustParse("200Mi"),
				generic.ObjectCountQuotaResourceNameFor(schema.GroupResource{Resource: "pods"}): resource.MustParse("1"),
			},
		},
	}
	t.Parallel()
	for _, enabled := range []bool{true, false} {
		for testName, testCase := range testCases {
			t.Run(testName, func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, enabled)
				actual, err := evaluator.Usage(testCase.pod)
				if err != nil {
					t.Error(err)
				}
				usage := testCase.usageFgEnabled
				if !enabled {
					usage = testCase.usageFgDisabled
				}
				if !quota.Equals(usage, actual) {
					t.Errorf("FG enabled: %v, expected: %v, actual: %v", enabled, usage, actual)
				}
			})
		}
	}
}

func BenchmarkPodMatchesScopeFunc(b *testing.B) {
	pod, _ := toExternalPodOrError(makePod("p1", "high-priority",
		api.ResourceList{api.ResourceCPU: resource.MustParse("1")}, api.PodRunning))

	tests := []struct {
		name     string
		selector corev1.ScopedResourceSelectorRequirement
	}{
		{
			name: "PriorityClass selector w/o operator",
			selector: corev1.ScopedResourceSelectorRequirement{
				ScopeName: corev1.ResourceQuotaScopePriorityClass,
			},
		},
		{
			name: "PriorityClass selector w/ 'Exists' operator",
			selector: corev1.ScopedResourceSelectorRequirement{
				ScopeName: corev1.ResourceQuotaScopePriorityClass,
				Operator:  corev1.ScopeSelectorOpExists,
			},
		},
		{
			name: "BestEfforts selector w/o operator",
			selector: corev1.ScopedResourceSelectorRequirement{
				ScopeName: corev1.ResourceQuotaScopeBestEffort,
			},
		},
		{
			name: "BestEfforts selector w/ 'Exists' operator",
			selector: corev1.ScopedResourceSelectorRequirement{
				ScopeName: corev1.ResourceQuotaScopeBestEffort,
				Operator:  corev1.ScopeSelectorOpExists,
			},
		},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_, _ = podMatchesScopeFunc(tt.selector, pod)
			}
		})
	}
}

func mockListerForResourceFunc(listerForResource map[schema.GroupVersionResource]cache.GenericLister) quota.ListerForResourceFunc {
	return func(gvr schema.GroupVersionResource) (cache.GenericLister, error) {
		lister, found := listerForResource[gvr]
		if !found {
			return nil, fmt.Errorf("no lister found for resource %v", gvr)
		}
		return lister, nil
	}
}

func newGenericLister(groupResource schema.GroupResource, items []runtime.Object) cache.GenericLister {
	store := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc})
	for _, item := range items {
		store.Add(item)
	}
	return cache.NewGenericLister(store, groupResource)
}

func makePod(name, pcName string, resList api.ResourceList, phase api.PodPhase) *api.Pod {
	return &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: api.PodSpec{
			PriorityClassName: pcName,
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Requests: resList,
						Limits:   resList,
					},
				},
			},
		},
		Status: api.PodStatus{
			Phase: phase,
		},
	}
}

func TestPodEvaluatorHandles(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	evaluator := NewPodEvaluator(nil, fakeClock)
	testCases := []struct {
		name  string
		attrs admission.Attributes
		want  bool
	}{
		{
			name:  "create",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Create, nil, false, nil),
			want:  true,
		},
		{
			name:  "update-activeDeadlineSeconds-to-nil",
			attrs: admission.NewAttributesRecord(&corev1.Pod{}, &corev1.Pod{Spec: corev1.PodSpec{ActiveDeadlineSeconds: ptr.To[int64](1)}}, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Update, nil, false, nil),
			want:  true,
		},
		{
			name:  "update-activeDeadlineSeconds-from-nil",
			attrs: admission.NewAttributesRecord(&corev1.Pod{Spec: corev1.PodSpec{ActiveDeadlineSeconds: ptr.To[int64](1)}}, &corev1.Pod{}, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Update, nil, false, nil),
			want:  true,
		},
		{
			name:  "update-activeDeadlineSeconds-with-different-values",
			attrs: admission.NewAttributesRecord(&corev1.Pod{Spec: corev1.PodSpec{ActiveDeadlineSeconds: ptr.To[int64](1)}}, &corev1.Pod{Spec: corev1.PodSpec{ActiveDeadlineSeconds: ptr.To[int64](2)}}, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Update, nil, false, nil),
			want:  false,
		},
		{
			name:  "update",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Update, nil, false, nil),
			want:  false,
		},
		{
			name:  "delete",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Delete, nil, false, nil),
			want:  false,
		},
		{
			name:  "connect",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "", admission.Connect, nil, false, nil),
			want:  false,
		},
		{
			name:  "create-subresource",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "subresource", admission.Create, nil, false, nil),
			want:  false,
		},
		{
			name:  "update-subresource",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "subresource", admission.Update, nil, false, nil),
			want:  false,
		},
		{
			name:  "update-resize",
			attrs: admission.NewAttributesRecord(nil, nil, schema.GroupVersionKind{Group: "core", Version: "v1", Kind: "Pod"}, "", "", schema.GroupVersionResource{Group: "core", Version: "v1", Resource: "pods"}, "resize", admission.Update, nil, false, nil),
			want:  true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := evaluator.Handles(tc.attrs)

			if tc.want != actual {
				t.Errorf("%s expected:\n%v\n, actual:\n%v", tc.name, tc.want, actual)
			}
		})
	}
}
