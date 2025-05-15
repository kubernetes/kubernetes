/*
Copyright 2025 The Kubernetes Authors.

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

package allocation

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/nodeshutdown"
	"k8s.io/kubernetes/pkg/kubelet/sysctl"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/tainttoleration"
)

func TestUpdatePodFromAllocation(t *testing.T) {
	containerRestartPolicyAlways := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345",
			Name:      "test",
			Namespace: "default",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(100, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(200, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(300, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(400, resource.DecimalSI),
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(600, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(700, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(800, resource.DecimalSI),
						},
					},
				},
			},
			InitContainers: []v1.Container{
				{
					Name: "c1-restartable-init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(200, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(300, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(400, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(500, resource.DecimalSI),
						},
					},
					RestartPolicy: &containerRestartPolicyAlways,
				},
				{
					Name: "c1-init",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(500, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(600, resource.DecimalSI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(700, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(800, resource.DecimalSI),
						},
					},
				},
			},
		},
	}

	resizedPod := pod.DeepCopy()
	resizedPod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(200, resource.DecimalSI)
	resizedPod.Spec.InitContainers[0].Resources.Requests[v1.ResourceCPU] = *resource.NewMilliQuantity(300, resource.DecimalSI)

	tests := []struct {
		name         string
		pod          *v1.Pod
		allocs       state.PodResourceInfoMap
		expectPod    *v1.Pod
		expectUpdate bool
	}{{
		name: "steady state",
		pod:  pod,
		allocs: state.PodResourceInfoMap{
			pod.UID: state.PodResourceInfo{
				ContainerResources: map[string]v1.ResourceRequirements{
					"c1":                  *pod.Spec.Containers[0].Resources.DeepCopy(),
					"c2":                  *pod.Spec.Containers[1].Resources.DeepCopy(),
					"c1-restartable-init": *pod.Spec.InitContainers[0].Resources.DeepCopy(),
					"c1-init":             *pod.Spec.InitContainers[1].Resources.DeepCopy(),
				},
			},
		},
		expectUpdate: false,
	}, {
		name:         "no allocations",
		pod:          pod,
		allocs:       state.PodResourceInfoMap{},
		expectUpdate: false,
	}, {
		name: "missing container allocation",
		pod:  pod,
		allocs: state.PodResourceInfoMap{
			pod.UID: state.PodResourceInfo{
				ContainerResources: map[string]v1.ResourceRequirements{
					"c2": *pod.Spec.Containers[1].Resources.DeepCopy(),
				},
			},
		},
		expectUpdate: false,
	}, {
		name: "resized container",
		pod:  pod,
		allocs: state.PodResourceInfoMap{
			pod.UID: state.PodResourceInfo{
				ContainerResources: map[string]v1.ResourceRequirements{
					"c1":                  *resizedPod.Spec.Containers[0].Resources.DeepCopy(),
					"c2":                  *resizedPod.Spec.Containers[1].Resources.DeepCopy(),
					"c1-restartable-init": *resizedPod.Spec.InitContainers[0].Resources.DeepCopy(),
					"c1-init":             *resizedPod.Spec.InitContainers[1].Resources.DeepCopy(),
				},
			},
		},
		expectUpdate: true,
		expectPod:    resizedPod,
	}}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pod := test.pod.DeepCopy()
			allocatedPod, updated := updatePodFromAllocation(pod, test.allocs)

			if test.expectUpdate {
				assert.True(t, updated, "updated")
				assert.Equal(t, test.expectPod, allocatedPod)
				assert.NotEqual(t, pod, allocatedPod)
			} else {
				assert.False(t, updated, "updated")
				assert.Same(t, pod, allocatedPod)
			}
		})
	}
}

func TestRecordAdmissionRejection(t *testing.T) {
	metrics.Register()

	testCases := []struct {
		name   string
		reason string
		wants  string
	}{
		{
			name:   "AppArmor",
			reason: lifecycle.AppArmorNotAdmittedReason,
			wants: `
				# HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
				# TYPE kubelet_admission_rejections_total counter
				kubelet_admission_rejections_total{reason="AppArmor"} 1
			`,
		},
		{
			name:   "PodOSSelectorNodeLabelDoesNotMatch",
			reason: lifecycle.PodOSSelectorNodeLabelDoesNotMatch,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="PodOSSelectorNodeLabelDoesNotMatch"} 1
            `,
		},
		{
			name:   "PodOSNotSupported",
			reason: lifecycle.PodOSNotSupported,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="PodOSNotSupported"} 1
            `,
		},
		{
			name:   "InvalidNodeInfo",
			reason: lifecycle.InvalidNodeInfo,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="InvalidNodeInfo"} 1
            `,
		},
		{
			name:   "InitContainerRestartPolicyForbidden",
			reason: lifecycle.InitContainerRestartPolicyForbidden,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="InitContainerRestartPolicyForbidden"} 1
            `,
		},
		{
			name:   "SupplementalGroupsPolicyNotSupported",
			reason: lifecycle.SupplementalGroupsPolicyNotSupported,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="SupplementalGroupsPolicyNotSupported"} 1
            `,
		},
		{
			name:   "UnexpectedAdmissionError",
			reason: lifecycle.UnexpectedAdmissionError,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="UnexpectedAdmissionError"} 1
            `,
		},
		{
			name:   "UnknownReason",
			reason: lifecycle.UnknownReason,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="UnknownReason"} 1
            `,
		},
		{
			name:   "UnexpectedPredicateFailureType",
			reason: lifecycle.UnexpectedPredicateFailureType,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="UnexpectedPredicateFailureType"} 1
            `,
		},
		{
			name:   "node(s) had taints that the pod didn't tolerate",
			reason: tainttoleration.ErrReasonNotMatch,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="node(s) had taints that the pod didn't tolerate"} 1
            `,
		},
		{
			name:   "Evicted",
			reason: eviction.Reason,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="Evicted"} 1
            `,
		},
		{
			name:   "SysctlForbidden",
			reason: sysctl.ForbiddenReason,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="SysctlForbidden"} 1
            `,
		},
		{
			name:   "TopologyAffinityError",
			reason: topologymanager.ErrorTopologyAffinity,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="TopologyAffinityError"} 1
            `,
		},
		{
			name:   "NodeShutdown",
			reason: nodeshutdown.NodeShutdownNotAdmittedReason,
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="NodeShutdown"} 1
            `,
		},
		{
			name:   "OutOfcpu",
			reason: "OutOfcpu",
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="OutOfcpu"} 1
            `,
		},
		{
			name:   "OutOfmemory",
			reason: "OutOfmemory",
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="OutOfmemory"} 1
            `,
		},
		{
			name:   "OutOfephemeral-storage",
			reason: "OutOfephemeral-storage",
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="OutOfephemeral-storage"} 1
            `,
		},
		{
			name:   "OutOfpods",
			reason: "OutOfpods",
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="OutOfpods"} 1
            `,
		},
		{
			name:   "OutOfgpu",
			reason: "OutOfgpu",
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="OutOfExtendedResources"} 1
            `,
		},
		{
			name:   "OtherReason",
			reason: "OtherReason",
			wants: `
                # HELP kubelet_admission_rejections_total [ALPHA] Cumulative number pod admission rejections by the Kubelet.
                # TYPE kubelet_admission_rejections_total counter
                kubelet_admission_rejections_total{reason="Other"} 1
            `,
		},
	}

	// Run tests.
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Clear the metrics after the test.
			metrics.AdmissionRejectionsTotal.Reset()

			// Call the function.
			recordAdmissionRejection(tc.reason)

			if err := testutil.GatherAndCompare(metrics.GetGather(), strings.NewReader(tc.wants), "kubelet_admission_rejections_total"); err != nil {
				t.Error(err)
			}
		})
	}
}
