/*
Copyright 2026 The Kubernetes Authors.

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

package podresize

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/apis/core"
)

func TestResizeAdmission(t *testing.T) {
	linuxNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "linux-node",
			Labels: map[string]string{
				corev1.LabelOSStable: "linux",
			},
		},
		Status: corev1.NodeStatus{
			Allocatable: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("4"),
				corev1.ResourceMemory: resource.MustParse("8Gi"),
			},
		},
	}

	windowsNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "windows-node",
			Labels: map[string]string{
				corev1.LabelOSStable: "windows",
			},
		},
		Status: corev1.NodeStatus{
			Allocatable: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("4"),
				corev1.ResourceMemory: resource.MustParse("8Gi"),
			},
		},
	}

	testPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "test-pod",
			Namespace:  "default",
			Generation: 1,
		},
		Spec: core.PodSpec{
			NodeName: "linux-node",
			Containers: []core.Container{
				{
					Name: "main",
					Resources: core.ResourceRequirements{
						Requests: core.ResourceList{
							core.ResourceCPU:    resource.MustParse("1"),
							core.ResourceMemory: resource.MustParse("1Gi"),
						},
					},
				},
			},
		},
	}

	testCases := []struct {
		name        string
		subresource string
		nodeName    string
		oldPod      *core.Pod
		newPod      func() *core.Pod
		expectErr   bool
		errContains string
	}{
		{
			name:        "Successful cpu resize on Linux node",
			subresource: "resize",
			nodeName:    "linux-node",
			oldPod:      testPod,
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("2")
				return p
			},
			expectErr: false,
		},
		{
			name:        "Successful memory resize on Linux node",
			subresource: "resize",
			nodeName:    "linux-node",
			oldPod:      testPod,
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceMemory] = resource.MustParse("2Gi")
				return p
			},
			expectErr: false,
		},
		{
			name:        "Skip validation if not resize subresource",
			subresource: "",
			nodeName:    "linux-node",
			oldPod:      testPod,
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("10") // Would fail if checked
				return p
			},
			expectErr: false,
		},
		{
			name:        "Skip validation if generation is unchanged",
			subresource: "",
			nodeName:    "linux-node",
			oldPod:      testPod,
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Generation = 1
				p.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("10") // Would fail if checked
				return p
			},
			expectErr: false,
		},
		{
			name:        "Skip validation if pod not bound to a node",
			subresource: "resize",
			nodeName:    "",
			oldPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Spec.NodeName = ""
				return p
			}(),
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Spec.NodeName = ""
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("10")
				return p
			},
			expectErr: false,
		},
		{
			name:        "Reject resize exceeding CPU allocatable",
			subresource: "resize",
			nodeName:    "linux-node",
			oldPod:      testPod,
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("10")
				return p
			},
			expectErr:   true,
			errContains: "Node didn't have enough allocatable resources: cpu",
		},
		{
			name:        "Reject resize exceeding memory allocatable",
			subresource: "resize",
			nodeName:    "linux-node",
			oldPod:      testPod,
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceMemory] = resource.MustParse("10Gi")
				return p
			},
			expectErr:   true,
			errContains: "Node didn't have enough allocatable resources: memory",
		},
		{
			name:        "Reject resize exceeding both CPU and memory allocatable",
			subresource: "resize",
			nodeName:    "linux-node",
			oldPod:      testPod,
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("10")
				p.Spec.Containers[0].Resources.Requests[core.ResourceMemory] = resource.MustParse("10Gi")
				return p
			},
			expectErr:   true,
			errContains: "Node didn't have enough allocatable resources: cpu, requested: 10000, allocatable: 4000; memory, requested: 10737418240, allocatable: 8589934592",
		},
		{
			name:        "Reject resize on non-Linux node",
			subresource: "resize",
			nodeName:    "windows-node",
			oldPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Spec.NodeName = "windows-node"
				return p
			}(),
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Spec.NodeName = "windows-node"
				p.Generation = 2
				p.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("2")
				return p
			},
			expectErr:   true,
			errContains: "pod resize is only supported on linux nodes",
		},
		{
			name:        "Reject if node not found",
			subresource: "resize",
			nodeName:    "missing-node",
			oldPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Spec.NodeName = "missing-node"
				return p
			}(),
			newPod: func() *core.Pod {
				p := testPod.DeepCopy()
				p.Spec.NodeName = "missing-node"
				p.Generation = 2
				return p
			},
			expectErr:   true,
			errContains: "node \"missing-node\" not found",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewClientset(linuxNode, windowsNode)
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			plugin, err := NewPlugin()
			require.NoError(t, err)
			plugin.SetExternalKubeInformerFactory(informerFactory)
			plugin.inPlacePodVerticalScalingEnabled = true

			stopCh := make(chan struct{})
			defer close(stopCh)
			informerFactory.Start(stopCh)
			informerFactory.WaitForCacheSync(stopCh)

			newPodObj := tc.newPod()
			attrs := admission.NewAttributesRecord(
				newPodObj,
				tc.oldPod,
				core.Kind("Pod").WithVersion("v1"),
				newPodObj.Namespace,
				newPodObj.Name,
				core.Resource("pods").WithVersion("v1"),
				tc.subresource,
				admission.Update,
				&metav1.UpdateOptions{},
				false,
				nil,
			)

			err = plugin.Validate(context.Background(), attrs, nil)

			if tc.expectErr {
				require.Error(t, err)
				if tc.errContains != "" {
					assert.Contains(t, err.Error(), tc.errContains)
				}
			} else {
				require.NoError(t, err)
			}
		})
	}
}
