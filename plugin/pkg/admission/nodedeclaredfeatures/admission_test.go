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

package nodedeclaredfeatures

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	ndf "k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/dranodeallocatableresources"
	ndftesting "k8s.io/component-helpers/nodedeclaredfeatures/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	apisresource "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/features"
)

func TestAdmission(t *testing.T) {
	nodeWithFeature := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
		Status: v1.NodeStatus{
			DeclaredFeatures: []string{"TestFeature"},
			NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.35.0"},
		},
	}
	nodeWithoutFeature := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "test-node-no-feature"},
		Status: v1.NodeStatus{
			DeclaredFeatures: []string{},
			NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.35.0"},
		},
	}

	client := fake.NewClientset(nodeWithFeature, nodeWithoutFeature)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	oldPod := &core.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pod", Namespace: "test-ns"},
		Spec: core.PodSpec{
			NodeName: "test-node",
			Containers: []core.Container{
				{
					Name:  "container",
					Image: "image",
					Resources: core.ResourceRequirements{
						Requests: core.ResourceList{
							core.ResourceCPU:    resource.MustParse("1000m"),
							core.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				}},
		},
	}
	newPod := oldPod.DeepCopy()
	newPod.Generation = oldPod.Generation + 1
	newPod.Spec.Containers[0].Resources.Requests[core.ResourceCPU] = resource.MustParse("2000m")
	podWithNoNode := oldPod.DeepCopy()
	podWithNoNode.Spec.NodeName = ""
	podWithInvalidNode := oldPod.DeepCopy()
	podWithInvalidNode.Spec.NodeName = "invalid-node"

	createMockFeature := func(t *testing.T, name string, inferForUpdate bool, maxVersionStr string) *ndftesting.MockFeature {
		m := ndftesting.NewMockFeature(t)
		m.SetName(name)
		m.SetInferForUpdate(func(_, _ *ndf.PodInfo) bool { return inferForUpdate })
		if maxVersionStr != "" {
			m.SetMaxVersion(version.MustParseSemantic(maxVersionStr))
		} else {
			m.SetMaxVersion(nil)
		}
		return m
	}

	testCases := []struct {
		name               string
		pod                *core.Pod
		oldPod             *core.Pod
		registeredFeatures []ndf.Feature
		featureGateEnabled bool
		componentVersion   string
		expectErr          bool
		errContains        string
		subresource        string
	}{
		{
			name:               "Feature gate disabled",
			pod:                newPod,
			oldPod:             oldPod,
			registeredFeatures: []ndf.Feature{},
			featureGateEnabled: false,
			expectErr:          false,
			componentVersion:   "1.35.0",
		},
		{
			name:               "skip validation when pod is not bound to node",
			pod:                podWithNoNode,
			oldPod:             oldPod,
			registeredFeatures: []ndf.Feature{},
			featureGateEnabled: true,
			expectErr:          false,
			componentVersion:   "1.35.0",
		},
		{
			name:               "skip validation on invalid node name",
			pod:                func() *core.Pod { p := newPod.DeepCopy(); p.Spec.NodeName = "not-found-node"; return p }(),
			oldPod:             oldPod,
			featureGateEnabled: true,
			registeredFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, "1.35.0"),
			},
			expectErr:        true,
			errContains:      "node \"not-found-node\" not found",
			componentVersion: "1.35.0",
		},
		{
			name:               "No feature requirements",
			pod:                newPod,
			oldPod:             oldPod,
			featureGateEnabled: true,
			registeredFeatures: []ndf.Feature{
				createMockFeature(t, "FeatureA", false, ""),
			},
			componentVersion: "1.35.0",
			expectErr:        false,
		},
		{
			name:               "Feature requirement met",
			pod:                newPod,
			oldPod:             oldPod,
			featureGateEnabled: true,
			componentVersion:   "1.35.0",
			registeredFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, "1.35.0"),
			},
			expectErr: false,
		},
		{
			name: "Feature requirement not met",
			pod: func() *core.Pod {
				p := newPod.DeepCopy()
				p.Spec.NodeName = "test-node-no-feature"
				return p
			}(),
			oldPod:             oldPod,
			featureGateEnabled: true,
			componentVersion:   "1.34.0",
			registeredFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, "1.35.0"),
			},
			expectErr:   true,
			errContains: "pod update requires features TestFeature which are not available on node \"test-node-no-feature\"",
		},
		{
			name: "skip validation when generation not updated",
			pod: func() *core.Pod {
				p := newPod.DeepCopy()
				p.Generation = oldPod.Generation
				return p
			}(),
			oldPod:             oldPod,
			featureGateEnabled: true,
			componentVersion:   "1.34.0",
			registeredFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, "1.35.0"),
			},
			expectErr: false,
		},
		{
			name: "Feature not need as its generally available",
			pod: func() *core.Pod {
				p := newPod.DeepCopy()
				p.Spec.NodeName = "test-node-no-feature"
				return p
			}(),
			oldPod:             oldPod,
			featureGateEnabled: true,
			componentVersion:   "1.35.0",
			registeredFeatures: []ndf.Feature{
				// Feature max version less than component version
				createMockFeature(t, "TestFeature", false, "1.34.0"),
			},
			expectErr: false,
		},
		{
			name: "skip validation for `status` subresource",
			pod: func() *core.Pod {
				p := newPod.DeepCopy()
				p.Spec.NodeName = "test-node-no-feature"
				return p
			}(),
			oldPod:             oldPod,
			featureGateEnabled: true,
			componentVersion:   "1.35.0",
			registeredFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, "1.35.0"),
			},
			subresource: "status",
			expectErr:   false,
		},
		{
			name:               "DO not skip validation for `resize` subresource",
			pod:                newPod,
			oldPod:             oldPod,
			featureGateEnabled: true,
			componentVersion:   "1.34.0",
			registeredFeatures: []ndf.Feature{
				createMockFeature(t, "TestFeature", true, "1.35.0"),
			},
			subresource: "resize",
			expectErr:   false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.featureGateEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.36"))
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeDeclaredFeatures, tc.featureGateEnabled)
			}

			target, err := NewPlugin()
			require.NoError(t, err)

			if tc.featureGateEnabled {
				framework := ndf.New(tc.registeredFeatures)
				target.nodeDeclaredFeatureFramework = framework
				target.version = version.MustParseSemantic(tc.componentVersion)
			}

			target.SetExternalKubeInformerFactory(informerFactory)
			target.InspectFeatureGates(utilfeature.DefaultFeatureGate)
			err = target.ValidateInitialization()
			require.NoError(t, err)

			stopCh := make(chan struct{})
			defer close(stopCh)
			informerFactory.Start(stopCh)
			informerFactory.WaitForCacheSync(stopCh)
			attrs := admission.NewAttributesRecord(tc.pod, tc.oldPod, core.Kind("Pod").WithVersion("v1"), tc.pod.Namespace, tc.pod.Name, core.Resource("pods").WithVersion("v1"), tc.subresource, admission.Update, &metav1.UpdateOptions{}, false, nil)
			err = target.Validate(context.Background(), attrs, nil)

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

func TestNodeAllocatableResourceSliceAdmission(t *testing.T) {
	nodeNameWithFeature := "node-with-feature"
	nodeNameWithFeature2 := "node-with-feature-2"
	nodeNameWithoutFeature := "node-without-feature"

	nodeWithFeature := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   nodeNameWithFeature,
			Labels: map[string]string{"pool": "pool1"},
		},
		Status: v1.NodeStatus{
			DeclaredFeatures: []string{dranodeallocatableresources.DRANodeAllocatableResourcesFeature},
			NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.37.0"},
		},
	}
	nodeWithFeature2 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   nodeNameWithFeature2,
			Labels: map[string]string{"pool": "pool1"},
		},
		Status: v1.NodeStatus{
			DeclaredFeatures: []string{dranodeallocatableresources.DRANodeAllocatableResourcesFeature},
			NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.37.0"},
		},
	}
	nodeWithoutFeature := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   nodeNameWithoutFeature,
			Labels: map[string]string{"pool": "pool2"},
		},
		Status: v1.NodeStatus{
			DeclaredFeatures: []string{},
			NodeInfo:         v1.NodeSystemInfo{KubeletVersion: "1.35.0"},
		},
	}

	trueVal := true
	nodeSelectorMatchNode := &core.NodeSelector{
		NodeSelectorTerms: []core.NodeSelectorTerm{
			{
				MatchExpressions: []core.NodeSelectorRequirement{
					{
						Key:      "pool",
						Operator: core.NodeSelectorOpIn,
						Values:   []string{"pool1"},
					},
				},
			},
		},
	}
	devSelectorDoesNoeMatchNode := &core.NodeSelector{
		NodeSelectorTerms: []core.NodeSelectorTerm{
			{
				MatchExpressions: []core.NodeSelectorRequirement{
					{
						Key:      "pool",
						Operator: core.NodeSelectorOpIn,
						Values:   []string{"pool2"},
					},
				},
			},
		},
	}

	oneQuantity := resource.MustParse("1")
	nodeAllocatable := apisresource.NodeAllocatableResource{
		Mapping: &apisresource.NodeAllocatableMapping{
			DeviceMultiplier: &oneQuantity,
		},
	}

	testCases := []struct {
		name                   string
		slice                  *apisresource.ResourceSlice
		nodes                  []runtime.Object
		draFeatureGateDisabled bool
		expectErr              bool
		errContains            string
		operation              admission.Operation
	}{
		{
			name: "DRANodeAllocatableResources gate disabled, skip validation",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-dra-gate-off"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					NodeName: &nodeNameWithoutFeature,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:                  []runtime.Object{nodeWithoutFeature},
			draFeatureGateDisabled: true,
			expectErr:              false,
			operation:              admission.Create,
		},
		{
			name: "mo mappings, no node validation needed",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-without-mappings"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					NodeName: &nodeNameWithoutFeature,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
						},
					},
				},
			},
			nodes:     []runtime.Object{nodeWithoutFeature},
			expectErr: false,
			operation: admission.Create,
		},
		{
			name: "node supports feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-with-mappings"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					NodeName: &nodeNameWithFeature,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes: []runtime.Object{nodeWithFeature},

			expectErr: false,
			operation: admission.Create,
		},
		{
			name: "node does not support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-no-feature-node"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					NodeName: &nodeNameWithoutFeature,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:       []runtime.Object{nodeWithoutFeature},
			expectErr:   true,
			errContains: "does not support DRANodeAllocatableResources feature",
			operation:   admission.Create,
		},
		{
			name: "node name is nil",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-no-node"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					NodeName: nil,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:       []runtime.Object{},
			expectErr:   true,
			errContains: "does not map to any nodes",
			operation:   admission.Create,
		},
		{
			name: "target node not found",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-with-mappings"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					NodeName: &nodeNameWithFeature,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:       []runtime.Object{},
			expectErr:   true,
			errContains: "node \"node-with-feature\" not found",
			operation:   admission.Create,
		},
		{
			name: "node selector in device matches node that has feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-with-selector-match"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:       "test-driver",
					NodeSelector: nodeSelectorMatchNode,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:     []runtime.Object{nodeWithFeature, nodeWithFeature2, nodeWithoutFeature},
			expectErr: false,
			operation: admission.Create,
		},
		{
			name: "nodeSelector matches node the does not support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-with-selector-no-match"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:       "test-driver",
					NodeSelector: devSelectorDoesNoeMatchNode,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:       []runtime.Object{nodeWithFeature, nodeWithoutFeature},
			expectErr:   true,
			errContains: "does not support DRANodeAllocatableResources feature",
			operation:   admission.Create,
		},
		{
			name: "slice-level AllNodes is true in slice, one node does not support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-with-all-nodes"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					AllNodes: &trueVal,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:       []runtime.Object{nodeWithFeature, nodeWithoutFeature},
			expectErr:   true,
			errContains: "does not support DRANodeAllocatableResources feature",
			operation:   admission.Create,
		},
		{
			name: "slice-level AllNodes is true, all nodes support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-with-all-nodes"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:   "test-driver",
					AllNodes: &trueVal,
					Devices: []apisresource.Device{
						{
							Name: "dev-0",
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:     []runtime.Object{nodeWithFeature, nodeWithFeature2},
			expectErr: false,
			operation: admission.Create,
		},
		{
			name: "device-level node name supports feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-per-device-nodename"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:                 "test-driver",
					PerDeviceNodeSelection: &trueVal,
					Devices: []apisresource.Device{
						{
							Name:     "dev-0",
							NodeName: &nodeNameWithFeature,
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:     []runtime.Object{nodeWithFeature},
			expectErr: false,
			operation: admission.Create,
		},
		{
			name: "device-level nodeName does not support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-per-device-nodename-no-feature"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:                 "test-driver",
					PerDeviceNodeSelection: &trueVal,
					Devices: []apisresource.Device{
						{
							Name:     "dev-0",
							NodeName: &nodeNameWithoutFeature,
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:       []runtime.Object{nodeWithoutFeature},
			expectErr:   true,
			errContains: "does not support DRANodeAllocatableResources feature",
			operation:   admission.Create,
		},
		{
			name: "device-level node selector matches matches nodes that support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-per-device-nodeselector"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:                 "test-driver",
					PerDeviceNodeSelection: &trueVal,
					Devices: []apisresource.Device{
						{
							Name:         "dev-0",
							NodeSelector: nodeSelectorMatchNode,
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:     []runtime.Object{nodeWithFeature, nodeWithFeature2, nodeWithoutFeature},
			expectErr: false,
			operation: admission.Create,
		},
		{
			name: "device-level node selector matches node that does not support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-per-device-nodeselector-no-match"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:                 "test-driver",
					PerDeviceNodeSelection: &trueVal,
					Devices: []apisresource.Device{
						{
							Name:         "dev-0",
							NodeSelector: devSelectorDoesNoeMatchNode,
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes: []runtime.Object{nodeWithFeature, nodeWithoutFeature}, expectErr: true,
			errContains: "does not support DRANodeAllocatableResources feature",
			operation:   admission.Create,
		},
		{
			name: "device-level AllNodes is true, one node does not support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-per-device-allnodes"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:                 "test-driver",
					PerDeviceNodeSelection: &trueVal,
					Devices: []apisresource.Device{
						{
							Name:     "dev-0",
							AllNodes: &trueVal,
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:       []runtime.Object{nodeWithFeature, nodeWithoutFeature},
			expectErr:   true,
			errContains: "does not support DRANodeAllocatableResources feature",
			operation:   admission.Create,
		},
		{
			name: "device-level AllNodes is true, all nodes support feature",
			slice: &apisresource.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "slice-per-device-allnodes"},
				Spec: apisresource.ResourceSliceSpec{
					Driver:                 "test-driver",
					PerDeviceNodeSelection: &trueVal,
					Devices: []apisresource.Device{
						{
							Name:     "dev-0",
							AllNodes: &trueVal,
							NodeAllocatableResources: map[v1.ResourceName]apisresource.NodeAllocatableResource{
								v1.ResourceCPU: nodeAllocatable,
							},
						},
					},
				},
			},
			nodes:     []runtime.Object{nodeWithFeature, nodeWithFeature2},
			expectErr: false,
			operation: admission.Create,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRANodeAllocatableResources, !tc.draFeatureGateDisabled)

			client := fake.NewClientset(tc.nodes...)
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			target, err := NewPlugin()
			require.NoError(t, err)

			target.SetExternalKubeInformerFactory(informerFactory)
			target.InspectFeatureGates(utilfeature.DefaultFeatureGate)
			err = target.ValidateInitialization()
			require.NoError(t, err)

			stopCh := make(chan struct{})
			defer close(stopCh)
			informerFactory.Start(stopCh)
			informerFactory.WaitForCacheSync(stopCh)

			attrs := admission.NewAttributesRecord(tc.slice, nil, apisresource.Kind("ResourceSlice").WithVersion("v1"), tc.slice.Namespace, tc.slice.Name, apisresource.Resource("resourceslices").WithVersion("v1"), "", tc.operation, &metav1.CreateOptions{}, false, nil)
			err = target.Validate(context.Background(), attrs, nil)

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
