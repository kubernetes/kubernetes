/*
Copyright The Kubernetes Authors.

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

package admission

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestCheck(t *testing.T) {
	nodeaffinityError := Result{Name: nodeaffinity.Name, Reason: nodeaffinity.ErrReasonPod}
	nodenameError := Result{Name: nodename.Name, Reason: nodename.ErrReason}
	nodeportsError := Result{Name: nodeports.Name, Reason: nodeports.ErrReason}
	podOverheadError := Result{InsufficientResource: &helper.InsufficientResource{ResourceName: v1.ResourceCPU, Reason: "Insufficient cpu", Requested: 2000, Used: 7000, Capacity: 8000}}
	extendedResourceError := Result{InsufficientResource: &helper.InsufficientResource{ResourceName: "foo.com/bar", Reason: "Insufficient foo.com/bar", Requested: 1, Unresolvable: true}}
	nodeCPUCapacity := map[v1.ResourceName]string{v1.ResourceCPU: "8"}
	extendedResource := map[v1.ResourceName]string{"foo.com/bar": "1"}
	nodeAllocatableResourceError := Result{InsufficientResource: &helper.InsufficientResource{ResourceName: v1.ResourceCPU, Reason: "Insufficient cpu", Requested: 9000, Used: 0, Capacity: 8000, Unresolvable: true}}
	tests := []struct {
		name                              string
		node                              *v1.Node
		existingPods                      []*v1.Pod
		pod                               *v1.Pod
		wantResults                       [][]Result
		enableDRAExtendedResource         bool
		enableDRANodeAllocatableResources bool
	}{
		{
			name: "check nodeAffinity and nodeports, nodeAffinity need fail quickly if includeAllFailures is false",
			node: st.MakeNode().Name("fake-node").Label("foo", "bar").Obj(),
			pod:  st.MakePod().Name("pod2").HostPort(80).NodeSelector(map[string]string{"foo": "bar1"}).Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("pod1").HostPort(80).Obj(),
			},
			wantResults: [][]Result{{nodeaffinityError, nodeportsError}, {nodeaffinityError}},
		},
		{
			name: "check PodOverhead and nodeAffinity, PodOverhead need fail quickly if includeAllFailures is false",
			node: st.MakeNode().Name("fake-node").Label("foo", "bar").Capacity(nodeCPUCapacity).Obj(),
			pod:  st.MakePod().Name("pod2").Container("c").Overhead(v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).NodeSelector(map[string]string{"foo": "bar1"}).Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "7"}).Node("fake-node").Obj(),
			},
			wantResults: [][]Result{{podOverheadError, nodeaffinityError}, {podOverheadError}},
		},
		{
			name: "check nodename and nodeports, nodename need fail quickly if includeAllFailures is false",
			node: st.MakeNode().Name("fake-node").Obj(),
			pod:  st.MakePod().Name("pod2").HostPort(80).Node("fake-node1").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("pod1").HostPort(80).Node("fake-node").Obj(),
			},
			wantResults: [][]Result{{nodenameError, nodeportsError}, {nodenameError}},
		},
		{
			name:        "check extended resource handling when node Allocatable doesn't have the resource",
			node:        st.MakeNode().Name("fake-node").Obj(),
			pod:         st.MakePod().Name("pod1").Req(extendedResource).Obj(),
			wantResults: [][]Result{{extendedResourceError}, {extendedResourceError}},
		},
		{
			name:                      "check extended resource handling when node Allocatable doesn't have the resource and DRAExtendedResource is enabled",
			node:                      st.MakeNode().Name("fake-node").Obj(),
			pod:                       st.MakePod().Name("pod1").Req(extendedResource).Obj(),
			wantResults:               [][]Result{{extendedResourceError}, {extendedResourceError}},
			enableDRAExtendedResource: true,
		},
		{
			name: "pod not rejected when DRANodeAllocatableResources flag is disabled",
			node: st.MakeNode().Name("fake-node").Capacity(nodeCPUCapacity).Obj(),
			pod: func() *v1.Pod {
				p := st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj()
				p.Status.NodeAllocatableResourceClaimStatuses = []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "node-allocatable-claim",
						Mapping: []v1.NodeAllocatableMappedResources{{
							Name:     v1.ResourceCPU,
							Quantity: new(resource.MustParse("8")),
						}},
					},
				}
				return p
			}(),
			wantResults:                       [][]Result{nil, nil},
			enableDRANodeAllocatableResources: false,
		},
		{
			name: "pod rejected when DRANodeAllocatableResources flag is enabled and pod's resource request exceeds node capacity",
			node: st.MakeNode().Name("fake-node").Capacity(nodeCPUCapacity).Obj(),
			pod: func() *v1.Pod {
				p := st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj()
				p.Status.NodeAllocatableResourceClaimStatuses = []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "node-allocatable-claim",
						Mapping: []v1.NodeAllocatableMappedResources{{
							Name:     v1.ResourceCPU,
							Quantity: new(resource.MustParse(nodeCPUCapacity[v1.ResourceCPU])), // We should exceed node capacity since we also request 1 CPU in standard request.
						}},
					},
				}
				return p
			}(),
			wantResults:                       [][]Result{{nodeAllocatableResourceError}, {nodeAllocatableResourceError}},
			enableDRANodeAllocatableResources: true,
		},
		{
			name: "pod not rejected when DRANodeAllocatableResources flag is enabled and pod's resource request fits within node capacity",
			node: st.MakeNode().Name("fake-node").Capacity(nodeCPUCapacity).Obj(),
			pod: func() *v1.Pod {
				p := st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj()
				cpuQty := resource.MustParse(nodeCPUCapacity[v1.ResourceCPU])
				cpuQty.Sub(resource.MustParse("1"))
				p.Status.NodeAllocatableResourceClaimStatuses = []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "node-allocatable-claim",
						Mapping: []v1.NodeAllocatableMappedResources{{
							Name:     v1.ResourceCPU,
							Quantity: new(cpuQty),
						}},
					},
				}
				return p
			}(),
			wantResults:                       [][]Result{nil, nil},
			enableDRANodeAllocatableResources: true,
		},
		{
			name: "pod rejected when DRANodeAllocatableResources flag is enabled and pod's resource request + DRA Overhead exceeds node capacity",
			node: st.MakeNode().Name("fake-node").Capacity(nodeCPUCapacity).Obj(),
			pod: func() *v1.Pod {
				p := st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj()
				p.Status.NodeAllocatableResourceClaimStatuses = []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "node-allocatable-claim",
						Containers:        []string{"bar"}, // Default container name created by st.MakePod() is usually "bar" (let's use that or empty containers since it's just PerPod)
						Overhead: []v1.NodeAllocatableOverheadResources{{
							Name:   v1.ResourceCPU,
							PerPod: new(resource.MustParse(nodeCPUCapacity[v1.ResourceCPU])), // 1 CPU + nodeCPUCapacity CPU > nodeCPUCapacity
						}},
					},
				}
				return p
			}(),
			wantResults:                       [][]Result{{nodeAllocatableResourceError}, {nodeAllocatableResourceError}},
			enableDRANodeAllocatableResources: true,
		},
		{
			name: "pod not rejected when DRANodeAllocatableResources flag is enabled and pod's resource request + DRA Overhead fits within node capacity",
			node: st.MakeNode().Name("fake-node").Capacity(nodeCPUCapacity).Obj(),
			pod: func() *v1.Pod {
				p := st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj()
				cpuQty := resource.MustParse(nodeCPUCapacity[v1.ResourceCPU])
				cpuQty.Sub(resource.MustParse("1")) // Now cpuQty + 1 CPU (request) = nodeCPUCapacity CPU
				p.Status.NodeAllocatableResourceClaimStatuses = []v1.NodeAllocatableResourceClaimStatus{
					{
						ResourceClaimName: "node-allocatable-claim",
						Containers:        []string{"bar"},
						Overhead: []v1.NodeAllocatableOverheadResources{{
							Name:   v1.ResourceCPU,
							PerPod: &cpuQty,
						}},
					},
				}
				return p
			}(),
			wantResults:                       [][]Result{nil, nil},
			enableDRANodeAllocatableResources: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !tt.enableDRAExtendedResource {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.36"))
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.DRAExtendedResource:         tt.enableDRAExtendedResource,
				features.DRANodeAllocatableResources: tt.enableDRANodeAllocatableResources,
			})
			nodeInfo := framework.NewNodeInfo(tt.existingPods...)
			nodeInfo.SetNode(tt.node)

			flags := []bool{true, false}
			for i := range flags {
				admissionResults := Check(tt.pod, nodeInfo, flags[i])

				if diff := cmp.Diff(tt.wantResults[i], admissionResults); diff != "" {
					t.Errorf("Unexpected admissionResults (-want, +got):\n%s", diff)
				}
			}
		})
	}
}
