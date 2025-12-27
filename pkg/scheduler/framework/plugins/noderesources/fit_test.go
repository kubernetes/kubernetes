/*
Copyright 2019 The Kubernetes Authors.

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

package noderesources

import (
	"fmt"
	"testing"
	"testing/synctest"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	plfeature "k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

var (
	extendedResourceA                 = v1.ResourceName("example.com/aaa")
	extendedResourceB                 = v1.ResourceName("example.com/bbb")
	kubernetesIOResourceA             = v1.ResourceName("kubernetes.io/something")
	kubernetesIOResourceB             = v1.ResourceName("subdomain.kubernetes.io/something")
	hugePageResourceA                 = v1.ResourceName(v1.ResourceHugePagesPrefix + "2Mi")
	extendedResourceName              = "extended.resource.dra.io/something"
	extendedResourceDRA               = v1.ResourceName(extendedResourceName)
	deviceClassName                   = "device-class-name"
	deviceClassWithExtendResourceName = &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: deviceClassName,
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: &extendedResourceName,
		},
	}
)

var clusterEventCmpOpts = []cmp.Option{
	cmpopts.EquateComparable(fwk.ClusterEvent{}),
}

func makeResources(milliCPU, memory, pods, extendedA, storage, hugePageA int64) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
		v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
		extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
		v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
		hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
	}
}

func makeAllocatableResources(milliCPU, memory, pods, extendedA, storage, hugePageA int64) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
		v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
		extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
		v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
		hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
	}
}

func newResourcePod(usage ...framework.Resource) *v1.Pod {
	var containers []v1.Container
	for _, req := range usage {
		rl := v1.ResourceList{
			v1.ResourceCPU:              *resource.NewMilliQuantity(req.MilliCPU, resource.DecimalSI),
			v1.ResourceMemory:           *resource.NewQuantity(req.Memory, resource.BinarySI),
			v1.ResourcePods:             *resource.NewQuantity(int64(req.AllowedPodNumber), resource.BinarySI),
			v1.ResourceEphemeralStorage: *resource.NewQuantity(req.EphemeralStorage, resource.BinarySI),
		}
		for rName, rQuant := range req.ScalarResources {
			if rName == hugePageResourceA {
				rl[rName] = *resource.NewQuantity(rQuant, resource.BinarySI)
			} else {
				rl[rName] = *resource.NewQuantity(rQuant, resource.DecimalSI)
			}
		}
		containers = append(containers, v1.Container{
			Resources: v1.ResourceRequirements{Requests: rl},
		})
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}
}

func newResourceInitPod(pod *v1.Pod, usage ...framework.Resource) *v1.Pod {
	pod.Spec.InitContainers = newResourcePod(usage...).Spec.Containers
	return pod
}

func newResourceOverheadPod(pod *v1.Pod, overhead v1.ResourceList) *v1.Pod {
	pod.Spec.Overhead = overhead
	return pod
}

func getErrReason(rn v1.ResourceName) string {
	return fmt.Sprintf("Insufficient %v", rn)
}

var defaultScoringStrategy = &config.ScoringStrategy{
	Type: config.LeastAllocated,
	Resources: []config.ResourceSpec{
		{Name: "cpu", Weight: 1},
		{Name: "memory", Weight: 1},
	},
}

func newPodLevelResourcesPod(pod *v1.Pod, podResources v1.ResourceRequirements) *v1.Pod {
	pod.Spec.Resources = &podResources
	return pod
}

func TestEnoughRequests(t *testing.T) { testEnoughRequests(ktesting.Init(t)) }
func testEnoughRequests(tCtx ktesting.TContext) {
	enoughPodsTests := []struct {
		pod                        *v1.Pod
		nodeInfo                   *framework.NodeInfo
		name                       string
		args                       config.NodeResourcesFitArgs
		podLevelResourcesEnabled   bool
		draExtendedResourceEnabled bool
		nilDRAManager              bool
		wantInsufficientResources  []InsufficientResource
		wantStatus                 *fwk.Status
	}{
		{
			pod: &v1.Pod{},
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 10, Memory: 20})),
			name:                      "no resources requested always fits",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 10, Memory: 20})),
			name:       "too many resources fails",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceCPU), getErrReason(v1.ResourceMemory)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceCPU, Reason: getErrReason(v1.ResourceCPU), Requested: 1, Used: 10, Capacity: 10},
				{ResourceName: v1.ResourceMemory, Reason: getErrReason(v1.ResourceMemory), Requested: 1, Used: 20, Capacity: 20},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}), framework.Resource{MilliCPU: 3, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 8, Memory: 19})),
			name:       "too many resources fails due to init container cpu",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceCPU)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceCPU, Reason: getErrReason(v1.ResourceCPU), Requested: 3, Used: 8, Capacity: 10},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}), framework.Resource{MilliCPU: 3, Memory: 1}, framework.Resource{MilliCPU: 2, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 8, Memory: 19})),
			name:       "too many resources fails due to highest init container cpu",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceCPU)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceCPU, Reason: getErrReason(v1.ResourceCPU), Requested: 3, Used: 8, Capacity: 10},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}), framework.Resource{MilliCPU: 1, Memory: 3}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 9, Memory: 19})),
			name:       "too many resources fails due to init container memory",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceMemory)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceMemory, Reason: getErrReason(v1.ResourceMemory), Requested: 3, Used: 19, Capacity: 20},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}), framework.Resource{MilliCPU: 1, Memory: 3}, framework.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 9, Memory: 19})),
			name:       "too many resources fails due to highest init container memory",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceMemory)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceMemory, Reason: getErrReason(v1.ResourceMemory), Requested: 3, Used: 19, Capacity: 20},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}), framework.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 9, Memory: 19})),
			name:                      "init container fits because it's the max, not sum, of containers and init containers",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}), framework.Resource{MilliCPU: 1, Memory: 1}, framework.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 9, Memory: 19})),
			name:                      "multiple init containers fit because it's the max, not sum, of containers and init containers",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 5})),
			name:                      "both resources fit",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourcePod(framework.Resource{MilliCPU: 2, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 9, Memory: 5})),
			name:       "one resource memory fits",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceCPU)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceCPU, Reason: getErrReason(v1.ResourceCPU), Requested: 2, Used: 9, Capacity: 10},
			},
		},
		{
			pod: newResourcePod(framework.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 19})),
			name:       "one resource cpu fits",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceMemory)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceMemory, Reason: getErrReason(v1.ResourceMemory), Requested: 2, Used: 19, Capacity: 20},
			},
		},
		{
			pod: newResourcePod(framework.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 19})),
			name:                      "equal edge case",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 4, Memory: 1}), framework.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 19})),
			name:                      "equal edge case for init container",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod:                       newResourcePod(framework.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo:                  framework.NewNodeInfo(newResourcePod(framework.Resource{})),
			name:                      "extended resource fits",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod:                       newResourceInitPod(newResourcePod(framework.Resource{}), framework.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo:                  framework.NewNodeInfo(newResourcePod(framework.Resource{})),
			name:                      "extended resource fits for init container",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 10}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 0}})),
			name:       "extended resource capacity enforced",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceA, Reason: getErrReason(extendedResourceA), Requested: 10, Used: 0, Capacity: 5, Unresolvable: true},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{}),
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 10}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 0}})),
			name:       "extended resource capacity enforced for init container",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceA, Reason: getErrReason(extendedResourceA), Requested: 10, Used: 0, Capacity: 5, Unresolvable: true},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 5}})),
			name:       "extended resource allocatable enforced",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(extendedResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceA, Reason: getErrReason(extendedResourceA), Requested: 1, Used: 5, Capacity: 5},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{}),
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 5}})),
			name:       "extended resource allocatable enforced for init container",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(extendedResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceA, Reason: getErrReason(extendedResourceA), Requested: 1, Used: 5, Capacity: 5},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}},
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			name:       "extended resource allocatable enforced for multiple containers",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceA, Reason: getErrReason(extendedResourceA), Requested: 6, Used: 2, Capacity: 5, Unresolvable: true},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{}),
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}},
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			name:                      "extended resource allocatable admits multiple init containers",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{}),
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 6}},
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			name:       "extended resource allocatable enforced for multiple init containers",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceA, Reason: getErrReason(extendedResourceA), Requested: 6, Used: 2, Capacity: 5, Unresolvable: true},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:       "extended resource allocatable enforced for unknown resource",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceB)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceB, Reason: getErrReason(extendedResourceB), Requested: 1, Used: 0, Capacity: 0, Unresolvable: true},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{}),
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:       "extended resource allocatable enforced for unknown resource for init container",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceB)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: extendedResourceB, Reason: getErrReason(extendedResourceB), Requested: 1, Used: 0, Capacity: 0, Unresolvable: true},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{kubernetesIOResourceA: 10}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:       "kubernetes.io resource capacity enforced",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(kubernetesIOResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: kubernetesIOResourceA, Reason: getErrReason(kubernetesIOResourceA), Requested: 10, Used: 0, Capacity: 0, Unresolvable: true},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{}),
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{kubernetesIOResourceB: 10}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:       "kubernetes.io resource capacity enforced for init container",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(kubernetesIOResourceB)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: kubernetesIOResourceB, Reason: getErrReason(kubernetesIOResourceB), Requested: 10, Used: 0, Capacity: 0, Unresolvable: true},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 10}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 0}})),
			name:       "hugepages resource capacity enforced",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(hugePageResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: hugePageResourceA, Reason: getErrReason(hugePageResourceA), Requested: 10, Used: 0, Capacity: 5, Unresolvable: true},
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{}),
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 10}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 0}})),
			name:       "hugepages resource capacity enforced for init container",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(hugePageResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: hugePageResourceA, Reason: getErrReason(hugePageResourceA), Requested: 10, Used: 0, Capacity: 5, Unresolvable: true},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 3}},
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 3}}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 2}})),
			name:       "hugepages resource allocatable enforced for multiple containers",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(hugePageResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: hugePageResourceA, Reason: getErrReason(hugePageResourceA), Requested: 6, Used: 2, Capacity: 5, Unresolvable: true},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			args: config.NodeResourcesFitArgs{
				IgnoredResources: []string{"example.com/bbb"},
			},
			name:                      "skip checking ignored extended resource",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourceOverheadPod(
				newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceList{v1.ResourceCPU: resource.MustParse("3m"), v1.ResourceMemory: resource.MustParse("13")},
			),
			nodeInfo:                  framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 5, Memory: 5})),
			name:                      "resources + pod overhead fits",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			pod: newResourceOverheadPod(
				newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m"), v1.ResourceMemory: resource.MustParse("15")},
			),
			nodeInfo:   framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 5, Memory: 5})),
			name:       "requests + overhead does not fit for memory",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceMemory)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: v1.ResourceMemory, Reason: getErrReason(v1.ResourceMemory), Requested: 16, Used: 5, Capacity: 20},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{
					MilliCPU: 1,
					Memory:   1,
					ScalarResources: map[v1.ResourceName]int64{
						extendedResourceB:     1,
						kubernetesIOResourceA: 1,
					}}),
			nodeInfo: framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			args: config.NodeResourcesFitArgs{
				IgnoredResourceGroups: []string{"example.com"},
			},
			name:       "skip checking ignored extended resource via resource groups",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(kubernetesIOResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{
					ResourceName: kubernetesIOResourceA,
					Reason:       getErrReason(kubernetesIOResourceA),
					Requested:    1,
					Used:         0,
					Capacity:     0,
					Unresolvable: true,
				},
			},
		},
		{
			pod: newResourcePod(
				framework.Resource{
					MilliCPU: 1,
					Memory:   1,
					ScalarResources: map[v1.ResourceName]int64{
						extendedResourceA: 0,
					}}),
			nodeInfo: framework.NewNodeInfo(newResourcePod(framework.Resource{
				MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 6}})),
			name:                      "skip checking extended resource request with quantity zero via resource groups",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newResourcePod(
				framework.Resource{
					ScalarResources: map[v1.ResourceName]int64{
						extendedResourceA: 1,
					}}),
			nodeInfo: framework.NewNodeInfo(newResourcePod(framework.Resource{
				MilliCPU: 20, Memory: 30, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}})),
			name:                      "skip checking resource request with quantity zero",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newPodLevelResourcesPod(
				newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m"), v1.ResourceMemory: resource.MustParse("2")},
				},
			),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 5})),
			name:                      "both pod-level and container-level resources fit",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newPodLevelResourcesPod(
				newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("7m"), v1.ResourceMemory: resource.MustParse("2")},
				},
			),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 5})),
			name:       "pod-level cpu resource not fit",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceCPU)),
			wantInsufficientResources: []InsufficientResource{{
				ResourceName: v1.ResourceCPU, Reason: getErrReason(v1.ResourceCPU), Requested: 7, Used: 5, Capacity: 10},
			},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newPodLevelResourcesPod(
				newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("3m"), v1.ResourceMemory: resource.MustParse("2")},
				},
			),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 19})),
			name:       "pod-level memory resource not fit",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceMemory)),
			wantInsufficientResources: []InsufficientResource{{
				ResourceName: v1.ResourceMemory, Reason: getErrReason(v1.ResourceMemory), Requested: 2, Used: 19, Capacity: 20},
			},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newPodLevelResourcesPod(
				newResourcePod(),
				v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("3m"), v1.ResourceMemory: resource.MustParse("2"), hugePageResourceA: *resource.NewQuantity(5, resource.BinarySI)},
				},
			),
			nodeInfo:                  framework.NewNodeInfo(),
			name:                      "pod-level hugepages resource fit",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newPodLevelResourcesPod(
				newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 3}}),
				v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("3m"), v1.ResourceMemory: resource.MustParse("2"), hugePageResourceA: *resource.NewQuantity(5, resource.BinarySI)},
				},
			),
			nodeInfo:                  framework.NewNodeInfo(),
			name:                      "both pod-level and container-level hugepages resource fit",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newPodLevelResourcesPod(
				newResourcePod(),
				v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("3m"), v1.ResourceMemory: resource.MustParse("2"), hugePageResourceA: *resource.NewQuantity(10, resource.BinarySI)},
				},
			),
			nodeInfo:   framework.NewNodeInfo(),
			name:       "pod-level hugepages resource not fit",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(hugePageResourceA)),
			wantInsufficientResources: []InsufficientResource{
				{ResourceName: hugePageResourceA, Reason: getErrReason(hugePageResourceA), Requested: 10, Used: 0, Capacity: 5, Unresolvable: true},
			},
		},
		{
			podLevelResourcesEnabled: true,
			pod: newResourceInitPod(newPodLevelResourcesPod(
				newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("3m"), v1.ResourceMemory: resource.MustParse("2")},
				},
			),
				framework.Resource{MilliCPU: 1, Memory: 1},
			),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 5, Memory: 19})),
			name:       "one pod-level cpu resource fits and all init and non-init containers resources fit",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, getErrReason(v1.ResourceMemory)),
			wantInsufficientResources: []InsufficientResource{{
				ResourceName: v1.ResourceMemory, Reason: getErrReason(v1.ResourceMemory), Requested: 2, Used: 19, Capacity: 20},
			},
		},
		{
			draExtendedResourceEnabled: true,
			pod:                        newResourcePod(framework.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo:                   framework.NewNodeInfo(newResourcePod(framework.Resource{})),
			name:                       "extended resource backed by device plugin",
			wantInsufficientResources:  []InsufficientResource{},
		},
		{
			draExtendedResourceEnabled: true,
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceDRA: 1}}),
			nodeInfo: framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:     "extended resource backed by DRA",
			// When DRAExtendedResource is enabled and there's a matching DeviceClass,
			// the noderesources plugin delegates to DRA instead of reporting insufficient resources.
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			draExtendedResourceEnabled: true,
			nilDRAManager:              true,
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:     "extended resource backed by DRA, nil draManager",
			// When DRAExtendedResource is enabled but draManager is nil (e.g., kubelet path),
			// the noderesources plugin delegates to DRA instead of reporting insufficient resources.
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			draExtendedResourceEnabled: false,
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceDRA: 1}}),
			nodeInfo:   framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:       "extended resource backed by DRA not enabled",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceDRA)),
			wantInsufficientResources: []InsufficientResource{
				{
					ResourceName: "extended.resource.dra.io/something",
					Reason:       "Insufficient extended.resource.dra.io/something",
					Requested:    1,
					Unresolvable: true,
				},
			},
		},
		{
			draExtendedResourceEnabled: true,
			pod: newResourcePod(
				framework.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo:   framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 0, Memory: 0})),
			name:       "extended resource NOT backed by DRA",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(extendedResourceB)),
			wantInsufficientResources: []InsufficientResource{
				{
					ResourceName: "example.com/bbb",
					Reason:       "Insufficient example.com/bbb",
					Requested:    1,
					Unresolvable: true,
				},
			},
		},
		{
			draExtendedResourceEnabled: true,
			pod: newResourcePod(
				framework.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceDRA: 1}}),
			nodeInfo:                  framework.NewNodeInfo(newResourcePod(framework.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceDRA: 0}})),
			name:                      "extended resource was backed by Device Plugin (Allocatable: 0)",
			wantInsufficientResources: []InsufficientResource{},
		},
		{
			draExtendedResourceEnabled: true,
			pod: newResourcePod(
				framework.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo:                  framework.NewNodeInfo(newResourcePod(framework.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 0}})),
			name:                      "extended resource was backed by Device Plugin (Allocatable: 0), global cache",
			wantInsufficientResources: []InsufficientResource{},
		},
	}

	for _, test := range enoughPodsTests {
		tCtx.SyncTest(test.name, func(tCtx ktesting.TContext) {

			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, test.draExtendedResourceEnabled)
			node := v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 5, 20, 5), Allocatable: makeAllocatableResources(10, 20, 32, 5, 20, 5)}}
			test.nodeInfo.SetNode(&node)

			if test.args.ScoringStrategy == nil {
				test.args.ScoringStrategy = defaultScoringStrategy
			}

			client := fake.NewSimpleClientset(deviceClassWithExtendResourceName)
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			claimsCache := assumecache.NewAssumeCache(tCtx.Logger(), informerFactory.Resource().V1().ResourceClaims().Informer(), "resource claim", "", nil)
			draManager := dynamicresources.NewDRAManager(tCtx, claimsCache, nil, informerFactory)
			if test.draExtendedResourceEnabled {
				cache := draManager.DeviceClassResolver().(*extendedresourcecache.ExtendedResourceCache)
				handle, err := informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(cache)
				tCtx.ExpectNoError(err, "add device class informer event handler")
				tCtx.Cleanup(func() {
					_ = informerFactory.Resource().V1().DeviceClasses().Informer().RemoveEventHandler(handle)
				})
			}
			informerFactory.Start(tCtx.Done())
			tCtx.Cleanup(func() {
				tCtx.Cancel("test has completed")
				// Now we can wait for all goroutines to stop.
				informerFactory.Shutdown()
			})
			informerFactory.WaitForCacheSync(tCtx.Done())
			// Wait for event delivery.
			synctest.Wait()

			runOpts := []runtime.Option{
				runtime.WithSharedDRAManager(draManager),
			}
			fh, _ := runtime.NewFramework(tCtx, nil, nil, runOpts...)
			defer func() {
				tCtx.Cancel("test has completed")
				runtime.WaitForShutdown(fh)
			}()
			p, err := NewFit(tCtx, &test.args, fh, plfeature.Features{EnablePodLevelResources: test.podLevelResourcesEnabled, EnableDRAExtendedResource: test.draExtendedResourceEnabled})
			tCtx.ExpectNoError(err, "create fit plugin")
			cycleState := framework.NewCycleState()
			_, preFilterStatus := p.(fwk.PreFilterPlugin).PreFilter(tCtx, cycleState, test.pod, nil)
			if !preFilterStatus.IsSuccess() {
				tCtx.Errorf("prefilter failed with status: %v", preFilterStatus)
			}

			gotStatus := p.(fwk.FilterPlugin).Filter(tCtx, cycleState, test.pod, test.nodeInfo)
			if diff := cmp.Diff(test.wantStatus, gotStatus); diff != "" {
				tCtx.Errorf("status does not match (-want,+got):\n%s", diff)
			}

			opts := ResourceRequestsOptions{EnablePodLevelResources: test.podLevelResourcesEnabled, EnableDRAExtendedResource: test.draExtendedResourceEnabled}
			state := computePodResourceRequest(test.pod, opts)
			var testDRAManager fwk.SharedDRAManager
			if !test.nilDRAManager {
				testDRAManager = draManager
			}
			gotInsufficientResources := fitsRequest(state, test.nodeInfo, p.(*Fit).ignoredResources, p.(*Fit).ignoredResourceGroups, testDRAManager, opts)
			if diff := cmp.Diff(test.wantInsufficientResources, gotInsufficientResources); diff != "" {
				tCtx.Errorf("insufficient resources do not match (-want,+got):\n%s", diff)
			}
		})
	}
}

func TestPreFilterDisabled(t *testing.T) { testPreFilterDisabled(ktesting.Init(t)) }
func testPreFilterDisabled(tCtx ktesting.TContext) {
	pod := &v1.Pod{}
	nodeInfo := framework.NewNodeInfo()
	node := v1.Node{}
	nodeInfo.SetNode(&node)
	p, err := NewFit(tCtx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, nil, plfeature.Features{})
	tCtx.ExpectNoError(err, "create fit plugin")
	cycleState := framework.NewCycleState()
	gotStatus := p.(fwk.FilterPlugin).Filter(tCtx, cycleState, pod, nodeInfo)
	wantStatus := fwk.AsStatus(fwk.ErrNotFound)
	if diff := cmp.Diff(wantStatus, gotStatus); diff != "" {
		tCtx.Errorf("status does not match (-want,+got):\n%s", diff)
	}
}

func TestNotEnoughRequests(t *testing.T) { testNotEnoughRequests(ktesting.Init(t)) }
func testNotEnoughRequests(tCtx ktesting.TContext) {
	notEnoughPodsTests := []struct {
		pod        *v1.Pod
		nodeInfo   *framework.NodeInfo
		fits       bool
		name       string
		wantStatus *fwk.Status
	}{
		{
			pod:        &v1.Pod{},
			nodeInfo:   framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 10, Memory: 20})),
			name:       "even without specified resources, predicate fails when there's no space for additional pod",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "Too many pods"),
		},
		{
			pod:        newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo:   framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 5, Memory: 5})),
			name:       "even if both resources fit, predicate fails when there's no space for additional pod",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "Too many pods"),
		},
		{
			pod:        newResourcePod(framework.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo:   framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 5, Memory: 19})),
			name:       "even for equal edge case, predicate fails when there's no space for additional pod",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "Too many pods"),
		},
		{
			pod:        newResourceInitPod(newResourcePod(framework.Resource{MilliCPU: 5, Memory: 1}), framework.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo:   framework.NewNodeInfo(newResourcePod(framework.Resource{MilliCPU: 5, Memory: 19})),
			name:       "even for equal edge case, predicate fails when there's no space for additional pod due to init container",
			wantStatus: fwk.NewStatus(fwk.Unschedulable, "Too many pods"),
		},
	}
	for _, test := range notEnoughPodsTests {
		tCtx.Run(test.name, func(tCtx ktesting.TContext) {
			node := v1.Node{Status: v1.NodeStatus{Capacity: v1.ResourceList{}, Allocatable: makeAllocatableResources(10, 20, 1, 0, 0, 0)}}
			test.nodeInfo.SetNode(&node)

			p, err := NewFit(tCtx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, nil, plfeature.Features{})
			tCtx.ExpectNoError(err, "create fit plugin")
			cycleState := framework.NewCycleState()
			_, preFilterStatus := p.(fwk.PreFilterPlugin).PreFilter(tCtx, cycleState, test.pod, nil)
			if !preFilterStatus.IsSuccess() {
				tCtx.Errorf("prefilter failed with status: %v", preFilterStatus)
			}

			gotStatus := p.(fwk.FilterPlugin).Filter(tCtx, cycleState, test.pod, test.nodeInfo)
			if diff := cmp.Diff(test.wantStatus, gotStatus); diff != "" {
				tCtx.Errorf("status does not match (-want,+got):\n%s", diff)
			}
		})
	}

}

func TestStorageRequests(t *testing.T) { testStorageRequests(ktesting.Init(t)) }
func testStorageRequests(tCtx ktesting.TContext) {
	storagePodsTests := []struct {
		pod        *v1.Pod
		nodeInfo   *framework.NodeInfo
		name       string
		wantStatus *fwk.Status
	}{
		{
			pod: newResourcePod(framework.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 2, Memory: 10})),
			name: "empty storage requested, and pod fits",
		},
		{
			pod: newResourcePod(framework.Resource{EphemeralStorage: 25}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 2, Memory: 2})),
			name:       "storage ephemeral local storage request exceeds allocatable",
			wantStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(v1.ResourceEphemeralStorage)),
		},
		{
			pod: newResourceInitPod(newResourcePod(framework.Resource{EphemeralStorage: 5})),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 2, Memory: 2, EphemeralStorage: 10})),
			name: "ephemeral local storage is sufficient",
		},
		{
			pod: newResourcePod(framework.Resource{EphemeralStorage: 10}),
			nodeInfo: framework.NewNodeInfo(
				newResourcePod(framework.Resource{MilliCPU: 2, Memory: 2})),
			name: "pod fits",
		},
	}

	for _, test := range storagePodsTests {
		tCtx.Run(test.name, func(tCtx ktesting.TContext) {
			node := v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 5, 20, 5), Allocatable: makeAllocatableResources(10, 20, 32, 5, 20, 5)}}
			test.nodeInfo.SetNode(&node)

			p, err := NewFit(tCtx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, nil, plfeature.Features{})
			tCtx.ExpectNoError(err, "create fit plugin")
			cycleState := framework.NewCycleState()
			_, preFilterStatus := p.(fwk.PreFilterPlugin).PreFilter(tCtx, cycleState, test.pod, nil)
			if !preFilterStatus.IsSuccess() {
				tCtx.Errorf("prefilter failed with status: %v", preFilterStatus)
			}

			gotStatus := p.(fwk.FilterPlugin).Filter(tCtx, cycleState, test.pod, test.nodeInfo)
			if diff := cmp.Diff(test.wantStatus, gotStatus); diff != "" {
				tCtx.Errorf("status does not match (-want,+got):\n%s", diff)
			}
		})
	}

}

func TestRestartableInitContainers(t *testing.T) { testRestartableInitContainers(ktesting.Init(t)) }
func testRestartableInitContainers(tCtx ktesting.TContext) {
	newPod := func() *v1.Pod {
		return &v1.Pod{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "regular"},
				},
			},
		}
	}
	newPodWithRestartableInitContainers := func(request, sidecarRequest *v1.ResourceList) *v1.Pod {
		restartPolicyAlways := v1.ContainerRestartPolicyAlways

		container := v1.Container{Name: "regular"}
		if request != nil {
			container.Resources = v1.ResourceRequirements{
				Requests: *request,
			}
		}

		sidecarContainer := v1.Container{
			Name:          "restartable-init",
			RestartPolicy: &restartPolicyAlways,
		}
		if sidecarRequest != nil {
			sidecarContainer.Resources = v1.ResourceRequirements{
				Requests: *sidecarRequest,
			}
		}
		return &v1.Pod{
			Spec: v1.PodSpec{
				Containers:     []v1.Container{container},
				InitContainers: []v1.Container{sidecarContainer},
			},
		}
	}

	testCases := []struct {
		name                    string
		pod                     *v1.Pod
		enableSidecarContainers bool
		wantPreFilterStatus     *fwk.Status
		wantFilterStatus        *fwk.Status
	}{
		{
			name: "allow pod without restartable init containers if sidecar containers is disabled",
			pod:  newPod(),
		},
		{
			name:                "not allow pod with restartable init containers if sidecar containers is disabled",
			pod:                 newPodWithRestartableInitContainers(nil, nil),
			wantPreFilterStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "Pod has a restartable init container and the SidecarContainers feature is disabled"),
		},
		{
			name:                    "allow pod without restartable init containers if sidecar containers is enabled",
			enableSidecarContainers: true,
			pod:                     newPod(),
		},
		{
			name:                    "allow pod with restartable init containers if sidecar containers is enabled",
			enableSidecarContainers: true,
			pod:                     newPodWithRestartableInitContainers(nil, nil),
		},
		{
			name:                    "allow pod if the total requested resources do not exceed the node's allocatable resources",
			enableSidecarContainers: true,
			pod: newPodWithRestartableInitContainers(
				&v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(1, resource.DecimalSI)},
				&v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(1, resource.DecimalSI)},
			),
		},
		{
			name:                    "not allow pod if the total requested resources do exceed the node's allocatable resources",
			enableSidecarContainers: true,
			pod: newPodWithRestartableInitContainers(
				&v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(1, resource.DecimalSI)},
				&v1.ResourceList{v1.ResourceCPU: *resource.NewMilliQuantity(2, resource.DecimalSI)},
			),
			wantFilterStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, getErrReason(v1.ResourceCPU)),
		},
	}

	for _, test := range testCases {
		tCtx.Run(test.name, func(tCtx ktesting.TContext) {
			node := v1.Node{Status: v1.NodeStatus{Capacity: v1.ResourceList{}, Allocatable: makeAllocatableResources(2, 0, 1, 0, 0, 0)}}
			nodeInfo := framework.NewNodeInfo()
			nodeInfo.SetNode(&node)

			p, err := NewFit(tCtx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, nil, plfeature.Features{EnableSidecarContainers: test.enableSidecarContainers})
			tCtx.ExpectNoError(err, "create fit plugin")
			cycleState := framework.NewCycleState()
			_, preFilterStatus := p.(fwk.PreFilterPlugin).PreFilter(tCtx, cycleState, test.pod, nil)
			if diff := cmp.Diff(test.wantPreFilterStatus, preFilterStatus); diff != "" {
				tCtx.Error("prefilter status does not match (-expected +actual):\n", diff)
			}
			if !preFilterStatus.IsSuccess() {
				return
			}

			filterStatus := p.(fwk.FilterPlugin).Filter(tCtx, cycleState, test.pod, nodeInfo)
			if diff := cmp.Diff(test.wantFilterStatus, filterStatus); diff != "" {
				tCtx.Error("filter status does not match (-expected +actual):\n", diff)
			}
		})
	}

}

func TestFitScore(t *testing.T) {
	testFitScore(ktesting.Init(t))
}
func testFitScore(tCtx ktesting.TContext) {
	tests := []struct {
		name                 string
		requestedPod         *v1.Pod
		nodes                []*v1.Node
		existingPods         []*v1.Pod
		expectedPriorities   fwk.NodeScoreList
		nodeResourcesFitArgs config.NodeResourcesFitArgs
		runPreScore          bool
		draObjects           []apiruntime.Object
	}{
		{
			name: "test case for ScoringStrategy RequestedToCapacityRatio case1",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "3000", "memory": "5000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "6000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 10}, {Name: "node2", Score: 32}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.RequestedToCapacityRatio,
					Resources: defaultResources,
					RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
						Shape: []config.UtilizationShapePoint{
							{Utilization: 0, Score: 10},
							{Utilization: 100, Score: 0},
						},
					},
				},
			},
			runPreScore: true,
		},
		{
			name: "test case for ScoringStrategy RequestedToCapacityRatio case2",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "3000", "memory": "5000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "6000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 95}, {Name: "node2", Score: 68}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.RequestedToCapacityRatio,
					Resources: defaultResources,
					RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
						Shape: []config.UtilizationShapePoint{
							{Utilization: 0, Score: 0},
							{Utilization: 100, Score: 10},
						},
					},
				},
			},
			runPreScore: true,
		},
		{
			name: "test case for ScoringStrategy MostAllocated",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "6000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 67}, {Name: "node2", Score: 36}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.MostAllocated,
					Resources: defaultResources,
				},
			},
			runPreScore: true,
		},
		{
			name: "test case for ScoringStrategy LeastAllocated",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "6000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 32}, {Name: "node2", Score: 63}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.LeastAllocated,
					Resources: defaultResources,
				},
			},
			runPreScore: true,
		},
		{
			name: "test case for ScoringStrategy RequestedToCapacityRatio case1 if PreScore is not called",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "3000", "memory": "5000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "6000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 10}, {Name: "node2", Score: 32}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.RequestedToCapacityRatio,
					Resources: defaultResources,
					RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
						Shape: []config.UtilizationShapePoint{
							{Utilization: 0, Score: 10},
							{Utilization: 100, Score: 0},
						},
					},
				},
			},
			runPreScore: false,
		},
		{
			name: "test case for ScoringStrategy MostAllocated if PreScore is not called",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "6000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 67}, {Name: "node2", Score: 36}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.MostAllocated,
					Resources: defaultResources,
				},
			},
			runPreScore: false,
		},
		{
			name: "test case for ScoringStrategy LeastAllocated if PreScore is not called",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "6000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 32}, {Name: "node2", Score: 63}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.LeastAllocated,
					Resources: defaultResources,
				},
			},
			runPreScore: false,
		},
		{
			name: "test case for ScoringStrategy MostAllocated with sidecar container",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
					SidecarReq(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 67}, {Name: "node2", Score: 45}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.MostAllocated,
					Resources: defaultResources,
				},
			},
			runPreScore: true,
		},
		{
			name: "test case for ScoringStrategy LeastAllocated with sidecar container",
			requestedPod: st.MakePod().
				Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
				Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).
					SidecarReq(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
				st.MakePod().Node("node2").Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 32}, {Name: "node2", Score: 55}},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.LeastAllocated,
					Resources: defaultResources,
				},
			},
			runPreScore: true,
		},
		{
			name:         "test case for ScoringStrategy LeastAllocated with Extended resources",
			requestedPod: st.MakePod().Req(map[v1.ResourceName]string{extendedResourceDRA: "1"}).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 66}, {Name: "node2", Score: 50}},
			draObjects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice("node1", "test-driver").Device("device-1").Device("device-2").Device("device-3").Obj(),
				st.MakeResourceSlice("node2", "test-driver").Device("device-1").Device("device-2").Obj(),
			},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type: config.LeastAllocated,
					Resources: []config.ResourceSpec{
						{Name: extendedResourceName, Weight: 1},
					},
				},
			},
			runPreScore: true,
		},
		{
			name:         "test case for ScoringStrategy LeastAllocated with Extended resources if PreScore is not called",
			requestedPod: st.MakePod().Req(map[v1.ResourceName]string{extendedResourceDRA: "1"}).Obj(),
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Obj(),
				st.MakeNode().Name("node2").Obj(),
			},
			expectedPriorities: []fwk.NodeScore{{Name: "node1", Score: 66}, {Name: "node2", Score: 50}},
			draObjects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice("node1", "test-driver").Device("device-1").Device("device-2").Device("device-3").Obj(),
				st.MakeResourceSlice("node2", "test-driver").Device("device-1").Device("device-2").Obj(),
			},
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type: config.LeastAllocated,
					Resources: []config.ResourceSpec{
						{Name: extendedResourceName, Weight: 1},
					},
				},
			},
		},
	}

	for _, test := range tests {
		tCtx.SyncTest(test.name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, test.draObjects != nil)
			state := framework.NewCycleState()
			snapshot := cache.NewSnapshot(test.existingPods, test.nodes)
			fh, _ := runtime.NewFramework(tCtx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			defer func() {
				tCtx.Cancel("test has completed")
				runtime.WaitForShutdown(fh)
			}()
			args := test.nodeResourcesFitArgs
			p, err := NewFit(tCtx, &args, fh, plfeature.Features{
				EnableDRAExtendedResource: test.draObjects != nil,
			})
			if err != nil {
				tCtx.Fatalf("unexpected error: %v", err)
			}

			if test.draObjects != nil {
				draManager := newTestDRAManager(tCtx, test.draObjects...)
				p.(*Fit).draManager = draManager
			}

			var gotPriorities fwk.NodeScoreList
			for _, n := range test.nodes {
				if test.runPreScore {
					status := p.(fwk.PreScorePlugin).PreScore(tCtx, state, test.requestedPod, tf.BuildNodeInfos(test.nodes))
					if !status.IsSuccess() {
						tCtx.Errorf("PreScore is expected to return success, but didn't. Got status: %v", status)
					}
				}
				nodeInfo, err := snapshot.Get(n.Name)
				if err != nil {
					tCtx.Errorf("failed to get node %q from snapshot: %v", n.Name, err)
				}
				score, status := p.(fwk.ScorePlugin).Score(tCtx, state, test.requestedPod, nodeInfo)
				if !status.IsSuccess() {
					tCtx.Errorf("Score is expected to return success, but didn't. Got status: %v", status)
				}
				gotPriorities = append(gotPriorities, fwk.NodeScore{Name: n.Name, Score: score})
			}

			if diff := cmp.Diff(test.expectedPriorities, gotPriorities); diff != "" {
				tCtx.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedPriorities, gotPriorities)
			}
		})
	}
}

var benchmarkResourceSet = []config.ResourceSpec{
	{Name: string(v1.ResourceCPU), Weight: 1},
	{Name: string(v1.ResourceMemory), Weight: 1},
	{Name: string(v1.ResourcePods), Weight: 1},
	{Name: string(v1.ResourceStorage), Weight: 1},
	{Name: string(v1.ResourceEphemeralStorage), Weight: 1},
	{Name: string(extendedResourceA), Weight: 1},
	{Name: string(extendedResourceB), Weight: 1},
	{Name: string(kubernetesIOResourceA), Weight: 1},
	{Name: string(kubernetesIOResourceB), Weight: 1},
	{Name: string(hugePageResourceA), Weight: 1},
}

func BenchmarkTestFitScore(b *testing.B) {
	tests := []struct {
		name                 string
		nodeResourcesFitArgs config.NodeResourcesFitArgs
	}{
		{
			name: "RequestedToCapacityRatio with defaultResources",
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.RequestedToCapacityRatio,
					Resources: defaultResources,
					RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
						Shape: []config.UtilizationShapePoint{
							{Utilization: 0, Score: 10},
							{Utilization: 100, Score: 0},
						},
					},
				},
			},
		},
		{
			name: "RequestedToCapacityRatio with 10 resources",
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.RequestedToCapacityRatio,
					Resources: benchmarkResourceSet,
					RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
						Shape: []config.UtilizationShapePoint{
							{Utilization: 0, Score: 10},
							{Utilization: 100, Score: 0},
						},
					},
				},
			},
		},
		{
			name: "MostAllocated with defaultResources",
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.MostAllocated,
					Resources: defaultResources,
				},
			},
		},
		{
			name: "MostAllocated with 10 resources",
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.MostAllocated,
					Resources: benchmarkResourceSet,
				},
			},
		},
		{
			name: "LeastAllocated with defaultResources",
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.LeastAllocated,
					Resources: defaultResources,
				},
			},
		},
		{
			name: "LeastAllocated with 10 resources",
			nodeResourcesFitArgs: config.NodeResourcesFitArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type:      config.LeastAllocated,
					Resources: benchmarkResourceSet,
				},
			},
		},
	}

	for _, test := range tests {
		b.Run(test.name, func(b *testing.B) {
			_, ctx := ktesting.NewTestContext(b)
			existingPods := []*v1.Pod{
				st.MakePod().Node("node1").Req(map[v1.ResourceName]string{"cpu": "2000", "memory": "4000"}).Obj(),
			}
			nodes := []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{"cpu": "4000", "memory": "10000"}).Obj(),
			}
			state := framework.NewCycleState()
			var nodeResourcesFunc = runtime.FactoryAdapter(plfeature.Features{}, NewFit)
			pl := plugintesting.SetupPlugin(ctx, b, nodeResourcesFunc, &test.nodeResourcesFitArgs, cache.NewSnapshot(existingPods, nodes))
			p := pl.(*Fit)
			nodeInfo, err := p.handle.SnapshotSharedLister().NodeInfos().Get(nodes[0].Name)
			if err != nil {
				b.Errorf("failed to get node %q from snapshot: %v", nodes[0].Name, err)
			}

			b.ResetTimer()

			requestedPod := st.MakePod().Req(map[v1.ResourceName]string{"cpu": "1000", "memory": "2000"}).Obj()
			for i := 0; i < b.N; i++ {
				_, status := p.Score(ctx, state, requestedPod, nodeInfo)
				if !status.IsSuccess() {
					b.Errorf("unexpected status: %v", status)
				}
			}
		})
	}
}

func TestEventsToRegister(t *testing.T) {
	tests := []struct {
		name                            string
		enableInPlacePodVerticalScaling bool
		enableSchedulingQueueHint       bool
		enableDRAExtendedResource       bool
		expectedClusterEvents           []fwk.ClusterEventWithHint
	}{
		{
			name:                            "Register events with InPlacePodVerticalScaling feature enabled",
			enableInPlacePodVerticalScaling: true,
			expectedClusterEvents: []fwk.ClusterEventWithHint{
				{Event: fwk.ClusterEvent{Resource: "Pod", ActionType: fwk.UpdatePodScaleDown | fwk.Delete}},
				{Event: fwk.ClusterEvent{Resource: "Node", ActionType: fwk.Add | fwk.UpdateNodeAllocatable | fwk.UpdateNodeTaint | fwk.UpdateNodeLabel}},
			},
		},
		{
			name:                      "Register events with SchedulingQueueHint feature enabled",
			enableSchedulingQueueHint: true,
			expectedClusterEvents: []fwk.ClusterEventWithHint{
				{Event: fwk.ClusterEvent{Resource: "Pod", ActionType: fwk.Delete}},
				{Event: fwk.ClusterEvent{Resource: "Node", ActionType: fwk.Add | fwk.UpdateNodeAllocatable}},
			},
		},
		{
			name:                            "Register events with InPlacePodVerticalScaling feature disabled",
			enableInPlacePodVerticalScaling: false,
			expectedClusterEvents: []fwk.ClusterEventWithHint{
				{Event: fwk.ClusterEvent{Resource: "Pod", ActionType: fwk.Delete}},
				{Event: fwk.ClusterEvent{Resource: "Node", ActionType: fwk.Add | fwk.UpdateNodeAllocatable | fwk.UpdateNodeTaint | fwk.UpdateNodeLabel}},
			},
		},
		{
			name:                      "Register events with DRAExtendedResource feature enabled",
			enableDRAExtendedResource: true,
			expectedClusterEvents: []fwk.ClusterEventWithHint{
				{Event: fwk.ClusterEvent{Resource: "Pod", ActionType: fwk.Delete}},
				{Event: fwk.ClusterEvent{Resource: "Node", ActionType: fwk.Add | fwk.UpdateNodeAllocatable | fwk.UpdateNodeTaint | fwk.UpdateNodeLabel}},
				{Event: fwk.ClusterEvent{Resource: fwk.DeviceClass, ActionType: fwk.Add | fwk.Update}},
			},
		},
		{
			name:                      "Register events with DRAExtendedResource feature disabled",
			enableDRAExtendedResource: false,
			expectedClusterEvents: []fwk.ClusterEventWithHint{
				{Event: fwk.ClusterEvent{Resource: "Pod", ActionType: fwk.Delete}},
				{Event: fwk.ClusterEvent{Resource: "Node", ActionType: fwk.Add | fwk.UpdateNodeAllocatable | fwk.UpdateNodeTaint | fwk.UpdateNodeLabel}},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fp := &Fit{enableInPlacePodVerticalScaling: test.enableInPlacePodVerticalScaling, enableSchedulingQueueHint: test.enableSchedulingQueueHint, enableDRAExtendedResource: test.enableDRAExtendedResource}
			_, ctx := ktesting.NewTestContext(t)
			actualClusterEvents, err := fp.EventsToRegister(ctx)
			if err != nil {
				t.Fatal(err)
			}
			for i := range actualClusterEvents {
				actualClusterEvents[i].QueueingHintFn = nil
			}
			if diff := cmp.Diff(test.expectedClusterEvents, actualClusterEvents, clusterEventCmpOpts...); diff != "" {
				t.Error("Cluster Events doesn't match extected events (-expected +actual):\n", diff)
			}
		})
	}
}

func Test_isSchedulableAfterPodChange(t *testing.T) {
	testcases := map[string]struct {
		pod                             *v1.Pod
		oldObj, newObj                  interface{}
		enableInPlacePodVerticalScaling bool
		expectedHint                    fwk.QueueingHint
		expectedErr                     bool
	}{
		"backoff-wrong-old-object": {
			pod:                             &v1.Pod{},
			oldObj:                          "not-a-pod",
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.Queue,
			expectedErr:                     true,
		},
		"backoff-wrong-new-object": {
			pod:                             &v1.Pod{},
			newObj:                          "not-a-pod",
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.Queue,
			expectedErr:                     true,
		},
		"queue-on-other-pod-deleted": {
			pod:                             st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj:                          st.MakePod().Name("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("fake").Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.Queue,
		},
		"skip-queue-on-unscheduled-pod-deleted": {
			pod:                             &v1.Pod{},
			oldObj:                          &v1.Pod{},
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.QueueSkip,
		},
		"skip-queue-on-disable-inplace-pod-vertical-scaling": {
			pod:    st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj: st.MakePod().Name("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("fake").Obj(),
			// (Actually, this scale down cannot happen when InPlacePodVerticalScaling is disabled.)
			newObj:                          st.MakePod().Name("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("fake").Obj(),
			enableInPlacePodVerticalScaling: false,
			expectedHint:                    fwk.QueueSkip,
		},
		"skip-queue-on-other-unscheduled-pod": {
			pod:                             st.MakePod().Name("pod1").UID("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).UID("uid0").Obj(),
			oldObj:                          st.MakePod().Name("pod2").UID("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).UID("uid1").Obj(),
			newObj:                          st.MakePod().Name("pod2").UID("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).UID("uid1").Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.QueueSkip,
		},
		"skip-queue-on-other-pod-unrelated-resource-scaled-down": {
			pod:                             st.MakePod().Name("pod1").UID("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj:                          st.MakePod().Name("pod2").UID("pod2").Req(map[v1.ResourceName]string{v1.ResourceMemory: "2"}).Node("fake").Obj(),
			newObj:                          st.MakePod().Name("pod2").UID("pod2").Req(map[v1.ResourceName]string{v1.ResourceMemory: "1"}).Node("fake").Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.QueueSkip,
		},
		"skip-queue-on-other-pod-unrelated-pod-level-resource-scaled-down": {
			pod:                             st.MakePod().Name("pod1").UID("pod1").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj:                          st.MakePod().Name("pod2").UID("pod2").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceMemory: "2"}).Node("fake").Obj(),
			newObj:                          st.MakePod().Name("pod2").UID("pod2").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceMemory: "1"}).Node("fake").Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.QueueSkip,
		},
		"queue-on-other-pod-some-resource-scale-down": {
			pod:                             st.MakePod().Name("pod1").UID("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj:                          st.MakePod().Name("pod2").UID("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("fake").Obj(),
			newObj:                          st.MakePod().Name("pod2").UID("pod2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("fake").Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.Queue,
		},
		"queue-on-other-pod-some-pod-level-resource-scale-down": {
			pod:                             st.MakePod().Name("pod1").UID("pod1").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj:                          st.MakePod().Name("pod2").UID("pod2").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("fake").Obj(),
			newObj:                          st.MakePod().Name("pod2").UID("pod2").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("fake").Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.Queue,
		},
		"queue-on-target-pod-some-resource-scale-down": {
			pod:                             st.MakePod().Name("pod1").UID("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj:                          st.MakePod().Name("pod1").UID("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			newObj:                          st.MakePod().Name("pod1").UID("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.Queue,
		},
		"queue-on-target-pod-some-pod-level-resource-scale-down": {
			pod:                             st.MakePod().Name("pod1").UID("pod1").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			oldObj:                          st.MakePod().Name("pod1").UID("pod1").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			newObj:                          st.MakePod().Name("pod1").UID("pod1").PodLevelResourceRequests(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			enableInPlacePodVerticalScaling: true,
			expectedHint:                    fwk.Queue,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			p, err := NewFit(ctx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, nil, plfeature.Features{
				EnableInPlacePodVerticalScaling: tc.enableInPlacePodVerticalScaling,
			})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*Fit).isSchedulableAfterPodEvent(logger, tc.pod, tc.oldObj, tc.newObj)
			if tc.expectedErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expectedHint, actualHint)
		})
	}
}

func Test_isSchedulableAfterNodeChange(t *testing.T) {
	testcases := map[string]struct {
		pod            *v1.Pod
		oldObj, newObj interface{}
		expectedHint   fwk.QueueingHint
		expectedErr    bool
	}{
		"backoff-wrong-new-object": {
			pod:          &v1.Pod{},
			newObj:       "not-a-node",
			expectedHint: fwk.Queue,
			expectedErr:  true,
		},
		"backoff-wrong-old-object": {
			pod:          &v1.Pod{},
			oldObj:       "not-a-node",
			newObj:       &v1.Node{},
			expectedHint: fwk.Queue,
			expectedErr:  true,
		},
		"skip-queue-on-node-add-without-sufficient-resources": {
			pod: newResourcePod(framework.Resource{Memory: 2}),
			newObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "1",
			}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		"skip-queue-on-node-add-without-required-resource-type": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}},
			),
			newObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				extendedResourceB: "1",
			}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		"queue-on-node-add-with-sufficient-resources": {
			pod: newResourcePod(framework.Resource{
				Memory:          2,
				ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1},
			}),
			newObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "4",
				extendedResourceA: "2",
			}).Obj(),
			expectedHint: fwk.Queue,
		},
		"skip-queue-on-node-unrelated-changes": {
			pod: newResourcePod(framework.Resource{
				Memory:          2,
				ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1},
			}),
			oldObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "2",
				extendedResourceA: "2",
			}).Obj(),
			newObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "2",
				extendedResourceA: "1",
				extendedResourceB: "2",
			}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		"queue-on-pod-requested-resources-increase": {
			pod: newResourcePod(framework.Resource{
				Memory:          2,
				ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1},
			}),
			oldObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "2",
				extendedResourceA: "1",
			}).Obj(),
			newObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "2",
				extendedResourceA: "2",
			}).Obj(),
			expectedHint: fwk.Queue,
		},
		"skip-queue-on-node-changes-from-suitable-to-unsuitable": {
			pod: newResourcePod(framework.Resource{
				Memory:          2,
				ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1},
			}),
			oldObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "4",
				extendedResourceA: "2",
			}).Obj(),
			newObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "1",
				extendedResourceA: "2",
			}).Obj(),
			expectedHint: fwk.QueueSkip,
		},
		"queue-on-node-changes-from-unsuitable-to-suitable": {
			pod: newResourcePod(framework.Resource{
				Memory:          2,
				ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1},
			}),
			oldObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "1",
				extendedResourceA: "2",
			}).Obj(),
			newObj: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceMemory: "4",
				extendedResourceA: "2",
			}).Obj(),
			expectedHint: fwk.Queue,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			p, err := NewFit(ctx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, nil, plfeature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*Fit).isSchedulableAfterNodeChange(logger, tc.pod, tc.oldObj, tc.newObj)
			if tc.expectedErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expectedHint, actualHint)
		})
	}
}

func Test_isSchedulableAfterDeviceClassChange(t *testing.T) {
	ern := "example.com/gpu"
	testcases := map[string]struct {
		pod            *v1.Pod
		oldObj, newObj any
		expectedHint   fwk.QueueingHint
		expectedErr    bool
	}{
		"backoff-wrong-new-object": {
			pod:          &v1.Pod{},
			newObj:       "not-a-class",
			expectedHint: fwk.Queue,
			expectedErr:  true,
		},
		"backoff-wrong-old-object": {
			pod:          &v1.Pod{},
			oldObj:       "not-a-class",
			newObj:       &resourceapi.DeviceClass{},
			expectedHint: fwk.Queue,
			expectedErr:  true,
		},
		"skip-queue-on-class-nil-extended-resource-name-pointer": {
			pod: newResourcePod(framework.Resource{Memory: 2}),
			newObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{},
			},
			oldObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{},
			},
			expectedHint: fwk.QueueSkip,
		},
		"skip-queue-on-class-same-extended-resource-name-pointer": {
			pod: newResourcePod(framework.Resource{Memory: 2}),
			newObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{
					ExtendedResourceName: &ern,
				},
			},
			oldObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{
					ExtendedResourceName: &ern,
				},
			},
			expectedHint: fwk.QueueSkip,
		},
		"skip-queue-on-class-same-extended-resource-name": {
			pod: newResourcePod(framework.Resource{Memory: 2}),
			newObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			oldObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			expectedHint: fwk.QueueSkip,
		},
		"queue-on-class-add-with-implicit-extended-resource-name": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{"deviceclass.resource.kubernetes.io/gpuclass": 1},
			}),
			newObj: &resourceapi.DeviceClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpuclass",
				},
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			expectedHint: fwk.Queue,
		},
		"skip-on-class-add-with-implicit-extended-resource-name-not-matching": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{"deviceclass.resource.kubernetes.io/gpuclass": 1},
			}),
			newObj: &resourceapi.DeviceClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myclass",
				},
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			expectedHint: fwk.QueueSkip,
		},
		"skip-on-class-add-with-explicit-extended-resource-name-not-matching": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{"example.com/othergpu": 1},
			}),
			newObj: &resourceapi.DeviceClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "myclass",
				},
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			expectedHint: fwk.QueueSkip,
		},
		"skip-on-class-update-with-implicit-extended-resource-name": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{"deviceclass.resource.kubernetes.io/gpuclass": 1},
			}),
			newObj: &resourceapi.DeviceClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpuclass",
				},
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			oldObj: &resourceapi.DeviceClass{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpuclass",
				},
			},
			expectedHint: fwk.QueueSkip,
		},
		"queue-on-class-add-with-extended-resource-name": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{"example.com/gpu": 1},
			}),
			newObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			expectedHint: fwk.Queue,
		},
		"queue-on-class-update-with-extended-resource-name": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{"example.com/gpu": 1},
			}),
			newObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu")},
			},
			oldObj: &resourceapi.DeviceClass{
				Spec: resourceapi.DeviceClassSpec{ExtendedResourceName: ptr.To("example.com/gpu1")},
			},
			expectedHint: fwk.Queue,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			p, err := NewFit(ctx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, nil, plfeature.Features{})
			if err != nil {
				t.Fatal(err)
			}
			actualHint, err := p.(*Fit).isSchedulableAfterDeviceClassEvent(logger, tc.pod, tc.oldObj, tc.newObj)
			if tc.expectedErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.expectedHint, actualHint)
		})
	}
}

func TestIsFit(t *testing.T) {
	testCases := map[string]struct {
		pod                      *v1.Pod
		node                     *v1.Node
		podLevelResourcesEnabled bool
		expected                 bool
	}{
		"nil node": {
			pod:      &v1.Pod{},
			expected: false,
		},
		"insufficient resource": {
			pod:      st.MakePod().Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			node:     st.MakeNode().Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			expected: false,
		},
		"sufficient resource": {
			pod:      st.MakePod().Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			node:     st.MakeNode().Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			expected: true,
		},
		"insufficient pod-level resource": {
			pod: st.MakePod().Resources(
				v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}},
			).Obj(),
			node:                     st.MakeNode().Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj(),
			podLevelResourcesEnabled: true,
			expected:                 false,
		},
		"sufficient pod-level resource": {
			pod: st.MakePod().Resources(
				v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")}},
			).Obj(),
			node:                     st.MakeNode().Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			podLevelResourcesEnabled: true,
			expected:                 true,
		},
		"sufficient pod-level resource hugepages": {
			pod: st.MakePod().Resources(
				v1.ResourceRequirements{Requests: v1.ResourceList{hugePageResourceA: resource.MustParse("2Mi")}},
			).Obj(),
			node:                     st.MakeNode().Capacity(map[v1.ResourceName]string{hugePageResourceA: "2Mi"}).Obj(),
			podLevelResourcesEnabled: true,
			expected:                 true,
		},
		"insufficient pod-level resource hugepages": {
			pod: st.MakePod().Resources(
				v1.ResourceRequirements{Requests: v1.ResourceList{hugePageResourceA: resource.MustParse("4Mi")}},
			).Obj(),
			node:                     st.MakeNode().Capacity(map[v1.ResourceName]string{hugePageResourceA: "2Mi"}).Obj(),
			podLevelResourcesEnabled: true,
			expected:                 false,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			if got := isFit(tc.pod, tc.node, nil, ResourceRequestsOptions{EnablePodLevelResources: tc.podLevelResourcesEnabled}); got != tc.expected {
				t.Errorf("expected: %v, got: %v", tc.expected, got)
			}
		})
	}
}

func TestHaveAnyRequestedResourcesIncreased(t *testing.T) {
	testCases := map[string]struct {
		pod                        *v1.Pod
		originalNode               *v1.Node
		modifiedNode               *v1.Node
		draExtendedResourceEnabled bool
		expected                   bool
	}{
		"no-requested-resources": {
			pod: newResourcePod(framework.Resource{}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:             "1",
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "1",
				v1.ResourceEphemeralStorage: "1",
				extendedResourceA:           "1",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:             "1",
				v1.ResourceCPU:              "2",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "2",
				extendedResourceA:           "2",
			}).Obj(),
			expected: false,
		},
		"no-requested-resources-pods-increased": {
			pod: newResourcePod(framework.Resource{}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:             "1",
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "1",
				v1.ResourceEphemeralStorage: "1",
				extendedResourceA:           "1",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourcePods:             "2",
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "1",
				v1.ResourceEphemeralStorage: "1",
				extendedResourceA:           "1",
			}).Obj(),
			expected: true,
		},
		"requested-resources-dra-in-node": {
			pod: newResourcePod(framework.Resource{
				MilliCPU:         2,
				Memory:           2,
				EphemeralStorage: 2,
				ScalarResources:  map[v1.ResourceName]int64{extendedResourceA: 2},
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "3",
				extendedResourceA:           "4",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "1",
				extendedResourceA:           "1",
			}).Obj(),
			draExtendedResourceEnabled: true,
			expected:                   false,
		},
		"requested-resources-dra-not-in-node": {
			pod: newResourcePod(framework.Resource{
				MilliCPU:         2,
				Memory:           2,
				EphemeralStorage: 2,
				ScalarResources:  map[v1.ResourceName]int64{extendedResourceA: 2},
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "3",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "1",
			}).Obj(),
			draExtendedResourceEnabled: true,
			expected:                   true,
		},
		"requested-resources-dra-zero-capacity": {
			pod: newResourcePod(framework.Resource{
				ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1},
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				extendedResourceA: "0",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				extendedResourceA: "0",
			}).Obj(),
			draExtendedResourceEnabled: true,
			expected:                   true,
		},
		"scalar-resources-not-in-node": {
			pod: newResourcePod(framework.Resource{
				MilliCPU:         2,
				Memory:           2,
				EphemeralStorage: 2,
				ScalarResources:  map[v1.ResourceName]int64{extendedResourceA: 2},
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "3",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "1",
			}).Obj(),
			draExtendedResourceEnabled: false,
			expected:                   false,
		},
		"requested-resources-decreased": {
			pod: newResourcePod(framework.Resource{
				MilliCPU:         2,
				Memory:           2,
				EphemeralStorage: 2,
				ScalarResources:  map[v1.ResourceName]int64{extendedResourceA: 2},
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "3",
				extendedResourceA:           "4",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "1",
				extendedResourceA:           "1",
			}).Obj(),
			expected: false,
		},
		"requested-resources-increased": {
			pod: newResourcePod(framework.Resource{
				MilliCPU:         2,
				Memory:           2,
				EphemeralStorage: 2,
				ScalarResources:  map[v1.ResourceName]int64{extendedResourceA: 2},
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "3",
				extendedResourceA:           "4",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "3",
				v1.ResourceMemory:           "4",
				v1.ResourceEphemeralStorage: "3",
				extendedResourceA:           "4",
			}).Obj(),
			expected: true,
		},
		"non-requested-resources-decreased": {
			pod: newResourcePod(framework.Resource{
				MilliCPU: 2,
				Memory:   2,
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "3",
				extendedResourceA:           "4",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "1",
				extendedResourceA:           "1",
			}).Obj(),
			expected: false,
		},
		"non-requested-resources-increased": {
			pod: newResourcePod(framework.Resource{
				MilliCPU: 2,
				Memory:   2,
			}),
			originalNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "3",
				extendedResourceA:           "4",
			}).Obj(),
			modifiedNode: st.MakeNode().Capacity(map[v1.ResourceName]string{
				v1.ResourceCPU:              "1",
				v1.ResourceMemory:           "2",
				v1.ResourceEphemeralStorage: "5",
				extendedResourceA:           "6",
			}).Obj(),
			expected: false,
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			if got := haveAnyRequestedResourcesIncreased(tc.pod, tc.originalNode, tc.modifiedNode, nil, ResourceRequestsOptions{EnableDRAExtendedResource: tc.draExtendedResourceEnabled}); got != tc.expected {
				t.Errorf("expected: %v, got: %v", tc.expected, got)
			}
		})
	}
}

func TestFitSignPod(t *testing.T) {
	tests := map[string]struct {
		name                      string
		pod                       *v1.Pod
		enableDRAExtendedResource bool
		expectedFragments         []fwk.SignFragment
		expectedStatusCode        fwk.Code
	}{
		"pod with CPU and memory requests": {
			pod: st.MakePod().Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "2000",
			}).Obj(),
			enableDRAExtendedResource: false,
			expectedFragments: []fwk.SignFragment{
				{Key: fwk.ResourcesSignerName, Value: computePodResourceRequest(st.MakePod().Req(map[v1.ResourceName]string{
					v1.ResourceCPU:    "1000m",
					v1.ResourceMemory: "2000",
				}).Obj(), ResourceRequestsOptions{})},
			},
			expectedStatusCode: fwk.Success,
		},
		"best-effort pod with no requests": {
			pod:                       st.MakePod().Obj(),
			enableDRAExtendedResource: false,
			expectedFragments: []fwk.SignFragment{
				{Key: fwk.ResourcesSignerName, Value: computePodResourceRequest(st.MakePod().Obj(), ResourceRequestsOptions{})},
			},
			expectedStatusCode: fwk.Success,
		},
		"pod with multiple containers": {
			pod: st.MakePod().Container("container1").Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "500m",
				v1.ResourceMemory: "1000",
			}).Container("container2").Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1500m",
				v1.ResourceMemory: "3000",
			}).Obj(),
			enableDRAExtendedResource: false,
			expectedFragments: []fwk.SignFragment{
				{Key: fwk.ResourcesSignerName, Value: computePodResourceRequest(st.MakePod().Container("container1").Req(map[v1.ResourceName]string{
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "1000",
				}).Container("container2").Req(map[v1.ResourceName]string{
					v1.ResourceCPU:    "1500m",
					v1.ResourceMemory: "3000",
				}).Obj(), ResourceRequestsOptions{})},
			},
			expectedStatusCode: fwk.Success,
		},
		"DRA extended resource enabled - returns unschedulable": {
			pod: st.MakePod().Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "2000",
			}).Obj(),
			enableDRAExtendedResource: true,
			expectedFragments:         nil,
			expectedStatusCode:        fwk.Unschedulable,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			fh, _ := runtime.NewFramework(ctx, nil, nil)
			p, err := NewFit(ctx, &config.NodeResourcesFitArgs{ScoringStrategy: defaultScoringStrategy}, fh, plfeature.Features{
				EnableDRAExtendedResource: test.enableDRAExtendedResource,
			})
			if err != nil {
				t.Fatalf("failed to create plugin: %v", err)
			}

			fit := p.(*Fit)
			fragments, status := fit.SignPod(ctx, test.pod)

			if status.Code() != test.expectedStatusCode {
				t.Errorf("unexpected status code, want: %v, got: %v, message: %v", test.expectedStatusCode, status.Code(), status.Message())
			}

			if test.expectedStatusCode == fwk.Success {
				if diff := cmp.Diff(test.expectedFragments, fragments); diff != "" {
					t.Errorf("unexpected fragments, diff (-want,+got):\n%s", diff)
				}
			}
		})
	}
}
