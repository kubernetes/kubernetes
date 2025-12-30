/*
Copyright 2022 The Kubernetes Authors.

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

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	goruntime "runtime"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	cgotesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/dynamic-resource-allocation/deviceclass/extendedresourcecache"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/dynamic-resource-allocation/structured"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	configv1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func init() {
	metrics.InitMetrics()
}

var (
	podKind = v1.SchemeGroupVersion.WithKind("Pod")

	nodeName                     = "worker"
	node2Name                    = "worker-2"
	node3Name                    = "worker-3"
	driver                       = "some-driver"
	driver2                      = "some-driver-2"
	podName                      = "my-pod"
	podUID                       = "1234"
	resourceName                 = "my-resource"
	resourceName2                = resourceName + "-2"
	claimName                    = podName + "-" + resourceName
	claimName2                   = podName + "-" + resourceName2
	className                    = "my-resource-class"
	namespace                    = "default"
	attrName                     = resourceapi.QualifiedName("healthy") // device attribute only available on non-default node
	extendedResourceName         = "example.com/gpu"
	extendedResourceName2        = "example.com/gpu2"
	implicitExtendedResourceName = "deviceclass.resource.kubernetes.io/my-resource-class"

	deviceClass = &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
	}
	deviceClassWithExtendResourceName = &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className,
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: &extendedResourceName,
		},
	}
	deviceClassWithExtendResourceName2 = &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: className + "2",
		},
		Spec: resourceapi.DeviceClassSpec{
			ExtendedResourceName: &extendedResourceName2,
		},
	}
	podWithClaimName = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
				Obj()
	podWithClaimTemplate = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimTemplateName: &claimName}).
				Obj()
	podWithClaimTemplateInStatus = func() *v1.Pod {
		pod := podWithClaimTemplate.DeepCopy()
		pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{
			{
				Name:              pod.Spec.ResourceClaims[0].Name,
				ResourceClaimName: &claimName,
			},
		}
		return pod
	}()
	podWithTwoClaimTemplates = st.MakePod().Name(podName).Namespace(namespace).
					UID(podUID).
					PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimTemplateName: &claimName}).
					PodResourceClaims(v1.PodResourceClaim{Name: resourceName2, ResourceClaimTemplateName: &claimName}).
					Obj()
	podWithTwoClaimNames = st.MakePod().Name(podName).Namespace(namespace).
				UID(podUID).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName2, ResourceClaimName: &claimName2}).
				Obj()
	podWithExtendedResourceName = st.MakePod().Name(podName).Namespace(namespace).
					UID(podUID).
					Req(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "1",
		}).
		Obj()
	podWithExtendedResourceName2 = st.MakePod().Name(podName).Namespace(namespace).
					UID(podUID).
					Req(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName):  "1",
			v1.ResourceName(extendedResourceName2): "1",
		}).
		Obj()
	podWithImplicitExtendedResourceName = st.MakePod().Name(podName).Namespace(namespace).
						UID(podUID).
						Req(map[v1.ResourceName]string{
			v1.ResourceName(implicitExtendedResourceName): "1",
			v1.ResourceName(extendedResourceName):         "2",
		}).
		Obj()
	podWithImplicitExtendedResourceNameTwoContainers = st.MakePod().Name(podName).Namespace(namespace).
								UID(podUID).
								Req(map[v1.ResourceName]string{
			v1.ResourceName(implicitExtendedResourceName): "1",
		}).
		Req(map[v1.ResourceName]string{
			v1.ResourceName(extendedResourceName): "2",
		}).
		Obj()

	// Node with "instance-1" device and no device attributes.
	workerNode           = &st.MakeNode().Name(nodeName).Label("kubernetes.io/hostname", nodeName).Node
	workerNodeSlice      = st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
	largeWorkerNodeSlice = st.MakeResourceSlice(nodeName, driver).Device("instance-1").Device("instance-2").Device("instance-3").Device("instance-4").Obj()

	// Node with same device, but now with a "healthy" boolean attribute.
	workerNode2      = &st.MakeNode().Name(node2Name).Label("kubernetes.io/hostname", node2Name).Node
	workerNode2Slice = st.MakeResourceSlice(node2Name, driver).Device("instance-1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{attrName: {BoolValue: ptr.To(true)}}).Obj()

	// Yet another node, same as the second one.
	workerNode3      = &st.MakeNode().Name(node3Name).Label("kubernetes.io/hostname", node3Name).Node
	workerNode3Slice = st.MakeResourceSlice(node3Name, driver).Device("instance-1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{attrName: {BoolValue: ptr.To(true)}}).Obj()

	workerNodeWithExtendedResource                = &st.MakeNode().Name(nodeName).Label("kubernetes.io/hostname", nodeName).Capacity(map[v1.ResourceName]string{v1.ResourceName(extendedResourceName): "1"}).Node
	workerNodeWithExtendedResourceZeroAllocatable = &st.MakeNode().Name(nodeName).Label("kubernetes.io/hostname", nodeName).Capacity(map[v1.ResourceName]string{v1.ResourceName(extendedResourceName): "0"}).Node
	brokenSelector                                = resourceapi.DeviceSelector{
		CEL: &resourceapi.CELDeviceSelector{
			// Not set for workerNode.
			Expression: fmt.Sprintf(`device.attributes["%s"].%s`, driver, attrName),
		},
	}

	claim = st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		Request(className).
		Obj()
	largeClaim = st.MakeResourceClaim().
			Name(claimName).
			Namespace(namespace).
			Request(className).
			Request(className).
			Request(className).
			Request(className).
			Request(className).
			Obj()
	claim2 = st.MakeResourceClaim().
		Name(claimName2).
		Namespace(namespace).
		Request(className).
		Obj()
	claimWithPrioritzedList = st.MakeResourceClaim().
				Name(claimName).
				Namespace(namespace).
				RequestWithPrioritizedList(
			st.SubRequest("subreq-1", className, 1),
		).
		Obj()
	claimWithPrioritizedListAndSelector = st.MakeResourceClaim().
						Name(claimName).
						Namespace(namespace).
						RequestWithPrioritizedList(
			st.SubRequestWithSelector("subreq-1", className, fmt.Sprintf(`device.attributes["%s"].%s`, driver, attrName)),
			st.SubRequest("subreq-2", className, 1),
		).
		Obj()
	claimWithMultiplePrioritizedListRequests = st.MakeResourceClaim().
							Name(claimName).
							Namespace(namespace).
							RequestWithPrioritizedList(
			st.SubRequest("subreq-1", className, 2),
			st.SubRequest("subreq-2", className, 1),
		).
		RequestWithPrioritizedList(
			st.SubRequest("subreq-1", className, 2),
			st.SubRequest("subreq-2", className, 1),
		).Obj()
	claim2WithPrioritizedListAndMultipleSubrequests = st.MakeResourceClaim().
							Name(claimName2).
							Namespace(namespace).
							RequestWithPrioritizedList(
			st.SubRequest("subreq-1", className, 4),
			st.SubRequest("subreq-2", className, 3),
			st.SubRequest("subreq-3", className, 2),
			st.SubRequest("subreq-4", className, 1),
		).Obj()

	pendingClaim = st.FromResourceClaim(claim).
			OwnerReference(podName, podUID, podKind).
			Obj()
	pendingClaim2 = st.FromResourceClaim(claim2).
			OwnerReference(podName, podUID, podKind).
			Obj()
	pendingClaimWithPrioritizedList = st.FromResourceClaim(claimWithPrioritzedList).
					OwnerReference(podName, podUID, podKind).
					Obj()
	pendingClaimWithPrioritizedListAndSelector = st.FromResourceClaim(claimWithPrioritizedListAndSelector).
							OwnerReference(podName, podUID, podKind).
							Obj()
	pendingClaim2WithPrioritizedListAndMultipleSubrequests = st.FromResourceClaim(claim2WithPrioritizedListAndMultipleSubrequests).
								OwnerReference(podName, podUID, podKind).
								Obj()
	pendingClaimWithMultiplePrioritizedListRequests = st.FromResourceClaim(claimWithMultiplePrioritizedListRequests).
							OwnerReference(podName, podUID, podKind).
							Obj()
	allocationResult = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance-1",
				Request: "req-1",
			}},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	allocationResult2 = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver2,
				Pool:    nodeName,
				Device:  "instance-2",
				Request: "req-2",
			}},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	extendedResourceAllocationResult = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance-1",
				Request: "container-0-request-0",
			}},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	extendedResourceAllocationResult2 = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance-1",
				Request: "container-0-request-1",
			}},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	implicitExtendedResourceAllocationResult = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-1",
					Request: "container-0-request-0",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-2",
					Request: "container-0-request-1",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-3",
					Request: "container-0-request-1",
				},
			},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	implicitExtendedResourceAllocationResultTwoContainers = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-1",
					Request: "container-0-request-0",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-2",
					Request: "container-1-request-0",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-3",
					Request: "container-1-request-0",
				},
			},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	extendedResourceAllocationResultNode2 = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance-1",
				Request: "container-0-request-0",
			}},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{node2Name}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}

	allocationResultWithPrioritizedList = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance-1",
				Request: "req-1/subreq-1",
			}},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	allocationResultWithPrioritizedListAndSelector = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:  driver,
				Pool:    nodeName,
				Device:  "instance-1",
				Request: "req-1/subreq-1",
			}},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	allocationResultWithPrioritizedListAndMultipleSubrequests = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-1",
					Request: "req-1/subreq-2",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-2",
					Request: "req-1/subreq-2",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-3",
					Request: "req-1/subreq-2",
				},
			},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	allocationResultWithMultiplePrioritizedListRequests = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-1",
					Request: "req-1/subreq-1",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-2",
					Request: "req-1/subreq-1",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-1",
					Request: "req-2/subreq-1",
				},
				{
					Driver:  driver,
					Pool:    nodeName,
					Device:  "instance-2",
					Request: "req-2/subreq-1",
				},
			},
		},
		NodeSelector: func() *v1.NodeSelector {
			return st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		}(),
	}
	inUseClaim = st.FromResourceClaim(pendingClaim).
			Allocation(allocationResult).
			ReservedForPod(podName, types.UID(podUID)).
			Obj()
	inUseClaimWithPrioritizedList = st.FromResourceClaim(pendingClaimWithPrioritizedList).
					Allocation(allocationResultWithPrioritizedList).
					ReservedForPod(podName, types.UID(podUID)).
					Obj()
	inUseClaimWithPrioritizedListAndSelector = st.FromResourceClaim(pendingClaimWithPrioritizedListAndSelector).
							Allocation(allocationResultWithPrioritizedListAndSelector).
							ReservedForPod(podName, types.UID(podUID)).
							Obj()
	inUseClaim2WithPrioritizedListAndMultipleSubrequests = st.FromResourceClaim(pendingClaim2WithPrioritizedListAndMultipleSubrequests).
								Allocation(allocationResultWithPrioritizedListAndMultipleSubrequests).
								ReservedForPod(podName, types.UID(podUID)).
								Obj()
	inUseClaimWithMultiplePrioritizedListRequests = st.FromResourceClaim(pendingClaimWithMultiplePrioritizedListRequests).
							Allocation(allocationResultWithMultiplePrioritizedListRequests).
							ReservedForPod(podName, types.UID(podUID)).
							Obj()
	allocatedClaim = st.FromResourceClaim(pendingClaim).
			Allocation(allocationResult).
			Obj()
	allocatedClaim2 = st.FromResourceClaim(pendingClaim2).
			Allocation(allocationResult2).
			Obj()
	allocatedClaimWithPrioritizedList = st.FromResourceClaim(pendingClaimWithPrioritizedList).
						Allocation(allocationResultWithPrioritizedList).
						Obj()
	allocatedClaimWithPrioritizedListAndSelector = st.FromResourceClaim(pendingClaimWithPrioritizedListAndSelector).
							Allocation(allocationResultWithPrioritizedListAndSelector).
							Obj()
	allocatedClaim2WithPrioritizedListAndMultipleSubrequests = st.FromResourceClaim(pendingClaim2WithPrioritizedListAndMultipleSubrequests).
									Allocation(allocationResultWithPrioritizedListAndMultipleSubrequests).
									Obj()
	allocatedClaimWithMultiplePrioritizedListRequests = st.FromResourceClaim(pendingClaimWithMultiplePrioritizedListRequests).
								Allocation(allocationResultWithMultiplePrioritizedListRequests).
								Obj()
	allocatedClaimWithWrongTopology = st.FromResourceClaim(allocatedClaim).
					Allocation(&resourceapi.AllocationResult{NodeSelector: st.MakeNodeSelector().In("no-such-label", []string{"no-such-value"}, st.NodeSelectorTypeMatchExpressions).Obj()}).
					Obj()
	allocatedClaimWithGoodTopology = st.FromResourceClaim(allocatedClaim).
					Allocation(&resourceapi.AllocationResult{NodeSelector: st.MakeNodeSelector().In("kubernetes.io/hostname", []string{nodeName}, st.NodeSelectorTypeMatchExpressions).Obj()}).
					Obj()
	otherClaim = st.MakeResourceClaim().
			Name("not-my-claim").
			Namespace(namespace).
			Request(className).
			Obj()
	otherAllocatedClaim = st.FromResourceClaim(otherClaim).
				Allocation(allocationResult).
				Obj()
	extendedResourceClaim = st.MakeResourceClaim().
				Name("my-pod-extended-resources-0").
				GenerateName("my-pod-extended-resources-").
				Namespace(namespace).
				Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
				OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-0", className).
		Allocation(extendedResourceAllocationResult).
		Obj()
	extendedResourceClaim2 = st.MakeResourceClaim().
				Name("my-pod-extended-resources-0").
				GenerateName("my-pod-extended-resources-").
				Namespace(namespace).
				Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
				OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-1", className+"2").
		Allocation(extendedResourceAllocationResult2).
		Obj()
	extendedResourceClaimNoName = st.MakeResourceClaim().
					Name(specialClaimInMemName).
					GenerateName("my-pod-extended-resources-").
					Namespace(namespace).
					Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
					OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-0", className).
		Allocation(extendedResourceAllocationResult).
		Obj()
	extendedResourceClaimNoName2 = st.MakeResourceClaim().
					Name(specialClaimInMemName).
					GenerateName("my-pod-extended-resources-").
					Namespace(namespace).
					Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
					OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-1", className+"2").
		Allocation(extendedResourceAllocationResult2).
		Obj()
	implicitExtendedResourceClaim = st.MakeResourceClaim().
					Name("my-pod-extended-resources-0").
					GenerateName("my-pod-extended-resources-").
					Namespace(namespace).
					Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
					OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-0", className).
		RequestWithNameCount("container-0-request-1", className, 2).
		Allocation(implicitExtendedResourceAllocationResult).
		Obj()
	implicitExtendedResourceClaimNoName = st.MakeResourceClaim().
						Name(specialClaimInMemName).
						GenerateName("my-pod-extended-resources-").
						Namespace(namespace).
						Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
						OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-0", className).
		RequestWithNameCount("container-0-request-1", className, 2).
		Allocation(implicitExtendedResourceAllocationResult).
		Obj()
	implicitExtendedResourceClaimTwoContainers = st.MakeResourceClaim().
							Name("my-pod-extended-resources-0").
							GenerateName("my-pod-extended-resources-").
							Namespace(namespace).
							Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
							OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-0", className).
		RequestWithNameCount("container-1-request-0", className, 2).
		Allocation(implicitExtendedResourceAllocationResultTwoContainers).
		Obj()
	implicitExtendedResourceClaimNoNameTwoContainers = st.MakeResourceClaim().
								Name(specialClaimInMemName).
								GenerateName("my-pod-extended-resources-").
								Namespace(namespace).
								Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
								OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-0", className).
		RequestWithNameCount("container-1-request-0", className, 2).
		Allocation(implicitExtendedResourceAllocationResultTwoContainers).
		Obj()
	extendedResourceClaimNode2 = st.MakeResourceClaim().
					Name("my-pod-extended-resources-0").
					GenerateName("my-pod-extended-resources-").
					Namespace(namespace).
					Annotations(map[string]string{"resource.kubernetes.io/extended-resource-claim": "true"}).
					OwnerRef(
			metav1.OwnerReference{
				APIVersion: "v1",
				Kind:       "Pod",
				Name:       podName,
				UID:        types.UID(podUID),
				Controller: ptr.To(true),
			}).
		RequestWithName("container-0-request-0", className).
		Allocation(extendedResourceAllocationResultNode2).
		Obj()

	deviceTaint = resourceapi.DeviceTaint{
		Key:    "taint-key",
		Value:  "taint-value",
		Effect: resourceapi.DeviceTaintEffectNoSchedule,
	}

	// for DRA Device Binding Conditions
	bindingConditions        = []string{"condition"}
	bindingFailureConditions = []string{"failed"}

	fabricSlice = func() *resourceapi.ResourceSlice {
		res := st.MakeResourceSlice(nodeName, driver).Device("instance-1").Obj()
		res.Spec.Devices[0].BindsToNode = ptr.To(true)
		res.Spec.Devices[0].BindingConditions = bindingConditions
		res.Spec.Devices[0].BindingFailureConditions = bindingFailureConditions
		res.Spec.NodeSelector = st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		return res
	}()

	fabricSlice2 = func() *resourceapi.ResourceSlice {
		res := st.MakeResourceSlice(nodeName, driver2).Device("instance-2").Obj()
		res.Spec.Devices[0].BindsToNode = ptr.To(true)
		res.Spec.Devices[0].BindingConditions = bindingConditions
		res.Spec.Devices[0].BindingFailureConditions = bindingFailureConditions
		res.Spec.NodeSelector = st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj()
		return res
	}()

	allocationResultWithBindingConditions = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:                   driver,
				Pool:                     nodeName,
				Device:                   "instance-1",
				Request:                  "req-1",
				BindingConditions:        bindingConditions,
				BindingFailureConditions: bindingFailureConditions,
			}},
		},
		NodeSelector: st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj(),
	}

	allocationResultWithBindingConditions2 = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: []resourceapi.DeviceRequestAllocationResult{{
				Driver:                   driver2,
				Pool:                     nodeName,
				Device:                   "instance-2",
				Request:                  "req-2",
				BindingConditions:        bindingConditions,
				BindingFailureConditions: bindingFailureConditions,
			}},
		},
		NodeSelector: st.MakeNodeSelector().In("metadata.name", []string{nodeName}, st.NodeSelectorTypeMatchFields).Obj(),
	}

	boundClaim = st.FromResourceClaim(allocatedClaim).
			Allocation(allocationResultWithBindingConditions).
			AllocatedDeviceStatuses([]resourceapi.AllocatedDeviceStatus{
			{
				Driver: driver,
				Pool:   nodeName,
				Device: "instance-1",
				Conditions: []metav1.Condition{
					{Type: "condition", Status: metav1.ConditionTrue},
					{Type: "failed", Status: metav1.ConditionFalse},
				},
			},
		}).
		Obj()

	boundClaim2 = st.FromResourceClaim(allocatedClaim2).
			Allocation(allocationResultWithBindingConditions2).
			AllocatedDeviceStatuses([]resourceapi.AllocatedDeviceStatus{
			{
				Driver: driver2,
				Pool:   nodeName,
				Device: "instance-2",
				Conditions: []metav1.Condition{
					{Type: "condition", Status: metav1.ConditionTrue},
					{Type: "failed", Status: metav1.ConditionFalse},
				},
			},
		}).
		Obj()

	failedBindingClaim = st.FromResourceClaim(allocatedClaim).
				Allocation(allocationResultWithBindingConditions).
				AllocatedDeviceStatuses([]resourceapi.AllocatedDeviceStatus{
			{
				Driver: driver,
				Pool:   nodeName,
				Device: "instance-1",
				Conditions: []metav1.Condition{
					{Type: "condition", Status: metav1.ConditionFalse},
					{Type: "failed", Status: metav1.ConditionTrue},
				},
			},
		}).
		Obj()

	failedBindingClaim2 = st.FromResourceClaim(allocatedClaim2).
				Allocation(allocationResultWithBindingConditions2).
				AllocatedDeviceStatuses([]resourceapi.AllocatedDeviceStatus{
			{
				Driver: driver2,
				Pool:   nodeName,
				Device: "instance-2",
				Conditions: []metav1.Condition{
					{Type: "condition", Status: metav1.ConditionFalse},
					{Type: "failed", Status: metav1.ConditionTrue},
				},
			},
		}).
		Obj()
)

func taintDevices(slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
	slice = slice.DeepCopy()
	for i := range slice.Spec.Devices {
		slice.Spec.Devices[i].Taints = append(slice.Spec.Devices[i].Taints, deviceTaint)
	}
	return slice
}

func reserve(claim *resourceapi.ResourceClaim, pod *v1.Pod) *resourceapi.ResourceClaim {
	return st.FromResourceClaim(claim).
		ReservedForPod(pod.Name, types.UID(pod.UID)).
		Obj()
}

func adminAccess(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	claim = claim.DeepCopy()
	for i := range claim.Spec.Devices.Requests {
		claim.Spec.Devices.Requests[i].Exactly.AdminAccess = ptr.To(true)
	}
	if claim.Status.Allocation != nil {
		for i := range claim.Status.Allocation.Devices.Results {
			claim.Status.Allocation.Devices.Results[i].AdminAccess = ptr.To(true)
		}
	}
	return claim
}

func breakCELInClaim(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	claim = claim.DeepCopy()
	for i := range claim.Spec.Devices.Requests {
		for e := range claim.Spec.Devices.Requests[i].Exactly.Selectors {
			claim.Spec.Devices.Requests[i].Exactly.Selectors[e] = brokenSelector
		}
		if len(claim.Spec.Devices.Requests[i].Exactly.Selectors) == 0 {
			claim.Spec.Devices.Requests[i].Exactly.Selectors = []resourceapi.DeviceSelector{brokenSelector}
		}
	}
	return claim
}

func breakCELInClass(class *resourceapi.DeviceClass) *resourceapi.DeviceClass {
	class = class.DeepCopy()
	for i := range class.Spec.Selectors {
		class.Spec.Selectors[i] = brokenSelector
	}
	if len(class.Spec.Selectors) == 0 {
		class.Spec.Selectors = []resourceapi.DeviceSelector{brokenSelector}
	}

	return class
}

func updateDeviceClassName(claim *resourceapi.ResourceClaim, deviceClassName string) *resourceapi.ResourceClaim {
	claim = claim.DeepCopy()
	for i := range claim.Spec.Devices.Requests {
		// If the firstAvailable list is empty we update the device class name
		// on the base request.
		if len(claim.Spec.Devices.Requests[i].FirstAvailable) == 0 {
			claim.Spec.Devices.Requests[i].Exactly.DeviceClassName = deviceClassName
		} else {
			// If subrequests are specified, update the device class name on
			// all of them.
			for j := range claim.Spec.Devices.Requests[i].FirstAvailable {
				claim.Spec.Devices.Requests[i].FirstAvailable[j].DeviceClassName = deviceClassName
			}
		}
	}
	return claim
}

func getDefaultDynamicResourcesArgs() *config.DynamicResourcesArgs {
	v1dra := &kubeschedulerconfigv1.DynamicResourcesArgs{}
	configv1.SetDefaults_DynamicResourcesArgs(v1dra)
	dra := &config.DynamicResourcesArgs{}
	_ = configv1.Convert_v1_DynamicResourcesArgs_To_config_DynamicResourcesArgs(v1dra, dra, nil)
	return dra
}

// result defines the expected outcome of some operation. It covers
// operation's status and the state of the world (= objects).
type result struct {
	status *fwk.Status
	// changes contains a mapping of name to an update function for
	// the corresponding object. These functions apply exactly the expected
	// changes to a copy of the object as it existed before the operation.
	changes change

	// added contains objects created by the operation.
	added []metav1.Object

	// removed contains objects deleted by the operation.
	removed []metav1.Object

	// assumedClaim is the one claim which is expected to be assumed,
	// nil if none.
	assumedClaim *resourceapi.ResourceClaim

	// inFlightClaims is a list of claims which are expected to be tracked as
	// in flight, nil if none.
	inFlightClaims []metav1.Object
}

// change contains functions for modifying objects of a certain type. These
// functions will get called for all objects of that type. If they needs to
// make changes only to a particular instance, then it must check the name.
type change struct {
	claim func(*resourceapi.ResourceClaim) *resourceapi.ResourceClaim
}
type perNodeResult map[string]result

func (p perNodeResult) forNode(nodeName string) result {
	if p == nil {
		return result{}
	}
	return p[nodeName]
}

type perNodeScoreResult map[string]int64

func (p perNodeScoreResult) forNode(nodeName string) int64 {
	if p == nil {
		return 0
	}
	return p[nodeName]
}

type want struct {
	preenqueue           result
	preFilterResult      *fwk.PreFilterResult
	prefilter            result
	filter               perNodeResult
	prescore             result
	scoreResult          perNodeScoreResult
	score                perNodeResult
	normalizeScoreResult fwk.NodeScoreList
	normalizeScore       result
	reserve              result
	unreserve            result
	prebindPreFlight     *fwk.Status
	prebind              result
	postbind             result
	postFilterResult     *fwk.PostFilterResult
	postfilter           result

	// unreserveAfterBindFailure, if set, triggers a call to Unreserve
	// after PreBind, as if the actual Bind had failed.
	unreserveAfterBindFailure *result

	// unreserveBeforePreBind, if set, triggers a call to Unreserve
	// before PreBind, as if the some other PreBind plugin had failed.
	unreserveBeforePreBind *result
}

// prepare contains changes for objects in the API server.
// Those changes are applied before running the steps. This can
// be used to simulate concurrent changes by some other entities
// like a resource driver.
type prepare struct {
	filter     change
	prescore   change
	reserve    change
	unreserve  change
	prebind    change
	postbind   change
	postfilter change
}

type testPluginCase struct {
	// patchTestCase gets called right before the test case is tested.
	// It can be used to update time stamps in those test cases
	// which are sensitive to the current time.
	patchTestCase func(tc *testPluginCase)

	args    *config.DynamicResourcesArgs
	nodes   []*v1.Node // default if unset is workerNode
	pod     *v1.Pod
	claims  []*resourceapi.ResourceClaim
	classes []*resourceapi.DeviceClass

	// objs get stored directly in the fake client, without passing
	// through reactors, in contrast to the types above.
	objs []apiruntime.Object

	prepare prepare
	want    want

	// Invoke Filter with a canceled context.
	cancelFilter bool

	// enableDRAAdminAccess is set to true if the DRAAdminAccess feature gate is enabled.
	enableDRAAdminAccess bool
	// enableDRADeviceBindingConditions is set to true if the DRADeviceBindingConditions feature gate is enabled.
	enableDRADeviceBindingConditions bool
	// EnableDRAResourceClaimDeviceStatus is set to true if the DRAResourceClaimDeviceStatus feature gate is enabled.
	enableDRAResourceClaimDeviceStatus bool
	// Feature gates. False is chosen so that the uncommon case
	// doesn't need to be set.
	disableDRA bool

	enableDRAExtendedResource        bool
	enableDRAPrioritizedList         bool
	enableDRADeviceTaints            bool
	disableDRASchedulerFilterTimeout bool
	skipOnWindows                    string
	failPatch                        bool
	reactors                         []cgotesting.Reactor
	metrics                          func(ktesting.TContext, compbasemetrics.Gatherer)
}

func TestPlugin(t *testing.T) {
	testPlugin(ktesting.Init(t))
}
func testPlugin(tCtx ktesting.TContext) {
	testcases := map[string]testPluginCase{
		"empty": {
			pod: st.MakePod().Name("foo").Namespace("default").Obj(),
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.Skip),
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable),
				},
				prebindPreFlight: fwk.NewStatus(fwk.Skip),
			},
		},
		"empty-with-extended-resources-enabled": {
			enableDRAExtendedResource: true,
			pod:                       st.MakePod().Name("foo").Namespace("default").Obj(),
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.Skip),
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable),
				},
				prebindPreFlight: fwk.NewStatus(fwk.Skip),
			},
		},
		"claim-reference": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{allocatedClaim, otherClaim},
			want: want{
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Status.ReservedFor = inUseClaim.Status.ReservedFor
							}
							return claim
						},
					},
				},
			},
		},
		"claim-template": {
			pod:    podWithClaimTemplateInStatus,
			claims: []*resourceapi.ResourceClaim{allocatedClaim, otherClaim},
			want: want{
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimTemplateInStatus),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Status.ReservedFor = inUseClaim.Status.ReservedFor
							}
							return claim
						},
					},
				},
			},
		},
		"missing-claim": {
			pod:    podWithClaimTemplate, // status not set
			claims: []*resourceapi.ResourceClaim{allocatedClaim, otherClaim},
			want: want{
				preenqueue: result{
					status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `pod "default/my-pod": ResourceClaim not created yet`),
				},
			},
		},
		"deleted-claim": {
			pod: podWithClaimTemplateInStatus,
			claims: func() []*resourceapi.ResourceClaim {
				claim := allocatedClaim.DeepCopy()
				claim.DeletionTimestamp = &metav1.Time{Time: time.Now()}
				return []*resourceapi.ResourceClaim{claim}
			}(),
			want: want{
				preenqueue: result{
					status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `resourceclaim "my-pod-my-resource" is being deleted`),
				},
			},
		},
		"wrong-claim": {
			pod: podWithClaimTemplateInStatus,
			claims: func() []*resourceapi.ResourceClaim {
				claim := allocatedClaim.DeepCopy()
				claim.OwnerReferences[0].UID += "123"
				return []*resourceapi.ResourceClaim{claim}
			}(),
			want: want{
				preenqueue: result{
					status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `ResourceClaim default/my-pod-my-resource was not created for pod default/my-pod (pod is not owner)`),
				},
			},
		},
		"no-resources": {
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `cannot allocate all claims`),
					},
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `still not schedulable`),
				},
			},
		},
		"with-resources": {
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaim.Finalizers
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},
		"with-resources-has-finalizer": {
			// As before. but the finalizer is already set. Could happen if
			// the scheduler got interrupted.
			pod: podWithClaimName,
			claims: func() []*resourceapi.ResourceClaim {
				claim := pendingClaim
				claim.Finalizers = allocatedClaim.Finalizers
				return []*resourceapi.ResourceClaim{claim}
			}(),
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},
		"with-resources-finalizer-gets-removed": {
			// As before. but the finalizer is already set. Then it gets
			// removed before the scheduler reaches PreBind.
			pod: podWithClaimName,
			claims: func() []*resourceapi.ResourceClaim {
				claim := pendingClaim
				claim.Finalizers = allocatedClaim.Finalizers
				return []*resourceapi.ResourceClaim{claim}
			}(),
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			prepare: prepare{
				prebind: change{
					claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
						claim.Finalizers = nil
						return claim
					},
				},
			},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaim.Finalizers
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},
		"with-resources-finalizer-gets-added": {
			// No finalizer initially, then it gets added before
			// the scheduler reaches PreBind. Shouldn't happen?
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			prepare: prepare{
				prebind: change{
					claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
						claim.Finalizers = allocatedClaim.Finalizers
						return claim
					},
				},
			},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},
		"skip-bind": {
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				unreserveBeforePreBind: &result{},
			},
		},
		"exhausted-resources": {
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim, otherAllocatedClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `cannot allocate all claims`),
					},
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `still not schedulable`),
				},
			},
		},

		// The two test cases for device tainting only need to cover
		// whether the feature gate is passed through to the allocator
		// correctly. The actual logic around device taints and allocation
		// is in the allocator.
		"tainted-device-disabled": {
			enableDRADeviceTaints: false,
			pod:                   podWithClaimName,
			claims:                []*resourceapi.ResourceClaim{pendingClaim},
			classes:               []*resourceapi.DeviceClass{deviceClass},
			objs:                  []apiruntime.Object{taintDevices(workerNodeSlice)},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaim.Finalizers
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},
		"tainted-device-enabled": {
			enableDRADeviceTaints: true,
			pod:                   podWithClaimName,
			claims:                []*resourceapi.ResourceClaim{pendingClaim},
			classes:               []*resourceapi.DeviceClass{deviceClass},
			objs:                  []apiruntime.Object{taintDevices(workerNodeSlice)},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `cannot allocate all claims`),
					},
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `still not schedulable`),
				},
			},
		},

		"request-admin-access-with-DRAAdminAccess-featuregate": {
			// When the DRAAdminAccess feature gate is enabled,
			// Because the pending claim asks for admin access,
			// allocation succeeds despite resources being exhausted.
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{adminAccess(pendingClaim), otherAllocatedClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{adminAccess(allocatedClaim)},
				},
				prebind: result{
					assumedClaim: reserve(adminAccess(allocatedClaim), podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaim.Finalizers
								claim.Status = adminAccess(inUseClaim).Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(adminAccess(allocatedClaim), podWithClaimName),
				},
			},
			enableDRAAdminAccess: true,
		},
		"request-admin-access-without-DRAAdminAccess-featuregate": {
			// When the DRAAdminAccess feature gate is disabled,
			// even though the pending claim requests admin access,
			// the scheduler returns an unschedulable status.
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{adminAccess(pendingClaim), otherAllocatedClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `claim default/my-pod-my-resource, request req-1: admin access is requested, but the feature is disabled`),
					},
				},
			},
			enableDRAAdminAccess: false,
		},

		"structured-ignore-allocated-admin-access": {
			// The allocated claim uses admin access, so a second claim may use
			// the same device.
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim, adminAccess(otherAllocatedClaim)},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaim.Finalizers
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},

		"claim-parameters-CEL-runtime-error": {
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{breakCELInClaim(pendingClaim)},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.AsStatus(errors.New(`claim default/my-pod-my-resource: selector #0: CEL runtime error: no such key: ` + string(attrName))),
					},
				},
			},
		},

		"class-parameters-CEL-runtime-error": {
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim},
			classes: []*resourceapi.DeviceClass{breakCELInClass(deviceClass)},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.AsStatus(errors.New(`class my-resource-class: selector #0: CEL runtime error: no such key: ` + string(attrName))),
					},
				},
			},
		},

		// When pod scheduling encounters CEL runtime errors for some nodes, but not all,
		// it should still not schedule the pod because there is something wrong with it.
		// Scheduling it would make it harder to detect that there is a problem.
		//
		// This matches the "keeps pod pending because of CEL runtime errors" E2E test.
		"CEL-runtime-error-for-one-of-two-nodes": {
			nodes:   []*v1.Node{workerNode, workerNode2},
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{breakCELInClaim(pendingClaim)},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice, workerNode2Slice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.AsStatus(errors.New(`claim default/my-pod-my-resource: selector #0: CEL runtime error: no such key: ` + string(attrName))),
					},
				},
			},
		},

		// When two nodes where found, PreScore gets called.
		"CEL-runtime-error-for-one-of-three-nodes": {
			nodes:   []*v1.Node{workerNode, workerNode2, workerNode3},
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{breakCELInClaim(pendingClaim)},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice, workerNode2Slice, workerNode3Slice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `claim default/my-pod-my-resource: selector #0: CEL runtime error: no such key: `+string(attrName)),
					},
				},
				prescore: result{
					// This is the error found during Filter.
					status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `filter node worker: claim default/my-pod-my-resource: selector #0: CEL runtime error: no such key: healthy`),
				},
			},
		},

		"missing-class": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{pendingClaim},
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("request req-1: device class %s does not exist", className)),
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `no new claims to deallocate`),
				},
			},
		},
		"wrong-topology": {
			// PostFilter tries to get the pod scheduleable by
			// deallocating the claim.
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{allocatedClaimWithWrongTopology},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `resourceclaim not available on the node`),
					},
				},
				postfilter: result{
					// Claims get deallocated immediately.
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								Allocation(nil).
								Obj()
						},
					},
					status: fwk.NewStatus(fwk.Unschedulable, `deallocation of ResourceClaim completed`),
				},
			},
		},
		"good-topology": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{allocatedClaimWithGoodTopology},
			want: want{
				prebind: result{
					assumedClaim: reserve(allocatedClaimWithGoodTopology, podWithClaimName),
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								ReservedFor(resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: podName, UID: types.UID(podUID)}).
								Obj()
						},
					},
				},
			},
		},
		"bind-failure": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{allocatedClaimWithGoodTopology},
			want: want{
				prebind: result{
					assumedClaim: reserve(allocatedClaimWithGoodTopology, podWithClaimName),
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								ReservedFor(resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: podName, UID: types.UID(podUID)}).
								Obj()
						},
					},
				},
				unreserveAfterBindFailure: &result{
					assumedClaim: reserve(allocatedClaimWithGoodTopology, podWithClaimName),
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							out := in.DeepCopy()
							out.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{}
							return out
						},
					},
				},
			},
		},
		"reserved-okay": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{inUseClaim},
		},
		"DRA-disabled": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{inUseClaim},
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.Skip),
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `plugin disabled`),
				},
				prebindPreFlight: fwk.NewStatus(fwk.Skip),
			},
			disableDRA: true,
		},
		"claim-with-request-with-unknown-device-class": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{updateDeviceClassName(claim, "does-not-exist")},
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `request req-1: device class does-not-exist does not exist`),
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `no new claims to deallocate`),
				},
			},
		},
		"claim-with-prioritized-list-feature-disabled": {
			enableDRAPrioritizedList: false,
			pod:                      podWithClaimName,
			claims:                   []*resourceapi.ResourceClaim{claimWithPrioritzedList},
			classes:                  []*resourceapi.DeviceClass{deviceClass},
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `claim default/my-pod-my-resource, request req-1: has subrequests, but the DRAPrioritizedList feature is disabled`),
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `no new claims to deallocate`),
				},
			},
		},
		"claim-with-prioritized-list-unknown-device-class": {
			enableDRAPrioritizedList: true,
			pod:                      podWithClaimName,
			claims:                   []*resourceapi.ResourceClaim{updateDeviceClassName(claimWithPrioritzedList, "does-not-exist")},
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `request req-1/subreq-1: device class does-not-exist does not exist`),
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `no new claims to deallocate`),
				},
			},
		},
		"claim-with-prioritized-list": {
			enableDRAPrioritizedList: true,
			pod:                      podWithClaimName,
			claims:                   []*resourceapi.ResourceClaim{pendingClaimWithPrioritizedList},
			classes:                  []*resourceapi.DeviceClass{deviceClass},
			objs:                     []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaimWithPrioritizedList},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaimWithPrioritizedList, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaimWithPrioritizedList.Finalizers
								claim.Status = inUseClaimWithPrioritizedList.Status
							}
							return claim
						},
					},
				},
			},
		},
		"extended-resource-name-with-node-resource": {
			enableDRAExtendedResource:          true,
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			nodes:                              []*v1.Node{workerNodeWithExtendedResource},
			pod:                                podWithExtendedResourceName,
			classes:                            []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			want:                               want{},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				_, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.ErrorContains(tCtx, err, "not found")
			},
		},
		"extended-resource-one-device-plugin-one-dra": {
			enableDRAExtendedResource:          true,
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			nodes:                              []*v1.Node{workerNodeWithExtendedResource},
			pod:                                podWithExtendedResourceName2,
			classes:                            []*resourceapi.DeviceClass{deviceClassWithExtendResourceName, deviceClassWithExtendResourceName2},
			objs:                               []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName2},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{extendedResourceClaimNoName2},
				},
				prebind: result{
					assumedClaim: reserve(extendedResourceClaim2, podWithExtendedResourceName2),
					added:        []metav1.Object{reserve(extendedResourceClaim2, podWithExtendedResourceName2)},
				},
				postbind: result{
					assumedClaim: reserve(extendedResourceClaim2, podWithExtendedResourceName2),
				},
			},
		},
		"extended-resource-name-with-zero-allocatable": {
			enableDRAExtendedResource: true,
			nodes:                     []*v1.Node{workerNodeWithExtendedResourceZeroAllocatable},
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{extendedResourceClaimNoName},
				},
				prebind: result{
					assumedClaim: reserve(extendedResourceClaim, podWithExtendedResourceName),
					added:        []metav1.Object{reserve(extendedResourceClaim, podWithExtendedResourceName)},
				},
				postbind: result{
					assumedClaim: reserve(extendedResourceClaim, podWithExtendedResourceName),
				},
			},
		},
		"non-DRA-extended-resource-name-with-zero-allocatable": {
			enableDRAExtendedResource: true,
			nodes:                     []*v1.Node{workerNodeWithExtendedResourceZeroAllocatable},
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClass},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				prefilter: result{
					status: fwk.NewStatus(fwk.Skip),
				},
				prebindPreFlight: fwk.NewStatus(fwk.Skip),
			},
		},
		"extended-resource-name-no-resource": {
			enableDRAExtendedResource: true,
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `cannot allocate all claims`),
					},
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `still not schedulable`),
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				_, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.ErrorContains(tCtx, err, "not found")
			},
		},
		"extended-resource-name-with-resources": {
			enableDRAExtendedResource: true,
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{extendedResourceClaimNoName},
				},
				prebind: result{
					assumedClaim: reserve(extendedResourceClaim, podWithExtendedResourceName),
					added:        []metav1.Object{reserve(extendedResourceClaim, podWithExtendedResourceName)},
				},
				postbind: result{
					assumedClaim: reserve(extendedResourceClaim, podWithExtendedResourceName),
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				metric, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.NoError(tCtx, err)
				require.Equal(tCtx, 1, int(metric["success"]))
			},
		},
		"implicit-extended-resource-name-with-resources": {
			enableDRAExtendedResource: true,
			pod:                       podWithImplicitExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{largeWorkerNodeSlice, podWithImplicitExtendedResourceName},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{implicitExtendedResourceClaimNoName},
				},
				prebind: result{
					assumedClaim: reserve(implicitExtendedResourceClaim, podWithImplicitExtendedResourceName),
					added:        []metav1.Object{reserve(implicitExtendedResourceClaim, podWithImplicitExtendedResourceName)},
				},
				postbind: result{
					assumedClaim: reserve(implicitExtendedResourceClaim, podWithImplicitExtendedResourceName),
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				metric, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.NoError(tCtx, err)
				require.Equal(tCtx, 1, int(metric["success"]))
			},
		},
		"implicit-extended-resource-name-two-containers-with-resources": {
			enableDRAExtendedResource: true,
			pod:                       podWithImplicitExtendedResourceNameTwoContainers,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{largeWorkerNodeSlice, podWithImplicitExtendedResourceNameTwoContainers},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{implicitExtendedResourceClaimNoNameTwoContainers},
				},
				prebind: result{
					assumedClaim: reserve(implicitExtendedResourceClaimTwoContainers, podWithImplicitExtendedResourceNameTwoContainers),
					added:        []metav1.Object{reserve(implicitExtendedResourceClaimTwoContainers, podWithImplicitExtendedResourceNameTwoContainers)},
				},
				postbind: result{
					assumedClaim: reserve(implicitExtendedResourceClaimTwoContainers, podWithImplicitExtendedResourceNameTwoContainers),
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				metric, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.NoError(tCtx, err)
				require.Equal(tCtx, 1, int(metric["success"]))
			},
		},
		"extended-resource-name-with-resources-fail-patch": {
			enableDRAExtendedResource: true,
			failPatch:                 true,
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{extendedResourceClaimNoName},
				},
				prebind: result{
					assumedClaim: reserve(extendedResourceClaim, podWithExtendedResourceName),
					added:        []metav1.Object{reserve(extendedResourceClaim, podWithExtendedResourceName)},
					status:       fwk.NewStatus(fwk.Unschedulable, `patch error`),
				},
				postbind: result{
					assumedClaim: reserve(extendedResourceClaim, podWithExtendedResourceName),
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				metric, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.NoError(tCtx, err)
				require.Equal(tCtx, 1, int(metric["success"]))
			},
		},
		"extended-resource-name-with-resources-has-claim": {
			enableDRAExtendedResource: true,
			pod:                       podWithExtendedResourceName,
			claims:                    []*resourceapi.ResourceClaim{extendedResourceClaim},
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `cannot schedule extended resource claim`),
					},
				},
				postfilter: result{
					status:  fwk.NewStatus(fwk.Unschedulable, `deletion of ResourceClaim completed`),
					removed: []metav1.Object{extendedResourceClaim},
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				_, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.ErrorContains(tCtx, err, "not found")
			},
		},
		"extended-resource-name-with-resources-delete-claim": {
			enableDRAExtendedResource: true,
			pod:                       podWithExtendedResourceName,
			claims:                    []*resourceapi.ResourceClaim{extendedResourceClaimNode2},
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `cannot schedule extended resource claim`),
					},
				},
				postfilter: result{
					status:  fwk.NewStatus(fwk.Unschedulable, `deletion of ResourceClaim completed`),
					removed: []metav1.Object{extendedResourceClaimNode2},
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				_, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.ErrorContains(tCtx, err, "not found")
			},
		},
		"extended-resource-name-bind-failure": {
			enableDRAExtendedResource: true,
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{extendedResourceClaimNoName},
				},
				prebind: result{
					assumedClaim: reserve(extendedResourceClaim, podWithExtendedResourceName),
					added:        []metav1.Object{reserve(extendedResourceClaim, podWithExtendedResourceName)},
				},
				unreserveAfterBindFailure: &result{
					removed: []metav1.Object{reserve(extendedResourceClaim, podWithExtendedResourceName)},
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				metric, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.NoError(tCtx, err)
				require.Equal(tCtx, 1, int(metric["success"]))
			},
		},
		"extended-resource-name-skip-bind": {
			enableDRAExtendedResource: true,
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{extendedResourceClaimNoName},
				},
				unreserveBeforePreBind: &result{},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				metric, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.NoError(tCtx, err)
				require.Equal(tCtx, 1, int(metric["success"]))
			},
		},
		"extended-resource-name-claim-creation-failure": {
			enableDRAExtendedResource: true,
			pod:                       podWithExtendedResourceName,
			classes:                   []*resourceapi.DeviceClass{deviceClassWithExtendResourceName},
			objs:                      []apiruntime.Object{workerNodeSlice, podWithExtendedResourceName},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{extendedResourceClaimNoName},
				},
				prebind: result{
					status: fwk.NewStatus(fwk.Unschedulable, `claim creation errors`),
				},
				unreserveAfterBindFailure: &result{
					removed: []metav1.Object{reserve(extendedResourceClaim, podWithExtendedResourceName)},
				},
			},
			reactors: []cgotesting.Reactor{
				&cgotesting.SimpleReactor{
					Verb:     "create",
					Resource: "resourceclaims",
					Reaction: func(action cgotesting.Action) (handled bool, ret apiruntime.Object, err error) {
						return true, nil, apierrors.NewBadRequest("claim creation errors")
					},
				},
			},
			metrics: func(tCtx ktesting.TContext, g compbasemetrics.Gatherer) {
				metric, err := testutil.GetCounterValuesFromGatherer(g, "scheduler_resourceclaim_creates_total", map[string]string{}, "status")
				require.NoError(tCtx, err)
				require.Equal(tCtx, 1, int(metric["failure"]))
			},
		},
		"canceled": {
			cancelFilter: true,
			args: &config.DynamicResourcesArgs{
				FilterTimeout: &metav1.Duration{Duration: time.Nanosecond},
			},
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{largeClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{largeWorkerNodeSlice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `asked by caller to stop allocating devices: test canceling Filter`),
					},
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `still not schedulable`),
				},
			},
		},
		"timeout": {
			args: &config.DynamicResourcesArgs{
				FilterTimeout: &metav1.Duration{Duration: time.Nanosecond},
			},
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{largeClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{largeWorkerNodeSlice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `timed out trying to allocate devices`),
					},
				},
				postfilter: result{
					status: fwk.NewStatus(fwk.Unschedulable, `still not schedulable`),
				},
			},
			// Skipping this test case on Windows as a 1ns timeout is not guaranteed to
			// expire immediately on Windows due to its coarser timer granularity -
			// typically in the range of 0.5 to 15.6 ms
			skipOnWindows: "coarse timer granularity",
		},
		"timeout_disabled": {
			// This variant uses the normal test objects to avoid excessive runtime.
			// It could theoretically pass even though the 1 ns limit is enforced
			// although it shouldn't be (which then would be a false positive),
			// but that's unlikely.
			disableDRASchedulerFilterTimeout: true,
			args:                             &config.DynamicResourcesArgs{},
			pod:                              podWithClaimName,
			claims:                           []*resourceapi.ResourceClaim{pendingClaim},
			classes:                          []*resourceapi.DeviceClass{deviceClass},
			objs:                             []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaim.Finalizers
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},
		"timeout_zero": {
			args: &config.DynamicResourcesArgs{
				FilterTimeout: &metav1.Duration{Duration: 0},
			},
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim},
			classes: []*resourceapi.DeviceClass{deviceClass},
			objs:    []apiruntime.Object{workerNodeSlice},
			want: want{
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaim},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaim.Finalizers
								claim.Status = inUseClaim.Status
							}
							return claim
						},
					},
				},
				postbind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimName),
				},
			},
		},
		"bound-claim-with-succeeded-binding-conditions": {
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			pod:                                podWithClaimName,
			claims:                             []*resourceapi.ResourceClaim{boundClaim},
			want: want{
				prebind: result{
					assumedClaim: reserve(boundClaim, podWithClaimName),
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								ReservedFor(resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: podName, UID: types.UID(podUID)}).
								Obj()
						},
					},
					status: nil,
				},
			},
		},
		"bound-claim-with-failed-binding": {
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			pod:                                podWithClaimName,
			claims:                             []*resourceapi.ResourceClaim{failedBindingClaim},
			objs:                               []apiruntime.Object{workerNodeSlice},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `resourceclaim not available on the node`),
					},
				},
				postfilter: result{
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								Allocation(nil).
								AllocatedDeviceStatuses(nil).
								Obj()
						},
					},
					status: fwk.NewStatus(fwk.Unschedulable, `deallocation of ResourceClaim completed`),
				},
			},
		},
		"bound-claim-with-timed-out-binding": {
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			pod:                                podWithClaimName,
			claims: func() []*resourceapi.ResourceClaim {
				claim := allocatedClaim.DeepCopy()
				claim.Status.Allocation = allocationResultWithBindingConditions.DeepCopy()
				// This claim has binding conditions but is timed out.
				claim.Status.Allocation.AllocationTimestamp = ptr.To(metav1.NewTime(time.Now().Add(-10 * time.Minute)))
				claim.Status.Devices = []resourceapi.AllocatedDeviceStatus{
					{
						Driver: driver,
						Pool:   nodeName,
						Device: "instance-1",
					},
				}
				return []*resourceapi.ResourceClaim{claim}
			}(),
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `resourceclaim not available on the node`),
					},
				},
				postfilter: result{
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								Allocation(nil).
								AllocatedDeviceStatuses(nil).
								Obj()
						},
					},
					status: fwk.NewStatus(fwk.Unschedulable, `deallocation of ResourceClaim completed`),
				},
			},
		},
		"prebind-fail-with-binding-timeout": {
			patchTestCase: func(tc *testPluginCase) {
				// The time stamps must be injected into the test case right
				// before it starts to get tested.
				now := time.Now()

				// Set the allocation time so that the claim is not timed out
				// yet when the test starts, but then times out relatively quickly (the 10 seconds)
				// when the test executes PreBind.
				bindingTimeout := tc.args.BindingTimeout.Duration
				timeoutAfter := 10 * time.Second
				allocatedAt := now.Add(-bindingTimeout).Add(timeoutAfter)

				claim := allocatedClaim.DeepCopy()
				claim.Status.Allocation = allocationResultWithBindingConditions.DeepCopy()
				// This claim has binding conditions but is not timed out.
				claim.Status.Allocation.AllocationTimestamp = ptr.To(metav1.NewTime(allocatedAt))
				claim.Status.Devices = []resourceapi.AllocatedDeviceStatus{
					{
						Driver: driver,
						Pool:   nodeName,
						Device: "instance-1",
					},
				}
				tc.claims = []*resourceapi.ResourceClaim{claim}

				claim = claim.DeepCopy()
				claim.Status.Devices = []resourceapi.AllocatedDeviceStatus{
					{
						Driver: driver,
						Pool:   nodeName,
						Device: "instance-1",
					},
				}
				tc.want.prebind.assumedClaim = reserve(claim, podWithClaimName)
			},

			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			args: &config.DynamicResourcesArgs{
				BindingTimeout: &metav1.Duration{Duration: 600 * time.Second},
			},
			pod:    podWithClaimName,
			claims: nil, // Set in patchTestCase.
			want: want{
				prebind: result{
					assumedClaim: nil, // Set in patchTestCase.
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								ReservedFor(resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: podName, UID: types.UID(podUID)}).
								Obj()
						},
					},
					status: fwk.AsStatus(errors.New("claim " + claim.Name + " binding timeout")),
				},
			},
		},
		"bound-claim-with-mixed-binding-conditions": {
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			pod:                                podWithClaimName,
			claims: func() []*resourceapi.ResourceClaim {
				claim := allocatedClaim.DeepCopy()
				claim.Status.Allocation = allocationResultWithBindingConditions.DeepCopy()
				// This claim has binding conditions but is timed out.
				claim.Status.Allocation.AllocationTimestamp = ptr.To(metav1.NewTime(time.Now().Add(-10 * time.Minute)))
				claim.Status.Devices = []resourceapi.AllocatedDeviceStatus{
					{
						Driver: driver,
						Pool:   nodeName,
						Device: "instance-1",
						Conditions: []metav1.Condition{
							{Type: "condition1", Status: metav1.ConditionTrue},
							{Type: "condition2", Status: metav1.ConditionFalse},
						},
					},
				}
				return []*resourceapi.ResourceClaim{claim}
			}(),
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `resourceclaim not available on the node`),
					},
				},
				postfilter: result{
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								Allocation(nil).
								AllocatedDeviceStatuses(nil).
								Obj()
						},
					},
					status: fwk.NewStatus(fwk.Unschedulable, `deallocation of ResourceClaim completed`),
				},
			},
		},
		"bound-claim-without-binding-conditions": {
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			// This test ensures that when DRADeviceBindingConditions is enabled,
			// but the claim has no binding conditions or binding failures,
			// the plugin proceeds as if all conditions are satisfied.
			pod:    podWithClaimTemplateInStatus,
			claims: []*resourceapi.ResourceClaim{allocatedClaim, otherClaim},
			want: want{
				prebind: result{
					assumedClaim: reserve(allocatedClaim, podWithClaimTemplateInStatus),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Status.ReservedFor = inUseClaim.Status.ReservedFor
							}
							return claim
						},
					},
					status: nil,
				},
			},
		},
		"multi-claims-binding-conditions-all-success": {
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			pod:                                podWithTwoClaimNames,
			claims:                             []*resourceapi.ResourceClaim{boundClaim, boundClaim2},
			classes:                            []*resourceapi.DeviceClass{deviceClass},
			nodes:                              []*v1.Node{workerNode},
			objs:                               []apiruntime.Object{fabricSlice, fabricSlice2},
			want: want{
				prebind: result{
					assumedClaim: reserve(boundClaim, podWithTwoClaimNames),
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							return st.FromResourceClaim(in).
								ReservedFor(resourceapi.ResourceClaimConsumerReference{Resource: "pods", Name: podName, UID: types.UID(podUID)}).
								Obj()
						},
					},
					status: nil,
				},
			},
		},
		"multi-claims-binding-conditions-one-fail": {
			enableDRADeviceBindingConditions:   true,
			enableDRAResourceClaimDeviceStatus: true,
			pod:                                podWithTwoClaimNames,
			claims:                             []*resourceapi.ResourceClaim{boundClaim, failedBindingClaim2},
			classes:                            []*resourceapi.DeviceClass{deviceClass},
			nodes:                              []*v1.Node{workerNode},
			objs:                               []apiruntime.Object{fabricSlice, fabricSlice2},
			want: want{
				filter: perNodeResult{
					workerNode.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `resourceclaim not available on the node`),
					},
				},
				postfilter: result{
					changes: change{
						claim: func(in *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if in.Name == claimName2 {
								return st.FromResourceClaim(in).
									Allocation(nil).
									AllocatedDeviceStatuses(nil).
									Obj()
							} else {
								return in
							}
						},
					},
					status: fwk.NewStatus(fwk.Unschedulable, `deallocation of ResourceClaim completed`),
				},
			},
		},
		"single-claim-prioritized-list-scoring": {
			enableDRAPrioritizedList: true,
			pod:                      podWithClaimName,
			claims:                   []*resourceapi.ResourceClaim{pendingClaimWithPrioritizedListAndSelector},
			classes:                  []*resourceapi.DeviceClass{deviceClass},
			nodes:                    []*v1.Node{workerNode, workerNode2},
			objs: []apiruntime.Object{
				st.MakeResourceSlice(nodeName, driver).Device("instance-1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{attrName: {BoolValue: ptr.To(true)}}).Obj(),
				st.MakeResourceSlice(node2Name, driver).Device("instance-1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{attrName: {BoolValue: ptr.To(false)}}).Obj(),
			},
			want: want{
				scoreResult: perNodeScoreResult{
					nodeName:  8,
					node2Name: 7,
				},
				normalizeScoreResult: fwk.NodeScoreList{
					{
						Name:  nodeName,
						Score: 100,
					},
					{
						Name:  node2Name,
						Score: 87,
					},
				},
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaimWithPrioritizedListAndSelector},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaimWithPrioritizedListAndSelector, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = allocatedClaimWithPrioritizedListAndSelector.Finalizers
								claim.Status = inUseClaimWithPrioritizedListAndSelector.Status
							}
							return claim
						},
					},
				},
			},
		},
		"multiple-claims-prioritized-list-scoring": {
			enableDRAPrioritizedList: true,
			pod:                      podWithTwoClaimNames,
			claims:                   []*resourceapi.ResourceClaim{pendingClaimWithPrioritizedList, pendingClaim2WithPrioritizedListAndMultipleSubrequests},
			classes:                  []*resourceapi.DeviceClass{deviceClass},
			nodes:                    []*v1.Node{workerNode, workerNode2, workerNode3},
			objs: []apiruntime.Object{
				st.MakeResourceSlice(nodeName, driver).
					Device("instance-1").
					Device("instance-2").
					Device("instance-3").
					Device("instance-4").Obj(),
				st.MakeResourceSlice(node2Name, driver).
					Device("instance-1").
					Device("instance-2").Obj(),
				st.MakeResourceSlice(node3Name, driver).
					Device("instance-1").Obj(),
			},
			want: want{
				filter: perNodeResult{
					workerNode3.Name: {
						status: fwk.NewStatus(fwk.UnschedulableAndUnresolvable, `cannot allocate all claims`),
					},
				},
				scoreResult: perNodeScoreResult{
					workerNode.Name:  15,
					workerNode2.Name: 13,
				},
				normalizeScoreResult: fwk.NodeScoreList{
					{
						Name:  workerNode.Name,
						Score: 100,
					},
					{
						Name:  workerNode2.Name,
						Score: 86,
					},
				},
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaimWithPrioritizedList, allocatedClaim2WithPrioritizedListAndMultipleSubrequests},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaimWithPrioritizedList, podWithTwoClaimNames),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = inUseClaimWithPrioritizedList.Finalizers
								claim.Status = inUseClaimWithPrioritizedList.Status
							}
							if claim.Name == claimName2 {
								claim = claim.DeepCopy()
								claim.Finalizers = inUseClaim2WithPrioritizedListAndMultipleSubrequests.Finalizers
								claim.Status = inUseClaim2WithPrioritizedListAndMultipleSubrequests.Status
							}
							return claim
						},
					},
				},
			},
		},
		"multiple-requests-prioritized-list-scoring": {
			enableDRAPrioritizedList: true,
			pod:                      podWithClaimName,
			claims:                   []*resourceapi.ResourceClaim{pendingClaimWithMultiplePrioritizedListRequests},
			classes:                  []*resourceapi.DeviceClass{deviceClass},
			nodes:                    []*v1.Node{workerNode, workerNode2, workerNode3},
			objs: []apiruntime.Object{
				st.MakeResourceSlice(nodeName, driver).
					Device("instance-1").
					Device("instance-2").
					Device("instance-3").
					Device("instance-4").Obj(),
				st.MakeResourceSlice(node2Name, driver).
					Device("instance-1").
					Device("instance-2").
					Device("instance-3").Obj(),
				st.MakeResourceSlice(node3Name, driver).
					Device("instance-1").
					Device("instance-2").Obj(),
			},
			want: want{
				scoreResult: perNodeScoreResult{
					workerNode.Name:  16,
					workerNode2.Name: 15,
					workerNode3.Name: 14,
				},
				normalizeScoreResult: fwk.NodeScoreList{
					{
						Name:  workerNode.Name,
						Score: 100,
					},
					{
						Name:  workerNode2.Name,
						Score: 93,
					},
					{
						Name:  workerNode3.Name,
						Score: 87,
					},
				},
				reserve: result{
					inFlightClaims: []metav1.Object{allocatedClaimWithMultiplePrioritizedListRequests},
				},
				prebind: result{
					assumedClaim: reserve(allocatedClaimWithMultiplePrioritizedListRequests, podWithClaimName),
					changes: change{
						claim: func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
							if claim.Name == claimName {
								claim = claim.DeepCopy()
								claim.Finalizers = inUseClaimWithMultiplePrioritizedListRequests.Finalizers
								claim.Status = inUseClaimWithMultiplePrioritizedListRequests.Status
							}
							return claim
						},
					},
				},
			},
		},
	}

	for name, tc := range testcases {
		if len(tc.skipOnWindows) > 0 && goruntime.GOOS == "windows" {
			tCtx.Skipf("Skipping '%s' test case on Windows, reason: %s", name, tc.skipOnWindows)
		}
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			if tc.patchTestCase != nil {
				tc.patchTestCase(&tc)
			}

			nodes := tc.nodes
			if nodes == nil {
				nodes = []*v1.Node{workerNode}
			}
			feats := feature.Features{
				EnableDRAAdminAccess:               tc.enableDRAAdminAccess,
				EnableDRADeviceBindingConditions:   tc.enableDRADeviceBindingConditions,
				EnableDRAResourceClaimDeviceStatus: tc.enableDRAResourceClaimDeviceStatus,
				EnableDRADeviceTaints:              tc.enableDRADeviceTaints,
				EnableDRASchedulerFilterTimeout:    !tc.disableDRASchedulerFilterTimeout,
				EnableDynamicResourceAllocation:    !tc.disableDRA,
				EnableDRAPrioritizedList:           tc.enableDRAPrioritizedList,
				EnableDRAExtendedResource:          tc.enableDRAExtendedResource,
			}

			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, tc.enableDRAExtendedResource)
			testCtx := setup(tCtx, tc.args, nodes, tc.claims, tc.classes, tc.objs, feats, tc.failPatch, tc.reactors)
			initialObjects := testCtx.listAll(tCtx)
			var registry compbasemetrics.KubeRegistry
			if tc.metrics != nil {
				registry = setupMetrics(feats)
			}

			status := testCtx.p.PreEnqueue(tCtx, tc.pod)
			tCtx.Run("PreEnqueue", func(tCtx ktesting.TContext) {
				testCtx.verify(tCtx, tc.want.preenqueue, initialObjects, nil, status)
			})
			if !status.IsSuccess() {
				return
			}

			nodeInfo := framework.NewNodeInfo()
			result, status := testCtx.p.PreFilter(tCtx, testCtx.state, tc.pod, []fwk.NodeInfo{nodeInfo})
			tCtx.Run("prefilter", func(tCtx ktesting.TContext) {
				assert.Equal(tCtx, tc.want.preFilterResult, result)
				testCtx.verify(tCtx, tc.want.prefilter, initialObjects, result, status)
			})
			unschedulable := status.IsRejected()

			var potentialNodes []fwk.NodeInfo

			initialObjects = testCtx.listAll(tCtx)
			testCtx.updateAPIServer(tCtx, initialObjects, tc.prepare.filter)
			if !unschedulable {
				for _, nodeInfo := range testCtx.nodeInfos {
					var status *fwk.Status
					tCtx.Run(fmt.Sprintf("filter/%s", nodeInfo.Node().Name), func(tCtx ktesting.TContext) {
						initialObjects = testCtx.listAll(tCtx)
						ctx := context.Context(tCtx)
						if tc.cancelFilter {
							c, cancel := context.WithCancelCause(ctx)
							ctx = c
							cancel(errors.New("test canceling Filter"))
						}
						status = testCtx.p.Filter(ctx, testCtx.state, tc.pod, nodeInfo)
						nodeName := nodeInfo.Node().Name
						testCtx.verify(tCtx, tc.want.filter.forNode(nodeName), initialObjects, nil, status)
					})
					if status.Code() == fwk.Success {
						potentialNodes = append(potentialNodes, nodeInfo)
					}
					if status.Code() == fwk.Error {
						// An error aborts scheduling.
						return
					}
				}
				if len(potentialNodes) == 0 {
					unschedulable = true
				}
			}

			var scores fwk.NodeScoreList
			if !unschedulable && len(potentialNodes) > 1 {
				initialObjects = testCtx.listAll(tCtx)
				initialObjects = testCtx.updateAPIServer(tCtx, initialObjects, tc.prepare.prescore)

				for _, potentialNode := range potentialNodes {
					initialObjects = testCtx.listAll(tCtx)
					score, status := testCtx.p.Score(tCtx, testCtx.state, tc.pod, potentialNode)
					nodeName := potentialNode.Node().Name
					tCtx.Run(fmt.Sprintf("score/%s", nodeName), func(tCtx ktesting.TContext) {
						assert.Equal(tCtx, tc.want.scoreResult.forNode(nodeName), score)
						testCtx.verify(tCtx, tc.want.score.forNode(nodeName), initialObjects, nil, status)
					})
					scores = append(scores, fwk.NodeScore{Name: nodeName, Score: score})
				}

				initialObjects = testCtx.listAll(tCtx)
				status := testCtx.p.NormalizeScore(tCtx, testCtx.state, tc.pod, scores)
				tCtx.Run("normalizeScore", func(tCtx ktesting.TContext) {
					assert.Equal(tCtx, tc.want.normalizeScoreResult, scores)
					testCtx.verify(tCtx, tc.want.normalizeScore, initialObjects, nil, status)
				})
			}

			var selectedNodeName string
			if !unschedulable && len(potentialNodes) > 0 {
				if len(scores) > 0 {
					nodeScore := scores[0]
					for _, score := range scores {
						if score.Score > nodeScore.Score {
							nodeScore = score
						}
					}
					selectedNodeName = nodeScore.Name
				} else {
					selectedNodeName = potentialNodes[0].Node().Name
				}

				initialObjects = testCtx.listAll(tCtx)
				initialObjects = testCtx.updateAPIServer(tCtx, initialObjects, tc.prepare.reserve)
				status := testCtx.p.Reserve(tCtx, testCtx.state, tc.pod, selectedNodeName)
				tCtx.Run("reserve", func(tCtx ktesting.TContext) {
					testCtx.verify(tCtx, tc.want.reserve, initialObjects, nil, status)
				})
				if status.Code() != fwk.Success {
					unschedulable = true
				}
			}

			if selectedNodeName != "" {
				if unschedulable {
					initialObjects = testCtx.listAll(tCtx)
					initialObjects = testCtx.updateAPIServer(tCtx, initialObjects, tc.prepare.unreserve)
					testCtx.p.Unreserve(tCtx, testCtx.state, tc.pod, selectedNodeName)
					tCtx.Run("unreserve", func(tCtx ktesting.TContext) {
						testCtx.verify(tCtx, tc.want.unreserve, initialObjects, nil, status)
					})
				} else {
					if tc.want.unreserveBeforePreBind != nil {
						initialObjects = testCtx.listAll(tCtx)
						testCtx.p.Unreserve(tCtx, testCtx.state, tc.pod, selectedNodeName)
						tCtx.Run("unreserveBeforePreBind", func(tCtx ktesting.TContext) {
							testCtx.verify(tCtx, *tc.want.unreserveBeforePreBind, initialObjects, nil, status)
						})
						return
					}

					initialObjects = testCtx.listAll(tCtx)
					initialObjects = testCtx.updateAPIServer(tCtx, initialObjects, tc.prepare.prebind)
					preBindPreFlightStatus := testCtx.p.PreBindPreFlight(tCtx, testCtx.state, tc.pod, selectedNodeName)
					tCtx.Run("prebindPreFlight", func(tContext ktesting.TContext) {
						assert.Equal(tCtx, tc.want.prebindPreFlight, preBindPreFlightStatus)
					})
					preBindStatus := testCtx.p.PreBind(tCtx, testCtx.state, tc.pod, selectedNodeName)
					tCtx.Run("prebind", func(tCtx ktesting.TContext) {
						testCtx.verify(tCtx, tc.want.prebind, initialObjects, nil, preBindStatus)
					})
					if tc.want.unreserveAfterBindFailure != nil {
						initialObjects = testCtx.listAll(tCtx)
						testCtx.p.Unreserve(tCtx, testCtx.state, tc.pod, selectedNodeName)
						tCtx.Run("unreserverAfterBindFailure", func(tCtx ktesting.TContext) {
							testCtx.verify(tCtx, *tc.want.unreserveAfterBindFailure, initialObjects, nil, status)
						})
					} else if status.IsSuccess() {
						initialObjects = testCtx.listAll(tCtx)
						initialObjects = testCtx.updateAPIServer(tCtx, initialObjects, tc.prepare.postbind)
					}
				}
			} else if len(potentialNodes) == 0 {
				initialObjects = testCtx.listAll(tCtx)
				initialObjects = testCtx.updateAPIServer(tCtx, initialObjects, tc.prepare.postfilter)
				result, status := testCtx.p.PostFilter(tCtx, testCtx.state, tc.pod, nil /* filteredNodeStatusMap not used by plugin */)
				tCtx.Run("postfilter", func(tCtx ktesting.TContext) {
					assert.Equal(tCtx, tc.want.postFilterResult, result)
					testCtx.verify(tCtx, tc.want.postfilter, initialObjects, nil, status)
				})
			}
			if tc.metrics != nil {
				tc.metrics(tCtx, registry)
			}
		})
	}
}

func setupMetrics(features feature.Features) compbasemetrics.KubeRegistry {
	// Since feature gate is not set globally, we can't use metrics.Register().
	// We use a new registry instead of using global registry.
	testRegistry := compbasemetrics.NewKubeRegistry()
	if features.EnableDRAExtendedResource {
		testRegistry.MustRegister(metrics.ResourceClaimCreatesTotal)
		metrics.ResourceClaimCreatesTotal.Reset()
	}
	return testRegistry
}

type testContext struct {
	client          *fake.Clientset
	informerFactory informers.SharedInformerFactory
	draManager      *DefaultDRAManager
	p               *DynamicResources
	nodeInfos       []fwk.NodeInfo
	state           fwk.CycleState
}

func (tc *testContext) verify(tCtx ktesting.TContext, expected result, initialObjects []metav1.Object, result interface{}, status *fwk.Status) {
	tCtx.Helper()
	if expected.status == nil {
		assert.Nil(tCtx, status)
	} else if actualErr := status.AsError(); actualErr != nil {
		// Compare only the error strings.
		assert.ErrorContains(tCtx, actualErr, expected.status.AsError().Error())
	} else {
		assert.Equal(tCtx, expected.status, status)
	}
	objects := tc.listAll(tCtx)
	wantObjects := update(initialObjects, expected.changes)
	wantObjects = append(wantObjects, expected.added...)
	for _, remove := range expected.removed {
		for i, obj := range wantObjects {
			// This is a bit relaxed (no GVR comparison, no UID
			// comparison) to simplify writing the test cases.
			if obj.GetName() == remove.GetName() && obj.GetNamespace() == remove.GetNamespace() {
				wantObjects = append(wantObjects[0:i], wantObjects[i+1:]...)
				break
			}
		}
	}
	sortObjects(wantObjects)
	if wantObjects == nil {
		wantObjects = []metav1.Object{}
	}
	if objects == nil {
		objects = []metav1.Object{}
	}

	// Sometimes assert strips the diff too much, let's do it ourselves...
	ignoreFieldsInResourceClaims := []cmp.Option{
		cmpopts.IgnoreFields(metav1.ObjectMeta{}, "UID", "ResourceVersion"),
		cmpopts.IgnoreFields(resourceapi.AllocationResult{}, "AllocationTimestamp"),
		// It does not matter which specific device is allocated for the testing purpose.
		cmpopts.IgnoreFields(resourceapi.DeviceRequestAllocationResult{}, "Device"),
	}
	if diff := cmp.Diff(wantObjects, objects, ignoreFieldsInResourceClaims...); diff != "" {
		tCtx.Errorf("Stored objects are different (- expected, + actual):\n%s", diff)
	}

	var expectAssumedClaims []metav1.Object
	if expected.assumedClaim != nil {
		expectAssumedClaims = append(expectAssumedClaims, expected.assumedClaim)
	}
	// actualAssumedClaims are claims in assumed cache with different latest and api object
	// sameAssumedClaims are claims in assumed cache with same latest and api object
	actualAssumedClaims, sameAssumedClaims := tc.listAssumedClaims()

	// error when expecting no claims in assumed cache with different latest and api object
	if len(expectAssumedClaims) == 0 && len(actualAssumedClaims) != 0 {
		// In case we delete the claim API object,  wait for assumed cache to sync with informer,
		// then assumed cache should be empty.
		err := wait.PollUntilContextTimeout(tCtx, 200*time.Millisecond, time.Minute, true,
			func(ctx context.Context) (bool, error) {
				actualAssumedClaims, sameAssumedClaims = tc.listAssumedClaims()
				return len(actualAssumedClaims) == 0, nil
			})
		if err != nil || len(actualAssumedClaims) != 0 {
			tCtx.Errorf("Assumed claims are different, err=%v, expected: nil, actual:\n%v", err, actualAssumedClaims)
		}
	}
	if len(expectAssumedClaims) > 0 {
		// it is not an error as long as the expected claim is present in the assumed cache, no
		// matter its latest and api object are different or not.
		for _, expected := range expectAssumedClaims {
			seen := false
			for _, actual := range actualAssumedClaims {
				if cmp.Equal(expected, actual, ignoreFieldsInResourceClaims...) {
					seen = true
				}
			}
			for _, same := range sameAssumedClaims {
				if cmp.Equal(expected, same, ignoreFieldsInResourceClaims...) {
					seen = true
				}
			}
			if !seen {
				tCtx.Errorf("Assumed claims are different, expected: %v not found", expected)
			}
		}
	}

	actualInFlightClaims := tc.listInFlightClaims()
	if diff := cmp.Diff(expected.inFlightClaims, actualInFlightClaims, ignoreFieldsInResourceClaims...); diff != "" {
		tCtx.Errorf("In-flight claims are different (- expected, + actual):\n%s", diff)
	}
}

func (tc *testContext) listAll(tCtx ktesting.TContext) (objects []metav1.Object) {
	tCtx.Helper()
	claims, err := tc.client.ResourceV1().ResourceClaims("").List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list claims")
	for _, claim := range claims.Items {
		claim := claim
		objects = append(objects, &claim)
	}
	sortObjects(objects)
	return
}

func (tc *testContext) listAssumedClaims() ([]metav1.Object, []metav1.Object) {
	var assumedClaims []metav1.Object
	var sameClaims []metav1.Object
	for _, obj := range tc.draManager.resourceClaimTracker.cache.List(nil) {
		claim := obj.(*resourceapi.ResourceClaim)
		obj, _ := tc.draManager.resourceClaimTracker.cache.Get(claim.Namespace + "/" + claim.Name)
		apiObj, _ := tc.draManager.resourceClaimTracker.cache.GetAPIObj(claim.Namespace + "/" + claim.Name)
		if obj != apiObj {
			assumedClaims = append(assumedClaims, claim)
		} else {
			sameClaims = append(sameClaims, claim)
		}
	}
	sortObjects(assumedClaims)
	sortObjects(sameClaims)
	return assumedClaims, sameClaims
}

func (tc *testContext) listInFlightClaims() []metav1.Object {
	var inFlightClaims []metav1.Object
	tc.draManager.resourceClaimTracker.inFlightAllocations.Range(func(key, value any) bool {
		inFlightClaims = append(inFlightClaims, value.(*resourceapi.ResourceClaim))
		return true
	})
	sortObjects(inFlightClaims)
	return inFlightClaims
}

// updateAPIServer modifies objects and stores any changed object in the API server.
func (tc *testContext) updateAPIServer(tCtx ktesting.TContext, objects []metav1.Object, updates change) []metav1.Object {
	modified := update(objects, updates)
	for i := range modified {
		obj := modified[i]
		if diff := cmp.Diff(objects[i], obj); diff != "" {
			tCtx.Logf("Updating %T %q, diff (-old, +new):\n%s", obj, obj.GetName(), diff)
			switch obj := obj.(type) {
			case *resourceapi.ResourceClaim:
				obj, err := tc.client.ResourceV1().ResourceClaims(obj.Namespace).Update(tCtx, obj, metav1.UpdateOptions{})
				tCtx.ExpectNoError(err, "prepare update")
				modified[i] = obj
			default:
				tCtx.Fatalf("unsupported object type %T", obj)
			}
		}
	}
	return modified
}

func sortObjects(objects []metav1.Object) {
	sort.Slice(objects, func(i, j int) bool {
		if objects[i].GetNamespace() < objects[j].GetNamespace() {
			return true
		}
		return objects[i].GetName() < objects[j].GetName()
	})
}

// update walks through all existing objects, finds the corresponding update
// function based on name and kind, and replaces those objects that have an
// update function. The rest is left unchanged.
func update(objects []metav1.Object, updates change) []metav1.Object {
	var updated []metav1.Object

	for _, obj := range objects {
		switch in := obj.(type) {
		case *resourceapi.ResourceClaim:
			if updates.claim != nil {
				obj = updates.claim(in)
			}
		}
		updated = append(updated, obj)
	}

	return updated
}

func setup(tCtx ktesting.TContext, args *config.DynamicResourcesArgs, nodes []*v1.Node, claims []*resourceapi.ResourceClaim, classes []*resourceapi.DeviceClass, objs []apiruntime.Object, features feature.Features, failPatch bool, apiReactors []cgotesting.Reactor) (result *testContext) {
	tCtx.Helper()

	tc := &testContext{}

	tc.client = fake.NewSimpleClientset(objs...)
	reactor := createReactor(tc.client.Tracker(), failPatch)
	tc.client.PrependReactor("*", "*", reactor)
	// Prepends reactors to the client.
	tc.client.ReactionChain = append(apiReactors, tc.client.ReactionChain...)

	tc.informerFactory = informers.NewSharedInformerFactory(tc.client, 0)
	resourceSliceTrackerOpts := resourceslicetracker.Options{
		EnableDeviceTaintRules: true,
		SliceInformer:          tc.informerFactory.Resource().V1().ResourceSlices(),
		TaintInformer:          tc.informerFactory.Resource().V1alpha3().DeviceTaintRules(),
		ClassInformer:          tc.informerFactory.Resource().V1().DeviceClasses(),
		KubeClient:             tc.client,
	}
	resourceSliceTracker, err := resourceslicetracker.StartTracker(tCtx, resourceSliceTrackerOpts)
	require.NoError(tCtx, err, "couldn't start resource slice tracker")

	claimsCache := assumecache.NewAssumeCache(tCtx.Logger(), tc.informerFactory.Resource().V1().ResourceClaims().Informer(), "resource claim", "", nil)
	// NewAssumeCache calls the informer's AddEventHandler method to register
	// a handler in order to stay in sync with the informer's store, but
	// NewAssumeCache does not return the ResourceEventHandlerRegistration.
	// We call AddEventHandler of the assume cache, passing it a noop
	// ResourceEventHandler in order to get access to the
	// ResourceEventHandlerRegistration returned by the informer.
	//
	// This is not the registered handler that is used by the DRA
	// manager, but it is close enough because the assume cache
	// uses a single boolean for "is synced" for all handlers.
	registeredHandler := claimsCache.AddEventHandler(cache.ResourceEventHandlerFuncs{})

	tc.draManager = NewDRAManager(tCtx, claimsCache, resourceSliceTracker, tc.informerFactory)
	if features.EnableDRAExtendedResource {
		cache := tc.draManager.DeviceClassResolver().(*extendedresourcecache.ExtendedResourceCache)
		if _, err := tc.informerFactory.Resource().V1().DeviceClasses().Informer().AddEventHandler(cache); err != nil {
			tCtx.Logger().Error(err, "failed to add device class informer event handler")
		}
	}

	opts := []runtime.Option{
		runtime.WithClientSet(tc.client),
		runtime.WithInformerFactory(tc.informerFactory),
		runtime.WithEventRecorder(&events.FakeRecorder{}),
		runtime.WithSharedDRAManager(tc.draManager),
	}
	fh, err := runtime.NewFramework(tCtx, nil, nil, opts...)
	tCtx.ExpectNoError(err, "create scheduler framework")
	tCtx.Cleanup(func() {
		tCtx.Cancel("test has completed")
		runtime.WaitForShutdown(fh)
	})

	if args == nil {
		args = getDefaultDynamicResourcesArgs()
	}
	pl, err := New(tCtx, args, fh, features)
	tCtx.ExpectNoError(err, "create plugin")
	tc.p = pl.(*DynamicResources)

	// The tests use the API to create the objects because then reactors
	// get triggered.
	for _, claim := range claims {
		_, err := tc.client.ResourceV1().ResourceClaims(claim.Namespace).Create(tCtx, claim, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create resource claim")
	}
	for _, class := range classes {
		_, err := tc.client.ResourceV1().DeviceClasses().Create(tCtx, class, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create resource class")
	}

	tc.informerFactory.Start(tCtx.Done())
	tCtx.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCtx.Cancel("test is done")
		// Now we can wait for all goroutines to stop.
		tc.informerFactory.Shutdown()
	})

	tc.informerFactory.WaitForCacheSync(tCtx.Done())
	// The above does not tell us if the registered handler (from NewAssumeCache)
	// is synced, we need to wait until HasSynced of the handler returns
	// true, this ensures that the assume cache is in sync with the informer's
	// store which has been informed by at least one full LIST of the underlying storage.
	cache.WaitForNamedCacheSyncWithContext(tCtx, registeredHandler.HasSynced, resourceSliceTracker.HasSynced)

	for _, node := range nodes {
		nodeInfo := framework.NewNodeInfo()
		nodeInfo.SetNode(node)
		tc.nodeInfos = append(tc.nodeInfos, nodeInfo)
	}
	tc.state = framework.NewCycleState()

	return tc
}

// createReactor implements the logic required for the UID and ResourceVersion
// fields to work when using the fake client. Add it with client.PrependReactor
// to your fake client. ResourceVersion handling is required for conflict
// detection during updates, which is covered by some scenarios.
func createReactor(tracker cgotesting.ObjectTracker, failPatch bool) func(action cgotesting.Action) (handled bool, ret apiruntime.Object, err error) {
	var nameCounter int
	var uidCounter int
	var resourceVersionCounter int
	var mutex sync.Mutex

	return func(action cgotesting.Action) (handled bool, ret apiruntime.Object, err error) {
		if failPatch {
			if _, ok := action.(cgotesting.PatchAction); ok {
				return true, nil, errors.New("patch error")
			}
		}

		createAction, ok := action.(cgotesting.CreateAction)
		if !ok {
			return false, nil, nil
		}
		obj, ok := createAction.GetObject().(metav1.Object)
		if !ok {
			return false, nil, nil
		}

		mutex.Lock()
		defer mutex.Unlock()
		switch action.GetVerb() {
		case "create":
			if obj.GetUID() != "" {
				return true, nil, errors.New("UID must not be set on create")
			}
			if obj.GetResourceVersion() != "" {
				return true, nil, errors.New("ResourceVersion must not be set on create")
			}
			obj.SetUID(types.UID(fmt.Sprintf("UID-%d", uidCounter)))
			uidCounter++
			obj.SetResourceVersion(fmt.Sprintf("%d", resourceVersionCounter))
			resourceVersionCounter++
			if obj.GetName() == "" {
				obj.SetName(obj.GetGenerateName() + fmt.Sprintf("%d", nameCounter))
				nameCounter++
			}
		case "update":
			uid := obj.GetUID()
			resourceVersion := obj.GetResourceVersion()
			if uid == "" {
				return true, nil, errors.New("UID must be set on update")
			}
			if resourceVersion == "" {
				return true, nil, errors.New("ResourceVersion must be set on update")
			}

			oldObj, err := tracker.Get(action.GetResource(), obj.GetNamespace(), obj.GetName())
			if err != nil {
				return true, nil, err
			}
			oldObjMeta, ok := oldObj.(metav1.Object)
			if !ok {
				return true, nil, errors.New("internal error: unexpected old object type")
			}
			if oldObjMeta.GetResourceVersion() != resourceVersion {
				return true, nil, errors.New("ResourceVersion must match the object that gets updated")
			}

			obj.SetResourceVersion(fmt.Sprintf("%d", resourceVersionCounter))
			resourceVersionCounter++
		}
		return false, nil, nil
	}
}

func TestIsSchedulableAfterClaimChange(t *testing.T) {
	testIsSchedulableAfterClaimChange(ktesting.Init(t))
}
func testIsSchedulableAfterClaimChange(tCtx ktesting.TContext) {
	testcases := map[string]struct {
		pod            *v1.Pod
		claims         []*resourceapi.ResourceClaim
		oldObj, newObj interface{}
		wantHint       fwk.QueueingHint
		wantErr        bool
	}{
		"skip-deletes": {
			pod:      podWithClaimTemplate,
			oldObj:   allocatedClaim,
			newObj:   nil,
			wantHint: fwk.QueueSkip,
		},
		"backoff-wrong-new-object": {
			pod:     podWithClaimTemplate,
			newObj:  "not-a-claim",
			wantErr: true,
		},
		"skip-wrong-claim": {
			pod: podWithClaimTemplate,
			newObj: func() *resourceapi.ResourceClaim {
				claim := allocatedClaim.DeepCopy()
				claim.OwnerReferences[0].UID += "123"
				return claim
			}(),
			wantHint: fwk.QueueSkip,
		},
		"skip-unrelated-claim": {
			pod:    podWithClaimTemplate,
			claims: []*resourceapi.ResourceClaim{allocatedClaim},
			newObj: func() *resourceapi.ResourceClaim {
				claim := allocatedClaim.DeepCopy()
				claim.Name += "-foo"
				claim.UID += "123"
				return claim
			}(),
			wantHint: fwk.QueueSkip,
		},
		"queue-on-add": {
			pod:      podWithClaimName,
			newObj:   pendingClaim,
			wantHint: fwk.Queue,
		},
		"backoff-wrong-old-object": {
			pod:     podWithClaimName,
			claims:  []*resourceapi.ResourceClaim{pendingClaim},
			oldObj:  "not-a-claim",
			newObj:  pendingClaim,
			wantErr: true,
		},
		"skip-adding-finalizer": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{pendingClaim},
			oldObj: pendingClaim,
			newObj: func() *resourceapi.ResourceClaim {
				claim := pendingClaim.DeepCopy()
				claim.Finalizers = append(claim.Finalizers, "foo")
				return claim
			}(),
			wantHint: fwk.QueueSkip,
		},
		"queue-on-status-change": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{pendingClaim},
			oldObj: pendingClaim,
			newObj: func() *resourceapi.ResourceClaim {
				claim := pendingClaim.DeepCopy()
				claim.Status.Allocation = &resourceapi.AllocationResult{}
				return claim
			}(),
			wantHint: fwk.Queue,
		},
		"claim-deallocate": {
			pod:    podWithClaimName,
			claims: []*resourceapi.ResourceClaim{pendingClaim, otherAllocatedClaim},
			oldObj: otherAllocatedClaim,
			newObj: func() *resourceapi.ResourceClaim {
				claim := otherAllocatedClaim.DeepCopy()
				claim.Status.Allocation = nil
				return claim
			}(),
			wantHint: fwk.Queue,
		},
	}

	for name, tc := range testcases {
		tCtx.SyncTest(name, func(tCtx ktesting.TContext) {
			features := feature.Features{
				EnableDRASchedulerFilterTimeout: true,
				EnableDynamicResourceAllocation: true,
			}
			testCtx := setup(tCtx, nil, nil, tc.claims, nil, nil, features, false, nil)
			oldObj := tc.oldObj
			newObj := tc.newObj
			if claim, ok := tc.newObj.(*resourceapi.ResourceClaim); ok {
				// Add or update through the client and wait until the event is processed.
				claimKey := claim.Namespace + "/" + claim.Name
				if tc.oldObj == nil {
					// Some test claims already have it. Clear for create.
					createClaim := claim.DeepCopy()
					createClaim.UID = ""
					storedClaim, err := testCtx.client.ResourceV1().ResourceClaims(createClaim.Namespace).Create(tCtx, createClaim, metav1.CreateOptions{})
					if err != nil {
						tCtx.Fatalf("create claim: expected no error, got: %v", err)
					}
					claim = storedClaim
				} else {
					cachedClaim, err := testCtx.draManager.resourceClaimTracker.cache.Get(claimKey)
					if err != nil {
						tCtx.Fatalf("retrieve old claim: expected no error, got: %v", err)
					}
					updateClaim := claim.DeepCopy()
					// The test claim doesn't have those (generated dynamically), so copy them.
					updateClaim.UID = cachedClaim.(*resourceapi.ResourceClaim).UID
					updateClaim.ResourceVersion = cachedClaim.(*resourceapi.ResourceClaim).ResourceVersion

					storedClaim, err := testCtx.client.ResourceV1().ResourceClaims(updateClaim.Namespace).Update(tCtx, updateClaim, metav1.UpdateOptions{})
					if err != nil {
						tCtx.Fatalf("update claim: expected no error, got: %v", err)
					}
					claim = storedClaim
				}

				// Eventually the assume cache will have it, too.
				tCtx.Wait()
				cachedClaim, err := testCtx.draManager.resourceClaimTracker.cache.Get(claimKey)
				tCtx.ExpectNoError(err, "retrieve claim")
				if cachedClaim.(*resourceapi.ResourceClaim).ResourceVersion != claim.ResourceVersion {
					tCtx.Errorf("cached claim not updated yet")
				}

				// This has the actual UID and ResourceVersion,
				// which is relevant for
				// isSchedulableAfterClaimChange.
				newObj = claim
			}
			gotHint, err := testCtx.p.isSchedulableAfterClaimChange(tCtx.Logger(), tc.pod, oldObj, newObj)
			if tc.wantErr {
				if err == nil {
					tCtx.Fatal("want an error, got none")
				}
				return
			}

			if err != nil {
				tCtx.Fatalf("want no error, got: %v", err)
			}
			if tc.wantHint != gotHint {
				tCtx.Fatalf("want %#v, got %#v", tc.wantHint.String(), gotHint.String())
			}
		})
	}
}

func TestIsSchedulableAfterPodChange(t *testing.T) {
	testIsSchedulableAfterPodChange(ktesting.Init(t))
}
func testIsSchedulableAfterPodChange(tCtx ktesting.TContext) {
	testcases := map[string]struct {
		objs     []apiruntime.Object
		pod      *v1.Pod
		claims   []*resourceapi.ResourceClaim
		obj      interface{}
		wantHint fwk.QueueingHint
		wantErr  bool
	}{
		"backoff-wrong-new-object": {
			pod:     podWithClaimTemplate,
			obj:     "not-a-claim",
			wantErr: true,
		},
		"complete": {
			objs:     []apiruntime.Object{pendingClaim},
			pod:      podWithClaimTemplate,
			obj:      podWithClaimTemplateInStatus,
			wantHint: fwk.Queue,
		},
		"wrong-pod": {
			objs: []apiruntime.Object{pendingClaim},
			pod: func() *v1.Pod {
				pod := podWithClaimTemplate.DeepCopy()
				pod.Name += "2"
				pod.UID += "2" // This is the relevant difference.
				return pod
			}(),
			obj:      podWithClaimTemplateInStatus,
			wantHint: fwk.QueueSkip,
		},
		"missing-claim": {
			objs:     nil,
			pod:      podWithClaimTemplate,
			obj:      podWithClaimTemplateInStatus,
			wantHint: fwk.QueueSkip,
		},
		"incomplete": {
			objs: []apiruntime.Object{pendingClaim},
			pod:  podWithTwoClaimTemplates,
			obj: func() *v1.Pod {
				pod := podWithTwoClaimTemplates.DeepCopy()
				// Only one of two claims created.
				pod.Status.ResourceClaimStatuses = []v1.PodResourceClaimStatus{{
					Name:              pod.Spec.ResourceClaims[0].Name,
					ResourceClaimName: &claimName,
				}}
				return pod
			}(),
			wantHint: fwk.QueueSkip,
		},
	}

	for name, tc := range testcases {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			features := feature.Features{
				EnableDRASchedulerFilterTimeout: true,
				EnableDynamicResourceAllocation: true,
			}
			testCtx := setup(tCtx, nil, nil, tc.claims, nil, tc.objs, features, false, nil)
			gotHint, err := testCtx.p.isSchedulableAfterPodChange(tCtx.Logger(), tc.pod, nil, tc.obj)
			if tc.wantErr {
				if err == nil {
					tCtx.Fatal("want an error, got none")
				}
				return
			}

			if err != nil {
				tCtx.Fatalf("want no error, got: %v", err)
			}
			if tc.wantHint != gotHint {
				tCtx.Fatalf("want %#v, got %#v", tc.wantHint.String(), gotHint.String())
			}
		})
	}
}

// mockDeviceClassResolver is a simple mock implementation of fwk.DeviceClassResolver for testing
type mockDeviceClassResolver struct {
	mapping map[v1.ResourceName]*resourceapi.DeviceClass
}

func (m *mockDeviceClassResolver) GetDeviceClass(resourceName v1.ResourceName) *resourceapi.DeviceClass {
	return m.mapping[resourceName]
}

// TestAllocatorSelection covers the selection of a structured allocation implementation
// based on actual Kubernetes feature gates. This test lives here instead of
// k8s.io/dynamic-resource-allocation/structured because that code has no access
// to feature gate definitions.
func TestAllocatorSelection(t *testing.T) {
	for name, tc := range map[string]struct {
		features             string
		expectImplementation string
	}{
		// The most conservative implementation: only used when explicitly asking
		// for the most stable Kubernetes (no alpha or beta features).
		"only-GA": {
			features:             "AllAlpha=false,AllBeta=false",
			expectImplementation: "stable",
		},

		// By default, some beta features are on and the incubating implementation
		// is used.
		"default": {
			features:             "",
			expectImplementation: "incubating",
		},

		// Alpha features need the experimental implementation.
		"alpha": {
			features:             "AllAlpha=true,AllBeta=true",
			expectImplementation: "experimental",
		},
	} {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			featureGate := utilfeature.DefaultFeatureGate.DeepCopy()
			tCtx.ExpectNoError(featureGate.Set(tc.features), "set features")
			fts := feature.NewSchedulerFeaturesFromGates(featureGate)
			features := AllocatorFeatures(fts)

			// Slightly hacky: most arguments are not valid and the constructor
			// is expected to not use them yet.
			allocator, err := structured.NewAllocator(tCtx, features, structured.AllocatedState{}, nil, nil, nil)
			tCtx.ExpectNoError(err, "create allocator")
			allocatorType := fmt.Sprintf("%T", allocator)
			if !strings.Contains(allocatorType, tc.expectImplementation) {
				tCtx.Fatalf("Expected allocator implementation %q, got %s", tc.expectImplementation, allocatorType)
			}
		})
	}
}

func Test_computesScore(t *testing.T) {
	testcases := map[string]struct {
		claims        []*resourceapi.ResourceClaim
		allocations   nodeAllocation
		expectedScore int64
		expectErr     bool
	}{
		"more-claims-than-allocations": {
			claims: []*resourceapi.ResourceClaim{
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-1",
						st.SubRequest("subreq-1", className, 1),
					).
					Obj(),
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-2",
						st.SubRequest("subreq-1", className, 1),
					).
					Obj(),
			},
			allocations: nodeAllocation{},
			expectErr:   true,
		},
		"single-request-only-subrequest-allocated": {
			claims: []*resourceapi.ResourceClaim{
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-1",
						st.SubRequest("subreq-1", className, 1),
					).
					Obj(),
			},
			allocations: nodeAllocation{
				allocationResults: []resourceapi.AllocationResult{
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1/subreq-1",
								},
							},
						},
					},
				},
			},
			expectedScore: 8,
		},
		"single-request-last-subrequest-allocated": {
			claims: []*resourceapi.ResourceClaim{
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-1",
						st.SubRequest("subreq-1", className, 1),
						st.SubRequest("subreq-2", className, 1),
						st.SubRequest("subreq-3", className, 1),
						st.SubRequest("subreq-4", className, 1),
						st.SubRequest("subreq-5", className, 1),
						st.SubRequest("subreq-6", className, 1),
						st.SubRequest("subreq-7", className, 1),
						st.SubRequest("subreq-8", className, 1),
					).
					Obj(),
			},
			allocations: nodeAllocation{
				allocationResults: []resourceapi.AllocationResult{
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1/subreq-8",
								},
							},
						},
					},
				},
			},
			expectedScore: 1,
		},
		"multiple-requests-with-middle-subrequests-allocated": {
			claims: []*resourceapi.ResourceClaim{
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-1",
						st.SubRequest("subreq-1", className, 1),
						st.SubRequest("subreq-2", className, 1),
						st.SubRequest("subreq-3", className, 1),
						st.SubRequest("subreq-4", className, 1),
					).
					NamedRequestWithPrioritizedList("req-2",
						st.SubRequest("subreq-1", className, 1),
						st.SubRequest("subreq-2", className, 1),
						st.SubRequest("subreq-3", className, 1),
						st.SubRequest("subreq-4", className, 1),
						st.SubRequest("subreq-5", className, 1),
					).
					Obj(),
			},
			allocations: nodeAllocation{
				allocationResults: []resourceapi.AllocationResult{
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1/subreq-4",
								},
								{
									Request: "req-2/subreq-5",
								},
							},
						},
					},
				},
			},
			expectedScore: 9,
		},
		"multiple-requests-with-top-subrequests-allocated": {
			claims: []*resourceapi.ResourceClaim{
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-1",
						st.SubRequest("subreq-1", className, 1),
						st.SubRequest("subreq-2", className, 1),
						st.SubRequest("subreq-3", className, 1),
						st.SubRequest("subreq-4", className, 1),
						st.SubRequest("subreq-5", className, 1),
						st.SubRequest("subreq-6", className, 1),
						st.SubRequest("subreq-7", className, 1),
						st.SubRequest("subreq-8", className, 1),
					).
					NamedRequestWithPrioritizedList("req-2",
						st.SubRequest("subreq-1", className, 1),
					).
					Obj(),
			},
			allocations: nodeAllocation{
				allocationResults: []resourceapi.AllocationResult{
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1/subreq-8",
								},
								{
									Request: "req-2/subreq-1",
								},
							},
						},
					},
				},
			},
			expectedScore: 9,
		},
		"multiple-claims-with-last-subrequests-allocated": {
			claims: []*resourceapi.ResourceClaim{
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-1",
						st.SubRequest("subreq-1", className, 1),
						st.SubRequest("subreq-2", className, 1),
						st.SubRequest("subreq-3", className, 1),
						st.SubRequest("subreq-4", className, 1),
						st.SubRequest("subreq-5", className, 1),
						st.SubRequest("subreq-6", className, 1),
						st.SubRequest("subreq-7", className, 1),
						st.SubRequest("subreq-8", className, 1),
					).
					Obj(),
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-2",
						st.SubRequest("subreq-1", className, 1),
						st.SubRequest("subreq-2", className, 1),
						st.SubRequest("subreq-3", className, 1),
						st.SubRequest("subreq-4", className, 1),
						st.SubRequest("subreq-5", className, 1),
						st.SubRequest("subreq-6", className, 1),
						st.SubRequest("subreq-7", className, 1),
						st.SubRequest("subreq-8", className, 1),
					).
					Obj(),
			},
			allocations: nodeAllocation{
				allocationResults: []resourceapi.AllocationResult{
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1/subreq-8",
								},
							},
						},
					},
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-2/subreq-8",
								},
							},
						},
					},
				},
			},
			expectedScore: 2,
		},
		"multiple-claims-with-top-subrequests-allocated": {
			claims: []*resourceapi.ResourceClaim{
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-1",
						st.SubRequest("subreq-1", className, 1),
					).
					Obj(),
				st.MakeResourceClaim().
					NamedRequestWithPrioritizedList("req-2",
						st.SubRequest("subreq-1", className, 1),
					).
					Obj(),
			},
			allocations: nodeAllocation{
				allocationResults: []resourceapi.AllocationResult{
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-1/subreq-1",
								},
							},
						},
					},
					{
						Devices: resourceapi.DeviceAllocationResult{
							Results: []resourceapi.DeviceRequestAllocationResult{
								{
									Request: "req-2/subreq-1",
								},
							},
						},
					},
				},
			},
			expectedScore: 16,
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			iterator := slices.All(tc.claims)
			score, err := computeScore(iterator, tc.allocations)
			if err != nil {
				if !tc.expectErr {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if tc.expectErr {
				t.Fatal("expected error, got none")
			}
			assert.Equal(t, tc.expectedScore, score)
		})
	}
}

func TestNormalizeScore(t *testing.T) {
	testcases := map[string]struct {
		scores         fwk.NodeScoreList
		expectedScores fwk.NodeScoreList
	}{
		"empty": {
			scores:         fwk.NodeScoreList{},
			expectedScores: fwk.NodeScoreList{},
		},
		"single-score": {
			scores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 42,
				},
			},
			expectedScores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 100,
				},
			},
		},
		"all-same": {
			scores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 8,
				},
				{
					Name:  "node-2",
					Score: 8,
				},
			},
			expectedScores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 100,
				},
				{
					Name:  "node-2",
					Score: 100,
				},
			},
		},
		"all-same-very-large": {
			scores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: math.MaxInt32,
				},
				{
					Name:  "node-2",
					Score: math.MaxInt32,
				},
			},
			expectedScores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 100,
				},
				{
					Name:  "node-2",
					Score: 100,
				},
			},
		},
		"max-and-min-values": {
			scores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: math.MaxInt32,
				},
				{
					Name:  "node-2",
					Score: 0,
				},
			},
			expectedScores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 100,
				},
				{
					Name:  "node-2",
					Score: 0,
				},
			},
		},
		"mid-value": {
			scores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 99,
				},
				{
					Name:  "node-2",
					Score: 98,
				},
				{
					Name:  "node-3",
					Score: 97,
				},
			},
			expectedScores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 100,
				},
				{
					Name:  "node-2",
					Score: 98,
				},
				{
					Name:  "node-3",
					Score: 97,
				},
			},
		},
		"large-spread-lost-precision": {
			scores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: math.MaxInt32,
				},
				{
					Name:  "node-2",
					Score: math.MaxInt32 - 1,
				},
				{
					Name:  "node-3",
					Score: 1,
				},
				{
					Name:  "node-4",
					Score: 0,
				},
			},
			expectedScores: fwk.NodeScoreList{
				{
					Name:  "node-1",
					Score: 100,
				},
				{
					Name:  "node-2",
					Score: 99,
				},
				{
					Name:  "node-3",
					Score: 0,
				},
				{
					Name:  "node-4",
					Score: 0,
				},
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			pl := &DynamicResources{
				enabled: true,
			}
			scores := tc.scores
			_ = pl.NormalizeScore(context.Background(), nil, nil, scores)
			assert.Equal(t, tc.expectedScores, scores)
		})
	}
}
