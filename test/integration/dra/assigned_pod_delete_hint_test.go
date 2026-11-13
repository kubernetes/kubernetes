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

package dra

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// Only public framework methods are supplied: tests exercise the plugin without
// exporting its claim store or binding readiness helpers.
type draPluginTestHandle struct {
	fwk.Handle
	manager  *draPluginTestManager
	client   kubernetes.Interface
	recorder events.EventRecorderLogger
}

func (h *draPluginTestHandle) ClientSet() kubernetes.Interface           { return h.client }
func (h *draPluginTestHandle) SharedDRAManager() fwk.SharedDRAManager    { return h.manager }
func (h *draPluginTestHandle) EventRecorder() events.EventRecorderLogger { return h.recorder }
func (h *draPluginTestHandle) ProfileName() string                       { return "dra-binding-test" }

type draPluginTestManager struct {
	fwk.SharedDRAManager
	tracker fwk.ResourceClaimTracker
	mapping draPluginTestResolver
}

func (m *draPluginTestManager) ResourceClaims() fwk.ResourceClaimTracker     { return m.tracker }
func (m *draPluginTestManager) DeviceClassResolver() fwk.DeviceClassResolver { return m.mapping }

type draPluginTestResolver map[v1.ResourceName]*resourceapi.DeviceClass

func (r draPluginTestResolver) GetDeviceClass(name v1.ResourceName) *resourceapi.DeviceClass {
	return r[name]
}

type draPluginTestClaims struct {
	fwk.ResourceClaimTracker
	current *resourceapi.ResourceClaim
}

func (c *draPluginTestClaims) Get(namespace, name string) (*resourceapi.ResourceClaim, error) {
	if c.current == nil || c.current.Namespace != namespace || c.current.Name != name {
		return nil, apierrors.NewNotFound(schema.GroupResource{Group: resourceapi.GroupName, Resource: "resourceclaims"}, name)
	}
	return c.current, nil
}
func (c *draPluginTestClaims) List() ([]*resourceapi.ResourceClaim, error) {
	if c.current == nil {
		return nil, nil
	}
	return []*resourceapi.ResourceClaim{c.current}, nil
}

func newDRAPluginForTest(t *testing.T, fts feature.Features, manager *draPluginTestManager) *dynamicresources.DynamicResources {
	t.Helper()
	metrics.Register()
	handle := &draPluginTestHandle{manager: manager, client: fake.NewClientset(), recorder: events.NewFakeRecorder(10)}
	args := &config.DynamicResourcesArgs{}
	if fts.EnableDRADeviceBindingConditions && fts.EnableDRAResourceClaimDeviceStatus {
		args.BindingTimeout = &metav1.Duration{Duration: time.Second}
	}
	plugin, err := dynamicresources.New(context.Background(), args, handle, fts)
	require.NoError(t, err)
	return plugin.(*dynamicresources.DynamicResources)
}

func assignedPodDeletionHint(t *testing.T, extended bool, mapping draPluginTestResolver) fwk.QueueingHintFn {
	t.Helper()
	plugin := newDRAPluginForTest(t, feature.Features{EnableDRAExtendedResource: extended}, &draPluginTestManager{mapping: mapping})
	registered, err := plugin.EventsToRegister(context.Background())
	require.NoError(t, err)
	for _, event := range registered {
		if event.Event.Resource == fwk.AssignedPod && event.Event.ActionType == fwk.Delete {
			require.NotNil(t, event.QueueingHintFn)
			return event.QueueingHintFn
		}
	}
	t.Fatal("DynamicResources must register an AssignedPod deletion hint to wake device waiters after Unreserve")
	return nil
}

func TestDRAAssignedPodDeletionHint(t *testing.T) {
	resource := v1.ResourceName("example.com/mapped")
	mapping := draPluginTestResolver{resource: {}}
	explicit := st.MakePod().Name("holder").Namespace("default").UID("holder").Obj()
	explicit.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "device", ResourceClaimName: new("claim")}}
	plain := st.MakePod().Name("plain").Namespace("default").Obj()
	extended := st.MakePod().Name("holder").Namespace("default").Container("container").Obj()
	extended.Spec.Containers[0].Resources.Requests = v1.ResourceList{resource: apiresource.MustParse("1")}
	init := extended.DeepCopy()
	init.Spec.InitContainers = init.Spec.Containers
	init.Spec.Containers = nil
	unmapped := extended.DeepCopy()
	unmapped.Spec.Containers[0].Resources.Requests = v1.ResourceList{"example.com/unmapped": apiresource.MustParse("1")}
	waiter := st.MakePod().Name("waiter").Namespace("default").Obj()
	for _, tc := range []struct {
		name    string
		obj     interface{}
		gate    bool
		want    fwk.QueueingHint
		wantErr bool
	}{
		{name: "explicit", obj: explicit, want: fwk.Queue},
		{name: "plain", obj: plain, want: fwk.QueueSkip},
		{name: "extended", obj: extended, gate: true, want: fwk.Queue},
		{name: "extended-gate-off", obj: extended, want: fwk.QueueSkip},
		{name: "unmapped", obj: unmapped, gate: true, want: fwk.Queue},
		{name: "init-container", obj: init, gate: true, want: fwk.Queue},
		{name: "tombstone-explicit", obj: cache.DeletedFinalStateUnknown{Obj: explicit}, want: fwk.Queue},
		{name: "tombstone-plain", obj: cache.DeletedFinalStateUnknown{Obj: plain}, want: fwk.QueueSkip},
		{name: "tombstone-extended", obj: cache.DeletedFinalStateUnknown{Obj: extended}, gate: true, want: fwk.Queue},
		{name: "tombstone-init-container", obj: cache.DeletedFinalStateUnknown{Obj: init}, gate: true, want: fwk.Queue},
		{name: "unexpected-object", obj: "not-a-pod", want: fwk.Queue, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hint := assignedPodDeletionHint(t, tc.gate, mapping)
			got, err := hint(klog.Background(), waiter, tc.obj, nil)
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, tc.want, got)
		})
	}
}

func TestDRAAssignedPodDeletionAfterMappingChange(t *testing.T) {
	for _, change := range []string{"removed", "renamed"} {
		for _, tombstone := range []bool{false, true} {
			name := change + "/pod"
			if tombstone {
				name = change + "/tombstone"
			}
			t.Run(name, func(t *testing.T) {
				resource := v1.ResourceName("example.com/held")
				mapping := draPluginTestResolver{resource: {}}
				hint := assignedPodDeletionHint(t, true, mapping)
				holder := st.MakePod().Name("holder").Namespace("default").Container("container").Obj()
				holder.Spec.Containers[0].Resources.Requests = v1.ResourceList{resource: apiresource.MustParse("1")}
				holder.Status.ExtendedResourceClaimStatus = &v1.PodExtendedResourceClaimStatus{ResourceClaimName: "implicit-claim"}
				waiter := st.MakePod().Name("waiter").Namespace("default").Obj()
				got, err := hint(klog.Background(), waiter, holder, nil)
				require.NoError(t, err)
				require.Equal(t, fwk.Queue, got)
				delete(mapping, resource)
				if change == "renamed" {
					mapping["example.com/renamed"] = &resourceapi.DeviceClass{}
				}
				var oldObj interface{} = holder
				if tombstone {
					oldObj = cache.DeletedFinalStateUnknown{Obj: holder}
				}
				got, err = hint(klog.Background(), waiter, oldObj, nil)
				require.NoError(t, err)
				require.Equal(t, fwk.Queue, got, "changing the mapping must not hide devices held through an existing implicit claim")
			})
		}
	}
}

func TestDRABindingWaitAfterClaimChange(t *testing.T) {
	for _, implicit := range []bool{false, true} {
		kind := "standalone"
		if implicit {
			kind = "implicit"
		}
		for _, change := range []string{"still-pending", "deallocated", "deleted", "ready"} {
			t.Run(kind+"/"+change, func(t *testing.T) {
				resource := v1.ResourceName("example.com/device")
				pod := st.MakePod().Name("holder").Namespace("default").UID("holder").Container("container").Obj()
				initial := &resourceapi.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: "claim", Namespace: pod.Namespace, UID: "claim"},
					Status: resourceapi.ResourceClaimStatus{
						ReservedFor: []resourceapi.ResourceClaimConsumerReference{{Resource: "pods", Name: pod.Name, UID: pod.UID}},
						Allocation: &resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{{
							Driver: "example.com", Pool: "pool", Device: "device", BindingConditions: []string{"Ready"},
						}}}},
					},
				}
				if implicit {
					pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{resource: apiresource.MustParse("1")}
					initial.Annotations = map[string]string{resourceapi.ExtendedResourceClaimAnnotation: "true"}
					initial.OwnerReferences = []metav1.OwnerReference{{APIVersion: "v1", Kind: "Pod", Name: pod.Name, UID: pod.UID, Controller: new(true)}}
				} else {
					pod.Spec.ResourceClaims = []v1.PodResourceClaim{{Name: "device", ResourceClaimName: &initial.Name}}
				}
				tracker := &draPluginTestClaims{current: initial}
				plugin := newDRAPluginForTest(t, feature.Features{EnableDRAExtendedResource: implicit,
					EnableDRADeviceBindingConditions: true, EnableDRAResourceClaimDeviceStatus: true},
					&draPluginTestManager{tracker: tracker, mapping: draPluginTestResolver{resource: {}}})
				state := framework.NewCycleState()
				_, status := plugin.PreFilter(context.Background(), state, pod, nil)
				require.True(t, status.IsSuccess(), "PreFilter must initialize the binding state: %v", status)
				latest := initial.DeepCopy()
				switch change {
				case "deallocated":
					latest.Status.Allocation = nil
				case "deleted":
					latest = nil
				case "ready":
					latest.Status.Devices = []resourceapi.AllocatedDeviceStatus{{Driver: "example.com", Pool: "pool", Device: "device",
						Conditions: []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue}},
					}}
				}
				tracker.current = latest
				status = plugin.PreBind(context.Background(), state, pod, "worker-0")
				switch change {
				case "deallocated":
					require.ErrorIs(t, status.AsError(), dynamicresources.ErrDeviceBindingFailed, "deallocation must abort the wait before bindingTimeout")
				case "deleted":
					require.True(t, apierrors.IsNotFound(status.AsError()), "claim deletion must abort the binding wait: %v", status)
				case "still-pending":
					require.ErrorIs(t, status.AsError(), dynamicresources.ErrDeviceBindingTimeout)
				case "ready":
					require.True(t, status.IsSuccess(), "ready devices must allow binding: %v", status)
				}
			})
		}
	}
}
