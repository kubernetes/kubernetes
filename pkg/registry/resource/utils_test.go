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

package resource

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
)

// TestGetModifiedDrivers contains the unit tests for the getModifiedDrivers function.
func TestGetModifiedDrivers(t *testing.T) {
	// Helper to create AllocatedDeviceStatus
	devStatus := func(driver, pool, device string, network *resource.NetworkDeviceData) resource.AllocatedDeviceStatus {
		return resource.AllocatedDeviceStatus{
			Driver:      driver,
			Pool:        pool,
			Device:      device,
			NetworkData: network,
		}
	}

	// Helper to create ResourceClaimStatus
	claimStatus := func(devices ...resource.AllocatedDeviceStatus) resource.ResourceClaimStatus {
		return resource.ResourceClaimStatus{
			Devices: devices,
		}
	}

	testCases := map[string]struct {
		newStatus resource.ResourceClaimStatus
		oldStatus resource.ResourceClaimStatus
		expected  sets.Set[string]
	}{
		"no changes": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
				devStatus("driver-b", "pool-1", "dev-2", nil),
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
				devStatus("driver-b", "pool-1", "dev-2", nil),
			),
			expected: sets.Set[string]{},
		},
		"add one device": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
				devStatus("driver-b", "pool-1", "dev-2", nil), // New
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
			),
			expected: sets.New[string]("driver-b"),
		},
		"add device for existing driver": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
				devStatus("driver-a", "pool-1", "dev-2", nil), // New
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
			),
			expected: sets.New[string]("driver-a"),
		},
		"remove one device": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
				devStatus("driver-b", "pool-1", "dev-2", nil), // Removed
			),
			expected: sets.New[string]("driver-b"),
		},
		"remove device for driver that still has other devices": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
				devStatus("driver-a", "pool-1", "dev-2", nil), // Removed
			),
			expected: sets.New[string]("driver-a"),
		},
		"modify one device": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0", IPs: []string{"192.168.7.1/24"}}), // Modified
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0"}),
			),
			expected: sets.New[string]("driver-a"),
		},
		"modify device for driver, no change for other driver": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0", IPs: []string{"192.168.7.1/24"}}), // Modified
				devStatus("driver-b", "pool-1", "dev-2", nil),
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0"}),
				devStatus("driver-b", "pool-1", "dev-2", nil),
			),
			expected: sets.New[string]("driver-a"),
		},
		"complex change (add, remove, modify)": {
			newStatus: claimStatus(
				// driver-a: dev-1 modified
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0", IPs: []string{"192.168.7.1/24"}}), // Modified
				// driver-b: dev-2 unchanged
				devStatus("driver-b", "pool-1", "dev-2", nil),
				// driver-c: dev-3 added
				devStatus("driver-c", "pool-1", "dev-3", nil),
			),
			oldStatus: claimStatus(
				// driver-a: dev-1 old state
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0"}),
				// driver-b: dev-2 unchanged
				devStatus("driver-b", "pool-1", "dev-2", nil),
				// driver-d: dev-4 removed
				devStatus("driver-d", "pool-1", "dev-4", nil),
			),
			expected: sets.New[string]("driver-a", "driver-c", "driver-d"),
		},
		"empty to empty": {
			newStatus: claimStatus(),
			oldStatus: claimStatus(),
			expected:  sets.Set[string]{},
		},
		"empty to one device": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
			),
			oldStatus: claimStatus(),
			expected:  sets.New[string]("driver-a"),
		},
		"one device to empty": {
			newStatus: claimStatus(),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", nil),
			),
			expected: sets.New[string]("driver-a"),
		},
		"replace device with same key but different content": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0", IPs: []string{"192.168.7.1/24"}}),
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-1", &resource.NetworkDeviceData{InterfaceName: "eth0"}),
			),
			expected: sets.New[string]("driver-a"),
		},
		"replace device with different key for same driver": {
			newStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-NEW", nil),
			),
			oldStatus: claimStatus(
				devStatus("driver-a", "pool-1", "dev-OLD", nil),
			),
			expected: sets.New[string]("driver-a"),
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			result := getModifiedDrivers(tc.newStatus, tc.oldStatus)
			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("Expected driver set %v, but got %v", tc.expected, result)
			}
		})
	}
}

type fakeAuthorizer struct {
	rules map[string]authorizer.Decision
	err   error
}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	if f.err != nil {
		return authorizer.DecisionDeny, "forced error", f.err
	}
	key := fmt.Sprintf("%s/%s/%s", a.GetVerb(), a.GetResource(), a.GetName())
	if decision, ok := f.rules[key]; ok {
		return decision, "", nil
	}
	wildcardKey := fmt.Sprintf("%s/%s/*", a.GetVerb(), a.GetResource())
	if decision, ok := f.rules[wildcardKey]; ok {
		return decision, "", nil
	}
	return authorizer.DecisionDeny, "no rule matched", nil
}

func TestAuthorizedForDeviceStatus(t *testing.T) {
	namespaceName := "test-ns"
	nodeName := "test-node"
	driverName := "test-driver"

	testcases := []struct {
		name                     string
		newAllocatedDeviceStatus resource.ResourceClaimStatus
		oldAllocatedDeviceStatus resource.ResourceClaimStatus
		user                     user.Info
		authz                    authorizer.Authorizer
		expectErrs               field.ErrorList
	}{
		{
			name: "no drivers modified",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device"},
				},
			},
		},
		{
			name: "user not found in context",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			expectErrs: field.ErrorList{
				field.InternalError(field.NewPath("status", "allocation", "devices"), fmt.Errorf("cannot determine calling user to check driver status update authorization")),
			},
		},
		{
			name: "new allocation is nil",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user: &user.DefaultInfo{Name: "test-user"},
			expectErrs: field.ErrorList{
				field.Forbidden(field.NewPath("status", "allocation", "devices"), fmt.Sprintf("user %q is not authorized to update device status for driver %q, requires permission for %q on resource %q", "test-user", driverName, resourcev1.VerbUpdateDriverStatus, resourcev1.ResourceUpdateDriverStatus)),
			},
		},
		{
			name: "new node selector is nil",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user: &user.DefaultInfo{Name: "test-user"},
			expectErrs: field.ErrorList{
				field.Forbidden(field.NewPath("status", "allocation", "devices"), fmt.Sprintf("user %q is not authorized to update device status for driver %q, requires permission for %q on resource %q", "test-user", driverName, resourcev1.VerbUpdateDriverStatus, resourcev1.ResourceUpdateDriverStatus)),
			},
		},
		{
			name: "request from node, selector matches",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{nodeName}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{"update-device-status/drivers/" + driverName: authorizer.DecisionAllow}},
			user:  &user.DefaultInfo{Name: "system:node:" + nodeName, Groups: []string{"system:nodes"}},
		},
		{
			name: "request from node, selector matches via hostname",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchExpressions: []core.NodeSelectorRequirement{
							{Key: "kubernetes.io/hostname", Operator: "In", Values: []string{nodeName}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{"update-device-status/drivers/" + driverName: authorizer.DecisionAllow}},
			user:  &user.DefaultInfo{Name: "system:node:" + nodeName, Groups: []string{"system:nodes"}},
		},
		{
			name: "request from node, selector matches with extra user info",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{nodeName}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{"update-device-status/drivers/" + driverName: authorizer.DecisionAllow}},
			user:  &user.DefaultInfo{Name: "test-node", Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
		},
		{
			name: "request from node, selector does not match, not authorized",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{"other-node"}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user:  &user.DefaultInfo{Name: "system:node:" + nodeName, Groups: []string{"system:nodes"}},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			expectErrs: field.ErrorList{
				field.Forbidden(field.NewPath("status", "allocation", "devices"), fmt.Sprintf("user %q on node %q is not authorized to update device status for drivers on ResourceClaim not allocated to this node", "system:node:"+nodeName, nodeName)),
			},
		},
		{
			name: "request from node, selector does not match, not authorized",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{"other-node"}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user:  &user.DefaultInfo{Name: "test-user", Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			expectErrs: field.ErrorList{
				field.Forbidden(field.NewPath("status", "allocation", "devices"), fmt.Sprintf("user %q on node %q is not authorized to update device status for drivers on ResourceClaim not allocated to this node", "test-user", nodeName)),
			},
		},
		{
			name: "not from node, authorized for specific driver",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{nodeName}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user:  &user.DefaultInfo{Name: "test-user"},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{"update-device-status/drivers/test-driver": authorizer.DecisionAllow}},
		},
		{
			name: "not from node, authorized with wildcard",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{nodeName}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user:  &user.DefaultInfo{Name: "test-user"},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{"update-device-status/drivers/*": authorizer.DecisionAllow}},
		},
		{
			name: "not from node, not authorized",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{nodeName}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user:  &user.DefaultInfo{Name: "test-user"},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			expectErrs: field.ErrorList{
				field.Forbidden(field.NewPath("status", "allocation", "devices"), fmt.Sprintf("user %q is not authorized to update device status for driver %q, requires permission for \"update-device-status\" on resource \"drivers\"", "test-user", driverName)),
			},
		},
		{
			name: "authorization error",
			newAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: "In", Values: []string{nodeName}},
						}},
					}},
				},
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-new"},
				},
			},
			oldAllocatedDeviceStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device-old"},
				},
			},
			user:  &user.DefaultInfo{Name: "test-user"},
			authz: &fakeAuthorizer{err: fmt.Errorf("authz error")},
			expectErrs: field.ErrorList{
				field.InternalError(field.NewPath("status", "allocation", "devices"), fmt.Errorf("authorization check failed for driver %q: authz error", driverName)),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			if tc.user != nil {
				ctx = genericapirequest.WithUser(ctx, tc.user)
			}
			if tc.authz == nil {
				tc.authz = &fakeAuthorizer{rules: map[string]authorizer.Decision{}}
			}
			errs := AuthorizedForDeviceStatus(ctx, field.NewPath("status", "allocation", "devices"), tc.authz, tc.newAllocatedDeviceStatus, tc.oldAllocatedDeviceStatus, namespaceName)
			if diff := cmp.Diff(tc.expectErrs, errs, cmp.Comparer(func(x, y error) bool {
				if x == nil || y == nil {
					return x == y
				}
				return x.Error() == y.Error()
			})); diff != "" {
				t.Errorf("unexpected errors diff (-want +got):\n%s", diff)
			}
		})
	}
}
