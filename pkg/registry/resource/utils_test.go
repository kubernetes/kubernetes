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

package resource

import (
	"context"
	"fmt"
	"net/http"
	"reflect"
	"testing"

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

// fakeAuthorizer records authorization calls and returns preconfigured decisions.
// The key format is "verb/resource/subresource/name".
type fakeAuthorizer struct {
	rules      map[string]authorizer.Decision
	err        error
	callCounts map[string]int
}

func (f *fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	if f.callCounts == nil {
		f.callCounts = make(map[string]int)
	}

	key := fmt.Sprintf("%s/%s/%s/%s", a.GetVerb(), a.GetResource(), a.GetSubresource(), a.GetName())
	f.callCounts[key]++

	if f.err != nil {
		return authorizer.DecisionDeny, "forced error", f.err
	}
	if decision, ok := f.rules[key]; ok {
		return decision, "", nil
	}
	return authorizer.DecisionDeny, "no rule matched", nil
}

// withRequestContext builds a context with user info and request info set,
// simulating what GetAuthorizerAttributes expects.
func withRequestContext(ctx context.Context, u user.Info, verb string) context.Context {
	ctx = genericapirequest.WithUser(ctx, u)
	ctx = genericapirequest.WithRequestInfo(ctx, &genericapirequest.RequestInfo{
		IsResourceRequest: true,
		Verb:              verb,
		APIGroup:          "resource.k8s.io",
		APIVersion:        "v1",
		Resource:          "resourceclaims",
		Subresource:       "status",
		Namespace:         "default",
		Name:              "test-claim",
	})
	// GetAuthorizerAttributes also needs an http.Request in context for audit, but
	// the function doesn't fail without it — we simulate by using a dummy request.
	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, "/", nil)
	ctx = req.Context()
	return ctx
}

func singleNodeAllocation(nodeName string) *resource.AllocationResult {
	return &resource.AllocationResult{
		NodeSelector: &core.NodeSelector{
			NodeSelectorTerms: []core.NodeSelectorTerm{
				{
					MatchFields: []core.NodeSelectorRequirement{
						{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{nodeName}},
					},
				},
			},
		},
	}
}

func TestAuthorizedForBinding(t *testing.T) {
	saName := "system:serviceaccount:kube-system:scheduler"
	testUser := &user.DefaultInfo{Name: saName}
	fp := field.NewPath("status")

	testcases := []struct {
		name       string
		newStatus  resource.ResourceClaimStatus
		oldStatus  resource.ResourceClaimStatus
		authz      *fakeAuthorizer
		expectErrs int
	}{
		{
			name: "no allocation or reservedFor change, no check needed",
			newStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{{Driver: "d", Pool: "p", Device: "dev"}},
			},
			oldStatus:  resource.ResourceClaimStatus{},
			authz:      &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			expectErrs: 0,
		},
		{
			name: "allocation changed, authorized",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation("node-1"),
			},
			oldStatus: resource.ResourceClaimStatus{},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				"update/resourceclaims/binding/": authorizer.DecisionAllow,
			}},
			expectErrs: 0,
		},
		{
			name: "allocation changed, not authorized",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation("node-1"),
			},
			oldStatus:  resource.ResourceClaimStatus{},
			authz:      &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			expectErrs: 1,
		},
		{
			name: "reservedFor changed, authorized",
			newStatus: resource.ResourceClaimStatus{
				ReservedFor: []resource.ResourceClaimConsumerReference{{Resource: "pods", Name: "pod-1", UID: "uid-1"}},
			},
			oldStatus: resource.ResourceClaimStatus{},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				"update/resourceclaims/binding/": authorizer.DecisionAllow,
			}},
			expectErrs: 0,
		},
		{
			name: "reservedFor changed, not authorized",
			newStatus: resource.ResourceClaimStatus{
				ReservedFor: []resource.ResourceClaimConsumerReference{{Resource: "pods", Name: "pod-1", UID: "uid-1"}},
			},
			oldStatus:  resource.ResourceClaimStatus{},
			authz:      &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			expectErrs: 1,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := withRequestContext(context.Background(), testUser, "update")
			errs := AuthorizedForBinding(ctx, fp, tc.authz, tc.newStatus, tc.oldStatus)
			if len(errs) != tc.expectErrs {
				t.Errorf("expected %d errors, got %d: %v", tc.expectErrs, len(errs), errs)
			}
		})
	}
}

func TestAuthorizedForDeviceStatus(t *testing.T) {
	saName := "system:serviceaccount:kube-system:dra-driver"
	nodeName := "test-node"
	driverName := "test-driver"
	fp := field.NewPath("status", "devices")

	testcases := []struct {
		name       string
		newStatus  resource.ResourceClaimStatus
		oldStatus  resource.ResourceClaimStatus
		user       user.Info
		authz      *fakeAuthorizer
		verb       string
		expectErrs int
	}{
		{
			name: "no drivers modified",
			newStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device"},
				},
			},
			oldStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{
					{Driver: driverName, Pool: "pool", Device: "device"},
				},
			},
			user:       &user.DefaultInfo{Name: saName},
			authz:      &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			verb:       "update",
			expectErrs: 0,
		},
		{
			name: "associated-node: SA on same node, allowed by associated-node verb",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user: &user.DefaultInfo{Name: saName, Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				fmt.Sprintf("associated-node:update/resourceclaims/driver/%s", driverName): authorizer.DecisionAllow,
			}},
			verb:       "update",
			expectErrs: 0,
		},
		{
			name: "associated-node: SA on same node, allowed by arbitrary-node fallback",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user: &user.DefaultInfo{Name: saName, Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				fmt.Sprintf("arbitrary-node:update/resourceclaims/driver/%s", driverName): authorizer.DecisionAllow,
			}},
			verb:       "update",
			expectErrs: 0,
		},
		{
			name: "associated-node: SA on same node, neither verb allowed",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user:       &user.DefaultInfo{Name: saName, Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
			authz:      &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			verb:       "update",
			expectErrs: 1,
		},
		{
			name: "SA on different node, only arbitrary-node checked",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation("other-node"),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user: &user.DefaultInfo{Name: saName, Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				fmt.Sprintf("arbitrary-node:update/resourceclaims/driver/%s", driverName): authorizer.DecisionAllow,
			}},
			verb:       "update",
			expectErrs: 0,
		},
		{
			name: "SA on different node, associated-node not checked, denied",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation("other-node"),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user: &user.DefaultInfo{Name: saName, Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
			// Only grant associated-node, which should NOT be checked since nodes differ
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				fmt.Sprintf("associated-node:update/resourceclaims/driver/%s", driverName): authorizer.DecisionAllow,
			}},
			verb:       "update",
			expectErrs: 1,
		},
		{
			name: "no node association (controller), only arbitrary-node checked, allowed",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user: &user.DefaultInfo{Name: saName}, // no node-name extra
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				fmt.Sprintf("arbitrary-node:update/resourceclaims/driver/%s", driverName): authorizer.DecisionAllow,
			}},
			verb:       "update",
			expectErrs: 0,
		},
		{
			name: "no node association (controller), denied",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user:       &user.DefaultInfo{Name: saName},
			authz:      &fakeAuthorizer{rules: map[string]authorizer.Decision{}},
			verb:       "update",
			expectErrs: 1,
		},
		{
			name: "multi-node claim (no single node in selector), only arbitrary-node",
			newStatus: resource.ResourceClaimStatus{
				Allocation: &resource.AllocationResult{
					NodeSelector: &core.NodeSelector{
						NodeSelectorTerms: []core.NodeSelectorTerm{
							{MatchFields: []core.NodeSelectorRequirement{
								{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"node-a", "node-b"}},
							}},
						},
					},
				},
				Devices: []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user: &user.DefaultInfo{Name: saName, Extra: map[string][]string{serviceaccount.NodeNameKey: {"node-a"}}},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				fmt.Sprintf("arbitrary-node:update/resourceclaims/driver/%s", driverName): authorizer.DecisionAllow,
			}},
			verb:       "update",
			expectErrs: 0,
		},
		{
			name: "patch verb propagated",
			newStatus: resource.ResourceClaimStatus{
				Allocation: singleNodeAllocation(nodeName),
				Devices:    []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-new"}},
			},
			oldStatus: resource.ResourceClaimStatus{
				Devices: []resource.AllocatedDeviceStatus{{Driver: driverName, Pool: "pool", Device: "dev-old"}},
			},
			user: &user.DefaultInfo{Name: saName, Extra: map[string][]string{serviceaccount.NodeNameKey: {nodeName}}},
			authz: &fakeAuthorizer{rules: map[string]authorizer.Decision{
				fmt.Sprintf("associated-node:patch/resourceclaims/driver/%s", driverName): authorizer.DecisionAllow,
			}},
			verb:       "patch",
			expectErrs: 0,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := withRequestContext(context.Background(), tc.user, tc.verb)
			errs := AuthorizedForDeviceStatus(ctx, fp, tc.authz, tc.newStatus, tc.oldStatus)
			if len(errs) != tc.expectErrs {
				t.Errorf("expected %d errors, got %d: %v", tc.expectErrs, len(errs), errs)
			}
		})
	}
}

func TestNodeNameFromAllocation(t *testing.T) {
	testCases := []struct {
		name       string
		allocation *resource.AllocationResult
		expected   string
	}{
		{
			name:       "nil allocation",
			allocation: nil,
			expected:   "",
		},
		{
			name:       "nil node selector",
			allocation: &resource.AllocationResult{},
			expected:   "",
		},
		{
			name:       "exact single-node match",
			allocation: singleNodeAllocation("worker-1"),
			expected:   "worker-1",
		},
		{
			name: "multiple values",
			allocation: &resource.AllocationResult{
				NodeSelector: &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"node-a", "node-b"}},
						}},
					},
				},
			},
			expected: "",
		},
		{
			name: "match expressions instead of match fields",
			allocation: &resource.AllocationResult{
				NodeSelector: &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchExpressions: []core.NodeSelectorRequirement{
							{Key: "kubernetes.io/hostname", Operator: core.NodeSelectorOpIn, Values: []string{"node-1"}},
						}},
					},
				},
			},
			expected: "",
		},
		{
			name: "multiple terms",
			allocation: &resource.AllocationResult{
				NodeSelector: &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"node-1"}},
						}},
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"node-2"}},
						}},
					},
				},
			},
			expected: "",
		},
		{
			name: "wrong key",
			allocation: &resource.AllocationResult{
				NodeSelector: &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.namespace", Operator: core.NodeSelectorOpIn, Values: []string{"node-1"}},
						}},
					},
				},
			},
			expected: "",
		},
		{
			name: "wrong operator",
			allocation: &resource.AllocationResult{
				NodeSelector: &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: core.NodeSelectorOpNotIn, Values: []string{"node-1"}},
						}},
					},
				},
			},
			expected: "",
		},
		{
			name: "extra match fields",
			allocation: &resource.AllocationResult{
				NodeSelector: &core.NodeSelector{
					NodeSelectorTerms: []core.NodeSelectorTerm{
						{MatchFields: []core.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"node-1"}},
							{Key: "metadata.name", Operator: core.NodeSelectorOpIn, Values: []string{"node-1"}},
						}},
					},
				},
			},
			expected: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := nodeNameFromAllocation(tc.allocation)
			if result != tc.expected {
				t.Errorf("expected %q, got %q", tc.expected, result)
			}
		})
	}
}

func TestSAAssociatedWithAllocatedNode(t *testing.T) {
	validSA := &user.DefaultInfo{
		Name: "system:serviceaccount:default:dra-driver-sa",
		Extra: map[string][]string{
			serviceaccount.NodeNameKey: {"worker-node-1"},
		},
	}

	testCases := []struct {
		name              string
		userInfo          user.Info
		allocatedNodeName string
		expected          bool
	}{
		{
			name:              "SA on matching node",
			userInfo:          validSA,
			allocatedNodeName: "worker-node-1",
			expected:          true,
		},
		{
			name:              "SA on different node",
			userInfo:          validSA,
			allocatedNodeName: "worker-node-2",
			expected:          false,
		},
		{
			name:              "empty allocated node name",
			userInfo:          validSA,
			allocatedNodeName: "",
			expected:          false,
		},
		{
			name: "not a service account (kubelet identity)",
			userInfo: &user.DefaultInfo{
				Name: "system:node:worker-node-1",
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"worker-node-1"},
				},
			},
			allocatedNodeName: "worker-node-1",
			expected:          false,
		},
		{
			name: "not a service account (regular user)",
			userInfo: &user.DefaultInfo{
				Name: "jane-doe",
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"worker-node-1"},
				},
			},
			allocatedNodeName: "worker-node-1",
			expected:          false,
		},
		{
			name: "service account missing node name extra attribute",
			userInfo: &user.DefaultInfo{
				Name:  "system:serviceaccount:default:dra-driver-sa",
				Extra: map[string][]string{},
			},
			allocatedNodeName: "worker-node-1",
			expected:          false,
		},
		{
			name: "service account with multiple node names in extra attribute",
			userInfo: &user.DefaultInfo{
				Name: "system:serviceaccount:default:dra-driver-sa",
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"worker-node-1", "worker-node-2"},
				},
			},
			allocatedNodeName: "worker-node-1",
			expected:          false,
		},
		{
			name: "service account with invalid node name format",
			userInfo: &user.DefaultInfo{
				Name: "system:serviceaccount:default:dra-driver-sa",
				Extra: map[string][]string{
					serviceaccount.NodeNameKey: {"invalid_node_name!"},
				},
			},
			allocatedNodeName: "invalid_node_name!",
			expected:          false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := saAssociatedWithAllocatedNode(tc.userInfo, tc.allocatedNodeName)
			if result != tc.expected {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}
}
