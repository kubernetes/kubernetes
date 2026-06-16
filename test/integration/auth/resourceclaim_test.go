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

package auth

import (
	"context"
	"fmt"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestResourceClaimGranularStatusAuthorization(t *testing.T) {
	// Enable Feature Gates Globally for the test run
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicResourceAllocation, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourceClaimDeviceStatus, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAResourceClaimGranularStatusAuthorization, true)

	const (
		ns        = "dra-authz-test"
		saName    = "dra-plugin-sa"
		claimName = "test-claim"
		nodeName  = "worker-1"
	)

	testcases := []struct {
		name             string
		preAllocate      bool
		impersonateExtra map[string][]string
		setupRBAC        func(t *testing.T, adminClient clientset.Interface)
		updateClaim      func(c *resourceapi.ResourceClaim)
		verifyErr        func(t *testing.T, err error)
	}{
		{
			name:        "fails to update status.devices without driver permission",
			preAllocate: true,
			setupRBAC:   func(t *testing.T, adminClient clientset.Interface) {}, // No extra RBAC beyond front-door
			updateClaim: func(c *resourceapi.ResourceClaim) {
				c.Status.Devices = []resourceapi.AllocatedDeviceStatus{
					{Driver: "test-driver", Pool: "pool1", Device: "dev1"},
				}
			},
			verifyErr: func(t *testing.T, err error) {
				if err == nil || !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "Forbidden: changing status.devices requires") {
					t.Errorf("Expected Invalid/Forbidden error, got: %v", err)
				}
			},
		},
		{
			name:        "succeeds with associated-node permission for same-node SA",
			preAllocate: true,
			impersonateExtra: map[string][]string{
				"authentication.kubernetes.io/node-name": {nodeName},
			},
			setupRBAC: func(t *testing.T, adminClient clientset.Interface) {
				createRoleAndBinding(t, adminClient, ns, saName, "node-local-driver",
					[]string{"resourceclaims/driver"}, []string{"associated-node:update"})
			},
			updateClaim: func(c *resourceapi.ResourceClaim) {
				c.Status.Devices = []resourceapi.AllocatedDeviceStatus{
					{Driver: "test-driver", Pool: "pool1", Device: "dev1"},
				}
			},
			verifyErr: func(t *testing.T, err error) {
				if err != nil {
					t.Errorf("Expected success via associated-node, got: %v", err)
				}
			},
		},
		{
			name:        "fails deallocation without binding permission",
			preAllocate: true,
			setupRBAC:   func(t *testing.T, adminClient clientset.Interface) {},
			updateClaim: func(c *resourceapi.ResourceClaim) {
				c.Status.Allocation = nil
			},
			verifyErr: func(t *testing.T, err error) {
				if err == nil || !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "Forbidden: changing status.allocation") {
					t.Errorf("Expected Invalid/Forbidden on unbind, got: %v", err)
				}
			},
		},
		{
			name:        "succeeds to update status.reservedFor with binding permission",
			preAllocate: true,
			setupRBAC: func(t *testing.T, adminClient clientset.Interface) {
				createClusterRoleAndBinding(t, adminClient, ns, saName, "cluster-binding-updater-reserved",
					[]string{"resourceclaims/binding"}, []string{"update"})
			},
			updateClaim: func(c *resourceapi.ResourceClaim) {
				c.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{
					{Resource: "pods", Name: "pod-1", UID: "uid-1"},
				}
			},
			verifyErr: func(t *testing.T, err error) {
				if err != nil {
					t.Errorf("Expected success, got: %v", err)
				}
			},
		},
		{
			name:        "fails when updating both allocation and devices but missing binding permission",
			preAllocate: true,
			setupRBAC: func(t *testing.T, adminClient clientset.Interface) {
				// Has driver permission, but LACKS binding permission
				createRoleAndBinding(t, adminClient, ns, saName, "driver-only",
					[]string{"resourceclaims/driver"}, []string{"arbitrary-node:update"})
			},
			updateClaim: func(c *resourceapi.ResourceClaim) {
				// Re-allocate to a different node (requires binding)
				if c.Status.Allocation != nil && c.Status.Allocation.NodeSelector != nil {
					c.Status.Allocation.NodeSelector.NodeSelectorTerms[0].MatchFields[0].Values = []string{"worker-2"}
				}
				// Change devices (requires driver)
				c.Status.Devices = []resourceapi.AllocatedDeviceStatus{
					{Driver: "test-driver", Pool: "pool1", Device: "dev2"},
				}
			},
			verifyErr: func(t *testing.T, err error) {
				if err == nil || !apierrors.IsInvalid(err) || !strings.Contains(err.Error(), "Forbidden: changing status.allocation") {
					t.Errorf("Expected Forbidden on simultaneous update missing binding, got: %v", err)
				}
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
				"--runtime-config=api/all=true",
				"--authorization-mode=RBAC",
			}, framework.SharedEtcd())
			t.Cleanup(server.TearDownFn)

			adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

			// Setup Namespace and Service Account
			_, err := adminClient.CoreV1().Namespaces().Create(context.TODO(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}}, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			_, err = adminClient.CoreV1().ServiceAccounts(ns).Create(context.TODO(), &corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: saName}}, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// Create the base ResourceClaim
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{Name: claimName},
				Spec: resourceapi.ResourceClaimSpec{
					Devices: resourceapi.DeviceClaim{
						Requests: []resourceapi.DeviceRequest{{
							Name: "req-1",
							FirstAvailable: []resourceapi.DeviceSubRequest{{
								Name:            "subreq-1",
								DeviceClassName: "test-class",
							}},
						}},
					},
				},
			}
			_, err = adminClient.ResourceV1().ResourceClaims(ns).Create(context.TODO(), claim, metav1.CreateOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// Admin Pre-allocation (if required by test)
			if tc.preAllocate {
				c, err := adminClient.ResourceV1().ResourceClaims(ns).Get(context.TODO(), claimName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Failed to fetch claim for pre-allocation: %v", err)
				}
				c.Status.Allocation = &resourceapi.AllocationResult{
					NodeSelector: &corev1.NodeSelector{
						NodeSelectorTerms: []corev1.NodeSelectorTerm{{
							MatchFields: []corev1.NodeSelectorRequirement{{Key: "metadata.name", Operator: corev1.NodeSelectorOpIn, Values: []string{nodeName}}},
						}},
					},
					Devices: resourceapi.DeviceAllocationResult{
						Results: []resourceapi.DeviceRequestAllocationResult{
							{Request: "req-1", Driver: "test-driver", Pool: "pool1", Device: "dev1"},
						},
					},
				}
				_, err = adminClient.ResourceV1().ResourceClaims(ns).UpdateStatus(context.TODO(), c, metav1.UpdateOptions{})
				if err != nil {
					t.Fatalf("Admin failed to set baseline allocation: %v", err)
				}
			}

			// Setup RBAC
			createRoleAndBinding(t, adminClient, ns, saName, "base-status-updater", []string{"resourceclaims/status"}, []string{"update", "patch"})
			createRoleAndBinding(t, adminClient, ns, saName, "base-claim-reader", []string{"resourceclaims"}, []string{"get"})
			tc.setupRBAC(t, adminClient)

			// Build the Impersonated Client
			saConfig := rest.CopyConfig(server.ClientConfig)
			saConfig.Impersonate = rest.ImpersonationConfig{
				UserName: fmt.Sprintf("system:serviceaccount:%s:%s", ns, saName),
				Extra:    tc.impersonateExtra,
			}
			saClient := clientset.NewForConfigOrDie(saConfig)

			// Execute Test Update
			cToUpdate, err := adminClient.ResourceV1().ResourceClaims(ns).Get(context.TODO(), claimName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Failed to fetch claim before test execution: %v", err)
			}
			tc.updateClaim(cToUpdate)
			_, testErr := saClient.ResourceV1().ResourceClaims(ns).UpdateStatus(context.TODO(), cToUpdate, metav1.UpdateOptions{})

			// 7. Verify Results
			tc.verifyErr(t, testErr)
		})
	}
}

// createRoleAndBinding is a quick helper to assign namespaced RBAC rules
func createRoleAndBinding(t *testing.T, client clientset.Interface, ns, saName, roleName string, resources, verbs []string) {
	role := &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{Name: roleName},
		Rules: []rbacv1.PolicyRule{{
			APIGroups: []string{"resource.k8s.io"},
			Resources: resources,
			Verbs:     verbs,
		}},
	}
	_, err := client.RbacV1().Roles(ns).Create(context.TODO(), role, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatal(err)
	}

	binding := &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: roleName + "-binding"},
		Subjects:   []rbacv1.Subject{{Kind: "ServiceAccount", Name: saName, Namespace: ns}},
		RoleRef:    rbacv1.RoleRef{APIGroup: "rbac.authorization.k8s.io", Kind: "Role", Name: roleName},
	}
	_, err = client.RbacV1().RoleBindings(ns).Create(context.TODO(), binding, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatal(err)
	}
}

// createClusterRoleAndBinding is a helper for cluster-scoped synthetic checks (like binding)
func createClusterRoleAndBinding(t *testing.T, client clientset.Interface, ns, saName, roleName string, resources, verbs []string) {
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{Name: roleName},
		Rules: []rbacv1.PolicyRule{{
			APIGroups: []string{"resource.k8s.io"},
			Resources: resources,
			Verbs:     verbs,
		}},
	}
	_, err := client.RbacV1().ClusterRoles().Create(context.TODO(), role, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatal(err)
	}

	binding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{Name: roleName + "-binding"},
		Subjects:   []rbacv1.Subject{{Kind: "ServiceAccount", Name: saName, Namespace: ns}},
		RoleRef:    rbacv1.RoleRef{APIGroup: "rbac.authorization.k8s.io", Kind: "ClusterRole", Name: roleName},
	}
	_, err = client.RbacV1().ClusterRoleBindings().Create(context.TODO(), binding, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatal(err)
	}
}
