/*
Copyright 2018 The Kubernetes Authors.

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

package lease

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/pointer"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
)

func TestNewNodeLease(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
			UID:  types.UID("foo-uid"),
		},
	}
	cases := []struct {
		desc       string
		controller *controller
		base       *coordinationv1.Lease
		expect     *coordinationv1.Lease
	}{
		{
			desc: "nil base without node",
			controller: &controller{
				client:               fake.NewSimpleClientset(),
				leaseName:            node.Name,
				holderIdentity:       node.Name,
				leaseDurationSeconds: 10,
				clock:                fakeClock,
			},
			base: nil,
			expect: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now()},
				},
			},
		},
		{
			desc: "nil base with node",
			controller: &controller{
				client:               fake.NewSimpleClientset(node),
				leaseName:            node.Name,
				holderIdentity:       node.Name,
				leaseDurationSeconds: 10,
				clock:                fakeClock,
			},
			base: nil,
			expect: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: corev1.SchemeGroupVersion.WithKind("Node").Version,
							Kind:       corev1.SchemeGroupVersion.WithKind("Node").Kind,
							Name:       node.Name,
							UID:        node.UID,
						},
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now()},
				},
			},
		},
		{
			desc: "non-nil base without owner ref, renew time is updated",
			controller: &controller{
				client:               fake.NewSimpleClientset(node),
				holderIdentity:       node.Name,
				leaseDurationSeconds: 10,
				clock:                fakeClock,
			},
			base: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-10 * time.Second)},
				},
			},
			expect: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: corev1.SchemeGroupVersion.WithKind("Node").Version,
							Kind:       corev1.SchemeGroupVersion.WithKind("Node").Kind,
							Name:       node.Name,
							UID:        node.UID,
						},
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now()},
				},
			},
		},
		{
			desc: "non-nil base with owner ref, renew time is updated",
			controller: &controller{
				client:               fake.NewSimpleClientset(node),
				holderIdentity:       node.Name,
				leaseDurationSeconds: 10,
				clock:                fakeClock,
			},
			base: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: corev1.SchemeGroupVersion.WithKind("Node").Version,
							Kind:       corev1.SchemeGroupVersion.WithKind("Node").Kind,
							Name:       node.Name,
							UID:        node.UID,
						},
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-10 * time.Second)},
				},
			},
			expect: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: corev1.SchemeGroupVersion.WithKind("Node").Version,
							Kind:       corev1.SchemeGroupVersion.WithKind("Node").Kind,
							Name:       node.Name,
							UID:        node.UID,
						},
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now()},
				},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			tc.controller.newLeasePostProcessFunc = setNodeOwnerFunc(logger, tc.controller.client, node.Name)
			tc.controller.leaseNamespace = corev1.NamespaceNodeLease
			newLease, _ := tc.controller.newLease(tc.base)
			if newLease == tc.base {
				t.Fatalf("the new lease must be newly allocated, but got same address as base")
			}
			if !apiequality.Semantic.DeepEqual(tc.expect, newLease) {
				t.Errorf("unexpected result from newLease: %s", cmp.Diff(tc.expect, newLease))
			}
		})
	}
}

func TestRetryUpdateNodeLease(t *testing.T) {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
			UID:  types.UID("foo-uid"),
		},
	}
	gr := schema.GroupResource{Group: "v1", Resource: "lease"}
	noConnectionUpdateErr := apierrors.NewServerTimeout(gr, "put", 1)
	optimistcLockUpdateErr := apierrors.NewConflict(gr, "lease", fmt.Errorf("conflict"))
	cases := []struct {
		desc                       string
		updateReactor              func(action clienttesting.Action) (bool, runtime.Object, error)
		getReactor                 func(action clienttesting.Action) (bool, runtime.Object, error)
		onRepeatedHeartbeatFailure func()
		expectErr                  bool
		client                     *fake.Clientset
	}{
		{
			desc: "no errors",
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, &coordinationv1.Lease{}, nil
			},
			getReactor:                 nil,
			onRepeatedHeartbeatFailure: nil,
			expectErr:                  false,
			client:                     fake.NewSimpleClientset(node),
		},
		{
			desc: "connection errors",
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, nil, noConnectionUpdateErr
			},
			getReactor:                 nil,
			onRepeatedHeartbeatFailure: nil,
			expectErr:                  true,
			client:                     fake.NewSimpleClientset(node),
		},
		{
			desc: "optimistic lock errors",
			updateReactor: func() func(action clienttesting.Action) (bool, runtime.Object, error) {
				i := 0
				return func(action clienttesting.Action) (bool, runtime.Object, error) {
					i++
					switch i {
					case 1:
						return true, nil, noConnectionUpdateErr
					case 2:
						return true, nil, optimistcLockUpdateErr
					default:
						return true, &coordinationv1.Lease{}, nil
					}
				}
			}(),
			getReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, &coordinationv1.Lease{}, nil
			},
			onRepeatedHeartbeatFailure: func() { t.Fatalf("onRepeatedHeartbeatFailure called") },
			expectErr:                  false,
			client:                     fake.NewSimpleClientset(node),
		},
		{
			desc: "node not found errors",
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				t.Fatalf("lease was updated when node does not exist!")
				return true, nil, nil
			},
			getReactor:                 nil,
			onRepeatedHeartbeatFailure: nil,
			expectErr:                  true,
			client:                     fake.NewSimpleClientset(),
		},
	}
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			cl := tc.client
			if tc.updateReactor != nil {
				cl.PrependReactor("update", "leases", tc.updateReactor)
			}
			if tc.getReactor != nil {
				cl.PrependReactor("get", "leases", tc.getReactor)
			}
			c := &controller{
				clock:                      testingclock.NewFakeClock(time.Now()),
				client:                     cl,
				leaseClient:                cl.CoordinationV1().Leases(corev1.NamespaceNodeLease),
				holderIdentity:             node.Name,
				leaseNamespace:             corev1.NamespaceNodeLease,
				leaseDurationSeconds:       10,
				onRepeatedHeartbeatFailure: tc.onRepeatedHeartbeatFailure,
				newLeasePostProcessFunc:    setNodeOwnerFunc(logger, cl, node.Name),
			}
			if err := c.retryUpdateLease(ctx, nil); tc.expectErr != (err != nil) {
				t.Fatalf("got %v, expected %v", err != nil, tc.expectErr)
			}
		})
	}
}

func TestUpdateUsingLatestLease(t *testing.T) {
	nodeName := "foo"
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
			UID:  types.UID("foo-uid"),
		},
	}

	notFoundErr := apierrors.NewNotFound(coordinationv1.Resource("lease"), nodeName)
	internalErr := apierrors.NewInternalError(errors.New("unreachable code"))

	makeLease := func(name, resourceVersion string) *coordinationv1.Lease {
		return &coordinationv1.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:       corev1.NamespaceNodeLease,
				Name:            name,
				ResourceVersion: resourceVersion,
			},
		}
	}

	cases := []struct {
		desc                       string
		existingObjs               []runtime.Object
		latestLease                *coordinationv1.Lease
		updateReactor              func(action clienttesting.Action) (bool, runtime.Object, error)
		getReactor                 func(action clienttesting.Action) (bool, runtime.Object, error)
		createReactor              func(action clienttesting.Action) (bool, runtime.Object, error)
		expectLatestLease          bool
		expectLeaseResourceVersion string
	}{
		{
			desc:          "latestLease is nil and need to create",
			existingObjs:  []runtime.Object{node},
			latestLease:   nil,
			updateReactor: nil,
			getReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, nil, notFoundErr
			},
			createReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "1"), nil
			},
			expectLatestLease:          true,
			expectLeaseResourceVersion: "1",
		},
		{
			desc:          "latestLease is nil and need to create, node doesn't exist",
			existingObjs:  nil,
			latestLease:   nil,
			updateReactor: nil,
			getReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, nil, notFoundErr
			},
			createReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "1"), nil
			},
			expectLatestLease:          false,
			expectLeaseResourceVersion: "1",
		},
		{
			desc:         "latestLease is nil and need to update",
			existingObjs: []runtime.Object{node},
			latestLease:  nil,
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "2"), nil
			},
			getReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "1"), nil
			},
			expectLatestLease:          true,
			expectLeaseResourceVersion: "2",
		},
		{
			desc:         "latestLease exist and need to update",
			existingObjs: []runtime.Object{node},
			latestLease:  makeLease(nodeName, "1"),
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "2"), nil
			},
			expectLatestLease:          true,
			expectLeaseResourceVersion: "2",
		},
		{
			desc:         "update with latest lease failed",
			existingObjs: []runtime.Object{node},
			latestLease:  makeLease(nodeName, "1"),
			updateReactor: func() func(action clienttesting.Action) (bool, runtime.Object, error) {
				i := 0
				return func(action clienttesting.Action) (bool, runtime.Object, error) {
					i++
					switch i {
					case 1:
						return true, nil, notFoundErr
					case 2:
						return true, makeLease(nodeName, "3"), nil
					default:
						t.Fatalf("unexpect call update lease")
						return true, nil, internalErr
					}
				}
			}(),
			getReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "2"), nil
			},
			expectLatestLease:          true,
			expectLeaseResourceVersion: "3",
		},
	}
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			cl := fake.NewSimpleClientset(tc.existingObjs...)
			if tc.updateReactor != nil {
				cl.PrependReactor("update", "leases", tc.updateReactor)
			}
			if tc.getReactor != nil {
				cl.PrependReactor("get", "leases", tc.getReactor)
			}
			if tc.createReactor != nil {
				cl.PrependReactor("create", "leases", tc.createReactor)
			}
			c := &controller{
				clock:                   testingclock.NewFakeClock(time.Now()),
				client:                  cl,
				leaseClient:             cl.CoordinationV1().Leases(corev1.NamespaceNodeLease),
				holderIdentity:          node.Name,
				leaseNamespace:          corev1.NamespaceNodeLease,
				leaseDurationSeconds:    10,
				latestLease:             tc.latestLease,
				newLeasePostProcessFunc: setNodeOwnerFunc(logger, cl, node.Name),
			}

			c.sync(ctx)

			if tc.expectLatestLease {
				if tc.expectLeaseResourceVersion != c.latestLease.ResourceVersion {
					t.Fatalf("latestLease RV got %v, expected %v", c.latestLease.ResourceVersion, tc.expectLeaseResourceVersion)
				}
			} else {
				if c.latestLease != nil {
					t.Fatalf("unexpected latestLease: %v", c.latestLease)
				}
			}
		})
	}
}

// setNodeOwnerFunc helps construct a newLeasePostProcessFunc which sets
// a node OwnerReference to the given lease object
func setNodeOwnerFunc(logger klog.Logger, c clientset.Interface, nodeName string) func(lease *coordinationv1.Lease) error {
	return func(lease *coordinationv1.Lease) error {
		// Setting owner reference needs node's UID. Note that it is different from
		// kubelet.nodeRef.UID. When lease is initially created, it is possible that
		// the connection between master and node is not ready yet. So try to set
		// owner reference every time when renewing the lease, until successful.
		if len(lease.OwnerReferences) == 0 {
			if node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{}); err == nil {
				lease.OwnerReferences = []metav1.OwnerReference{
					{
						APIVersion: corev1.SchemeGroupVersion.WithKind("Node").Version,
						Kind:       corev1.SchemeGroupVersion.WithKind("Node").Kind,
						Name:       nodeName,
						UID:        node.UID,
					},
				}
			} else {
				logger.Error(err, "failed to get node when trying to set owner ref to the node lease", "node", nodeName)
				return err
			}
		}
		return nil
	}
}
