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

package nodelease

import (
	"errors"
	"fmt"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/utils/pointer"
)

func TestNewLease(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())
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
			newLease := tc.controller.newLease(tc.base)
			if newLease == tc.base {
				t.Fatalf("the new lease must be newly allocated, but got same address as base")
			}
			if !apiequality.Semantic.DeepEqual(tc.expect, newLease) {
				t.Errorf("unexpected result from newLease: %s", diff.ObjectDiff(tc.expect, newLease))
			}
		})
	}
}

func TestRetryUpdateLease(t *testing.T) {
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
	}{
		{
			desc: "no errors",
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, &coordinationv1.Lease{}, nil
			},
			getReactor:                 nil,
			onRepeatedHeartbeatFailure: nil,
			expectErr:                  false,
		},
		{
			desc: "connection errors",
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, nil, noConnectionUpdateErr
			},
			getReactor:                 nil,
			onRepeatedHeartbeatFailure: nil,
			expectErr:                  true,
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
		},
	}
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			cl := fake.NewSimpleClientset(node)
			if tc.updateReactor != nil {
				cl.PrependReactor("update", "leases", tc.updateReactor)
			}
			if tc.getReactor != nil {
				cl.PrependReactor("get", "leases", tc.getReactor)
			}
			c := &controller{
				clock:                      clock.NewFakeClock(time.Now()),
				client:                     cl,
				leaseClient:                cl.CoordinationV1().Leases(corev1.NamespaceNodeLease),
				holderIdentity:             node.Name,
				leaseDurationSeconds:       10,
				onRepeatedHeartbeatFailure: tc.onRepeatedHeartbeatFailure,
			}
			if err := c.retryUpdateLease(nil); tc.expectErr != (err != nil) {
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
		latestLease                *coordinationv1.Lease
		updateReactor              func(action clienttesting.Action) (bool, runtime.Object, error)
		getReactor                 func(action clienttesting.Action) (bool, runtime.Object, error)
		createReactor              func(action clienttesting.Action) (bool, runtime.Object, error)
		expectLeaseResourceVersion string
	}{
		{
			desc:          "latestLease is nil and need to create",
			latestLease:   nil,
			updateReactor: nil,
			getReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, nil, notFoundErr
			},
			createReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "1"), nil
			},
			expectLeaseResourceVersion: "1",
		},
		{
			desc:        "latestLease is nil and need to update",
			latestLease: nil,
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "2"), nil
			},
			getReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "1"), nil
			},
			expectLeaseResourceVersion: "2",
		},
		{
			desc:        "latestLease exist and need to update",
			latestLease: makeLease(nodeName, "1"),
			updateReactor: func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, makeLease(nodeName, "2"), nil
			},
			expectLeaseResourceVersion: "2",
		},
		{
			desc:        "update with latest lease failed",
			latestLease: makeLease(nodeName, "1"),
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
			expectLeaseResourceVersion: "3",
		},
	}
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			cl := fake.NewSimpleClientset(node)
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
				clock:                clock.NewFakeClock(time.Now()),
				client:               cl,
				leaseClient:          cl.CoordinationV1().Leases(corev1.NamespaceNodeLease),
				holderIdentity:       node.Name,
				leaseDurationSeconds: 10,
				latestLease:          tc.latestLease,
			}

			c.sync()

			if tc.expectLeaseResourceVersion != c.latestLease.ResourceVersion {
				t.Fatalf("latestLease RV got %v, expected %v", c.latestLease.ResourceVersion, tc.expectLeaseResourceVersion)
			}
		})
	}
}
