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
	"testing"
	"time"

	coordv1beta1 "k8s.io/api/coordination/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/client-go/kubernetes/fake"
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
		base       *coordv1beta1.Lease
		expect     *coordv1beta1.Lease
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
			expect: &coordv1beta1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
				},
				Spec: coordv1beta1.LeaseSpec{
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
			expect: &coordv1beta1.Lease{
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
				Spec: coordv1beta1.LeaseSpec{
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
			base: &coordv1beta1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      node.Name,
					Namespace: corev1.NamespaceNodeLease,
				},
				Spec: coordv1beta1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-10 * time.Second)},
				},
			},
			expect: &coordv1beta1.Lease{
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
				Spec: coordv1beta1.LeaseSpec{
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
			base: &coordv1beta1.Lease{
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
				Spec: coordv1beta1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr(node.Name),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-10 * time.Second)},
				},
			},
			expect: &coordv1beta1.Lease{
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
				Spec: coordv1beta1.LeaseSpec{
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
