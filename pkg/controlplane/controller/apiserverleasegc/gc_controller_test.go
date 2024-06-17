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

package apiserverleasegc

import (
	"context"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	testingclock "k8s.io/utils/clock/testing"
	"k8s.io/utils/pointer"
)

// Test_Controller validates the garbage collection logic for the apiserverleasegc controller.
func Test_Controller(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	tests := []struct {
		name          string
		lease         *coordinationv1.Lease
		expectDeleted bool
	}{
		{
			name: "lease not expired",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver-12345",
					Namespace: metav1.NamespaceSystem,
					Labels: map[string]string{
						"apiserver.kubernetes.io/identity": "kube-apiserver",
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("kube-apiserver-12345"),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now()},
				},
			},
			expectDeleted: false,
		},
		{
			name: "expired lease but with a different component label",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver-12345",
					Namespace: metav1.NamespaceSystem,
					Labels: map[string]string{
						"apiserver.kubernetes.io/identity": "kube-controller-manager",
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("kube-apiserver-12345"),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-time.Minute)},
				},
			},
			expectDeleted: false,
		},
		{
			name: "lease expired due to expired renew time",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver-12345",
					Namespace: metav1.NamespaceSystem,
					Labels: map[string]string{
						"apiserver.kubernetes.io/identity": "kube-apiserver",
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("kube-apiserver-12345"),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-time.Minute)},
				},
			},
			expectDeleted: true,
		},
		{
			name: "lease expired due to nil renew time",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver-12345",
					Namespace: metav1.NamespaceSystem,
					Labels: map[string]string{
						"apiserver.kubernetes.io/identity": "kube-apiserver",
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("kube-apiserver-12345"),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            nil,
				},
			},
			expectDeleted: true,
		},
		{
			name: "lease expired due to nil lease duration seconds",
			lease: &coordinationv1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "kube-apiserver-12345",
					Namespace: metav1.NamespaceSystem,
					Labels: map[string]string{
						"apiserver.kubernetes.io/identity": "kube-apiserver",
					},
				},
				Spec: coordinationv1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("kube-apiserver-12345"),
					LeaseDurationSeconds: nil,
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-time.Minute)},
				},
			},
			expectDeleted: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clientset := fake.NewSimpleClientset(test.lease)
			controller := NewAPIServerLeaseGC(clientset, 100*time.Millisecond, metav1.NamespaceSystem, "apiserver.kubernetes.io/identity=kube-apiserver")
			go controller.Run(nil)

			time.Sleep(time.Second)

			_, err := clientset.CoordinationV1().Leases(test.lease.Namespace).Get(context.TODO(), test.lease.Name, metav1.GetOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				t.Errorf("unexpected error %v", err)
			}

			if apierrors.IsNotFound(err) && !test.expectDeleted {
				t.Errorf("lease was not deleted")
			}

			if err == nil && test.expectDeleted {
				t.Error("lease was not deleted")
			}
		})
	}
}
