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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/utils/pointer"
)

func TestNewLease(t *testing.T) {
	fakeClock := clock.NewFakeClock(time.Now())
	cases := []struct {
		desc       string
		controller *controller
		base       *coordv1beta1.Lease
		expect     *coordv1beta1.Lease
	}{
		{
			desc: "nil base",
			controller: &controller{
				holderIdentity:       "foo",
				leaseDurationSeconds: 10,
				clock:                fakeClock,
			},
			base: nil,
			expect: &coordv1beta1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: coordv1beta1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("foo"),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now()},
				},
			},
		},
		{
			desc: "non-nil base renew time is updated",
			controller: &controller{
				holderIdentity:       "foo",
				leaseDurationSeconds: 10,
				clock:                fakeClock,
			},
			base: &coordv1beta1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: coordv1beta1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("foo"),
					LeaseDurationSeconds: pointer.Int32Ptr(10),
					RenewTime:            &metav1.MicroTime{Time: fakeClock.Now().Add(-10 * time.Second)},
				},
			},
			expect: &coordv1beta1.Lease{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
				Spec: coordv1beta1.LeaseSpec{
					HolderIdentity:       pointer.StringPtr("foo"),
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
