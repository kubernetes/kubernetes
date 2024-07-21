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

package validation

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/utils/ptr"
)

func TestValidateLease(t *testing.T) {
	lease := &coordination.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "invalidName++",
			Namespace: "==invalid_Namespace==",
		},
	}
	errs := ValidateLease(lease)
	if len(errs) != 2 {
		t.Errorf("unexpected list of errors: %#v", errs.ToAggregate().Error())
	}
}

func TestValidateLeaseSpec(t *testing.T) {
	holder := "holder"
	leaseDuration := int32(0)
	leaseTransitions := int32(-1)
	preferredHolder := "holder2"

	testcases := []struct {
		spec coordination.LeaseSpec
		err  bool
	}{
		{
			// valid
			coordination.LeaseSpec{
				HolderIdentity:       &holder,
				LeaseDurationSeconds: ptr.To[int32](10),
				LeaseTransitions:     ptr.To[int32](1),
			},
			false,
		},
		{
			// valid with PreferredHolder
			coordination.LeaseSpec{
				HolderIdentity:       &holder,
				LeaseDurationSeconds: ptr.To[int32](10),
				LeaseTransitions:     ptr.To[int32](1),
				Strategy:             ptr.To(coordination.OldestEmulationVersion),
				PreferredHolder:      ptr.To("someotherholder"),
			},
			false,
		},
		{
			coordination.LeaseSpec{
				HolderIdentity:       &holder,
				LeaseDurationSeconds: &leaseDuration,
				LeaseTransitions:     &leaseTransitions,
			},
			true,
		},
		{
			coordination.LeaseSpec{
				HolderIdentity:       &holder,
				LeaseDurationSeconds: &leaseDuration,
				LeaseTransitions:     &leaseTransitions,
				PreferredHolder:      &preferredHolder,
			},
			true,
		},
	}

	for _, tc := range testcases {
		errs := ValidateLeaseSpec(&tc.spec, field.NewPath("foo"))
		if tc.err && len(errs) == 0 {
			t.Error("Expected err, got no err")
		} else if !tc.err && len(errs) != 0 {
			t.Errorf("Expected no err, got err %v", errs)
		}
	}
}

func TestValidateLeaseSpecUpdate(t *testing.T) {
	holder := "holder"
	leaseDuration := int32(0)
	leaseTransitions := int32(-1)
	lease := &coordination.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "holder",
			Namespace: "holder-namespace",
		},
		Spec: coordination.LeaseSpec{
			HolderIdentity:       &holder,
			LeaseDurationSeconds: &leaseDuration,
			LeaseTransitions:     &leaseTransitions,
		},
	}
	oldHolder := "oldHolder"
	oldLeaseDuration := int32(3)
	oldLeaseTransitions := int32(3)
	oldLease := &coordination.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "holder",
			Namespace: "holder-namespace",
		},
		Spec: coordination.LeaseSpec{
			HolderIdentity:       &oldHolder,
			LeaseDurationSeconds: &oldLeaseDuration,
			LeaseTransitions:     &oldLeaseTransitions,
		},
	}
	errs := ValidateLeaseUpdate(lease, oldLease)
	if len(errs) != 3 {
		t.Errorf("unexpected list of errors: %#v", errs.ToAggregate().Error())
	}

	validLeaseDuration := int32(10)
	validLeaseTransitions := int32(20)
	validLease := &coordination.Lease{
		ObjectMeta: oldLease.ObjectMeta,
		Spec: coordination.LeaseSpec{
			HolderIdentity:       &holder,
			LeaseDurationSeconds: &validLeaseDuration,
			LeaseTransitions:     &validLeaseTransitions,
		},
	}
	validLease.ObjectMeta.ResourceVersion = "2"
	errs = ValidateLeaseUpdate(validLease, oldLease)
	if len(errs) != 0 {
		t.Errorf("unexpected list of errors for valid update: %#v", errs.ToAggregate().Error())
	}
}

func TestValidateLeaseCandidate(t *testing.T) {
	lease := &coordination.LeaseCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "invalidName++",
			Namespace: "==invalid_Namespace==",
		},
	}
	errs := ValidateLeaseCandidate(lease)
	if len(errs) == 0 {
		t.Errorf("expected invalid LeaseCandidate")
	}
}

func TestValidateLeaseCandidateSpec(t *testing.T) {
	testcases := []struct {
		name      string
		shouldErr bool
		spec      *coordination.LeaseCandidateSpec
	}{
		{
			"valid",
			false,
			&coordination.LeaseCandidateSpec{
				BinaryVersion:       "1.30.0",
				EmulationVersion:    "1.30.0",
				LeaseName:           "test",
				PreferredStrategies: []coordination.CoordinatedLeaseStrategy{coordination.OldestEmulationVersion},
			},
		},
		{
			"valid custom strategy should not require binaryVersion and emulationVersion",
			false,
			&coordination.LeaseCandidateSpec{
				LeaseName:           "test",
				PreferredStrategies: []coordination.CoordinatedLeaseStrategy{"custom.com/foo"},
			},
		},

		{
			"no lease name",
			true,
			&coordination.LeaseCandidateSpec{
				EmulationVersion: "1.30.0",
			},
		},
		{
			"bad binaryVersion",
			true,
			&coordination.LeaseCandidateSpec{
				BinaryVersion: "1.30.1.6",
				LeaseName:     "test",
			},
		},
		{
			"emulation should be greater than or equal to binary version",
			true,
			&coordination.LeaseCandidateSpec{
				EmulationVersion: "1.30.0",
				BinaryVersion:    "1.29.0",
				LeaseName:        "test",
			},
		},
		{
			"preferredStrategies bad",
			true,
			&coordination.LeaseCandidateSpec{
				BinaryVersion:       "1.30.1",
				EmulationVersion:    "1.30.1",
				LeaseName:           "test",
				PreferredStrategies: []coordination.CoordinatedLeaseStrategy{"foo"},
			},
		},
		{
			"preferredStrategies good but emulationVersion missing",
			true,
			&coordination.LeaseCandidateSpec{
				BinaryVersion:       "1.30.1",
				LeaseName:           "test",
				PreferredStrategies: []coordination.CoordinatedLeaseStrategy{coordination.OldestEmulationVersion},
			},
		},
	}

	for _, tc := range testcases {
		errs := ValidateLeaseCandidateSpec(tc.spec, field.NewPath("foo"))
		if len(errs) > 0 && !tc.shouldErr {
			t.Errorf("unexpected list of errors: %#v", errs.ToAggregate().Error())
		} else if len(errs) == 0 && tc.shouldErr {
			t.Errorf("Expected err, got no error for tc: %s", tc.name)
		}
	}
}

func TestValidateLeaseCandidateUpdate(t *testing.T) {
	testcases := []struct {
		name   string
		old    coordination.LeaseCandidate
		update coordination.LeaseCandidate
		err    bool
	}{
		{
			name: "valid update",
			old: coordination.LeaseCandidate{
				Spec: coordination.LeaseCandidateSpec{
					BinaryVersion:    "1.30.0",
					EmulationVersion: "1.30.0",
					LeaseName:        "test",
				},
			},
			update: coordination.LeaseCandidate{
				Spec: coordination.LeaseCandidateSpec{
					BinaryVersion:    "1.30.0",
					EmulationVersion: "1.30.0",
					LeaseName:        "test",
				},
			},
			err: false,
		},
		{
			name: "update LeaseName should fail",
			old: coordination.LeaseCandidate{
				Spec: coordination.LeaseCandidateSpec{
					BinaryVersion:    "1.30.0",
					EmulationVersion: "1.30.0",
					LeaseName:        "test",
				},
			},
			update: coordination.LeaseCandidate{
				Spec: coordination.LeaseCandidateSpec{
					BinaryVersion:    "1.30.0",
					EmulationVersion: "1.30.0",
					LeaseName:        "test-update",
				},
			},
			err: true,
		},
	}

	for _, tc := range testcases {
		tc.old.ResourceVersion = "1"
		tc.update.ResourceVersion = "1"
		errs := ValidateLeaseCandidateUpdate(&tc.update, &tc.old)
		if tc.err && len(errs) == 0 {
			t.Errorf("Expected err, got no err for tc: %s", tc.name)
		} else if !tc.err && len(errs) != 0 {
			t.Errorf("Expected no err, got err %v for tc: %s", errs, tc.name)

		}
	}
}

func TestValidateCoordinatedLeaseStrategy(t *testing.T) {
	testcases := []struct {
		strategy coordination.CoordinatedLeaseStrategy
		err      bool
	}{
		{
			coordination.CoordinatedLeaseStrategy("foobar"),
			true,
		},
		{
			coordination.CoordinatedLeaseStrategy("example.com/foobar/toomanyslashes"),
			true,
		},
		{

			coordination.CoordinatedLeaseStrategy(coordination.OldestEmulationVersion),
			false,
		},
		{
			coordination.CoordinatedLeaseStrategy("example.com/foobar"),
			false,
		},
	}

	for _, tc := range testcases {
		errs := ValidateCoordinatedLeaseStrategy(tc.strategy, field.NewPath("foo"))
		if tc.err && len(errs) == 0 {
			t.Error("Expected err, got no err")
		} else if !tc.err && len(errs) != 0 {
			t.Errorf("Expected no err, got err %v", errs)
		}
	}
}
