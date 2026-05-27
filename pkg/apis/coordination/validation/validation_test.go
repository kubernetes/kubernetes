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
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/coordination"
	utilsclock "k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"
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
				BinaryVersion:    "1.30.0",
				EmulationVersion: "1.30.0",
				LeaseName:        "test",
				Strategy:         coordination.OldestEmulationVersion,
			},
		},
		{
			"valid custom strategy should not require emulationVersion",
			false,
			&coordination.LeaseCandidateSpec{
				LeaseName:     "test",
				BinaryVersion: "1.30.0",
				Strategy:      coordination.CoordinatedLeaseStrategy("custom.com/foo"),
			},
		},
		{
			"binaryVersion is required",
			true,
			&coordination.LeaseCandidateSpec{
				LeaseName: "test",
				Strategy:  coordination.CoordinatedLeaseStrategy("custom.com/foo"),
			},
		},
		{
			"no lease name",
			true,
			&coordination.LeaseCandidateSpec{
				BinaryVersion:    "1.30.0",
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
			"strategy bad",
			true,
			&coordination.LeaseCandidateSpec{
				BinaryVersion:    "1.30.1",
				EmulationVersion: "1.30.1",
				LeaseName:        "test",
				Strategy:         coordination.CoordinatedLeaseStrategy("foo"),
			},
		},
		{
			"strategy missing",
			true,
			&coordination.LeaseCandidateSpec{
				BinaryVersion:    "1.30.1",
				EmulationVersion: "1.30.1",
				LeaseName:        "test",
			},
		},
		{
			"strategy good but emulationVersion missing",
			true,
			&coordination.LeaseCandidateSpec{
				BinaryVersion: "1.30.1",
				LeaseName:     "test",
				Strategy:      coordination.OldestEmulationVersion,
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
					Strategy:         coordination.OldestEmulationVersion,
				},
			},
			update: coordination.LeaseCandidate{
				Spec: coordination.LeaseCandidateSpec{
					BinaryVersion:    "1.30.0",
					EmulationVersion: "1.30.0",
					LeaseName:        "test",
					Strategy:         coordination.OldestEmulationVersion,
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
					Strategy:         coordination.OldestEmulationVersion,
				},
			},
			update: coordination.LeaseCandidate{
				Spec: coordination.LeaseCandidateSpec{
					BinaryVersion:    "1.30.0",
					EmulationVersion: "1.30.0",
					LeaseName:        "test-update",
					Strategy:         coordination.OldestEmulationVersion,
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

const valiUIDName = "bd23f542-ac79-4b44-a628-9735e18b8037"

func TestValidateEviction(t *testing.T) {
	successCases := map[string]struct {
		input *coordination.Eviction
	}{
		"valid: pod target": {
			input: mkValidEviction(),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateEviction(tc.input)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
			}
		})
	}

	failureCases := map[string]struct {
		input *coordination.Eviction

		errors []*field.Error
	}{
		"name is not valid": {
			input: mkValidEviction(setName("-invalid-name-test", "")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("metadata", "name"), "", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"),
			},
		},
		"missing namespace": {
			input: mkValidEviction(setNamespace("")),
			errors: []*field.Error{
				field.Required(field.NewPath("metadata", "namespace"), ""),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateEviction(tc.input)
			if len(errs) == 0 {
				t.Errorf("Expected failure")
				return
			}
			if len(errs) != len(tc.errors) {
				t.Errorf("Expected %d errors, got %d: %v", len(tc.errors), len(errs), errs)
				return
			}
			matcher := field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring()
			matcher.Test(t, tc.errors, errs)

			for i, err := range errs {
				expectedErr := tc.errors[i]
				if err.CoveredByDeclarative != expectedErr.CoveredByDeclarative {
					t.Errorf("Error %d: expected CoveredByDeclarative=%v, got %v for error: %v",
						i, expectedErr.CoveredByDeclarative, err.CoveredByDeclarative, err)
				}
			}
		})
	}
}

func TestValidateEvictionUpdate(t *testing.T) {
	successCases := map[string]struct {
		input    *coordination.Eviction
		oldInput *coordination.Eviction
	}{
		"set requester annotation": {
			oldInput: mkValidEviction(),
			input:    mkValidEviction(addAnnotation("acme.io/foo", "requester")),
		},
		"update requester annotations": {
			oldInput: mkValidEviction(addAnnotation("acme.io/foo", "requester")),
			input:    mkValidEviction(addAnnotation("acme.io/foo", "requesterresponder")),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			tc.oldInput.ResourceVersion = "0"
			tc.input.ResourceVersion = "1"
			errs := ValidateEvictionUpdate(tc.input, tc.oldInput)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
				return
			}
		})
	}
}

func TestValidateEvictionStatusUpdate(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())
	clockBefore := func(duration time.Duration) utilsclock.PassiveClock {
		return testing2.NewFakePassiveClock(clock.Now().Add(-duration))
	}
	clockAfter := func(duration time.Duration) utilsclock.PassiveClock {
		return testing2.NewFakePassiveClock(clock.Now().Add(duration))
	}
	clock2 := clockAfter(5 * time.Second)

	successCases := map[string]struct {
		clock    utilsclock.PassiveClock
		input    *coordination.EvictionStatus
		oldInput *coordination.EvictionStatus
	}{
		// conditions
		"Evicted condition can be set": {
			oldInput: mkValidEvictionStatus(1),
			input:    mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionEvicted, true)),
		},
		"Evicted condition can be changed to true": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionEvicted, false)),
			input:    mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionEvicted, true)),
		},
		"Failed condition can be set": {
			oldInput: mkValidEvictionStatus(1),
			input:    mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionFailed, true)),
		},
		"Failed condition can be changed to true": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionFailed, false)),
			input:    mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionFailed, true)),
		},
		"Non terminal condition can be added when immutable condition exists": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionFailed, true)),
			input:    mkValidEvictionStatus(1, addCondition(clock, "NewCondition", true), addCondition(clock, coordination.EvictionConditionFailed, true)),
		},
		"Non terminal condition can be changed when immutable condition exists": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, "NewCondition", true), addCondition(clock, coordination.EvictionConditionFailed, true)),
			input:    mkValidEvictionStatus(1, addCondition(clock, "NewCondition", false), addCondition(clock, coordination.EvictionConditionFailed, true)),
		},
		"A new condition can be added when observedGeneration is nil": {
			oldInput: mkValidEvictionStatus(0, setObservedGeneration(nil), addCondition(clock, "NewCondition", true), addCondition(clock, coordination.EvictionConditionFailed, true)),
			input:    mkValidEvictionStatus(0, setObservedGeneration(nil), addCondition(clock, "NewCondition", false), addCondition(clock, coordination.EvictionConditionFailed, true)),
		},
		// observedGeneration
		"set initial generation": {
			oldInput: &coordination.EvictionStatus{},
			input:    mkValidEvictionStatus(0),
		},
		"update generation": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](5))),
		},
		// requesters
		"initialize requester with eviction intent": {
			oldInput: mkValidEvictionStatus(0, clearRequesters()),
			input:    mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
		},
		"initialize requester with withdrawn intent": {
			oldInput: mkValidEvictionStatus(0, clearRequesters()),
			input:    mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentWithdrawn, "foo.example.com/baz")),
		},
		"change requester intent from eviction to withdrawn": {
			oldInput: mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentWithdrawn, "foo.example.com/baz")),
		},
		"change requester intent from withdrawn to eviction": {
			oldInput: mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentWithdrawn, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
		},
		"add a second requester with eviction intent": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
		},
		"change a last requester eviction intent to withdrawn": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentWithdrawn, "foo.example.com/bar")),
		},
		// first sync
		"responders initialization": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(3),
		},
		"16 responders": { // 1 more than Pod's spec.evictionResponders
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(16),
		},
		// Full Eviction progression/lifecycle
		"mark active and started": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
		},
		"responder reports progress": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
		},
		"responder reports next progress": {
			clock: clockAfter(time.Minute + 5*time.Second),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(time.Minute+5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
		},
		"responder reports complete": {
			clock: clockAfter(5 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(time.Minute+5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
		},
		"move active complete to processed and set new active": {
			clock: clockAfter(5*time.Minute + 27*time.Second),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
		},
		"next responder reports next progress": {
			clock: clockAfter(25 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setRespondersHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
		},
		"next responder reports complete": {
			clock: clockAfter(28 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setRespondersHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
		},
		"finalize eviction request active": {
			clock: clockAfter(35 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
			input: mkValidEvictionStatus(2,
				setCompletedCount(2),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
		},
		"finalize eviction request active - large number of responders": {
			clock: clockAfter(5 * time.Minute),
			oldInput: mkValidEvictionStatus(16,
				setStateFor(coordination.ResponderStateActive, 15),
				setCompletedCount(15),
				setRespondersFullStatus(clockAfter(time.Minute), clockAfter(2*time.Minute), 0, 7),
				setRespondersFullStatus(clockAfter(3*time.Minute), clockAfter(4*time.Minute), 7, 16)),
			input: mkValidEvictionStatus(16,
				setCompletedCount(16),
				setRespondersFullStatus(clockAfter(time.Minute), clockAfter(2*time.Minute), 0, 7),
				setRespondersFullStatus(clockAfter(3*time.Minute), clockAfter(4*time.Minute), 7, 16)),
		},
		// allowed update after the heartbeat deadline
		"processedResponders can be updated when the deadline (start time fallback) is reached": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+32*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+32*time.Second), 0, 1)),
		},
		"processedResponders can be updated when the deadline is reached": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Hour), 0, 1),
				setRespondersHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(time.Hour), 0, 1),
				setRespondersHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
		},
		// startTime
		"startTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(29*time.Second), 0, 1)),
		},
		"responder startTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockAfter(27*time.Second), 0, 1)),
		},
		// heartbeatTime
		"heartbeatTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(27*time.Second), 0, 1)),
		},
		"heartbeatTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockAfter(27*time.Second), 0, 1)),
		},
		"heartbeatTime can be updated after one minute": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
		},
		// expectedCompletionTime
		"expectedCompletionTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1),
				setRespondersExpectedCompletionTime(clockBefore(27*time.Second), 0, 1)),
		},
		"expectedCompletionTime can be set to the future": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(time.Hour*24*365*10-time.Minute), 0, 1)),
		},
		// completionTime
		"completionTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(27*time.Second), 0, 1)),
		},
		// processedResponders + conditions
		"processedResponders must allow active responder transition even with an exceeded deadline (start time fallback) - Failed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionConditionFailed, true)),
		},
		"processedResponders must allow active responder transition even with an exceeded deadline (start time fallback) - Evicted": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionConditionEvicted, true)),
		},
		"processedResponders must allow active responder transition even with an exceeded deadline - Failed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionConditionFailed, true)),
		},
		"processedResponders must allow active responder transition even with an exceeded deadline - Evicted": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionConditionEvicted, true)),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			if validate.SemanticDeepEqual(tc.oldInput, tc.input) {
				t.Errorf("Expected oldInput and input to differ")
			}
			oldEviction := mkValidEviction()
			oldEviction.ResourceVersion = "0"
			oldEviction.Status = *tc.oldInput
			eviction := mkValidEviction()
			eviction.ResourceVersion = "1"
			eviction.Status = *tc.input
			opts := EvictionStatusValidationOptions{clock}
			if tc.clock != nil {
				opts.Clock = tc.clock
			}
			errs := ValidateEvictionStatusUpdate(eviction, oldEviction, opts)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
			}
		})
	}

	failureCases := map[string]struct {
		input    *coordination.EvictionStatus
		oldInput *coordination.EvictionStatus

		errors []*field.Error
	}{
		// conditions
		"add invalid condition": {
			oldInput: mkValidEvictionStatus(1),
			input: mkValidEvictionStatus(1, func(obj *coordination.EvictionStatus) {
				obj.Conditions = append(obj.Conditions, metav1.Condition{
					Type:               "-bad-name",
					Status:             "invalid",
					ObservedGeneration: -1,
					Reason:             "-Reason",
				})
			}),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("type"), "", "name part must consist of alphanumeric characters, '-', '_' or '.', and "),
				field.NotSupported(field.NewPath("status", "conditions").Index(0).Child("status"), "", []string{"False", "True", "Unknown"}),
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("observedGeneration"), "", "must be greater than or equal to zero"),
				field.Required(field.NewPath("status", "conditions").Index(0).Child("lastTransitionTime"), ""),
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("reason"), "", "a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and "),
			},
		},
		"Evicted condition cannot be removed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionEvicted, true)),
			input:    mkValidEvictionStatus(1),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition status cannot be reverted"),
			},
		},
		"Evicted condition cannot be changed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionEvicted, true)),
			input:    mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionEvicted, false)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition status cannot be reverted"),
			},
		},
		"Failed condition cannot be removed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionFailed, true)),
			input:    mkValidEvictionStatus(1),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Failed condition status cannot be reverted"),
			},
		},
		"Failed condition cannot be changed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionFailed, true)),
			input:    mkValidEvictionStatus(1, addCondition(clock, coordination.EvictionConditionFailed, false)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Failed condition status cannot be reverted"),
			},
		},
		// observedGeneration
		"clear generation": {
			oldInput: mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](1))),
			input:    mkValidEvictionStatus(0, setObservedGeneration(nil)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), 0, "cannot decrement, must be greater than or equal to 1"),
			},
		},
		"decrease generation": {
			oldInput: mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](2))),
			input:    mkValidEvictionStatus(0, setObservedGeneration(ptr.To[int64](1))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), 1, "cannot decrement, must be greater than or equal to 2"),
			},
		},
		// requesters
		"requester with withdrawn intent cannot be removed": {
			oldInput: mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentWithdrawn, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, clearRequesters()),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "requesters"), "", "requesters cannot be removed"),
			},
		},
		"requester with eviction intent cannot be removed": {
			oldInput: mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, clearRequesters()),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "requesters"), "", "requesters cannot be removed"),
			},
		},
		"secondary requester cannot be removed": {
			oldInput: mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "requesters"), "", "requesters cannot be removed"),
			},
		},
		"change and remove a requester": {
			oldInput: mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentEviction, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, clearRequesters(), addRequesters(coordination.RequesterIntentEviction, "bar.example.com/bay")),
			errors: field.ErrorList{
				field.Invalid(field.NewPath("status", "requesters"), "", "requesters cannot be removed"),
			},
		},
		"requester with non conformant name cannot be added": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addRequesters(coordination.RequesterIntentEviction, "foo.example.com")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "requesters").Index(0).Child("name"), "", "must be a domain-prefixed key"),
			},
		},
		// targetResponders and responders
		"cannot change keys in targetResponders and responders - remove": {
			oldInput: mkValidEvictionStatus(5),
			input:    mkValidEvictionStatus(4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders"), "", "must preserve the same length and the same keys in the same order"),
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order"),
			},
		},
		"cannot change keys in targetResponders and responders - add": {
			oldInput: mkValidEvictionStatus(4),
			input:    mkValidEvictionStatus(5),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders"), "", "must preserve the same length and the same keys in the same order"),
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order"),
			},
		},
		"invalid targetResponders and responders": { // required and duplicate is tested in the declarative validation test
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(1, addTargetAndStatusResponders("test.net", "invalid", "/invalid", "underscores_are_bad.k8s.io/bar", "foo.example.com/bar")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("name"), "test.net", "must be a domain-prefixed key"),
				field.Invalid(field.NewPath("status", "targetResponders").Index(2).Child("name"), "invalid", "must be a domain-prefixed key"),
				field.Invalid(field.NewPath("status", "targetResponders").Index(3).Child("name"), "/invalid", "prefix part must be non-empty"),
				field.Invalid(field.NewPath("status", "targetResponders").Index(4).Child("name"), "underscores_are_bad.k8s.io/ba", "prefix part a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters"),
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order"), // triggered by fallback to oldTargetResponders
			},
		},
		// targetResponders
		"cannot change keys in targetResponders - change": {
			oldInput: mkValidEvictionStatus(4),
			input:    mkValidEvictionStatus(3, addTargetResponders("foo/bar"), addStatusResponders(responderName(3))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders"), "", "must preserve the same keys in the same order"),
			},
		},
		"targetResponders must have matching status responder": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addTargetResponders("foo/bar")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0), "", "has to be tracked in status.responders first"),
			},
		},
		"Active responder must exceed the deadline (start time fallback) before it is marked Completed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "targetResponders").Index(0).Child("state"), "must stay Active because the responder is in progress"),
			},
		},
		"Active responder must exceed the deadline before it is marked Completed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(coordination.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "targetResponders").Index(0).Child("state"), "must stay Active because the responder is in progress"),
			},
		},
		// responders
		"responders must be initialized at the same time as targetResponders": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatusWithStatuses(0, 3),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order"),
			},
		},
		"cannot change keys in responders - change": {
			oldInput: mkValidEvictionStatus(4),
			input:    mkValidEvictionStatus(3, addTargetResponders(responderName(3)), addStatusResponders("foo/bar")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(3), "", "has to be tracked in status.responders first"),
				field.Invalid(field.NewPath("status", "responders"), "", "must contain the same keys in the same order as status.targetResponders"),
			},
		},
		"status responders cannot remove an item": {
			oldInput: mkValidEvictionStatus(5),
			input:    mkValidEvictionStatusWithStatuses(5, 4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(4), "", "has to be tracked in status.responders first"),
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order"),
			},
		},
		"status responders cannot be mutated unless present in active responders": {
			oldInput: mkValidEvictionStatus(5),
			input: mkValidEvictionStatus(5,
				setRespondersHeartBeatTime(clock, 1, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(1), "", validation.FieldImmutableErrorMsg).WithOrigin("immutable"),
			},
		},
		// status responder name
		"invalid status responders names - short circuited by targetResponders key order": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatusWithStatuses(1, 0,
				addStatusResponders("foo")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0), "", "has to be tracked in status.responders first"),
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys in the same order"),
			},
		},
		// startTime
		"startTime is required for an active responder": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setStateFor(coordination.ResponderStateActive, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(1).Child("startTime"), "is required for an active responder"),
			},
		},
		"startTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionStatus(2, setRespondersFullStatus(clock, clock2, 0, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setStateFor(coordination.ResponderStateActive, 1),
				setRespondersStartTime(clockBefore(31*time.Second), 1, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(1).Child("startTime"), "", "must be set to the present time"),
			},
		},
		"startTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionStatus(2, setRespondersFullStatus(clock, clock2, 0, 1),
				setStateFor(coordination.ResponderStateCompleted, 0),
				setStateFor(coordination.ResponderStateActive, 1),
				setRespondersStartTime(clockAfter(31*time.Second), 1, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(1).Child("startTime"), "", "must be set to the present time"),
			},
		},
		// heartbeatTime
		"cannot remove heartbeatTime when previously set": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "is required once set"),
			},
		},
		"heartbeatTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("startTime"), "is required for an active responder"),
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "", "cannot be set before status.responders[0].startTime"),
			},
		},
		"heartbeatTime cannot be decreased": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Second), 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Second), 0, 1),
				setRespondersHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "", "cannot be decreased"),
			},
		},
		"heartbeatTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "", "must occur after status.responders[0].startTime"),
			},
		},
		"heartbeatTime must be set to the present time with skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "", "must be set to the present time"),
			},
		},
		// expectedCompletionTime
		"expectedCompletionTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("startTime"), "is required for an active responder"),
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("expectedCompletionTime"), "", "cannot be set before status.responders[0].startTime"),
			},
		},
		"expectedCompletionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("expectedCompletionTime"), "", "must occur after status.responders[0].startTime"),
			},
		},
		"expectedCompletionTime cannot be set to the past with skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(32*time.Second), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(32*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("expectedCompletionTime"), "", "cannot be set to the past time"),
			},
		},
		"expectedCompletionTime must complete within 10 years": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(time.Hour*24*365*10+time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("expectedCompletionTime"), "", "must complete within 10 years"),
			},
		},
		// completionTime
		"completionTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("startTime"), "is required for an active responder"),
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), "", "cannot be set before status.responders[0].startTime"),
			},
		},
		"completionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), "", "must occur after status.responders[0].startTime"),
			},
		},
		"completionTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), "", "must be set to the present time"),
			},
		},
		"completionTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(1*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(coordination.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(1*time.Minute), 0, 1),
				setRespondersCompletionTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), "", "must be set to the present time"),
			},
		},
		// message
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			oldEviction := mkValidEviction()
			oldEviction.ResourceVersion = "0"
			oldEviction.Status = *tc.oldInput
			eviction := mkValidEviction()
			eviction.ResourceVersion = "1"
			eviction.Status = *tc.input
			errs := ValidateEvictionStatusUpdate(eviction, oldEviction, EvictionStatusValidationOptions{clock})
			if len(errs) == 0 {
				t.Errorf("Expected failure")
				return
			}
			if len(errs) != len(tc.errors) {
				errsFormated := "\n"
				for _, err := range errs {
					errsFormated += err.Error() + "\n"
				}
				t.Errorf("Expected %d errors, got %d: %v", len(tc.errors), len(errs), errsFormated)
				return
			}
			matcher := field.ErrorMatcher{}.ByType().ByField().ByDetailSubstring()
			matcher.Test(t, tc.errors, errs)

			for i, err := range errs {
				expectedErr := tc.errors[i]
				if err.CoveredByDeclarative != expectedErr.CoveredByDeclarative {
					t.Errorf("Error %d: expected CoveredByDeclarative=%v, got %v for error: %v",
						i, expectedErr.CoveredByDeclarative, err.CoveredByDeclarative, err)
				}
			}
		})
	}
}

func mkValidEviction(tweaks ...func(obj *coordination.Eviction)) *coordination.Eviction {
	obj := coordination.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDName, Namespace: "foo"},
		Spec: coordination.EvictionSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.EvictionPodReference{
					UID:  valiUIDName,
					Name: "foo.pod",
				},
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}

func setName(name, generateName string) func(obj *coordination.Eviction) {
	return func(obj *coordination.Eviction) {
		obj.Name = name
		obj.GenerateName = generateName
	}
}
func setNamespace(namespace string) func(obj *coordination.Eviction) {
	return func(obj *coordination.Eviction) {
		obj.Namespace = namespace
	}
}
func addAnnotation(key, value string) func(obj *coordination.Eviction) {
	return func(obj *coordination.Eviction) {
		if obj.Labels == nil {
			obj.Labels = make(map[string]string)
		}
		obj.Labels[key] = value
	}
}

func responderName(i int) string {
	return fmt.Sprintf("responder.example.com/bar%d", i)
}
func mkValidEvictionStatus(responders int, tweaks ...func(obj *coordination.EvictionStatus)) *coordination.EvictionStatus {
	return mkValidEvictionStatusWithStatuses(responders, responders, tweaks...)
}
func mkValidEvictionStatusWithStatuses(responders, statuses int, tweaks ...func(obj *coordination.EvictionStatus)) *coordination.EvictionStatus {
	obj := coordination.EvictionStatus{
		ObservedGeneration: ptr.To[int64](1),
	}
	for i := range responders {
		obj.TargetResponders = append(obj.TargetResponders, coordination.TargetResponder{
			Name:  responderName(i),
			State: coordination.ResponderStateInactive,
		})
		if i == 0 {
			obj.TargetResponders[i].State = coordination.ResponderStateActive
		}
	}
	for i := range statuses {
		obj.Responders = append(obj.Responders, coordination.ResponderStatus{
			Name: responderName(i),
		})
		if i == 0 {
			obj.Responders[i].StartTime = ptr.To(metav1.Now())
		}
	}
	addRequesters(coordination.RequesterIntentEviction, "foo.example.com/bar")
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}
func addCondition(clock utilsclock.PassiveClock, name coordination.EvictionConditionType, status bool) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		newCond := metav1.Condition{
			Type:               string(name),
			Status:             metav1.ConditionFalse,
			Reason:             string(name) + "Reason",
			LastTransitionTime: metav1.Time{Time: clock.Now()},
		}
		if status {
			newCond.Status = metav1.ConditionTrue
		}
		obj.Conditions = append(obj.Conditions, newCond)
	}
}

func setObservedGeneration(generation *int64) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		obj.ObservedGeneration = generation
	}
}
func addTargetResponders(responders ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for _, responderName := range responders {
			obj.TargetResponders = append(obj.TargetResponders, coordination.TargetResponder{Name: responderName, State: coordination.ResponderStateInactive})
		}
	}
}
func clearRequesters() func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		obj.Requesters = nil
	}
}
func addRequesters(intent coordination.RequesterIntent, names ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for _, name := range names {
			obj.Requesters = append(obj.Requesters, coordination.Requester{Name: name, Intent: intent})
		}
	}
}
func addTargetAndStatusResponders(responders ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		addTargetResponders(responders...)(obj)
		addStatusResponders(responders...)(obj)
	}
}
func setStateFor(state coordination.ResponderStateType, idx int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		obj.TargetResponders[idx].State = state
	}
}
func setCompletedCount(count int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := range count {
			if i < len(obj.TargetResponders) {
				obj.TargetResponders[i].State = coordination.ResponderStateCompleted
			}
		}
	}
}
func addStatusResponders(responders ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for _, responder := range responders {
			obj.Responders = append(obj.Responders, coordination.ResponderStatus{Name: responder})
		}
	}
}
func setRespondersStartTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].StartTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersHeartBeatTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].HeartbeatTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersExpectedCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].ExpectedCompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].CompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersMessage(from, to int, suffixes ...string) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].Message = fmt.Sprintf("message %d", i)
			for _, suffix := range suffixes {
				obj.Responders[i].Message += suffix
			}
		}
	}
}
func setRespondersFullStatus(startTimeClock, clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionStatus) {
	return func(obj *coordination.EvictionStatus) {
		setRespondersStartTime(startTimeClock, from, to)(obj)
		setRespondersHeartBeatTime(clock, from, to)(obj)
		setRespondersExpectedCompletionTime(clock, from, to)(obj)
		setRespondersCompletionTime(clock, from, to)(obj)
		setRespondersMessage(from, to, clock.Now().String())(obj)
	}
}
