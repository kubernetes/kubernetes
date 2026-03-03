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
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/core"
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

const valiUIDdName = "bd23f542-ac79-4b44-a628-9735e18b8037"

func TestValidateEvictionRequest(t *testing.T) {
	successCases := map[string]struct {
		input *coordination.EvictionRequest
	}{
		"valid: one requester": {
			input: mkValidEvictionRequest(1),
		},
		"valid: two requesters": {
			input: mkValidEvictionRequest(2),
		},
		"valid: 100 requesters": {
			input: mkValidEvictionRequest(100),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateEvictionRequest(tc.input)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
			}
		})
	}

	failureCases := map[string]struct {
		input *coordination.EvictionRequest

		errors []*field.Error
	}{
		"name target uid mismatch": {
			input: mkValidEvictionRequest(1, setName("4fa67f6f-da60-4748-bccd-1525dab1bfee", "")),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid"),
			},
		},
		"name is not valid": {
			input: mkValidEvictionRequest(1, setName("invalid-name-test", "")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("metadata", "name"), "invalid-name-test", "must be UUID in RFC 4122 normalized form, `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with lowercase hexadecimal characters"),
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid"),
			},
		},
		"missing namespace": {
			input: mkValidEvictionRequest(1, setNamespace("")),
			errors: []*field.Error{
				field.Required(field.NewPath("metadata", "namespace"), ""),
			},
		},
		"generateName not supported": {
			input: mkValidEvictionRequest(1, setName("", "invalid-generate-name")),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("metadata", "generateName"), "").MarkCoveredByDeclarative(),
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid"),
				field.Required(field.NewPath("metadata", "name"), "name or generateName is required"),
			},
		},
		"generateName with name not supported": {
			input: mkValidEvictionRequest(1, setName(valiUIDdName, "invalid-generate-name")),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("metadata", "generateName"), "").MarkCoveredByDeclarative(),
			},
		},
		"missing target": {
			input: mkValidEvictionRequest(1, clearTarget()),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "target"), "invalid-name-test", "must specify one of: `pod`").MarkCoveredByDeclarative(),
			},
		},
		"missing target name": {
			input: mkValidEvictionRequest(1, setTarget("", valiUIDdName)),
			errors: []*field.Error{
				field.Required(field.NewPath("spec", "target", "pod", "name"), "").MarkCoveredByDeclarative(),
			},
		},
		"missing target uid": {
			input: mkValidEvictionRequest(1, setTarget("bar", "")),
			errors: []*field.Error{
				field.Required(field.NewPath("spec", "target", "pod", "uid"), "").MarkCoveredByDeclarative(),
			},
		},
		"invalid target uid": {
			input: mkValidEvictionRequest(1, setTarget("bar", "invalid-uid")),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("metadata", "name"), "must be the same value as spec.target.pod.uid"),
				field.Invalid(field.NewPath("spec", "target", "pod", "uid"), "invalid-uid", "must be UUID in RFC 4122 normalized form, `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` with lowercase hexadecimal characters"),
			},
		},
		"requesters are required": {
			input: mkValidEvictionRequest(0),
			errors: []*field.Error{
				field.Required(field.NewPath("spec", "requesters"), "must have at least one requester on EvictionRequest creation"),
			},
		},
		"too many requesters": {
			input: mkValidEvictionRequest(101),
			errors: []*field.Error{
				field.TooMany(field.NewPath("spec", "requesters"), 101, 100).MarkCoveredByDeclarative(),
			},
		},
		"duplicate requesters": {
			input: mkValidEvictionRequest(3, addRequesters("foo.example.com", "foo.example.com")),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("spec", "requesters").Index(4), "").MarkCoveredByDeclarative(),
			},
		},
		"requester without a name": {
			input: mkValidEvictionRequest(0, addRequesters("")),
			errors: []*field.Error{
				field.Required(field.NewPath("spec", "requesters").Index(0).Child("name"), "").MarkCoveredByDeclarative(),
			},
		},
		"invalid requester, 2 segments": {
			input: mkValidEvictionRequest(1, addRequesters("example.com")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "example.com", "should be a domain with at least three segments separated by dots"),
			},
		},
		"invalid requester, reserved domain": {
			input: mkValidEvictionRequest(1, addRequesters("requester.k8s.io")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "requester.k8s.io", "domain names *.k8s.io, *.kubernetes.io are reserved"),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			errs := ValidateEvictionRequest(tc.input)
			if len(errs) == 0 {
				t.Errorf("Expected failure")
				return
			}
			if len(errs) != len(tc.errors) {
				t.Errorf("Expected %d errors, got %d: %v", len(tc.errors), len(errs), errs)
				return
			}
			matcher := field.ErrorMatcher{}.ByType().ByField().ByDetailExact()
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

func TestValidateEvictionRequestUpdate(t *testing.T) {
	successCases := map[string]struct {
		input    *coordination.EvictionRequest
		oldInput *coordination.EvictionRequest
	}{
		"valid: decrease two to one requester": {
			oldInput: mkValidEvictionRequest(2),
			input:    mkValidEvictionRequest(1),
		},
		"valid: cancellation two to zero requesters": {
			oldInput: mkValidEvictionRequest(2),
			input:    mkValidEvictionRequest(0),
		},
		"valid: increase to 100 requesters": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(100),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			tc.oldInput.ResourceVersion = "0"
			tc.input.ResourceVersion = "1"
			errs := ValidateEvictionRequestUpdate(tc.input, tc.oldInput)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
			}
		})
	}

	failureCases := map[string]struct {
		input    *coordination.EvictionRequest
		oldInput *coordination.EvictionRequest

		errors []*field.Error
	}{
		"clear target": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, clearTarget()),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "target"), "", validation.FieldImmutableErrorMsg).MarkCoveredByDeclarative(),
			},
		},
		"change target name": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, setTarget("change", valiUIDdName)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "target"), "", validation.FieldImmutableErrorMsg).MarkCoveredByDeclarative(),
			},
		},
		"change target uid": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, setTarget("bar", "")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "target"), "", validation.FieldImmutableErrorMsg).MarkCoveredByDeclarative(),
			},
		},
		"increase requesters in a canceled eviction request": {
			oldInput: mkValidEvictionRequest(0),
			input:    mkValidEvictionRequest(1),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "requesters"), "", validation.FieldImmutableErrorMsg),
			},
		},
		"add too many requesters": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(101),
			errors: []*field.Error{
				field.TooMany(field.NewPath("spec", "requesters"), 101, 100).MarkCoveredByDeclarative(),
			},
		},
		"add a duplicate requesters": {
			oldInput: mkValidEvictionRequest(3, addRequesters("foo.example.com")),
			input:    mkValidEvictionRequest(3, addRequesters("foo.example.com", "foo.example.com")),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("spec", "requesters").Index(4), "").MarkCoveredByDeclarative(),
			},
		},
		"add a requester without a name": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, addRequesters("")),
			errors: []*field.Error{
				field.Required(field.NewPath("spec", "requesters").Index(1).Child("name"), "").MarkCoveredByDeclarative(),
			},
		},
		"add an invalid requester, 2 segments": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, addRequesters("example.com")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "example.com", "should be a domain with at least three segments separated by dots"),
			},
		},
		"add an invalid requester, reserved domain": {
			oldInput: mkValidEvictionRequest(1),
			input:    mkValidEvictionRequest(1, addRequesters("requester.k8s.io")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("spec", "requesters").Index(1).Child("name"), "requester.k8s.io", "domain names *.k8s.io, *.kubernetes.io are reserved"),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			tc.oldInput.ResourceVersion = "0"
			tc.input.ResourceVersion = "1"
			errs := ValidateEvictionRequestUpdate(tc.input, tc.oldInput)
			if len(errs) == 0 {
				t.Errorf("Expected failure")
				return
			}
			if len(errs) != len(tc.errors) {
				t.Errorf("Expected %d errors, got %d: %v", len(tc.errors), len(errs), errs)
				return
			}
			matcher := field.ErrorMatcher{}.ByType().ByField().ByDetailExact()
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

func TestValidateEvictionRequestStatusUpdate(t *testing.T) {
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
		input    *coordination.EvictionRequestStatus
		oldInput *coordination.EvictionRequestStatus
	}{
		"set initial generation": {
			oldInput: &coordination.EvictionRequestStatus{},
			input:    mkValidEvictionRequestStatus(0),
		},
		"update generation": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(5)),
		},
		"interceptors initialization": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(3),
		},
		"16 interceptors": { // 1 more than Pod's spec.evictionInterceptors
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(16),
		},
		"reserved names can be used in the status": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatus(0,
				addTargetInterceptors("foo.k8s.io", "foo.kubernetes.io"),
				addInterceptorStatusesNames("foo.k8s.io", "foo.kubernetes.io"),
				addActiveInterceptors("foo.k8s.io"),
				setInterceptorsStartTime(clock, 0, 1)),
		},
		// Full EvictionRequest progression/lifecycle
		"mark active and started": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 1)),
		},
		"interceptor reports progress": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setInterceptorsExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setInterceptorsMessage(0, 1)),
		},
		"interceptor reports next progress": {
			clock: clockAfter(time.Minute + 5*time.Second),
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setInterceptorsExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setInterceptorsMessage(0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clockAfter(time.Minute+5*time.Second), 0, 1),
				setInterceptorsExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setInterceptorsMessage(0, 1)),
		},
		"interceptor reports complete": {
			clock: clockAfter(5 * time.Minute),
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clockAfter(time.Minute+5*time.Second), 0, 1),
				setInterceptorsExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setInterceptorsMessage(0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
		},
		"move active complete to processed": {
			clock: clockAfter(5*time.Minute + 27*time.Second),
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
		},
		"next active interceptor when the old one has progressed": {
			clock: clockAfter(5*time.Minute + 30*time.Second),
			oldInput: mkValidEvictionRequestStatus(2,
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setInterceptorsStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
		},
		"next interceptor reports next progress": {
			clock: clockAfter(25 * time.Minute),
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setInterceptorsStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setInterceptorsStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setInterceptorsHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
		},
		"next interceptor reports complete": {
			clock: clockAfter(28 * time.Minute),
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setInterceptorsStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setInterceptorsHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setInterceptorsFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
		},
		"finalize eviction request active": {
			clock: clockAfter(35 * time.Minute),
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setInterceptorsFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
			input: mkValidEvictionRequestStatus(2,
				addProcessedInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setInterceptorsFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
		},
		"finalize eviction request active - large number of interceptors": {
			clock: clockAfter(5 * time.Minute),
			oldInput: mkValidEvictionRequestStatus(16,
				addActiveInterceptors(interceptorName(15)),
				addProcessedInterceptorsCount(15),
				setInterceptorsFullStatus(clockAfter(time.Minute), clockAfter(2*time.Minute), 0, 7),
				setInterceptorsFullStatus(clockAfter(3*time.Minute), clockAfter(4*time.Minute), 7, 16)),
			input: mkValidEvictionRequestStatus(16,
				addProcessedInterceptorsCount(16),
				setInterceptorsFullStatus(clockAfter(time.Minute), clockAfter(2*time.Minute), 0, 7),
				setInterceptorsFullStatus(clockAfter(3*time.Minute), clockAfter(4*time.Minute), 7, 16)),
		},
		// allowed update after the heartbeat deadline
		"processedInterceptors can be updated when the deadline (start time fallback) is reached": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+32*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+32*time.Second), 0, 1)),
		},
		"processedInterceptors can be updated when the deadline is reached": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Hour), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Hour), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
		},
		// startTime
		"startTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(29*time.Second), 0, 1)),
		},
		"interceptor startTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockAfter(27*time.Second), 0, 1)),
		},
		// heartbeatTime
		"heartbeatTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(27*time.Second), 0, 1)),
		},
		"heartbeatTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockAfter(27*time.Second), 0, 1)),
		},
		"heartbeatTime can be updated after one minute": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clock, 0, 1)),
		},
		// expectedCompletionTime
		"expectedCompletionTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1),
				setInterceptorsExpectedCompletionTime(clockBefore(27*time.Second), 0, 1)),
		},
		"expectedCompletionTime can be set to the future": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1),
				setInterceptorsExpectedCompletionTime(clockAfter(time.Hour*24*365*10-time.Minute), 0, 1)),
		},
		// completionTime
		"completionTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(27*time.Second), 0, 1)),
		},
		// message
		"message can be very long - truncated in the strategy": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 1), setInterceptorsMessage(0, 1, strings.Repeat("a", 40000))),
		},
		// conditions
		"Evicted condition can be set": {
			oldInput: mkValidEvictionRequestStatus(1),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
		},
		"Evicted condition can be changed to true": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, false)),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
		},
		"Canceled condition ca be set": {
			oldInput: mkValidEvictionRequestStatus(1),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
		},
		"Canceled condition can be changed to true": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, false)),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
		},
		"Non terminal condition can be added when immutable condition exists": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, "NewCondition", true), addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
		},
		// processedInterceptors + conditions
		"processedInterceptors must allow active interceptor transition even with an exceeded deadline (start time fallback) - Canceled": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
		},
		"processedInterceptors must allow active interceptor transition even with an exceeded deadline (start time fallback) - Evicted": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
		},
		"processedInterceptors must allow active interceptor transition even with an exceeded deadline - Canceled": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
		},
		"processedInterceptors must allow active interceptor transition even with an exceeded deadline - Evicted": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			if validate.SemanticDeepEqual(tc.oldInput, tc.input) {
				t.Errorf("Expected oldInput and input to differ")
			}
			oldEvictionRequest := mkValidEvictionRequest(2)
			oldEvictionRequest.ResourceVersion = "0"
			oldEvictionRequest.Status = *tc.oldInput
			evictionRequest := mkValidEvictionRequest(2)
			evictionRequest.ResourceVersion = "1"
			evictionRequest.Status = *tc.input
			opts := EvictionRequestStatusValidationOptions{clock}
			if tc.clock != nil {
				opts.Clock = tc.clock
			}
			errs := ValidateEvictionRequestStatusUpdate(evictionRequest, oldEvictionRequest, opts)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
			}
		})
	}

	failureCases := map[string]struct {
		input    *coordination.EvictionRequestStatus
		oldInput *coordination.EvictionRequestStatus

		errors []*field.Error
	}{
		"clear generation": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(1)),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), 0, "cannot decrement, must be greater than or equal to 1"),
			},
		},
		"decrease generation": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(5)),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(4)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), 4, "cannot decrement, must be greater than or equal to 5"),
			},
		},
		"decrease generation to negative": {
			oldInput: mkValidEvictionRequestStatus(0, setObservedGeneration(5)),
			input:    mkValidEvictionRequestStatus(0, setObservedGeneration(-1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "observedGeneration"), -1, "must be greater than or equal to 0").MarkCoveredByDeclarative(),
			},
		},
		// all interceptors
		"immutable interceptor fields when targetInterceptors is missing": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatusWithStatuses(0, 3, addActiveInterceptors(interceptorName(1)), addProcessedInterceptorsCount(2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors"), "", validation.FieldImmutableErrorMsg),
				field.Invalid(field.NewPath("status", "processedInterceptors"), "", validation.FieldImmutableErrorMsg),
				field.Invalid(field.NewPath("status", "interceptors"), "", validation.FieldImmutableErrorMsg),
			},
		},
		// targetInterceptors
		"too many targetInterceptors": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatusWithStatuses(17, 0),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "targetInterceptors"), 17, 16).MarkCoveredByDeclarative(),
				field.Required(field.NewPath("status", "interceptors"), ""),
			},
		},
		"invalid targetInterceptors and required status interceptors": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(0, addTargetInterceptors("f.ba.com", "f.ba.com", "", "invalid", "foo.k8s.io", "example.com", "foo.example.com/bar")),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "targetInterceptors").Index(1), "").MarkCoveredByDeclarative(),
				field.Required(field.NewPath("status", "targetInterceptors").Index(2).Child("name"), ""),
				field.Invalid(field.NewPath("status", "targetInterceptors").Index(3).Child("name"), "invalid", "should be a domain with at least three segments separated by dots"),
				field.Invalid(field.NewPath("status", "targetInterceptors").Index(5).Child("name"), "example.com", "should be a domain with at least three segments separated by dots"),
				field.Invalid(field.NewPath("status", "targetInterceptors").Index(6).Child("name"), "foo.example.com/bar", "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"),
				field.Required(field.NewPath("status", "interceptors"), ""),
			},
		},
		"immutable targetInterceptors - remove": {
			oldInput: mkValidEvictionRequestStatus(5),
			input:    mkValidEvictionRequestStatus(4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetInterceptors"), "", validation.FieldImmutableErrorMsg),
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order"),
			},
		},
		"immutable targetInterceptors - add": {
			oldInput: mkValidEvictionRequestStatus(4),
			input:    mkValidEvictionRequestStatus(5),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetInterceptors"), "", validation.FieldImmutableErrorMsg),
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order"),
			},
		},
		// activeInterceptors
		"no new activeInterceptors are allowed when the eviction request has evicted status": {
			oldInput: mkValidEvictionRequestStatus(1),
			input: mkValidEvictionRequestStatus(1,
				addActiveInterceptors(interceptorName(0)),
				addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors"), "", validation.FieldImmutableErrorMsg),
			},
		},
		"no new activeInterceptors are allowed when the eviction request has canceled status": {
			oldInput: mkValidEvictionRequestStatus(1),
			input: mkValidEvictionRequestStatus(1,
				addActiveInterceptors(interceptorName(0)),
				addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors"), "", validation.FieldImmutableErrorMsg),
			},
		},
		"invalid activeInterceptors as all interceptors have been processed": {
			oldInput: mkValidEvictionRequestStatus(2,
				addProcessedInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptors(interceptorName(0)),
				addProcessedInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors"), "must not be set because all interceptors have been processed"),
			},
		},
		"activeInterceptors items cannot be removed unless processed": {
			oldInput: mkValidEvictionRequestStatus(3, addActiveInterceptors(interceptorName(0)), setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors(), setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors"), "items cannot be removed, unless they are added to status.processedInterceptors"),
			},
		},
		"too many activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatus(5, addActiveInterceptorsCount(2)),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "activeInterceptors"), 2, 1).MarkCoveredByDeclarative(),
			},
		},
		"duplicate activeInterceptors -short circuited by too many": {
			oldInput: mkValidEvictionRequestStatus(3),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors(interceptorName(0), interceptorName(0))),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "activeInterceptors"), 2, 1).MarkCoveredByDeclarative(),
			},
		},
		"invalid activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors("invalid")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "activeInterceptors").Index(0), "", "is not a valid interceptor from status.targetInterceptors"),
			},
		},
		"set activeInterceptors to old processed interceptor": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addActiveInterceptors(interceptorName(0)),
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors").Index(0), "must be \"interceptor-1.example.com\" because this interceptor is next in line"),
			},
		},
		"out of order activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3),
			input:    mkValidEvictionRequestStatus(3, addActiveInterceptors(interceptorName(1))),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors").Index(0), "must be \"interceptor-0.example.com\" because this interceptor is next in line"),
			},
		},
		"out of order activeInterceptors with one interceptor processed": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addActiveInterceptors(interceptorName(2)),
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "activeInterceptors").Index(0), "must be \"interceptor-1.example.com\" because this interceptor is next in line"),
			},
		},
		// processedInterceptors
		"removing processedInterceptors items": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(0), interceptorName(1)),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors"), "items cannot be removed"),
			},
		},
		"adding processedInterceptors items can not be done in bulk": {
			oldInput: mkValidEvictionRequestStatus(3,
				addActiveInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(2),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors"), "items can only be added one at a time"),
			},
		},
		"too many processedInterceptors": {
			// invalid input, but allows us to set too many processedInterceptors
			oldInput: mkValidEvictionRequestStatus(16,
				addActiveInterceptorsCount(16),
				addProcessedInterceptorsCount(16),
				setInterceptorsStartTime(clock, 0, 16),
				setInterceptorsCompletionTime(clock, 0, 16)),
			input: mkValidEvictionRequestStatus(16,
				addProcessedInterceptorsCount(17),
				setInterceptorsStartTime(clock, 0, 16),
				setInterceptorsCompletionTime(clock, 0, 16)),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "processedInterceptors"), 17, 16).MarkCoveredByDeclarative(),
			},
		},
		"duplicate processedInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3,
				addActiveInterceptors(interceptorName(0)),
				addProcessedInterceptors(interceptorName(0)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(0), interceptorName(0)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "processedInterceptors").Index(1), "").MarkCoveredByDeclarative(),
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(1), "is immutable because a \"status.interceptors[1]\" does not have a matching name"),
			},
		},
		"old processedInterceptors are immutable": {
			oldInput: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptors(interceptorName(1)),
				setInterceptorsFullStatus(clock, clock2, 0, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "processedInterceptors").Index(0), "", validation.FieldImmutableErrorMsg),
			},
		},
		"invalid processedInterceptors - must be a target interceptor": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addActiveInterceptors("invalid"),
				addInterceptorStatusesNames("invalid"),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addProcessedInterceptors("invalid"),
				addInterceptorStatusesNames("invalid"),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is not a valid interceptor from status.targetInterceptors"),
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors"),
			},
		},
		"processedInterceptors require a status status.interceptors": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addProcessedInterceptorsCount(1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because a \"interceptor-0.example.com\" has to be tracked in status.interceptors first"),
				field.Required(field.NewPath("status", "interceptors"), ""),
			},
		},
		"processedInterceptors must have a status with the same name and status interceptors should contain the same keys in the same order": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(3, 0,
				addActiveInterceptors(interceptorName(1)),
				addProcessedInterceptorsCount(1),
				addInterceptorStatusesNames(interceptorName(0), interceptorName(2), interceptorName(5)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatusWithStatuses(3, 0,
				addProcessedInterceptorsCount(2),
				addInterceptorStatusesNames(interceptorName(0), interceptorName(2), interceptorName(5)),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors"),
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(1), "is immutable because a \"status.interceptors[1]\" does not have a matching name"),
			},
		},
		"new processedInterceptors must be moved from activeInterceptors": {
			oldInput: mkValidEvictionRequestStatus(3,
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionRequestStatus(3,
				addProcessedInterceptorsCount(1),
				setInterceptorsFullStatus(clock, clock2, 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "\"interceptor-0.example.com\" should have been active and present in status.activeInterceptors"),
			},
		},
		"processedInterceptors must have a status with start time": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because \"interceptor-0.example.com\" interceptor should have status.interceptors[0].startTime"),
			},
		},
		"processedInterceptors must exceed the deadline (start time fallback) before it is marked processed": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because \"interceptor-0.example.com\" interceptor is in progress and it should report status.interceptors[0].heartbeatTime or status.interceptors[0].completionTime"),
			},
		},
		"processedInterceptors must exceed the deadline before it is marked processed": {
			oldInput: mkValidEvictionRequestStatus(1,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(1,
				addProcessedInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "processedInterceptors").Index(0), "is immutable because \"interceptor-0.example.com\" interceptor is in progress and it should report status.interceptors[0].heartbeatTime or status.interceptors[0].completionTime"),
			},
		},
		// interceptors
		"required status interceptors when targetInterceptors are set": {
			oldInput: mkValidEvictionRequestStatus(0),
			input:    mkValidEvictionRequestStatusWithStatuses(1, 0),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors"), ""),
			},
		},
		"status interceptors should contain the same keys in the same order - different length": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(2, 0,
				addInterceptorStatusesNames(interceptorName(1))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order"),
			},
		},
		"status interceptors should contain the same keys in the same order": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(2, 0,
				addInterceptorStatusesNames(interceptorName(1), interceptorName(0))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors"),
			},
		},
		"status interceptors cannot remove an item - target interceptors are immutable": {
			oldInput: mkValidEvictionRequestStatus(5),
			input:    mkValidEvictionRequestStatusWithStatuses(5, 4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should be the same length as status.targetInterceptors and contain the same keys in the same order"),
			},
		},
		"too many status interceptors": {
			oldInput: mkValidEvictionRequestStatusWithStatuses(17, 16),
			input:    mkValidEvictionRequestStatus(17),
			errors: []*field.Error{
				field.TooMany(field.NewPath("status", "interceptors"), 17, 16).MarkCoveredByDeclarative(),
			},
		},
		"duplicate status interceptors - short circuited by targetInterceptors key order": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(2, 0,
				addInterceptorStatusesNames(interceptorName(0), interceptorName(0))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors"),
			},
		},
		"status interceptors cannot be mutated unless present in active interceptors": {
			oldInput: mkValidEvictionRequestStatus(5),
			input: mkValidEvictionRequestStatus(5,
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0), "", validation.FieldImmutableErrorMsg),
			},
		},
		// status interceptor name
		"status interceptors cannot mutate name when active - short circuited by targetInterceptors key order": {
			oldInput: mkValidEvictionRequestStatus(2, addActiveInterceptorsCount(1), setInterceptorsStartTime(clock, 0, 2)),
			input: mkValidEvictionRequestStatusWithStatuses(2, 0,
				addActiveInterceptorsCount(1),
				addInterceptorStatusesNames(interceptorName(1), interceptorName(0)),
				setInterceptorsStartTime(clock, 0, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors"),
			},
		},
		"required status interceptors names - short circuited by targetInterceptors key order": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addInterceptorStatusesNames("")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors"),
			},
		},
		"invalid status interceptors names - short circuited by targetInterceptors key order": {
			oldInput: mkValidEvictionRequestStatus(0),
			input: mkValidEvictionRequestStatusWithStatuses(1, 0,
				addInterceptorStatusesNames("foo.k8s.io")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors"), "", "should contain the same keys in the same order as status.targetInterceptors"),
			},
		},
		// startTime
		"startTime cannot be removed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", validation.FieldImmutableErrorMsg),
			},
		},
		"startTime cannot be changed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockAfter(15*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", validation.FieldImmutableErrorMsg),
			},
		},
		"startTime is required for an active interceptor": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor"),
			},
		},
		"startTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", "should be set to the present time"),
			},
		},
		"startTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionRequestStatus(2),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "", "should be set to the present time"),
			},
		},
		// heartbeatTime
		"cannot remove heartbeatTime when previously set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "is required once set"),
			},
		},
		"heartbeatTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor"),
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "cannot be set before status.interceptors[0].startTime is set"),
			},
		},
		"heartbeatTime cannot be decreased": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Second), 0, 1),
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Second), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "cannot be decreased"),
			},
		},
		"heartbeatTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "must occur after status.interceptors[0].startTime"),
			},
		},
		"heartbeatTime must be updated after 1m at the earliest": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(59*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "there must be at least 1m0s increments during subsequent updates"),
			},
		},
		"heartbeatTime must be set to the present time with skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(20*time.Minute), 0, 1),
				setInterceptorsHeartBeatTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("heartbeatTime"), "", "should be set to the present time"),
			},
		},
		// expectedCompletionTime
		"expectedCompletionTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor"),
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "cannot be set before status.interceptors[0].startTime is set"),
			},
		},
		"expectedCompletionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "must occur after status.interceptors[0].startTime"),
			},
		},
		"expectedCompletionTime cannot be set to the past with skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(32*time.Second), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(32*time.Second), 0, 1),
				setInterceptorsExpectedCompletionTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "cannot be set to the past time"),
			},
		},
		"expectedCompletionTime must complete within 10 years": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(time.Minute), 0, 1),
				setInterceptorsExpectedCompletionTime(clockAfter(time.Hour*24*365*10+time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("expectedCompletionTime"), "", "must complete within 10 years"),
			},
		},
		// completionTime
		"completionTime cannot be changed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(4*time.Minute), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", validation.FieldImmutableErrorMsg),
			},
		},
		"completionTime cannot be removed once set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(5*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", validation.FieldImmutableErrorMsg),
			},
		},
		"completionTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "interceptors").Index(0).Child("startTime"), "is required for an active interceptor"),
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "cannot be set before status.interceptors[0].startTime is set"),
			},
		},
		"completionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "must occur after status.interceptors[0].startTime"),
			},
		},
		"completionTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clock, 0, 1),
				setInterceptorsCompletionTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "should be set to the present time"),
			},
		},
		"completionTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(1*time.Minute), 0, 1)),
			input: mkValidEvictionRequestStatus(2,
				addActiveInterceptorsCount(1),
				setInterceptorsStartTime(clockBefore(1*time.Minute), 0, 1),
				setInterceptorsCompletionTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "interceptors").Index(0).Child("completionTime"), "", "should be set to the present time"),
			},
		},
		// conditions
		"duplicate condition": {
			oldInput: mkValidEvictionRequestStatus(1),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true), addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			errors: []*field.Error{
				field.Duplicate(field.NewPath("status", "conditions").Index(1), "").MarkCoveredByDeclarative(),
			},
		},
		"add invalid condition": {
			oldInput: mkValidEvictionRequestStatus(1),
			input: mkValidEvictionRequestStatus(1, func(obj *coordination.EvictionRequestStatus) {
				obj.Conditions = append(obj.Conditions, metav1.Condition{
					Type:               "-bad-name",
					Status:             "invalid",
					ObservedGeneration: -1,
					Reason:             "-Reason",
				})
			}),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("type"), "", "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')"),
				field.NotSupported(field.NewPath("status", "conditions").Index(0).Child("status"), "", []string{"False", "True", "Unknown"}),
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("observedGeneration"), "", "must be greater than or equal to zero"),
				field.Required(field.NewPath("status", "conditions").Index(0).Child("lastTransitionTime"), ""),
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("reason"), "", "a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and must end with an alphanumeric character or '_' (e.g. 'my_name',  or 'MY_NAME',  or 'MyName',  or 'ReasonA,ReasonB',  or 'ReasonA:ReasonB', regex used for validation is '[A-Za-z]([A-Za-z0-9_,:]*[A-Za-z0-9_])?')"),
			},
		},
		"Evicted condition cannot be removed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			input:    mkValidEvictionRequestStatus(1),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition is immutable"),
			},
		},
		"Evicted condition cannot be changed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, true)),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionEvicted, false)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition is immutable"),
			},
		},
		"Canceled condition cannot be removed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
			input:    mkValidEvictionRequestStatus(1),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Canceled condition is immutable"),
			},
		},
		"Canceled condition cannot be changed": {
			oldInput: mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, true)),
			input:    mkValidEvictionRequestStatus(1, addCondition(clock, coordination.EvictionRequestConditionCanceled, false)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Canceled condition is immutable"),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			oldEvictionRequest := mkValidEvictionRequest(2)
			oldEvictionRequest.ResourceVersion = "0"
			oldEvictionRequest.Status = *tc.oldInput
			evictionRequest := mkValidEvictionRequest(2)
			evictionRequest.ResourceVersion = "1"
			evictionRequest.Status = *tc.input
			errs := ValidateEvictionRequestStatusUpdate(evictionRequest, oldEvictionRequest, EvictionRequestStatusValidationOptions{clock})
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
			matcher := field.ErrorMatcher{}.ByType().ByField().ByDetailExact()
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

func mkValidEvictionRequest(requesters int, tweaks ...func(obj *coordination.EvictionRequest)) *coordination.EvictionRequest {
	obj := coordination.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: valiUIDdName, Namespace: "foo"},
		Spec: coordination.EvictionRequestSpec{
			Target: coordination.EvictionTarget{
				Pod: &coordination.LocalTargetReference{
					UID:  valiUIDdName,
					Name: "foo.pod",
				},
			},
		},
	}
	for i := range requesters {
		obj.Spec.Requesters = append(obj.Spec.Requesters, coordination.Requester{
			Name: fmt.Sprintf("requester-%d.example.com", i),
		})
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}

func setName(name, generateName string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Name = name
		obj.GenerateName = generateName
	}
}
func setNamespace(namespace string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Namespace = namespace
	}
}
func clearTarget() func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Spec.Target.Pod = nil
	}
}
func setTarget(name, uid string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		obj.Spec.Target.Pod = &coordination.LocalTargetReference{
			UID:  uid,
			Name: name,
		}
	}
}
func addRequesters(names ...string) func(obj *coordination.EvictionRequest) {
	return func(obj *coordination.EvictionRequest) {
		for _, name := range names {
			obj.Spec.Requesters = append(obj.Spec.Requesters, coordination.Requester{Name: name})
		}
	}
}

func interceptorName(i int) string {
	return fmt.Sprintf("interceptor-%d.example.com", i)
}
func mkValidEvictionRequestStatus(interceptors int, tweaks ...func(obj *coordination.EvictionRequestStatus)) *coordination.EvictionRequestStatus {
	return mkValidEvictionRequestStatusWithStatuses(interceptors, interceptors, tweaks...)
}
func mkValidEvictionRequestStatusWithStatuses(interceptors, statuses int, tweaks ...func(obj *coordination.EvictionRequestStatus)) *coordination.EvictionRequestStatus {
	obj := coordination.EvictionRequestStatus{
		ObservedGeneration: 1,
	}
	for i := range interceptors {
		obj.TargetInterceptors = append(obj.TargetInterceptors, core.EvictionInterceptor{
			Name: interceptorName(i),
		})
	}
	for i := range statuses {
		obj.Interceptors = append(obj.Interceptors, coordination.InterceptorStatus{
			Name: interceptorName(i),
		})
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}
func addCondition(clock utilsclock.PassiveClock, name coordination.EvictionRequestConditionType, status bool) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
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

func setObservedGeneration(generation int64) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		obj.ObservedGeneration = generation
	}
}
func addTargetInterceptors(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, interceptorName := range interceptors {
			obj.TargetInterceptors = append(obj.TargetInterceptors, core.EvictionInterceptor{Name: interceptorName})
		}
	}
}
func addActiveInterceptors(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, interceptor := range interceptors {
			obj.ActiveInterceptors = append(obj.ActiveInterceptors, interceptor)
		}
	}
}
func addActiveInterceptorsCount(count int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := range count {
			obj.ActiveInterceptors = append(obj.ActiveInterceptors, interceptorName(i))
		}
	}
}
func addProcessedInterceptors(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, interceptor := range interceptors {
			obj.ProcessedInterceptors = append(obj.ProcessedInterceptors, interceptor)
		}
	}
}
func addProcessedInterceptorsCount(count int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := range count {
			obj.ProcessedInterceptors = append(obj.ProcessedInterceptors, interceptorName(i))
		}
	}
}
func addInterceptorStatusesNames(interceptors ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for _, interceptor := range interceptors {
			obj.Interceptors = append(obj.Interceptors, coordination.InterceptorStatus{Name: interceptor})
		}
	}
}
func setInterceptorsStartTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].StartTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsHeartBeatTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].HeartbeatTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsExpectedCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].ExpectedCompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].CompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setInterceptorsMessage(from, to int, suffixes ...string) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		for i := from; i < to; i++ {
			obj.Interceptors[i].Message = fmt.Sprintf("message %d", i)
			for _, suffix := range suffixes {
				obj.Interceptors[i].Message += suffix
			}
		}
	}
}
func setInterceptorsFullStatus(startTimeClock, clock utilsclock.PassiveClock, from, to int) func(obj *coordination.EvictionRequestStatus) {
	return func(obj *coordination.EvictionRequestStatus) {
		setInterceptorsStartTime(startTimeClock, from, to)(obj)
		setInterceptorsHeartBeatTime(clock, from, to)(obj)
		setInterceptorsExpectedCompletionTime(clock, from, to)(obj)
		setInterceptorsCompletionTime(clock, from, to)(obj)
		setInterceptorsMessage(from, to, clock.Now().String())(obj)
	}
}
