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

package validation

import (
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/lifecycle"
	utilsclock "k8s.io/utils/clock"
	testing2 "k8s.io/utils/clock/testing"
)

const validUID = "bd23f542-ac79-4b44-a628-9735e18b8037"

func TestValidateEvictionRequest(t *testing.T) {
	successCases := map[string]struct {
		input *lifecycle.EvictionRequest
	}{
		"valid: pod target": {
			input: mkValidEvictionRequest(),
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
		input *lifecycle.EvictionRequest

		errors []*field.Error
	}{
		"missing namespace": {
			input: mkValidEvictionRequest(setERNamespace("")),
			errors: []*field.Error{
				field.Required(field.NewPath("metadata", "namespace"), ""),
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

func TestValidateEvictionRequestUpdate(t *testing.T) {
	successCases := map[string]struct {
		input    *lifecycle.EvictionRequest
		oldInput *lifecycle.EvictionRequest
	}{
		"withdraw a requester intent": {
			oldInput: mkValidEvictionRequest(),
			input:    mkValidEvictionRequest(setRequesterIntent(lifecycle.EvictionRequestIntentWithdrawn)),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			tc.oldInput.ResourceVersion = "0"
			tc.input.ResourceVersion = "1"
			errs := ValidateEvictionRequestUpdate(tc.input, tc.oldInput)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
				return
			}
		})
	}
}

func TestValidateEvictionRequestStatusUpdate(t *testing.T) {
	clock := testing2.NewFakePassiveClock(time.Now())

	successCases := map[string]struct {
		clock    utilsclock.PassiveClock
		input    *lifecycle.EvictionRequestStatus
		oldInput *lifecycle.EvictionRequestStatus
	}{
		// conditions
		// this is different from Eviction
		"Evicted condition can be reverted to false": {
			oldInput: mkValidEvictionRequestStatus(addERCondition(clock, lifecycle.EvictionConditionTargetEvicted, true)),
			input:    mkValidEvictionRequestStatus(addERCondition(clock, lifecycle.EvictionConditionTargetEvicted, false)),
		},
		// this is different from Eviction
		"Failed condition can be reverted to false": {
			oldInput: mkValidEvictionRequestStatus(addERCondition(clock, lifecycle.EvictionConditionFailed, true)),
			input:    mkValidEvictionRequestStatus(addERCondition(clock, lifecycle.EvictionConditionFailed, false)),
		},
		// observedGeneration
		"set initial generation": {
			oldInput: &lifecycle.EvictionRequestStatus{},
			input:    mkValidEvictionRequestStatus(),
		},
		"update generation": {
			oldInput: mkValidEvictionRequestStatus(),
			input:    mkValidEvictionRequestStatus(setERObservedGeneration(new(int64(5)))),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			if validate.SemanticDeepEqual(tc.oldInput, tc.input) {
				t.Errorf("Expected oldInput and input to differ")
			}
			oldEvictionRequest := mkValidEvictionRequest()
			oldEvictionRequest.ResourceVersion = "0"
			oldEvictionRequest.Status = *tc.oldInput
			evictionRequest := mkValidEvictionRequest()
			evictionRequest.ResourceVersion = "1"
			evictionRequest.Status = *tc.input
			errs := ValidateEvictionRequestStatusUpdate(evictionRequest, oldEvictionRequest)
			if len(errs) != 0 {
				t.Errorf("Expected success for %q: %v", name, errs)
			}
		})
	}

	failureCases := map[string]struct {
		input    *lifecycle.EvictionRequestStatus
		oldInput *lifecycle.EvictionRequestStatus

		errors []*field.Error
	}{
		// conditions
		"add invalid condition": {
			oldInput: mkValidEvictionRequestStatus(),
			input: mkValidEvictionRequestStatus(func(obj *lifecycle.EvictionRequestStatus) {
				obj.Conditions = append(obj.Conditions, metav1.Condition{
					Type:               "-bad-name",
					Status:             metav1.ConditionTrue,
					Reason:             "-Reason",
					LastTransitionTime: metav1.Time{Time: clock.Now()},
				})
			}),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("type"), "", "name part must consist of alphanumeric characters, '-', '_' or '.', and "),
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("reason"), "", "a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and "),
			},
		},
	}

	for name, tc := range failureCases {
		t.Run(name, func(t *testing.T) {
			oldEvictionRequest := mkValidEvictionRequest()
			oldEvictionRequest.ResourceVersion = "0"
			oldEvictionRequest.Status = *tc.oldInput
			evictionRequest := mkValidEvictionRequest()
			evictionRequest.ResourceVersion = "1"
			evictionRequest.Status = *tc.input
			errs := ValidateEvictionRequestStatusUpdate(evictionRequest, oldEvictionRequest)
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

func TestValidateEviction(t *testing.T) {
	successCases := map[string]struct {
		input *lifecycle.Eviction
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
		input *lifecycle.Eviction

		errors []*field.Error
	}{
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
		input    *lifecycle.Eviction
		oldInput *lifecycle.Eviction
	}{
		"set requester annotation": {
			oldInput: mkValidEviction(),
			input:    mkValidEviction(addAnnotation("acme.io/foo", "requester")),
		},
		"update requester annotations": {
			oldInput: mkValidEviction(addAnnotation("acme.io/foo", "requester")),
			input:    mkValidEviction(addAnnotation("acme.io/foo", "requester-responder")),
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
		clock               utilsclock.PassiveClock
		input               *lifecycle.EvictionStatus
		oldInput            *lifecycle.EvictionStatus
		explicitEmptyUpdate bool
	}{
		// conditions
		"Evicted condition can be set": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
		},
		"Evicted condition can be changed to true": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionTargetEvicted, false),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
		},
		"Failed condition can be set": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
		},
		"Failed condition can be changed to true": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionFailed, false),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
		},
		"Non terminal condition can be added when immutable condition exists": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, "NewCondition", true), addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
		},
		"Non terminal condition can be changed when immutable condition exists": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, "NewCondition", true), addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, "NewCondition", false), addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
		},
		"A new condition can be added when observedGeneration is nil": {
			oldInput: mkValidEvictionStatus(0, setObservedGeneration(nil), addCondition(clock, "NewCondition", true), addCondition(clock, lifecycle.EvictionConditionFailed, true)),
			input:    mkValidEvictionStatus(0, setObservedGeneration(nil), addCondition(clock, "NewCondition", false), addCondition(clock, lifecycle.EvictionConditionFailed, true)),
		},
		// observedGeneration
		"set initial generation": {
			oldInput: &lifecycle.EvictionStatus{},
			input:    mkValidEvictionStatus(0),
		},
		"update generation": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, setObservedGeneration(new(int64(5)))),
		},
		// requesters
		"initialize requester with eviction intent": {
			oldInput: mkValidEvictionStatus(0, clearRequesters()),
			input:    mkValidEvictionStatus(0, addRequesters(lifecycle.RequesterIntentEviction, "foo.example.com/baz")),
		},
		"initialize requester with withdrawn intent": {
			oldInput: mkValidEvictionStatus(0, clearRequesters()),
			input:    mkValidEvictionStatus(0, addRequesters(lifecycle.RequesterIntentWithdrawn, "foo.example.com/baz")),
		},
		"change requester intent from eviction to withdrawn": {
			oldInput: mkValidEvictionStatus(0, clearRequesters(), addRequesters(lifecycle.RequesterIntentEviction, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, clearRequesters(), addRequesters(lifecycle.RequesterIntentWithdrawn, "foo.example.com/baz")),
		},
		"change requester intent from withdrawn to eviction": {
			oldInput: mkValidEvictionStatus(0, clearRequesters(), addRequesters(lifecycle.RequesterIntentWithdrawn, "foo.example.com/baz")),
			input:    mkValidEvictionStatus(0, clearRequesters(), addRequesters(lifecycle.RequesterIntentEviction, "foo.example.com/baz")),
		},
		"add a second requester with eviction intent": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addRequesters(lifecycle.RequesterIntentEviction, "foo.example.com/baz")),
		},
		"change a last requester eviction intent to withdrawn": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, clearRequesters(), addRequesters(lifecycle.RequesterIntentWithdrawn, "foo.example.com/bar")),
		},
		// first sync
		"responders initialization": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(3),
		},
		"11 responders": { // 1 more than Pod's spec.evictionResponders
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(11),
		},
		"first targetResponder can be inactive during the first sync if the request has failed": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
			),
		},
		"first targetResponder can be inactive during the first sync if the request has completed": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
			),
		},
		// Full Eviction progression/lifecycle
		"mark active and started": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
		},
		"responder reports progress": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
		},
		"responder reports next progress": {
			clock: clockAfter(time.Minute + 5*time.Second),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(time.Minute+5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
		},
		"cancel responder": {
			clock: clockAfter(time.Minute + 5*time.Second),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
			input: mkValidEvictionStatus(2,
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCanceled, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
		},

		"interrupt responder": {
			clock: clockAfter(time.Minute + 5*time.Second),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clock, 0, 1)),
		},
		"responder reports complete": {
			clock: clockAfter(5 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockAfter(time.Minute+5*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(10*time.Minute), 0, 1),
				setRespondersMessage(0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
		},
		"move active complete to processed and set new active": {
			clock: clockAfter(5*time.Minute + 27*time.Second),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
		},
		"next responder reports next progress": {
			clock: clockAfter(25 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setRespondersHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
		},
		"next responder reports complete": {
			clock: clockAfter(28 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setRespondersHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
		},
		"finalize eviction request active": {
			clock: clockAfter(35 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
			input: mkValidEvictionStatus(2,
				setCompletedCount(2),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersFullStatus(clockAfter(5*time.Minute+30*time.Second), clockAfter(27*time.Minute+32*time.Second), 1, 2)),
		},
		"cancel eviction request for the next responder": {
			clock: clockAfter(35 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setRespondersHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
			input: mkValidEvictionStatus(2,
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCanceled, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2),
				setRespondersHeartBeatTime(clockAfter(25*time.Minute), 1, 2)),
		},
		"interrupt eviction request for the next responder": {
			clock: clockAfter(35 * time.Minute),
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
			input: mkValidEvictionStatus(2,
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateInterrupted, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersFullStatus(clock, clockAfter(5*time.Minute+27*time.Second), 0, 1),
				setRespondersStartTime(clockAfter(5*time.Minute+30*time.Second), 1, 2)),
		},
		"finalize eviction request active - large number of responders": {
			clock: clockAfter(5 * time.Minute),
			oldInput: mkValidEvictionStatus(10,
				setStateFor(lifecycle.ResponderStateActive, 9),
				setCompletedCount(9),
				setRespondersFullStatus(clockAfter(time.Minute), clockAfter(2*time.Minute), 0, 4),
				setRespondersFullStatus(clockAfter(3*time.Minute), clockAfter(4*time.Minute), 4, 10)),
			input: mkValidEvictionStatus(10,
				setCompletedCount(10),
				setRespondersFullStatus(clockAfter(time.Minute), clockAfter(2*time.Minute), 0, 4),
				setRespondersFullStatus(clockAfter(3*time.Minute), clockAfter(4*time.Minute), 4, 10)),
		},
		// allowed update after the heartbeat deadline
		"processedResponders can be updated when the deadline (start time fallback) is reached": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+32*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+32*time.Second), 0, 1)),
		},
		"processedResponders can be updated when the deadline is reached": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Hour), 0, 1),
				setRespondersHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clockBefore(time.Hour), 0, 1),
				setRespondersHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
		},
		// startTime
		"startTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(29*time.Second), 0, 1)),
		},
		"responder startTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionStatus(2),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockAfter(27*time.Second), 0, 1)),
		},
		// heartbeatTime
		"heartbeatTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(27*time.Second), 0, 1)),
		},
		"heartbeatTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockAfter(27*time.Second), 0, 1)),
		},
		"heartbeatTime can be updated after one minute": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
		},
		// expectedCompletionTime
		"expectedCompletionTime can be set to the present time with an allowed skew - slower clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1),
				setRespondersExpectedCompletionTime(clockBefore(27*time.Second), 0, 1)),
		},
		"expectedCompletionTime can be set to the future": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1),
				setRespondersExpectedCompletionTime(clockAfter(time.Hour*24*365*10-time.Minute), 0, 1)),
		},
		// completionTime
		"completionTime can be set to the present time with an allowed skew - faster clock": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(27*time.Second), 0, 1)),
		},
		// processedResponders + conditions
		"processedResponders must allow active responder transition even with an exceeded deadline (start time fallback) - Failed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, lifecycle.EvictionConditionFailed, true)),
		},
		"processedResponders must allow active responder transition even with an exceeded deadline (start time fallback) - Evicted": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true)),
		},
		"processedResponders must allow active responder transition even with an exceeded deadline - Failed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, lifecycle.EvictionConditionFailed, true)),
		},
		"processedResponders must allow active responder transition even with an exceeded deadline - Evicted": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true)),
		},
		"second targetResponder completes without a final condition": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
		},
		// empty syncs
		"empty sync during heartbeat wait": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			explicitEmptyUpdate: true,
		},
		"all targetResponder completed without a final condition: empty sync": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			explicitEmptyUpdate: true,
		},
		"all targetResponder completed with a final condition: empty sync": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			explicitEmptyUpdate: true,
		},
		// k8s responders with assigned priorities
		"imperative-eviction.k8s.io/evictor k8s responder cannot have other priority than the assigned one": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addTargetAndStatusResponders(string(lifecycle.EvictionResponderImperativeEviction)), setPriorityFor(100, 0), setStateFor(lifecycle.ResponderStateActive, 0), setRespondersStartTime(clock, 0, 1)),
		},
	}
	for name, tc := range successCases {
		t.Run(name, func(t *testing.T) {
			if !tc.explicitEmptyUpdate && validate.SemanticDeepEqual(tc.oldInput, tc.input) {
				t.Errorf("Expected oldInput and input to differ")
			}
			// Ensure that the validation doesn't depend on the order of list elements.
			shuffleTargetResponders()(tc.oldInput)
			shuffleResponders()(tc.oldInput)
			shuffleRequesters()(tc.oldInput)
			shuffleTargetResponders()(tc.input)
			shuffleResponders()(tc.input)
			shuffleRequesters()(tc.input)
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
		input    *lifecycle.EvictionStatus
		oldInput *lifecycle.EvictionStatus

		errors []*field.Error
	}{
		// conditions
		"add invalid condition": {
			oldInput: mkValidEvictionStatus(1),
			input: mkValidEvictionStatus(1, func(obj *lifecycle.EvictionStatus) {
				obj.Conditions = append(obj.Conditions, metav1.Condition{
					Type:               "bad-name",
					Status:             metav1.ConditionTrue,
					Reason:             "-Reason",
					LastTransitionTime: metav1.Time{Time: clock.Now()},
				})
			}),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions").Index(0).Child("reason"), "", "a condition reason must start with alphabetic character, optionally followed by a string of alphanumeric characters or '_,:', and "),
			},
		},
		"Evicted condition cannot be removed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition status cannot be reverted"),
			},
		},
		"Evicted condition cannot be changed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionTargetEvicted, false),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Evicted condition status cannot be reverted"),
			},
		},
		"Failed condition cannot be removed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Failed condition status cannot be reverted"),
			},
		},
		"Failed condition cannot be changed": {
			oldInput: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			input: mkValidEvictionStatus(1, addCondition(clock, lifecycle.EvictionConditionFailed, false),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "conditions"), "", "Failed condition status cannot be reverted"),
			},
		},
		// requesters
		// targetResponders and responders
		"cannot change keys in targetResponders - remove": {
			oldInput: mkValidEvictionStatus(5),
			input:    mkValidEvictionStatus(4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders"), "", "must preserve the same length and the same keys"),
			},
		},
		"cannot change keys in targetResponders - add": {
			oldInput: mkValidEvictionStatus(4),
			input:    mkValidEvictionStatus(5),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders"), "", "must preserve the same length and the same keys"),
			},
		},
		// targetResponders
		"cannot change keys in targetResponders - change": {
			oldInput: mkValidEvictionStatus(4),
			input:    mkValidEvictionStatus(3, addTargetResponders("foo/bar"), addStatusResponders(responderName(3))),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders"), "", "must preserve the same keys"),
				field.Invalid(field.NewPath("status", "responders"), "", "must contain the same keys as status.targetResponders"),
			},
		},
		"targetResponders must have matching status responder": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addTargetResponders("foo/bar")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0), "", "has to be tracked in status.responders first"),
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys"),
			},
		},
		// priority
		"non k8s responder cannot have priority below 1000: 0": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(1, setPriorityFor(0, 0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("priority"), "", "priorities 0-999 are reserved for responders with *.k8s.io suffix"),
			},
		},
		"non k8s responder cannot have priority below 1000: 999": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(1, setPriorityFor(0, 0)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("priority"), "", "priorities 0-999 are reserved for responders with *.k8s.io suffix"),
			},
		},
		"imperative-eviction.k8s.io/evictor k8s responder cannot have other priority than the assigned one": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatus(0, addTargetAndStatusResponders(string(lifecycle.EvictionResponderImperativeEviction)), setPriorityFor(101, 0), setStateFor(lifecycle.ResponderStateActive, 0), setRespondersStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("priority"), "", "core k8s responder is expected to have priority: 100"),
			},
		},
		// targetResponder state transitions
		"first targetResponder should become active first ": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateInactive, 0), setStateFor(lifecycle.ResponderStateActive, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Inactive, because the previously active has not finished processing"),
				field.Required(field.NewPath("status", "responders").Index(1).Child("startTime"), ""),
			},
		},
		"first targetResponder should become active first during the first sync and not completed": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"first targetResponder should become active first during the first sync and not interrupted": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"first targetResponder should become active first during the first sync and not canceled": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCanceled, 0),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"first targetResponder should become active first during the first sync and not inactive": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateInactive, 0),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"first targetResponder should not become active first during the first sync with final condition": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Inactive, because the eviction request has finished processing"),
			},
		},
		"first targetResponder should not become completed first during the first sync with final condition": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Inactive, because the eviction request has finished processing"),
			},
		},
		"first targetResponder should not become interrupted first during the first sync with final condition": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Inactive, because the eviction request has finished processing"),
			},
		},
		"first targetResponder should not become canceled first during the first sync with final condition": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				addCondition(clock, lifecycle.EvictionConditionFailed, true),
				setStateFor(lifecycle.ResponderStateCanceled, 0),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Inactive, because the eviction request has finished processing"),
			},
		},
		"second targetResponder should not become active while the first reponder is active": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Inactive, because the previously active has not finished processing"),
				field.Required(field.NewPath("status", "responders").Index(1).Child("startTime"), ""),
			},
		},
		"second targetResponder should not become completed while the first reponder is active": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				setRespondersStartTime(clock, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Inactive, because the previously active has not finished processing"),
			},
		},
		"second targetResponder should not become interrupted while the first reponder is active": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0), setStateFor(lifecycle.ResponderStateInterrupted, 1),
				setRespondersStartTime(clock, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Inactive, because the previously active has not finished processing"),
			},
		},
		"second targetResponder should not become canceled while the first reponder is active": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0), setStateFor(lifecycle.ResponderStateCanceled, 1),
				setRespondersStartTime(clock, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Inactive, because the previously active has not finished processing"),
			},
		},
		"first targetResponder should not be inactive once it completes": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateInactive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Canceled, Completed, Interrupted, because this responder must reach a final state"),
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"second targetResponder should not be inactive once the first reponder completes": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"second targetResponder should not be completed once the first reponder completes": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"second targetResponder should not be interrupted once the first reponder completes": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateInterrupted, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"second targetResponder should not be canceled once the first reponder completes": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCanceled, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"first targetResponder is immutable once it completes": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateActive, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Completed, because final state is immutable"),
			},
		},
		"second targetResponder completes, should not allow a change of the first one": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateActive, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "must be one of: Completed, because final state is immutable"),
			},
		},
		"second targetResponder completes, should not stay active": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Canceled, Completed, Interrupted, because the eviction request has finished processing"),
			},
		},
		"second targetResponder completes, should not become inactive": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateInactive, 1),
				addCondition(clock, lifecycle.EvictionConditionTargetEvicted, true),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(1).Child("state"), "", "must be one of: Canceled, Completed, Interrupted, because this responder must reach a final state"),
			},
		},
		"second targetResponder completes, third should become active": {
			oldInput: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1), responderName(2)),
				setStateFor(lifecycle.ResponderStateCompleted, 0), setStateFor(lifecycle.ResponderStateCompleted, 1),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clock2, 0, 1),
				setRespondersStartTime(clock2, 1, 2),
				setRespondersCompletionTime(clock2, 1, 2),
			),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(2).Child("state"), "", "must be one of: Active, because this responder is next in line"),
			},
		},
		"Active responder must exceed the deadline (start time fallback) before it is marked Completed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "targetResponders").Index(0).Child("state"), "must stay Active because the responder is in progress"),
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "cannot become Completed because the responder didn't report status.responders.completionTime"),
			},
		},
		"Active responder must exceed the deadline before it is marked Completed": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "targetResponders").Index(0).Child("state"), "must stay Active because the responder is in progress"),
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "cannot become Completed because the responder didn't report status.responders.completionTime"),
			},
		},
		"Active responder must exceed the deadline before it is marked Interrupted": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateInterrupted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "targetResponders").Index(0).Child("state"), "must stay Active because the responder is in progress"),
			},
		},
		"Active responder must exceed the deadline before it is marked Canceled": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCanceled, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute+29*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Forbidden(field.NewPath("status", "targetResponders").Index(0).Child("state"), "must stay Active because the responder is in progress"),
			},
		},
		"Active responder exceeds the deadline, but cannot be marked Completed without CompletionTime": {
			oldInput: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(1,
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(20*time.Minute), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(0).Child("state"), "", "cannot become Completed because the responder didn't report status.responders.completionTime"),
			},
		},
		// responders
		"responders must be initialized at the same time as targetResponders": {
			oldInput: mkValidEvictionStatus(0),
			input:    mkValidEvictionStatusWithStatuses(0, 3),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys"),
			},
		},
		"cannot change keys in responders - change": {
			oldInput: mkValidEvictionStatus(4),
			input:    mkValidEvictionStatus(3, addTargetResponders(responderName(3)), addStatusResponders("foo/bar")),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(3), "", "has to be tracked in status.responders first"),
				field.Invalid(field.NewPath("status", "responders"), "", "must contain the same keys as status.targetResponders"),
			},
		},
		"status responders cannot remove an item": {
			oldInput: mkValidEvictionStatus(5),
			input:    mkValidEvictionStatusWithStatuses(5, 4),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "targetResponders").Index(4), "", "has to be tracked in status.responders first"),
				field.Invalid(field.NewPath("status", "responders"), "", "must be the same length as status.targetResponders and contain the same keys"),
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
				field.Invalid(field.NewPath("status", "responders"), "", "must contain the same keys as status.targetResponders"),
			},
		},
		// startTime
		"startTime is required for an active responder": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setStateFor(lifecycle.ResponderStateActive, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(1).Child("startTime"), "is required for an active responder"),
			},
		},
		"startTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionStatus(2, setRespondersFullStatus(clock, clock2, 0, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clockBefore(31*time.Second), 1, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(1).Child("startTime"), "", "must be set to the present time"),
			},
		},
		"startTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionStatus(2,
				setRespondersFullStatus(clock, clock2, 0, 1)),
			input: mkValidEvictionStatus(2, setRespondersFullStatus(clock, clock2, 0, 1),
				setStateFor(lifecycle.ResponderStateCompleted, 0),
				setStateFor(lifecycle.ResponderStateActive, 1),
				setRespondersStartTime(clockAfter(31*time.Second), 1, 2)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(1).Child("startTime"), "", "must be set to the present time"),
			},
		},
		// heartbeatTime
		"cannot remove heartbeatTime when previously set": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "is required once set"),
			},
		},
		"heartbeatTime cannot be set before startTime is set": {
			oldInput: mkValidEvictionStatus(0),
			input: mkValidEvictionStatus(0,
				addTargetAndStatusResponders(responderName(0), responderName(1)),
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("startTime"), "is required for an active responder"),
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "", "cannot be set before status.responders[0].startTime"),
			},
		},
		"heartbeatTime cannot be decreased": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Second), 0, 1),
				setRespondersHeartBeatTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Second), 0, 1),
				setRespondersHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "", "cannot be decreased"),
			},
		},
		"heartbeatTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersHeartBeatTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("heartbeatTime"), "", "must occur after status.responders[0].startTime"),
			},
		},
		"heartbeatTime must be set to the present time with skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(20*time.Minute), 0, 1),
				setRespondersHeartBeatTime(clockBefore(19*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
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
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("startTime"), "is required for an active responder"),
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("expectedCompletionTime"), "", "cannot be set before status.responders[0].startTime"),
			},
		},
		"expectedCompletionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersExpectedCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("expectedCompletionTime"), "", "must occur after status.responders[0].startTime"),
			},
		},
		"expectedCompletionTime cannot be set to the past with skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(32*time.Second), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(32*time.Second), 0, 1),
				setRespondersExpectedCompletionTime(clockBefore(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("expectedCompletionTime"), "", "cannot be set to the past time"),
			},
		},
		"expectedCompletionTime must complete within 10 years": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
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
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Required(field.NewPath("status", "responders").Index(0).Child("startTime"), "is required for an active responder"),
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), "", "cannot be set before status.responders[0].startTime"),
			},
		},
		"completionTime cannot be set before startTime even with a skew": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockBefore(2*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), "", "must occur after status.responders[0].startTime"),
			},
		},
		"completionTime must be set to the present time with skew; too new": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clock, 0, 1),
				setRespondersCompletionTime(clockAfter(31*time.Second), 0, 1)),
			errors: []*field.Error{
				field.Invalid(field.NewPath("status", "responders").Index(0).Child("completionTime"), "", "must be set to the present time"),
			},
		},
		"completionTime must be set to the present time with skew; too old": {
			oldInput: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
				setRespondersStartTime(clockBefore(1*time.Minute), 0, 1)),
			input: mkValidEvictionStatus(2,
				setStateFor(lifecycle.ResponderStateActive, 0),
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

func mkValidEvictionRequest(tweaks ...func(obj *lifecycle.EvictionRequest)) *lifecycle.EvictionRequest {
	obj := lifecycle.EvictionRequest{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "foo"},
		Spec: lifecycle.EvictionRequestSpec{
			Target: lifecycle.EvictionRequestTarget{
				Pod: &lifecycle.EvictionRequestPodReference{
					UID:  validUID,
					Name: "foo.pod",
				},
			},
			Requester: "foo.example.com/bar",
			Intent:    lifecycle.EvictionRequestIntentEviction,
		},
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}

func setERNamespace(namespace string) func(obj *lifecycle.EvictionRequest) {
	return func(obj *lifecycle.EvictionRequest) {
		obj.Namespace = namespace
	}
}

func setRequesterIntent(intent lifecycle.EvictionRequestIntent) func(obj *lifecycle.EvictionRequest) {
	return func(obj *lifecycle.EvictionRequest) {
		obj.Spec.Intent = intent
	}
}

func mkValidEvictionRequestStatus(tweaks ...func(obj *lifecycle.EvictionRequestStatus)) *lifecycle.EvictionRequestStatus {
	obj := lifecycle.EvictionRequestStatus{
		ObservedGeneration: new(int64(1)),
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}
func addERCondition(clock utilsclock.PassiveClock, name lifecycle.EvictionConditionType, status bool) func(obj *lifecycle.EvictionRequestStatus) {
	return func(obj *lifecycle.EvictionRequestStatus) {
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

func setERObservedGeneration(generation *int64) func(obj *lifecycle.EvictionRequestStatus) {
	return func(obj *lifecycle.EvictionRequestStatus) {
		obj.ObservedGeneration = generation
	}
}

func mkValidEviction(tweaks ...func(obj *lifecycle.Eviction)) *lifecycle.Eviction {
	obj := lifecycle.Eviction{
		ObjectMeta: metav1.ObjectMeta{Name: validUID, Namespace: "foo"},
		Spec: lifecycle.EvictionSpec{
			Target: lifecycle.EvictionTarget{
				Pod: &lifecycle.EvictionPodReference{
					UID:  validUID,
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

func setNamespace(namespace string) func(obj *lifecycle.Eviction) {
	return func(obj *lifecycle.Eviction) {
		obj.Namespace = namespace
	}
}
func addAnnotation(key, value string) func(obj *lifecycle.Eviction) {
	return func(obj *lifecycle.Eviction) {
		if obj.Labels == nil {
			obj.Labels = make(map[string]string)
		}
		obj.Labels[key] = value
	}
}

func responderName(i int) string {
	return fmt.Sprintf("responder.example.com/bar%d", i)
}
func mkValidEvictionStatus(responders int, tweaks ...func(obj *lifecycle.EvictionStatus)) *lifecycle.EvictionStatus {
	return mkValidEvictionStatusWithStatuses(responders, responders, tweaks...)
}
func mkValidEvictionStatusWithStatuses(responders, statuses int, tweaks ...func(obj *lifecycle.EvictionStatus)) *lifecycle.EvictionStatus {
	obj := lifecycle.EvictionStatus{
		ObservedGeneration: new(int64(1)),
	}
	for i := range responders {
		obj.TargetResponders = append(obj.TargetResponders, lifecycle.TargetResponder{
			Name:     responderName(i),
			Priority: new(5000 - int32(i)),
			State:    lifecycle.ResponderStateInactive,
		})
		if i == 0 {
			obj.TargetResponders[i].State = lifecycle.ResponderStateActive
		}
	}
	for i := range statuses {
		obj.Responders = append(obj.Responders, lifecycle.ResponderStatus{
			Name: responderName(i),
		})
		if i == 0 {
			obj.Responders[i].StartTime = new(metav1.Now())
		}
	}
	addRequesters(lifecycle.RequesterIntentEviction, "foo.example.com/bar")
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return &obj
}
func addCondition(clock utilsclock.PassiveClock, name lifecycle.EvictionConditionType, status bool) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
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

func setObservedGeneration(generation *int64) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		obj.ObservedGeneration = generation
	}
}
func addTargetResponders(responders ...string) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		lastPriority := len(obj.Responders)
		for i, name := range responders {
			obj.TargetResponders = append(obj.TargetResponders, lifecycle.TargetResponder{
				Name:     name,
				Priority: new(5000 - int32(lastPriority+i)),
				State:    lifecycle.ResponderStateInactive,
			})
		}
	}
}
func clearRequesters() func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		obj.Requesters = nil
	}
}
func addRequesters(intent lifecycle.RequesterIntent, names ...string) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for _, name := range names {
			obj.Requesters = append(obj.Requesters, lifecycle.Requester{Name: name, Intent: intent})
		}
	}
}
func addTargetAndStatusResponders(responders ...string) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		addTargetResponders(responders...)(obj)
		addStatusResponders(responders...)(obj)
	}
}
func setStateFor(state lifecycle.ResponderStateType, idx int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		obj.TargetResponders[idx].State = state
	}
}
func setPriorityFor(priority int32, idx int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		obj.TargetResponders[idx].Priority = new(priority)
	}
}
func setCompletedCount(count int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for i := range count {
			if i < len(obj.TargetResponders) {
				obj.TargetResponders[i].State = lifecycle.ResponderStateCompleted
			}
		}
	}
}
func addStatusResponders(responders ...string) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for _, responder := range responders {
			obj.Responders = append(obj.Responders, lifecycle.ResponderStatus{Name: responder})
		}
	}
}
func setRespondersStartTime(clock utilsclock.PassiveClock, from, to int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].StartTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersHeartBeatTime(clock utilsclock.PassiveClock, from, to int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].HeartbeatTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersExpectedCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].ExpectedCompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersCompletionTime(clock utilsclock.PassiveClock, from, to int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for i := from; i < to; i++ {
			obj.Responders[i].CompletionTime = &metav1.Time{Time: clock.Now().Add(time.Duration(i) * time.Second)}
		}
	}
}
func setRespondersMessage(from, to int, suffixes ...string) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		for i := from; i < to; i++ {
			msg := fmt.Sprintf("message %d", i)
			for _, suffix := range suffixes {
				msg += suffix
			}
			obj.Responders[i].Message = new(msg)
		}
	}
}
func setRespondersFullStatus(startTimeClock, clock utilsclock.PassiveClock, from, to int) func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		setRespondersStartTime(startTimeClock, from, to)(obj)
		setRespondersHeartBeatTime(clock, from, to)(obj)
		setRespondersExpectedCompletionTime(clock, from, to)(obj)
		setRespondersCompletionTime(clock, from, to)(obj)
		setRespondersMessage(from, to, clock.Now().String())(obj)
	}
}

func shuffleRequesters() func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		rand.Shuffle(len(obj.Requesters), func(i, j int) {
			obj.Requesters[i], obj.Requesters[j] = obj.Requesters[j], obj.Requesters[i]
		})
	}
}
func shuffleTargetResponders() func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		rand.Shuffle(len(obj.TargetResponders), func(i, j int) {
			obj.TargetResponders[i], obj.TargetResponders[j] = obj.TargetResponders[j], obj.TargetResponders[i]
		})
	}
}
func shuffleResponders() func(obj *lifecycle.EvictionStatus) {
	return func(obj *lifecycle.EvictionStatus) {
		rand.Shuffle(len(obj.Responders), func(i, j int) {
			obj.Responders[i], obj.Responders[j] = obj.Responders[j], obj.Responders[i]
		})
	}
}

func TestSortTargetResponders(t *testing.T) {
	testCases := map[string]struct {
		expectedResponders []lifecycle.TargetResponder
	}{
		"sorts by priority": {
			expectedResponders: []lifecycle.TargetResponder{
				{Name: responderName(1), Priority: new(int32(100000))},
				{Name: responderName(9), Priority: new(int32(99999))},
				{Name: responderName(10), Priority: new(int32(10001))},
				{Name: responderName(2), Priority: new(int32(10000))},
				{Name: responderName(13), Priority: new(int32(5))},
				{Name: responderName(4), Priority: new(int32(4))},
				{Name: responderName(12), Priority: new(int32(3))},
				{Name: responderName(6), Priority: new(int32(3))},
				{Name: responderName(11), Priority: new(int32(2))},
				{Name: responderName(8), Priority: new(int32(1))},
				{Name: responderName(7), Priority: new(int32(0))},
				{Name: responderName(3), Priority: new(int32(-1))},     // doesn't pass API validation
				{Name: responderName(14), Priority: new(int32(-1000))}, // doesn't pass API validation
				{Name: responderName(5), Priority: nil},                // doesn't pass API validation
			},
		},
		"sorts by domain name prefix and key when priority is the same": {
			expectedResponders: []lifecycle.TargetResponder{
				{Name: "responder-x.example.com/bar", Priority: new(int32(16))},
				{Name: "responder-c.fooexample.com/bar", Priority: new(int32(16))},
				{Name: "example.com/bar", Priority: new(int32(15))},
				{Name: "example.com/baz", Priority: new(int32(15))},
				{Name: "responder-y.example.com/bar", Priority: new(int32(15))},
				{Name: "responder-y.example.com/baz", Priority: new(int32(15))},
				{Name: "responder-z.example.com/bar", Priority: new(int32(15))},
				{Name: "responder-z.example.com/baz", Priority: new(int32(15))},
				{Name: "fooexample.com/bar", Priority: new(int32(15))},
				{Name: "responder-a.fooexample.com/bar", Priority: new(int32(15))},
				{Name: "responder-b.fooexample.com/bar", Priority: new(int32(15))},
				{Name: "responder-b.fooexample.com/baz", Priority: new(int32(15))},
				{Name: "responder-b.barexample.io/bar", Priority: new(int32(15))},
				{Name: "responder-b.fooexample.io/bar", Priority: new(int32(15))},
				{Name: "responder-b.fooexample.io/baz", Priority: new(int32(15))},
				{Name: "responder-k.kubernetes.io/y", Priority: new(int32(14))},
				{Name: "responder-k.kubernetes.io/y", Priority: new(int32(14))},
				{Name: "responder-k.kubernetes.io/z", Priority: new(int32(14))},
				{Name: "responder-k.kubernetes.io/x", Priority: new(int32(13))},
			},
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			input := append([]lifecycle.TargetResponder{}, tc.expectedResponders...)
			rand.Shuffle(len(input), func(i, j int) {
				input[i], input[j] = input[j], input[i]
			})
			SortTargetResponders(input)
			if diff := cmp.Diff(input, tc.expectedResponders); len(diff) > 0 {
				t.Errorf("unexpected TargetResponders sort: %v", diff)
			}
		})
	}
}
