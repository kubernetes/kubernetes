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

package authorizer_test

import (
	"reflect"
	"slices"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

// TestConditionsAwareDecisionUnionAdd exercises the Add method's bookkeeping behavior:
// it must reject duplicate authorizer names and stop appending after the first Allow/Deny leaf.
func TestConditionsAwareDecisionUnionAdd(t *testing.T) {
	noOp := authorizer.ConditionsAwareDecisionNoOpinion("", nil)
	allow := authorizer.ConditionsAwareDecisionAllow("a", nil)
	deny := authorizer.ConditionsAwareDecisionDeny("d", nil)
	condMapAllow := authorizer.ConditionsAwareDecisionConditionsMap(nil, nil, []authorizer.Condition{authorizer.GenericCondition{ID: "allow-1"}})
	condMapDeny := authorizer.ConditionsAwareDecisionConditionsMap([]authorizer.Condition{authorizer.GenericCondition{ID: "deny-1"}}, nil, nil)

	t.Run("duplicate conditional authorizers fails closed", func(t *testing.T) {
		var u authorizer.ConditionsAwareDecisionUnion
		u.Add("nop", noOp)
		u.Add("dup", condMapAllow)
		u.Add("dup", condMapAllow)

		d := u.ToDecision()
		if !d.IsNoOpinion() {
			t.Errorf("expected NoOpinion (no Deny leaf), got %s", d.String())
		}
		if d.Error().Error() != `duplicate conditionalAuthorizerName "dup"` {
			t.Errorf("expected aggregated duplicate error, got %v", d.Error())
		}
	})

	t.Run("duplicate when outcome could be deny fails closed", func(t *testing.T) {
		var u authorizer.ConditionsAwareDecisionUnion
		u.Add("dup", condMapAllow)
		u.Add("dup", condMapDeny)
		u.Add("a", allow)

		d := u.ToDecision()
		if !d.IsDeny() {
			t.Errorf("expected Deny (deny leaf present), got %s", d.String())
		}
		if d.Error().Error() != `duplicate conditionalAuthorizerName "dup"` {
			t.Errorf("expected aggregated duplicate error, got %v", d.Error())
		}
	})

	t.Run("Allow leaf short-circuits subsequent Adds", func(t *testing.T) {
		var u authorizer.ConditionsAwareDecisionUnion
		u.Add("0.example.com", noOp)
		u.Add("1.example.com", allow)
		// These additions must be silently dropped by the ContainsUnconditionalAllowOrDeny short-circuit.
		u.Add("2.example.com", deny)
		u.Add("3.example.com", condMapAllow)

		d := u.ToDecision()
		if !d.IsAllow() {
			t.Errorf("expected Allow after short-circuit, got %s", d.String())
		}

		// Iterating the resulting union should only see the entries before/at the short-circuit point.
		seen := map[string]bool{}
		for name := range d.UnionedDecisions() {
			seen[name] = true
		}
		// The wrapped decision is unconditional Allow, so UnionedDecisions is empty.
		if len(seen) != 0 {
			t.Errorf("expected unconditional Allow to expose no unioned decisions, got %v", seen)
		}
	})

	t.Run("Deny leaf short-circuits subsequent Adds", func(t *testing.T) {
		var u authorizer.ConditionsAwareDecisionUnion
		u.Add("", noOp)
		u.Add("", deny)
		u.Add("", allow)
		u.Add("3.example.com", condMapAllow)

		d := u.ToDecision()
		if !d.IsDeny() {
			t.Errorf("expected Deny after short-circuit, got %s", d.String())
		}
	})

	t.Run("ConditionsMap leaves do not short-circuit Add", func(t *testing.T) {
		var u authorizer.ConditionsAwareDecisionUnion
		u.Add("0.example.com", condMapAllow)
		u.Add("1.example.com", condMapAllow) // distinct authorizerName, still appended
		u.Add("2.example.com", noOp)

		d := u.ToDecision()
		if !d.IsUnion() {
			t.Fatalf("expected Union, got %s", d.String())
		}

		var names []string
		for name := range d.UnionedDecisions() {
			names = append(names, name)
		}
		want := []string{"0.example.com", "1.example.com", "2.example.com"}
		if !reflect.DeepEqual(names, want) {
			t.Errorf("UnionedDecisions names = %v, want %v", names, want)
		}
	})
}

// TestConditionsAwareDecisionUnion_AddAfterToDecisionDoesNotMutate verifies that once
// ConditionsAwareDecisionUnion.ToDecision has produced a decision, further calls to
// the builder's Add cannot mutate that decision — the builder's internal state
// (including its memoized possible-decisions set) is defensively copied.
func TestConditionsAwareDecisionUnion_AddAfterToDecisionDoesNotMutate(t *testing.T) {
	// Build a union with two ConditionsMap sub-decisions. Neither is an unconditional
	// Allow/Deny leaf, so the builder does NOT short-circuit subsequent Add calls.
	cmAllow1 := authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil, []authorizer.Condition{authorizer.GenericCondition{ID: "allow-1"}},
	)
	cmAllow2 := authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil, []authorizer.Condition{authorizer.GenericCondition{ID: "allow-2"}},
	)

	var u authorizer.ConditionsAwareDecisionUnion
	u.Add("first.example.com", cmAllow1)
	u.Add("second.example.com", cmAllow2)
	d := u.ToDecision()
	if !d.IsUnion() {
		t.Fatalf("expected Union decision, got %s", d.String())
	}

	collectNames := func(d authorizer.ConditionsAwareDecision) []string {
		var names []string
		for name := range d.UnionedDecisions() {
			names = append(names, name)
		}
		return names
	}

	namesBefore := []string{"first.example.com", "second.example.com"}
	possibleBefore := d.PossibleDecisions()

	// Add a further sub-decision to the builder, introducing a new possible outcome
	// (Deny) not previously present. If ToDecision failed to defensively copy the
	// builder's memoized state, this Add would leak into the already-returned decision.
	cmDeny := authorizer.ConditionsAwareDecisionConditionsMap(
		[]authorizer.Condition{authorizer.GenericCondition{ID: "deny-1"}}, nil, nil,
	)
	u.Add("third.example.com", cmDeny)

	if got := collectNames(d); !slices.Equal(got, namesBefore) {
		t.Errorf("UnionedDecisions names changed after builder Add: got %v, want %v (builder mutation must not leak)", got, namesBefore)
	}
	if got := d.PossibleDecisions(); !got.Equal(possibleBefore) {
		t.Errorf("PossibleDecisions changed after builder Add: got %v, want %v (builder mutation must not leak)",
			got.UnsortedList(), possibleBefore.UnsortedList())
	}
}
