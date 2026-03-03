/*
Copyright 2025 The Kubernetes Authors.

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

package authorizer

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authentication/user"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestDecisionZeroValueIsDeny(t *testing.T) {
	d := Decision{}
	if !d.IsDenied() {
		t.Fatal("Expected the zero value of Decision{} to be a Deny")
	}
	if d.String() != "Deny" {
		t.Fatal("Expected the zero value to string encode to 'Deny'")
	}
}

// TODO: Check that the Decision can only return true for one of the Is.. methods

var _ Authorizer = sampleAuthorizer{}

type sampleAuthorizer struct{}

func (a sampleAuthorizer) Authorize(ctx context.Context, attrs Attributes) (Decision, error) {
	switch attrs.GetUser().GetName() {
	case "alice":
		return DecisionAllow(), nil
	case "bob":
		return DecisionDeny(), nil
	case "carol":
		// allow carol to read anything, but require seting the owner=carol label on writes
		switch attrs.GetVerb() {
		case "list":
			return DecisionAllow(), nil
		case "update":
			// the authorizer is misbehaving here, it SHOULD check attrs.GetConditionsMode() and
			// fail closed to NoOpinion or Deny whenever it would like to return a conditional decision,
			// but the client does not support it. However, this check is also done in DecisionConditional here.
			return DecisionConditional(attrs, "labelSelectorApplies", maps.All(map[string]Condition{
				"owner-label-is-set": {
					// (oldobject != nil && has(oldobject.metadata.labels.owner) && oldobject.metadata.labels.owner == "carol") &&
					// (object != nil && has(object.metadata.labels.owner) && object.metadata.labels.owner == "carol")
					Condition: "owner=carol|owner=carol",
					Effect:    ConditionEffectAllow,
				},
			}))
		default:
			return DecisionNoOpinion(), nil
		}
	case "dave":
		// allow dave to read anything, but never set the classified label on writes
		switch attrs.GetVerb() {
		case "list":
			return DecisionAllow(), nil
		case "create", "update", "delete":
			// the authorizer is misbehaving here, it SHOULD check attrs.GetConditionsMode() and
			// fail closed to NoOpinion or Deny whenever it would like to return a conditional decision,
			// but the client does not support it. However, this check is also done in DecisionConditional here.
			return DecisionConditional(attrs, "labelSelectorApplies", maps.All(map[string]Condition{
				"deny-supersecret-label-on-oldobject": {
					// (oldobject != nil && has(oldobject.metadata.labels.supersecret)) && true
					Condition: "supersecret|",
					Effect:    ConditionEffectDeny,
				},
				"deny-supersecret-label-on-object": {
					// true && (object != nil && has(object.metadata.labels.supersecret))
					Condition: "|supersecret",
					Effect:    ConditionEffectDeny,
				},
			}))
		default:
			return DecisionNoOpinion(), nil
		}
	default:
		return DecisionNoOpinion(), nil
	}
}

func (a sampleAuthorizer) EvaluateConditions(ctx context.Context, d Decision, data ConditionData) (Decision, error) {
	if d.IsAllowed() || d.IsDenied() || d.IsNoOpinion() {
		return d, nil
	}
	if d.IsConditionalChain() {
		return d.FailClosedDecision(), errors.New("conditionschain unsupported")
	}
	// TODO: improve this
	if data.WriteRequest() == nil {
		return d.FailClosedDecision(), errors.New("only supports conditions for write requests")
	}

	enforceObjects := []runtime.Object{
		data.WriteRequest().GetOldObject(),
		data.WriteRequest().GetObject(),
	}

	decision, _, err := EvaluateConditionSet(d.ConditionSet(), "labelSelectorApplies", func(condition string) (bool, error) {
		// condition is of form: "label-selector-for-oldobject|label-selector-for-object"
		// if label-selector-for-oldobject is empty, it means "true"
		selectorStrs := strings.Split(condition, "|")
		if len(selectorStrs) != 2 {
			return false, errors.New("invalid labelselector condition")
		}
		for i, selectorStr := range selectorStrs {
			if len(selectorStr) == 0 {
				continue
			}
			runtimeObj := enforceObjects[i]
			if runtimeObj == nil {
				return false, nil
			}
			obj, ok := runtimeObj.(metav1.Object)
			if !ok {
				return false, errors.New("only supports objects with metadata")
			}
			selector, err := labels.Parse(selectorStr)
			if err != nil {
				return false, err
			}

			if !selector.Matches(labels.Set(obj.GetLabels())) {
				return false, nil
			}
		}
		return true, nil
	})
	return decision, err
}

// testConditionData implements ConditionData for testing.
type testConditionData struct {
	writeReq *testWriteRequest
}

func (d *testConditionData) WriteRequest() WriteRequestConditionData {
	if d.writeReq == nil {
		return nil
	}
	return d.writeReq
}

func (d *testConditionData) ImpersonationRequest() ImpersonationRequestConditionData {
	return nil
}

// testWriteRequest implements WriteRequestConditionData for testing.
type testWriteRequest struct {
	object    runtime.Object
	oldObject runtime.Object
}

func (r *testWriteRequest) GetOperation() string {
	switch {
	case r.object != nil && r.oldObject == nil:
		return "CREATE"
	case r.object != nil && r.oldObject != nil:
		return "UPDATE"
	case r.object == nil && r.oldObject != nil:
		return "DELETE"
	default:
		return "UNKNOWN"
	}
}
func (r *testWriteRequest) GetOperationOptions() runtime.Object { return nil }
func (r *testWriteRequest) GetObject() runtime.Object           { return r.object }
func (r *testWriteRequest) GetOldObject() runtime.Object        { return r.oldObject }

func objWithLabels(lbls map[string]string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{Object: map[string]any{}}
	if len(lbls) > 0 {
		obj.SetLabels(lbls)
	}
	return obj
}

// if we just cast a nil but u to runtime.Object, the runtimeObj == nil gives false when
// one wants true.
func runtimeObj(u *unstructured.Unstructured) runtime.Object {
	if u == nil {
		return nil
	}
	return u
}

func TestSampleAuthorizer(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
	type evalCase struct {
		name      string
		object    *unstructured.Unstructured
		oldObject *unstructured.Unstructured
		// the first case is with ConditionsModeNone, the second with ConditionsModeHumanReadable
		authorizeDecision [2]string
		finalDecision     [2]string
	}

	tests := []struct {
		name  string
		attrs AttributesRecord
		cases []evalCase
	}{
		// alice: unconditional allow for all verbs
		{
			name: "alice list",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "alice"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{"Allow", "Allow"}},
			},
		},
		{
			name: "alice create",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "alice"},
				Verb: "create",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{"Allow", "Allow"}},
			},
		},
		// bob: unconditional deny for all verbs
		{
			name: "bob list",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "bob"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "deny", authorizeDecision: [2]string{"Deny", "Deny"}},
			},
		},
		{
			name: "bob create",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "bob"},
				Verb: "create",
			},
			cases: []evalCase{
				{name: "deny", authorizeDecision: [2]string{"Deny", "Deny"}},
			},
		},
		// carol: allow reads, conditional writes (EffectAllow on owner=carol)
		{
			name: "carol list",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "carol"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{"Allow", "Allow"}},
			},
		},
		{
			name: "carol update",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "carol"},
				Verb: "update",
			},
			cases: []evalCase{
				{
					name:              "both objects with owner=carol",
					object:            objWithLabels(map[string]string{"owner": "carol"}),
					oldObject:         objWithLabels(map[string]string{"owner": "carol"}),
					authorizeDecision: [2]string{"NoOpinion", `Conditional(type="labelSelectorApplies", len=1)`},
					finalDecision:     [2]string{"NoOpinion", "Allow"},
				},
				{
					name:              "old with owner=carol, new without",
					object:            objWithLabels(map[string]string{"owner": "carol"}),
					oldObject:         objWithLabels(nil),
					authorizeDecision: [2]string{"NoOpinion", `Conditional(type="labelSelectorApplies", len=1)`},
					finalDecision:     [2]string{"NoOpinion", "NoOpinion"},
				},
				{
					name:              "new with owner=carol, old with owner=alice",
					object:            objWithLabels(map[string]string{"owner": "alice"}),
					oldObject:         objWithLabels(map[string]string{"owner": "carol"}),
					authorizeDecision: [2]string{"NoOpinion", `Conditional(type="labelSelectorApplies", len=1)`},
					finalDecision:     [2]string{"NoOpinion", "NoOpinion"},
				},
			},
		},
		{
			name: "carol unsupported verb",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "carol"},
				Verb: "patch",
			},
			cases: []evalCase{
				{name: "no opinion", authorizeDecision: [2]string{"NoOpinion", "NoOpinion"}},
			},
		},
		// dave: allow reads, conditional writes (EffectDeny on supersecret label)
		{
			name: "dave list",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "allow", authorizeDecision: [2]string{"Allow", "Allow"}},
			},
		},

		{
			name: "dave update",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "update",
			},
			cases: []evalCase{
				{
					name:              "both objects with supersecret",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "Deny"},
				},
				{
					name:              "new with supersecret old without",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					oldObject:         objWithLabels(nil),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "Deny"},
				},
				{
					name:              "new without old with supersecret",
					object:            objWithLabels(nil),
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "Deny"},
				},
				{
					name:              "both without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "NoOpinion"},
				},
			},
		},
		{
			name: "dave create",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "create",
			},
			cases: []evalCase{
				{
					name:              "create with supersecret",
					object:            objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "Deny"},
				},
				{
					name:              "create without supersecret",
					object:            objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "NoOpinion"},
				},
			},
		},
		{
			name: "dave delete",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "delete",
			},
			cases: []evalCase{
				{
					name:              "delete with supersecret on old object",
					oldObject:         objWithLabels(map[string]string{"supersecret": "yes"}),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "Deny"},
				},
				{
					name:              "delete without supersecret on old object",
					oldObject:         objWithLabels(map[string]string{"safe": "true"}),
					authorizeDecision: [2]string{"Deny", `Conditional(type="labelSelectorApplies", len=2)`},
					finalDecision:     [2]string{"Deny", "NoOpinion"},
				},
			},
		},
		{
			name: "dave unsupported verb",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "dave"},
				Verb: "patch",
			},
			cases: []evalCase{
				{name: "no opinion", authorizeDecision: [2]string{"NoOpinion", "NoOpinion"}},
			},
		},
		// unknown user: no opinion
		{
			name: "unknown user get",
			attrs: AttributesRecord{
				User: &user.DefaultInfo{Name: "unknown"},
				Verb: "list",
			},
			cases: []evalCase{
				{name: "no opinion", authorizeDecision: [2]string{"NoOpinion", "NoOpinion"}},
			},
		},
	}

	authz := sampleAuthorizer{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, tc := range tt.cases {
				// if only the authorization decision is specified, the final one is the same
				if len(tc.finalDecision[0]) == 0 && len(tc.finalDecision[1]) == 0 {
					tc.finalDecision[0] = tc.authorizeDecision[0]
					tc.finalDecision[1] = tc.authorizeDecision[1]
				}
				for i, conditionsMode := range [2]ConditionsMode{ConditionsModeNone, ConditionsModeHumanReadable} {
					t.Run(fmt.Sprintf("%s/%s", tc.name, conditionsMode), func(t *testing.T) {
						localAttrs := tt.attrs
						localAttrs.ConditionsMode = conditionsMode

						ctx := context.Background()
						decision, err := authz.Authorize(ctx, localAttrs)
						if err != nil {
							t.Fatalf("Authorize() returned unexpected error: %v", err)
						}

						if decision.String() != tc.authorizeDecision[i] {
							t.Errorf("got Authorize() decision %s (reason: %q), want %s", decision.String(), decision.Reason(), tc.authorizeDecision[i])
						}

						data := &testConditionData{
							writeReq: &testWriteRequest{
								object:    runtimeObj(tc.object),
								oldObject: runtimeObj(tc.oldObject),
							},
						}

						final, err := authz.EvaluateConditions(ctx, decision, data)
						if err != nil {
							t.Fatalf("Evaluate() returned unexpected error: %v", err)
						}
						if final.String() != tc.finalDecision[i] {
							t.Errorf("got Evaluate() decision %s (reason: %q), want %s", final.String(), final.Reason(), tc.finalDecision[i])
						}
					})
				}
			}
		})
	}
}

func TestEvaluateConditionSet(t *testing.T) {
	evalErr := errors.New("eval error")

	tests := []struct {
		name          string
		conditionSet  *ConditionSet
		supportedType ConditionType
		eval          func(string) (bool, error)
		wantDecision  string
		wantWarnings  bool
		wantErr       bool
		wantReason    string
	}{
		// nil conditionSet
		{
			name:          "nil condition set",
			conditionSet:  nil,
			supportedType: "test",
			eval:          func(string) (bool, error) { panic("should not be called") },
			wantDecision:  "NoOpinion",
		},

		// wrong condition type
		{
			name: "wrong type with only allow conditions",
			conditionSet: &ConditionSet{
				conditionType: "wrong-type",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { panic("should not be called") },
			wantDecision:  "NoOpinion",
			wantErr:       true,
		},
		{
			name: "wrong type with only noopinion conditions",
			conditionSet: &ConditionSet{
				conditionType: "wrong-type",
				conditions: map[string]Condition{
					"nop-cond": {Condition: "x", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { panic("should not be called") },
			wantDecision:  "NoOpinion",
			wantErr:       true,
		},
		{
			name: "wrong type with deny conditions",
			conditionSet: &ConditionSet{
				conditionType: "wrong-type",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { panic("should not be called") },
			wantDecision:  "Deny",
			wantErr:       true,
		},

		// Deny conditions
		{
			name: "deny condition matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-cond": {Condition: "x", Effect: ConditionEffectDeny, Description: "access denied"},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return true, nil },
			wantDecision:  "Deny",
			wantReason:    `condition "deny-cond" denied the request with description "access denied"`,
		},
		{
			name: "deny condition does not match",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-cond": {Condition: "x", Effect: ConditionEffectDeny},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return false, nil },
			wantDecision:  "NoOpinion",
			wantReason:    "no conditions matched",
		},
		{
			name: "deny condition eval error; deny error trumps matching allow",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "err", Effect: ConditionEffectDeny},
					"allow-cond": {Condition: "ok", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				if cond == "err" {
					return false, evalErr
				}
				return true, nil
			},
			wantDecision: "Deny",
			wantErr:      true,
			wantReason:   "one or more conditional evaluation errors occurred",
		},
		{
			name: "first deny no match second deny matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-no":  {Condition: "no-match", Effect: ConditionEffectDeny},
					"deny-yes": {Condition: "match", Effect: ConditionEffectDeny},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				return cond == "match", nil
			},
			wantDecision: "Deny",
			wantReason:   `condition "deny-yes" denied the request`,
		},

		// NoOpinion conditions
		{
			name: "noopinion condition matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"nop-cond": {Condition: "x", Effect: ConditionEffectNoOpinion, Description: "not relevant"},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return true, nil },
			wantDecision:  "NoOpinion",
			wantReason:    `condition "nop-cond" evaluated to NoOpinion with description "not relevant"`,
		},
		{
			name: "noopinion condition does not match",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"nop-cond": {Condition: "x", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return false, nil },
			wantDecision:  "NoOpinion",
			wantReason:    "no conditions matched",
		},
		{
			name: "noopinion condition eval error, noopinion error trumps matching allow",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"nop-cond":   {Condition: "err", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "ok", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				if cond == "err" {
					return false, evalErr
				}
				return true, nil
			},
			wantDecision: "NoOpinion",
			wantErr:      true,
			wantReason:   "one or more conditional evaluation errors occurred",
		},
		{
			name: "first noopinion no match second noopinion matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"nop-no":  {Condition: "no-match", Effect: ConditionEffectNoOpinion},
					"nop-yes": {Condition: "match", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				return cond == "match", nil
			},
			wantDecision: "NoOpinion",
			wantReason:   `condition "nop-yes" evaluated to NoOpinion`,
		},

		// Allow conditions
		{
			name: "allow condition matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow, Description: "access granted"},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return true, nil },
			wantDecision:  "Allow",
			wantReason:    `condition "allow-cond" allowed the request with description "access granted"`,
		},
		{
			name: "allow condition does not match",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return false, nil },
			wantDecision:  "NoOpinion",
			wantReason:    "no conditions matched",
		},
		{
			name: "allow condition eval error",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"allow-err1": {Condition: "x", Effect: ConditionEffectAllow},
					"allow-err2": {Condition: "y", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return false, evalErr },
			wantDecision:  "NoOpinion",
			wantErr:       true,
			wantReason:    "one or more conditional evaluation errors occurred",
		},
		{
			name: "allow first errors second matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"allow-err": {Condition: "error-cond", Effect: ConditionEffectAllow},
					"allow-ok":  {Condition: "match-cond", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				if cond == "error-cond" {
					return false, evalErr
				}
				return true, nil
			},
			wantDecision: "Allow",
			wantWarnings: true,
			wantReason:   `condition "allow-ok" allowed the request`,
		},

		// Precedence: Deny > NoOpinion > Allow
		{
			name: "deny takes precedence over allow",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return true, nil },
			wantDecision:  "Deny",
			wantReason:    `condition "deny-cond" denied the request`,
		},
		{
			name: "deny takes precedence over noopinion and allow",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
					"nop-cond":   {Condition: "x", Effect: ConditionEffectNoOpinion},
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return true, nil },
			wantDecision:  "Deny",
			wantReason:    `condition "deny-cond" denied the request`,
		},
		{
			name: "noopinion takes precedence over allow",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
					"nop-cond":   {Condition: "x", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return true, nil },
			wantDecision:  "NoOpinion",
			wantReason:    `condition "nop-cond" evaluated to NoOpinion`,
		},

		// Selective matching across effect types
		{
			name: "deny no match, noopinion matches, allow matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "deny-check", Effect: ConditionEffectDeny},
					"nop-cond":   {Condition: "nop-check", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "allow-check", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				return cond != "deny-check", nil
			},
			wantDecision: "NoOpinion",
			wantReason:   `condition "nop-cond" evaluated to NoOpinion`,
		},
		{
			name: "only allow matches",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "deny-check", Effect: ConditionEffectDeny},
					"nop-cond":   {Condition: "nop-check", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "allow-check", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				return cond == "allow-check", nil
			},
			wantDecision: "Allow",
			wantReason:   `condition "allow-cond" allowed the request`,
		},
		{
			name: "no conditions match across all effects",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-cond":  {Condition: "x", Effect: ConditionEffectDeny},
					"nop-cond":   {Condition: "x", Effect: ConditionEffectNoOpinion},
					"allow-cond": {Condition: "x", Effect: ConditionEffectAllow},
				},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { return false, nil },
			wantDecision:  "NoOpinion",
			wantReason:    "no conditions matched",
		},

		// Empty condition set (non-nil but no conditions)
		{
			name: "empty condition set with no conditions",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions:    map[string]Condition{},
			},
			supportedType: "test",
			eval:          func(string) (bool, error) { panic("should not be called") },
			wantDecision:  "NoOpinion",
			wantReason:    "no conditions matched",
		},

		// Deny no match, noopinion errors: fail closed to NoOpinion
		{
			name: "deny no match noopinion errors",
			conditionSet: &ConditionSet{
				conditionType: "test",
				conditions: map[string]Condition{
					"deny-cond": {Condition: "deny-check", Effect: ConditionEffectDeny},
					"nop-cond":  {Condition: "nop-check", Effect: ConditionEffectNoOpinion},
				},
			},
			supportedType: "test",
			eval: func(cond string) (bool, error) {
				if cond == "nop-check" {
					return false, evalErr
				}
				return false, nil
			},
			wantDecision: "NoOpinion",
			wantErr:      true,
			wantReason:   "one or more conditional evaluation errors occurred",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision, warnings, err := EvaluateConditionSet(tt.conditionSet, tt.supportedType, tt.eval)

			if (err != nil) != tt.wantErr {
				t.Fatalf("EvaluateConditionSet() error = %v, wantErr %v", err, tt.wantErr)
			}
			if (len(warnings) > 0) != tt.wantWarnings {
				t.Fatalf("EvaluateConditionSet() warnings = %v, wantWarnings %v", warnings, tt.wantWarnings)
			}
			if decision.String() != tt.wantDecision {
				t.Errorf("got decision %s (reason: %q), want %s", decision.String(), decision.Reason(), tt.wantDecision)
			}
			if decision.Reason() != tt.wantReason {
				t.Errorf("got reason %q, want %q", decision.Reason(), tt.wantReason)
			}
		})
	}
}
