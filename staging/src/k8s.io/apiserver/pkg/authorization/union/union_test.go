/*
Copyright 2014 The Kubernetes Authors.

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

package union

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

type mockAuthzHandler struct {
	decision authorizer.Decision
	err      error
}

func (mock *mockAuthzHandler) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return mock.decision, "", mock.err
}

// ConditionsAwareAuthorize is not conditions-aware, converts the Authorize decision.
func (mock *mockAuthzHandler) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionFromParts(mock.Authorize(ctx, a))
}

// EvaluateConditions is not supported by this authorizer.
func (*mockAuthzHandler) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

// mustNewIndexed wraps each authorizer with an auto-generated name and constructs
// a union authorizer. Intended for tests that don't care about specific names.
func mustNewIndexed(t *testing.T, handlers ...authorizer.Authorizer) authorizer.Authorizer {
	t.Helper()
	named := make([]NamedAuthorizer, len(handlers))
	for i, h := range handlers {
		named[i] = NamedAuthorizer{AuthorizerName: fmt.Sprintf("authz%d", i), Authorizer: h}
	}
	u, err := New(named...)
	if err != nil {
		t.Fatalf("union.New returned an unexpected error: %v", err)
	}
	return u
}

func TestAuthorizationSecondPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionAllow}
	authzHandler := mustNewIndexed(t, handler1, handler2)

	authorized, _, _ := authzHandler.Authorize(context.Background(), nil)
	if authorized != authorizer.DecisionAllow {
		t.Errorf("Unexpected authorization failure")
	}
}

func TestAuthorizationFirstPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionAllow}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	authzHandler := mustNewIndexed(t, handler1, handler2)

	authorized, _, _ := authzHandler.Authorize(context.Background(), nil)
	if authorized != authorizer.DecisionAllow {
		t.Errorf("Unexpected authorization failure")
	}
}

func TestAuthorizationNonePasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	authzHandler := mustNewIndexed(t, handler1, handler2)

	authorized, _, _ := authzHandler.Authorize(context.Background(), nil)
	if authorized == authorizer.DecisionAllow {
		t.Errorf("Expected failed authorization")
	}
}

func TestAuthorizationError(t *testing.T) {
	handler1 := &mockAuthzHandler{err: fmt.Errorf("foo")}
	handler2 := &mockAuthzHandler{err: fmt.Errorf("foo")}
	authzHandler := mustNewIndexed(t, handler1, handler2)

	_, _, err := authzHandler.Authorize(context.Background(), nil)
	if err == nil {
		t.Errorf("Expected error: %v", err)
	}
}

type mockAuthzRuleHandler struct {
	resourceRules    []authorizer.ResourceRuleInfo
	nonResourceRules []authorizer.NonResourceRuleInfo
	err              error
}

func (mock *mockAuthzRuleHandler) RulesFor(ctx context.Context, user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	if mock.err != nil {
		return []authorizer.ResourceRuleInfo{}, []authorizer.NonResourceRuleInfo{}, false, mock.err
	}
	return mock.resourceRules, mock.nonResourceRules, false, nil
}

func TestAuthorizationResourceRules(t *testing.T) {
	handler1 := &mockAuthzRuleHandler{
		resourceRules: []authorizer.ResourceRuleInfo{
			&authorizer.DefaultResourceRuleInfo{
				Verbs:     []string{"*"},
				APIGroups: []string{"*"},
				Resources: []string{"bindings"},
			},
			&authorizer.DefaultResourceRuleInfo{
				Verbs:     []string{"get", "list", "watch"},
				APIGroups: []string{"*"},
				Resources: []string{"*"},
			},
		},
	}
	handler2 := &mockAuthzRuleHandler{
		resourceRules: []authorizer.ResourceRuleInfo{
			&authorizer.DefaultResourceRuleInfo{
				Verbs:     []string{"*"},
				APIGroups: []string{"*"},
				Resources: []string{"events"},
			},
			&authorizer.DefaultResourceRuleInfo{
				Verbs:         []string{"get"},
				APIGroups:     []string{"*"},
				Resources:     []string{"*"},
				ResourceNames: []string{"foo"},
			},
		},
	}

	expected := []authorizer.DefaultResourceRuleInfo{
		{
			Verbs:     []string{"*"},
			APIGroups: []string{"*"},
			Resources: []string{"bindings"},
		},
		{
			Verbs:     []string{"get", "list", "watch"},
			APIGroups: []string{"*"},
			Resources: []string{"*"},
		},
		{
			Verbs:     []string{"*"},
			APIGroups: []string{"*"},
			Resources: []string{"events"},
		},
		{
			Verbs:         []string{"get"},
			APIGroups:     []string{"*"},
			Resources:     []string{"*"},
			ResourceNames: []string{"foo"},
		},
	}

	authzRulesHandler := NewRuleResolvers(handler1, handler2)

	rules, _, _, _ := authzRulesHandler.RulesFor(genericapirequest.NewContext(), nil, "")
	actual := getResourceRules(rules)
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Expected: \n%#v\n but actual: \n%#v\n", expected, actual)
	}
}

func TestAuthorizationNonResourceRules(t *testing.T) {
	handler1 := &mockAuthzRuleHandler{
		nonResourceRules: []authorizer.NonResourceRuleInfo{
			&authorizer.DefaultNonResourceRuleInfo{
				Verbs:           []string{"get"},
				NonResourceURLs: []string{"/api"},
			},
		},
	}

	handler2 := &mockAuthzRuleHandler{
		nonResourceRules: []authorizer.NonResourceRuleInfo{
			&authorizer.DefaultNonResourceRuleInfo{
				Verbs:           []string{"get"},
				NonResourceURLs: []string{"/api/*"},
			},
		},
	}

	expected := []authorizer.DefaultNonResourceRuleInfo{
		{
			Verbs:           []string{"get"},
			NonResourceURLs: []string{"/api"},
		},
		{
			Verbs:           []string{"get"},
			NonResourceURLs: []string{"/api/*"},
		},
	}

	authzRulesHandler := NewRuleResolvers(handler1, handler2)

	_, rules, _, _ := authzRulesHandler.RulesFor(genericapirequest.NewContext(), nil, "")
	actual := getNonResourceRules(rules)
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("Expected: \n%#v\n but actual: \n%#v\n", expected, actual)
	}
}

func getResourceRules(infos []authorizer.ResourceRuleInfo) []authorizer.DefaultResourceRuleInfo {
	rules := make([]authorizer.DefaultResourceRuleInfo, len(infos))
	for i, info := range infos {
		rules[i] = authorizer.DefaultResourceRuleInfo{
			Verbs:         info.GetVerbs(),
			APIGroups:     info.GetAPIGroups(),
			Resources:     info.GetResources(),
			ResourceNames: info.GetResourceNames(),
		}
	}
	return rules
}

func getNonResourceRules(infos []authorizer.NonResourceRuleInfo) []authorizer.DefaultNonResourceRuleInfo {
	rules := make([]authorizer.DefaultNonResourceRuleInfo, len(infos))
	for i, info := range infos {
		rules[i] = authorizer.DefaultNonResourceRuleInfo{
			Verbs:           info.GetVerbs(),
			NonResourceURLs: info.GetNonResourceURLs(),
		}
	}
	return rules
}

func TestAuthorizationUnequivocalDeny(t *testing.T) {
	cs := []struct {
		authorizers []authorizer.Authorizer
		decision    authorizer.Decision
	}{
		{
			authorizers: []authorizer.Authorizer{},
			decision:    authorizer.DecisionNoOpinion,
		},
		{
			authorizers: []authorizer.Authorizer{
				&mockAuthzHandler{decision: authorizer.DecisionNoOpinion},
				&mockAuthzHandler{decision: authorizer.DecisionAllow},
				&mockAuthzHandler{decision: authorizer.DecisionDeny},
			},
			decision: authorizer.DecisionAllow,
		},
		{
			authorizers: []authorizer.Authorizer{
				&mockAuthzHandler{decision: authorizer.DecisionNoOpinion},
				&mockAuthzHandler{decision: authorizer.DecisionDeny},
				&mockAuthzHandler{decision: authorizer.DecisionAllow},
			},
			decision: authorizer.DecisionDeny,
		},
		{
			authorizers: []authorizer.Authorizer{
				&mockAuthzHandler{decision: authorizer.DecisionNoOpinion},
				&mockAuthzHandler{decision: authorizer.DecisionDeny, err: errors.New("webhook failed closed")},
				&mockAuthzHandler{decision: authorizer.DecisionAllow},
			},
			decision: authorizer.DecisionDeny,
		},
	}
	for i, c := range cs {
		t.Run(fmt.Sprintf("case %v", i), func(t *testing.T) {
			authzHandler := mustNewIndexed(t, c.authorizers...)

			decision, _, _ := authzHandler.Authorize(context.Background(), nil)
			if decision != c.decision {
				t.Errorf("Unexpected authorization failure: %v, expected: %v", decision, c.decision)
			}
		})
	}
}

type evalTestEffect string

const (
	effectNone      evalTestEffect = ""
	effectAllow     evalTestEffect = "Allow"
	effectDeny      evalTestEffect = "Deny"
	effectNoOpinion evalTestEffect = "NoOpinion"
)

// evalTestAuthz is a configurable authorizer for testing the union evaluation flow.
type evalTestAuthz struct {
	// conditionEffect, if non-empty, makes AuthorizeConditionsAware return a ConditionsMap decision
	// with a single condition of this effect. If empty, decision is returned instead.
	conditionEffect evalTestEffect
	// decision is returned from AuthorizeConditionsAware when conditionEffect is empty.
	decision authorizer.Decision
	// authorizeErr is returned as the error from AuthorizeConditionsAware.
	authorizeErr error

	// evalDecision is returned from EvaluateConditions.
	evalDecision authorizer.Decision
	// evalErr is returned as the error from EvaluateConditions.
	evalErr error
}

func (a *evalTestAuthz) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	if a.conditionEffect != effectNone {
		return a.ConditionsAwareAuthorize(ctx, attrs).FailureDecision(), "failed closed", fmt.Errorf("wanted to return a condition, but the client doesn't support it")
	}
	return a.decision, "", a.authorizeErr
}

func (a *evalTestAuthz) ConditionsAwareAuthorize(ctx context.Context, attrs authorizer.Attributes) authorizer.ConditionsAwareDecision {
	if a.conditionEffect != effectNone {
		cond := []authorizer.Condition{authorizer.GenericCondition{ID: "test-cond", Condition: "test"}}
		var deny, noOpinion, allow []authorizer.Condition
		switch a.conditionEffect {
		case effectAllow:
			allow = cond
		case effectDeny:
			deny = cond
		case effectNoOpinion:
			noOpinion = cond
		}
		return authorizer.ConditionsAwareDecisionConditionsMap(deny, noOpinion, allow)
	}
	return authorizer.ConditionsAwareDecisionFromParts(a.decision, "", a.authorizeErr)
}

func (a *evalTestAuthz) EvaluateConditions(ctx context.Context, decision authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	if !decision.IsConditionsMap() {
		return decision.FailureDecision(), "failed closed", errors.New("evalTestAuthz.EvaluateConditions must be called on a ConditionsMap, which was returned previously")
	}
	return a.evalDecision, "", a.evalErr
}

// TestUnionEvaluateConditions tests the full Authorize + EvaluateConditions flow
// through a DAG of nested union authorizers:
//
//		union0 = [union1, union3, authz5]
//		union1 = [union2, authz3]
//		union2 = [authz1, authz2]
//	    union3 = [authz4]
func TestUnionEvaluateConditions(t *testing.T) {

	noOpinion := func() *evalTestAuthz {
		return &evalTestAuthz{decision: authorizer.DecisionNoOpinion}
	}

	tests := []struct {
		name                                   string
		authz1, authz2, authz3, authz4, authz5 *evalTestAuthz
		wantAuthorizeDecision                  string
		wantFinalDecision                      string
		wantAuthorizeErr                       bool
		wantFinalErr                           bool
	}{
		// === Concrete decisions (no conditions) ===

		{
			name:                  "all noopinion",
			authz1:                noOpinion(),
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `NoOpinion`,
			wantFinalDecision:     `NoOpinion`,
		},
		{
			name:                  "authz1 allow short-circuits everything",
			authz1:                &evalTestAuthz{decision: authorizer.DecisionAllow},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Allow`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:                  "authz1 deny short-circuits everything",
			authz1:                &evalTestAuthz{decision: authorizer.DecisionDeny},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Deny`,
			wantFinalDecision:     `Deny`,
		},
		{
			name:                  "authz1 noopinion authz2 allow",
			authz1:                noOpinion(),
			authz2:                &evalTestAuthz{decision: authorizer.DecisionAllow},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Allow`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:                  "authz1 authz2 noopinion authz3 allow",
			authz1:                noOpinion(),
			authz2:                noOpinion(),
			authz3:                &evalTestAuthz{decision: authorizer.DecisionAllow},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Allow`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:                  "all inner noopinion authz5 allow",
			authz1:                noOpinion(),
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                &evalTestAuthz{decision: authorizer.DecisionAllow},
			wantAuthorizeDecision: `Allow`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:                  "authz1 noopinion authz2 deny",
			authz1:                noOpinion(),
			authz2:                &evalTestAuthz{decision: authorizer.DecisionDeny},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Deny`,
			wantFinalDecision:     `Deny`,
		},

		// === Conditional decisions ===

		{
			name: "authz2 conditional allow evals to allow",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(allows=1), authz2: NoOpinion], authz3: NoOpinion], union3: NoOpinion, authz5: NoOpinion]`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:   "authz2 conditional allow evals to noopinion",
			authz1: noOpinion(),
			authz2: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: NoOpinion, authz2: ConditionsMap(allows=1)], authz3: NoOpinion], union3: NoOpinion, authz5: NoOpinion]`,
			wantFinalDecision:     `NoOpinion`,
		},
		{
			name:   "authz3 conditional deny evals to deny",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: NoOpinion, authz3: ConditionsMap(denies=1)], union3: NoOpinion, authz5: NoOpinion]`,
			wantFinalDecision:     `Deny`,
		},
		{
			name:   "authz4 conditional deny evals to noopinion",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: noOpinion(),
			authz4: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: NoOpinion, union3: Union[authz4: ConditionsMap(denies=1)], authz5: NoOpinion]`,
			wantFinalDecision:     `NoOpinion`,
		},
		{
			name:   "authz5 conditional noopinion evals to noopinion",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: noOpinion(),
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: effectNoOpinion,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			wantAuthorizeDecision: `NoOpinion(reason="authz5: {only NoOpinion conditions always evaluate to NoOpinion}")`,
			wantFinalDecision:     `NoOpinion(reason="authz5: {only NoOpinion conditions always evaluate to NoOpinion}")`,
		},

		// === Conditional + concrete mixes ===

		// conditional => noopinion, noopinion 	=== NoOpinion ok
		// conditional => noopinion, allow		=== Allow ok
		// conditional => noopinion, deny		=== Deny ok
		// conditional => allow, noopinion		=== Allow ok
		// conditional => allow, allow			=== Allow ok
		// conditional => allow, deny			=== Allow ok
		// conditional => deny, noopinion		=== Deny ok
		// conditional => deny, allow			=== Deny ok
		// conditional => deny, deny			=== Deny ok
		//
		// conditional => noopinion, conditional => noopinion	=== NoOpinion ok
		// conditional => noopinion, conditional => allow		=== Allow ok
		// conditional => noopinion, conditional => deny		=== Deny ok
		// ==> summarized as conditional => noopinion, X		=== X
		// conditional => allow, conditional => noopinion		=== Allow
		// conditional => allow, conditional => allow			=== Allow
		// conditional => allow, conditional => deny			=== Allow
		// ==> summarized as conditional => allow, <anything> 	=== Allow
		// conditional => deny, conditional => noopinion		=== Deny
		// conditional => deny, conditional => allow			=== Deny
		// conditional => deny, conditional => deny				=== Deny
		// ==> summarized as conditional => deny, <anything> 	=== Deny

		// Theorem: NoOpinion decisions can be inserted at any point
		//		    === Final decision the same regardless of permutation
		// Theorem: The final Decision is always the same, no matter if the
		//			authorizer list is flat or chopped into a DAG
		//
		// Theorem: The suffix after a concrete Allow or Deny does not matter:
		//
		// allow, <anything>		=== Allow
		// deny, <anything>			=== Deny

		// TODO(luxas): Implement differential testing
		// a) For a chain of length N, create all permutations of N^6 as test cases
		//	  ==> or instead work through all the possible combinations, and then insert NoOpinions?
		//	  ==> e.g. start with "I want an Allow response, what are all the ways I can get that?"
		// b) Using the formal model, compute the final decision
		// c) For each test case, create each possible DAG combination of that chain
		// d) For each DAG combo, run Authorize + Evaluate, and assert the final decision

		{
			// PossibleDecisions of [CM(allow), Allow] is {Allow} (the CM either matches → Allow,
			// or evaluates to NoOpinion → next leaf Allow), so each nested union eagerly
			// simplifies to Allow.
			name: "authz1 conditional => noopinion, authz2 allow (eagerly simplifies)",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz2:                &evalTestAuthz{decision: authorizer.DecisionAllow},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Allow`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:   "authz3 conditional => noopinion, authz5 deny",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz4:                noOpinion(),
			authz5:                &evalTestAuthz{decision: authorizer.DecisionDeny},
			wantAuthorizeDecision: `Union[union1: Union[union2: NoOpinion, authz3: ConditionsMap(allows=1)], union3: NoOpinion, authz5: Deny]`,
			wantFinalDecision:     `Deny`,
		},
		{
			name: "authz1 conditional => allow, authz2 allow (eagerly simplifies)",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			authz2:                &evalTestAuthz{decision: authorizer.DecisionAllow},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Allow`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:   "authz3 conditional => allow, authz5 deny",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			authz4:                noOpinion(),
			authz5:                &evalTestAuthz{decision: authorizer.DecisionDeny},
			wantAuthorizeDecision: `Union[union1: Union[union2: NoOpinion, authz3: ConditionsMap(allows=1)], union3: NoOpinion, authz5: Deny]`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:   "authz2 conditional => deny, authz3 allow",
			authz1: noOpinion(),
			authz2: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			authz3:                &evalTestAuthz{decision: authorizer.DecisionAllow},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: NoOpinion, authz2: ConditionsMap(denies=1)], authz3: Allow]]`,
			wantFinalDecision:     `Deny`,
		},
		{
			// inner=[union1=NoOpinion, union3=Union[CM(deny)], authz5=Deny].
			// PossibleDecisions: NoOpinion → {NoOpinion}; Union[CM(deny)] → {NoOpinion, Deny};
			// Deny → {Deny} (short-circuits, NoOpinion deleted) ⇒ {Deny}. Eagerly simplifies.
			name:   "authz4 conditional => deny, authz5 deny (eagerly simplifies to Deny)",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: noOpinion(),
			authz4: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			authz5:                &evalTestAuthz{decision: authorizer.DecisionDeny},
			wantAuthorizeDecision: `Deny`,
			wantFinalDecision:     `Deny`,
		},

		// === Multiple conditionals ===

		{
			name: "authz1 conditional => noopinion, authz3 conditional => allow",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(allows=1), authz2: NoOpinion], authz3: ConditionsMap(allows=1)], union3: NoOpinion, authz5: NoOpinion]`,
			wantFinalDecision:     `Allow`,
		},
		{
			name:   "authz1 conditional => noopinion, authz3 conditional => deny",
			authz1: noOpinion(),
			authz2: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz3: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: NoOpinion, authz2: ConditionsMap(allows=1)], authz3: ConditionsMap(denies=1)], union3: NoOpinion, authz5: NoOpinion]`,
			wantFinalDecision:     `Deny`,
		},
		{
			name: "authz1 conditional => allow, authz5 conditional => allow",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(allows=1), authz2: NoOpinion], authz3: ConditionsMap(allows=1)], union3: NoOpinion, authz5: ConditionsMap(allows=1)]`,
			wantFinalDecision:     `Allow`,
		},
		{
			name: "authz1 conditional => allow, authz5 conditional => deny",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(allows=1), authz2: NoOpinion], authz3: ConditionsMap(allows=1)], union3: NoOpinion, authz5: ConditionsMap(denies=1)]`,
			wantFinalDecision:     `Allow`,
		},
		{
			name: "all conditionals eval noopinion",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz2: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz3: &evalTestAuthz{
				conditionEffect: effectNoOpinion,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(allows=1), authz2: ConditionsMap(denies=1)], authz3: NoOpinion(reason="only NoOpinion conditions always evaluate to NoOpinion")], union3: NoOpinion, authz5: ConditionsMap(allows=1)]`,
			wantFinalDecision:     `NoOpinion(reason="union1: authz3: only NoOpinion conditions always evaluate to NoOpinion")`,
		},

		// === Conditional deny in the chain ===

		{
			name: "authz1 conditional => deny, authz5 conditional => allow",
			authz1: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
			},
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(denies=1), authz2: NoOpinion], authz3: ConditionsMap(allows=1)], union3: NoOpinion, authz5: ConditionsMap(allows=1)]`,
			wantFinalDecision:     `Deny`,
		},
		{
			name: "authz1 conditional => deny, authz5 conditional => deny",
			authz1: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionDeny,
			},
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(denies=1), authz2: NoOpinion], authz3: ConditionsMap(allows=1)], union3: NoOpinion, authz5: ConditionsMap(denies=1)]`,
			wantFinalDecision:     `Deny`,
		},

		// === Error handling ===

		{
			// authz1's error rides through three layers of union ToDecision (union2, union1, union0).
			// Each layer prepends the source authorizer name when collecting deciding errors,
			// so the final error string is "union1: union2: authz1: authz error".
			name: "authorize error propagated",
			authz1: &evalTestAuthz{
				decision:     authorizer.DecisionAllow,
				authorizeErr: errors.New("authz error"),
			},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Allow(err="union1: union2: authz1: authz error")`,
			wantAuthorizeErr:      true,
			wantFinalDecision:     `Allow(err="union1: union2: authz1: authz error")`,
			wantFinalErr:          true,
		},
		{
			name: "evaluate error propagated",
			authz1: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionAllow,
				evalErr:         errors.New("eval error"),
			},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(allows=1), authz2: NoOpinion], authz3: NoOpinion], union3: NoOpinion, authz5: NoOpinion]`,
			wantFinalDecision:     `Allow(err="eval error")`,
			wantFinalErr:          true,
		},
		// === Impossible evaluations ===
		{
			name: "evaluate error propagated",
			authz1: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionAllow,
			},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(denies=1), authz2: NoOpinion], authz3: NoOpinion], union3: NoOpinion, authz5: NoOpinion]`,
			wantFinalDecision:     `Deny(reason="failed closed", err="evaluated to decision Allow, but only [Deny NoOpinion] were possible")`,
			wantFinalErr:          true,
		},
		{
			// Regression guard: an impossible sub-decision must fail closed to the
			// SUB-decision's FailureDecision, not the OUTER union's. Here authz5's
			// Allow-only ConditionsMap has FailureDecision=NoOpinion, while the outer
			// union0 has FailureDecision=Deny (authz1 contributes Deny to its possible
			// set). If the guard regressed to using the outer FailureDecision, the
			// result below would be Deny instead of NoOpinion.
			name: "impossible sub-decision fails closed to sub's FailureDecision, not outer union's",
			authz1: &evalTestAuthz{
				conditionEffect: effectDeny,
				evalDecision:    authorizer.DecisionNoOpinion,
			},
			authz2: noOpinion(),
			authz3: noOpinion(),
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: effectAllow,
				evalDecision:    authorizer.DecisionDeny,
			},
			wantAuthorizeDecision: `Union[union1: Union[union2: Union[authz1: ConditionsMap(denies=1), authz2: NoOpinion], authz3: NoOpinion], union3: NoOpinion, authz5: ConditionsMap(allows=1)]`,
			wantFinalDecision:     `NoOpinion(reason="authz5: failed closed", err="evaluated to decision Deny, but only [Allow NoOpinion] were possible")`,
			wantFinalErr:          true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			union3, err := New(NamedAuthorizer{AuthorizerName: "authz4", Authorizer: tt.authz4})
			if err != nil {
				t.Fatalf("union3 New: %v", err)
			}
			union2, err := New(
				NamedAuthorizer{AuthorizerName: "authz1", Authorizer: tt.authz1},
				NamedAuthorizer{AuthorizerName: "authz2", Authorizer: tt.authz2},
			)
			if err != nil {
				t.Fatalf("union2 New: %v", err)
			}
			union1, err := New(
				NamedAuthorizer{AuthorizerName: "union2", Authorizer: union2},
				NamedAuthorizer{AuthorizerName: "authz3", Authorizer: tt.authz3},
			)
			if err != nil {
				t.Fatalf("union1 New: %v", err)
			}
			union0, err := New(
				NamedAuthorizer{AuthorizerName: "union1", Authorizer: union1},
				NamedAuthorizer{AuthorizerName: "union3", Authorizer: union3},
				NamedAuthorizer{AuthorizerName: "authz5", Authorizer: tt.authz5},
			)
			if err != nil {
				t.Fatalf("union0 New: %v", err)
			}

			attrs := authorizer.AttributesRecord{
				User: &user.DefaultInfo{Name: "testuser"},
				Verb: "get",
			}

			ctx := context.Background()
			authzDecision := union0.ConditionsAwareAuthorize(ctx, attrs)

			authzErr := authzDecision.Error()
			if (authzErr != nil) != tt.wantAuthorizeErr {
				t.Fatalf("ConditionsAwareAuthorize() error = %v, wantErr %v", authzErr, tt.wantAuthorizeErr)
			}
			if authzDecision.String() != tt.wantAuthorizeDecision {
				t.Errorf("ConditionsAwareAuthorize() = %s, want %s", authzDecision.String(), tt.wantAuthorizeDecision)
			}

			// Wrap in a ConditionsAwareDecision just to get a unified string formatting for assertions.
			finalDecision := authzDecision
			if !authzDecision.IsUnconditional() {
				finalDecision = authorizer.ConditionsAwareDecisionFromParts(union0.EvaluateConditions(ctx, authzDecision, nil))
			}
			finalErr := finalDecision.Error()
			if (finalErr != nil) != tt.wantFinalErr {
				t.Fatalf("EvaluateConditions() error = %v, wantErr %v", finalErr, tt.wantFinalErr)
			}
			if finalDecision.String() != tt.wantFinalDecision {
				t.Errorf("EvaluateConditions() = %s, want %s", finalDecision.String(), tt.wantFinalDecision)
			}
		})
	}
}

// countingAuthz wraps another Authorizer and records how many times each method was invoked.
type countingAuthz struct {
	inner             authorizer.Authorizer
	conditionsAware   int
	evaluateCondCalls int
}

func (c *countingAuthz) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return c.inner.Authorize(ctx, attrs)
}

func (c *countingAuthz) ConditionsAwareAuthorize(ctx context.Context, attrs authorizer.Attributes) authorizer.ConditionsAwareDecision {
	c.conditionsAware++
	return c.inner.ConditionsAwareAuthorize(ctx, attrs)
}

func (c *countingAuthz) EvaluateConditions(ctx context.Context, d authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	c.evaluateCondCalls++
	return c.inner.EvaluateConditions(ctx, d, data)
}

// TestUnionConditionsAwareAuthorizeShortCircuit verifies that once an authorizer returns
// an unconditional Allow or Deny, the union authorizer does not call any later authorizers
// in the chain.
func TestUnionConditionsAwareAuthorizeShortCircuit(t *testing.T) {
	cases := []struct {
		name       string
		decisions  []authorizer.Decision // decisions returned by the first N authorizers
		wantCalled []int                 // how many times each authorizer was called
	}{
		{
			name:       "Allow short-circuits later authorizers",
			decisions:  []authorizer.Decision{authorizer.DecisionNoOpinion, authorizer.DecisionAllow, authorizer.DecisionDeny, authorizer.DecisionAllow},
			wantCalled: []int{1, 1, 0, 0},
		},
		{
			name:       "Deny short-circuits later authorizers",
			decisions:  []authorizer.Decision{authorizer.DecisionNoOpinion, authorizer.DecisionDeny, authorizer.DecisionAllow, authorizer.DecisionAllow},
			wantCalled: []int{1, 1, 0, 0},
		},
		{
			name:       "all NoOpinions exhausts the chain",
			decisions:  []authorizer.Decision{authorizer.DecisionNoOpinion, authorizer.DecisionNoOpinion, authorizer.DecisionNoOpinion},
			wantCalled: []int{1, 1, 1},
		},
		{
			name:       "first authorizer Allow short-circuits everything",
			decisions:  []authorizer.Decision{authorizer.DecisionAllow, authorizer.DecisionAllow, authorizer.DecisionAllow},
			wantCalled: []int{1, 0, 0},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			handlers := make([]authorizer.Authorizer, len(tc.decisions))
			counters := make([]*countingAuthz, len(tc.decisions))
			for i, d := range tc.decisions {
				counters[i] = &countingAuthz{inner: &mockAuthzHandler{decision: d}}
				handlers[i] = counters[i]
			}

			authz := mustNewIndexed(t, handlers...)
			_ = authz.ConditionsAwareAuthorize(context.Background(), nil)

			for i, c := range counters {
				if c.conditionsAware != tc.wantCalled[i] {
					t.Errorf("authorizer[%d] ConditionsAwareAuthorize called %d times, want %d", i, c.conditionsAware, tc.wantCalled[i])
				}
			}
		})
	}
}

// TestUnionConditionsAwareAuthorizeEmptyChain verifies the union authorizer returns NoOpinion
// when given zero sub-authorizers.
func TestUnionConditionsAwareAuthorizeEmptyChain(t *testing.T) {
	authz, err := New()
	if err != nil {
		t.Fatalf("union.New returned an unexpected error: %v", err)
	}
	d := authz.ConditionsAwareAuthorize(context.Background(), nil)
	if !d.IsNoOpinion() {
		t.Errorf("expected NoOpinion from empty chain, got %s", d.String())
	}
}

// TestUnionEvaluateConditionsRoutesByAuthorizerName verifies that EvaluateConditions uses the
// authorizer index parsed from the authorizerName when dispatching ConditionsMap evaluation.
func TestUnionEvaluateConditionsRoutesByAuthorizerName(t *testing.T) {
	// First authorizer returns a ConditionsMap that, when evaluated, yields NoOpinion.
	// Second authorizer returns a ConditionsMap that, when evaluated, yields Allow.
	// We verify that EvaluateConditions correctly routes each unevaluated sub-decision back to
	// its originating authorizer (matching by authorizerName).
	first := &evalTestAuthz{conditionEffect: effectAllow, evalDecision: authorizer.DecisionNoOpinion}
	second := &evalTestAuthz{conditionEffect: effectAllow, evalDecision: authorizer.DecisionAllow}
	authz, err := New(
		NamedAuthorizer{AuthorizerName: "first", Authorizer: first},
		NamedAuthorizer{AuthorizerName: "second", Authorizer: second},
	)
	if err != nil {
		t.Fatalf("union.New returned an unexpected error: %v", err)
	}

	d := authz.ConditionsAwareAuthorize(context.Background(), nil)
	if !d.IsUnion() {
		t.Fatalf("expected Union, got %s", d.String())
	}

	gotDecision, _, err := authz.EvaluateConditions(context.Background(), d, nil)
	if err != nil {
		t.Fatalf("EvaluateConditions returned error: %v", err)
	}
	if gotDecision != authorizer.DecisionAllow {
		t.Errorf("EvaluateConditions decision = %v, want Allow (routed via second authorizer)", gotDecision)
	}
}

// TestUnionEvaluateConditionsUnconditionalShortCircuit verifies that EvaluateConditions
// returns a wrapped unconditional decision immediately without descending into the chain.
func TestUnionEvaluateConditionsUnconditionalShortCircuit(t *testing.T) {
	a1 := &countingAuthz{inner: &evalTestAuthz{decision: authorizer.DecisionAllow}}
	a2 := &countingAuthz{inner: &evalTestAuthz{decision: authorizer.DecisionNoOpinion}}
	authz := mustNewIndexed(t, a1, a2)

	d := authz.ConditionsAwareAuthorize(context.Background(), nil)
	// Should be unconditional Allow after short-circuit.
	if !d.IsAllow() {
		t.Fatalf("expected Allow, got %s", d.String())
	}
	if a1.conditionsAware != 1 || a2.conditionsAware != 0 {
		t.Errorf("ConditionsAwareAuthorize should not have called sub-authorizer ConditionsAwareAuthorize (a1=%d, a2=%d)", a1.conditionsAware, a2.conditionsAware)
	}
}
