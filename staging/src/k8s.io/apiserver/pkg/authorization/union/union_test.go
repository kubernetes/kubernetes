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
	"maps"
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

type mockAuthzHandler struct {
	decision authorizer.Decision
	err      error
}

func (mock *mockAuthzHandler) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, error) {
	return mock.decision, mock.err
}

func (mock *mockAuthzHandler) EvaluateConditions(ctx context.Context, decision authorizer.Decision, data authorizer.ConditionData) (authorizer.Decision, error) {
	return authorizer.DecisionDeny(), authorizer.ErrorConditionEvaluationNotSupported
}

func TestAuthorizationSecondPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion("")}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionAllow("")}
	authzHandler := New(handler1, handler2)

	authorized, _ := authzHandler.Authorize(context.Background(), nil)
	if !authorized.IsAllowed() {
		t.Errorf("Unexpected authorization failure")
	}
}

func TestAuthorizationFirstPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionAllow("")}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion("")}
	authzHandler := New(handler1, handler2)

	authorized, _ := authzHandler.Authorize(context.Background(), nil)
	if !authorized.IsAllowed() {
		t.Errorf("Unexpected authorization failure")
	}
}

func TestAuthorizationNonePasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion("")}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion("")}
	authzHandler := New(handler1, handler2)

	authorized, _ := authzHandler.Authorize(context.Background(), nil)
	if authorized.IsAllowed() {
		t.Errorf("Expected failed authorization")
	}
}

func TestAuthorizationError(t *testing.T) {
	handler1 := &mockAuthzHandler{err: fmt.Errorf("foo")}
	handler2 := &mockAuthzHandler{err: fmt.Errorf("foo")}
	authzHandler := New(handler1, handler2)

	_, err := authzHandler.Authorize(context.Background(), nil)
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

// evalTestAuthz is a configurable authorizer for testing the union evaluation flow.
type evalTestAuthz struct {
	// conditionEffect, if non-empty, makes Authorize return a Conditional decision
	// with a single condition of this effect. If empty, decision is returned instead.
	conditionEffect authorizer.ConditionEffect
	// decision is returned from Authorize when conditionEffect is empty.
	decision authorizer.Decision
	// authorizeErr is returned as the error from Authorize.
	authorizeErr error

	// evalDecision is returned from EvaluateConditions.
	evalDecision authorizer.Decision
	// evalErr is returned as the error from EvaluateConditions.
	evalErr error
}

func (a *evalTestAuthz) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, error) {
	if a.conditionEffect != "" {
		d, err := authorizer.DecisionConditional(attrs, "test-type", maps.All(map[string]authorizer.Condition{
			"test-cond": {Condition: "test", Effect: a.conditionEffect},
		}))
		if err != nil {
			panic(err) // these test conditions are always valid
		}
		return d, a.authorizeErr
	}
	return a.decision, a.authorizeErr
}

func (a *evalTestAuthz) EvaluateConditions(ctx context.Context, decision authorizer.Decision, data authorizer.ConditionData) (authorizer.Decision, error) {
	// Concrete decisions need no evaluation, return as-is.
	if decision.IsAllowed() || decision.IsDenied() || decision.IsNoOpinion() {
		return decision, nil
	}
	return a.evalDecision, a.evalErr
}

// TestUnionEvaluateConditions tests the full Authorize + EvaluateConditions flow
// through a DAG of nested union authorizers:
//
//		union0 = [union1, union3, authz5]
//		union1 = [union2, authz3]
//		union2 = [authz1, authz2]
//	    union3 = [authz4]
func TestUnionEvaluateConditions(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
	noOpinion := func() authorizer.Authorizer {
		return &evalTestAuthz{decision: authorizer.DecisionNoOpinion()}
	}

	tests := []struct {
		name                                   string
		authz1, authz2, authz3, authz4, authz5 authorizer.Authorizer
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
			wantAuthorizeDecision: "NoOpinion",
			wantFinalDecision:     "NoOpinion",
		},
		{
			name:                  "authz1 allow short-circuits everything",
			authz1:                &evalTestAuthz{decision: authorizer.DecisionAllow()},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: "Allow",
			wantFinalDecision:     "Allow",
		},
		{
			name:                  "authz1 deny short-circuits everything",
			authz1:                &evalTestAuthz{decision: authorizer.DecisionDeny()},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: "Deny",
			wantFinalDecision:     "Deny",
		},
		{
			name:                  "authz1 noopinion authz2 allow",
			authz1:                noOpinion(),
			authz2:                &evalTestAuthz{decision: authorizer.DecisionAllow()},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: "Allow",
			wantFinalDecision:     "Allow",
		},
		{
			name:                  "authz1 authz2 noopinion authz3 allow",
			authz1:                noOpinion(),
			authz2:                noOpinion(),
			authz3:                &evalTestAuthz{decision: authorizer.DecisionAllow()},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: "Allow",
			wantFinalDecision:     "Allow",
		},
		{
			name:                  "all inner noopinion authz5 allow",
			authz1:                noOpinion(),
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                &evalTestAuthz{decision: authorizer.DecisionAllow()},
			wantAuthorizeDecision: "Allow",
			wantFinalDecision:     "Allow",
		},
		{
			name:                  "authz1 noopinion authz2 deny",
			authz1:                noOpinion(),
			authz2:                &evalTestAuthz{decision: authorizer.DecisionDeny()},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: "Deny",
			wantFinalDecision:     "Deny",
		},

		// === Conditional decisions ===

		{
			name: "authz2 conditional allow evals to allow",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), NoOpinion], NoOpinion], NoOpinion, NoOpinion]`,
			wantFinalDecision:     "Allow",
		},
		{
			name:   "authz2 conditional allow evals to noopinion",
			authz1: noOpinion(),
			authz2: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[NoOpinion, Conditional(type="test-type", len=1)], NoOpinion], NoOpinion, NoOpinion]`,
			wantFinalDecision:     "NoOpinion",
		},
		{
			name:   "authz3 conditional deny evals to deny",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[NoOpinion, Conditional(type="test-type", len=1)], NoOpinion, NoOpinion]`,
			wantFinalDecision:     "Deny",
		},
		{
			name:   "authz4 conditional deny evals to noopinion",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: noOpinion(),
			authz4: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[NoOpinion, ConditionalChain[Conditional(type="test-type", len=1)], NoOpinion]`,
			wantFinalDecision:     "NoOpinion",
		},
		{
			name:   "authz5 conditional noopinion evals to noopinion",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: noOpinion(),
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectNoOpinion,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			wantAuthorizeDecision: `ConditionalChain[NoOpinion, NoOpinion, Conditional(type="test-type", len=1)]`,
			wantFinalDecision:     "NoOpinion",
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

		// TODO: Implement differential testing
		// a) For a chain of length N, create all permutations of N^6 as test cases
		//	  ==> or instead work through all the possible combinations, and then insert NoOpinions?
		//	  ==> e.g. start with "I want an Allow response, what are all the ways I can get that?"
		// b) Using the formal model, compute the final decision
		// c) For each test case, create each possible DAG combination of that chain
		// d) For each DAG combo, run Authorize + Evaluate, and assert the final decision

		{
			name: "authz1 conditional => noopinion, authz2 allow",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			// TODO: Here we could, using eager evaluation, directly fold to Allow
			authz2:                &evalTestAuthz{decision: authorizer.DecisionAllow()},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), Allow]]]`,
			wantFinalDecision:     "Allow",
		},
		{
			name:   "authz3 conditional => noopinion, authz5 deny",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz4:                noOpinion(),
			authz5:                &evalTestAuthz{decision: authorizer.DecisionDeny()},
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[NoOpinion, Conditional(type="test-type", len=1)], NoOpinion, Deny]`,
			wantFinalDecision:     "Deny",
		},
		{
			name: "authz1 conditional => allow, authz2 allow",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			// TODO: Here we could, using eager evaluation, directly fold to Allow
			authz2:                &evalTestAuthz{decision: authorizer.DecisionAllow()},
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), Allow]]]`,
			wantFinalDecision:     "Allow",
		},
		{
			name:   "authz3 conditional => allow, authz5 deny",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			authz4:                noOpinion(),
			authz5:                &evalTestAuthz{decision: authorizer.DecisionDeny()},
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[NoOpinion, Conditional(type="test-type", len=1)], NoOpinion, Deny]`,
			wantFinalDecision:     "Allow",
		},
		{
			name:   "authz2 conditional => deny, authz3 allow",
			authz1: noOpinion(),
			authz2: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			authz3:                &evalTestAuthz{decision: authorizer.DecisionAllow()},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[NoOpinion, Conditional(type="test-type", len=1)], Allow]]`,
			wantFinalDecision:     "Deny",
		},
		{
			name:   "authz4 conditional => deny, authz5 deny",
			authz1: noOpinion(),
			authz2: noOpinion(),
			authz3: noOpinion(),
			authz4: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			authz5:                &evalTestAuthz{decision: authorizer.DecisionDeny()},
			wantAuthorizeDecision: `ConditionalChain[NoOpinion, ConditionalChain[Conditional(type="test-type", len=1)], Deny]`,
			wantFinalDecision:     "Deny",
		},

		// === Multiple conditionals ===

		{
			name: "authz1 conditional => noopinion, authz3 conditional => allow",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), NoOpinion], Conditional(type="test-type", len=1)], NoOpinion, NoOpinion]`,
			wantFinalDecision:     "Allow",
		},
		{
			name:   "authz1 conditional => noopinion, authz3 conditional => deny",
			authz1: noOpinion(),
			authz2: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[NoOpinion, Conditional(type="test-type", len=1)], Conditional(type="test-type", len=1)], NoOpinion, NoOpinion]`,
			wantFinalDecision:     "Deny",
		},
		{
			name: "authz1 conditional => allow, authz5 conditional => allow",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), NoOpinion], Conditional(type="test-type", len=1)], NoOpinion, Conditional(type="test-type", len=1)]`,
			wantFinalDecision:     "Allow",
		},
		{
			name: "authz1 conditional => allow, authz5 conditional => deny",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), NoOpinion], Conditional(type="test-type", len=1)], NoOpinion, Conditional(type="test-type", len=1)]`,
			wantFinalDecision:     "Allow",
		},
		{
			name: "all conditionals eval noopinion",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz2: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectNoOpinion,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), Conditional(type="test-type", len=1)], Conditional(type="test-type", len=1)], NoOpinion, Conditional(type="test-type", len=1)]`,
			wantFinalDecision:     "NoOpinion",
		},

		// === Conditional deny in the chain ===

		{
			name: "authz1 conditional => deny, authz5 conditional => allow",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
			},
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), NoOpinion], Conditional(type="test-type", len=1)], NoOpinion, Conditional(type="test-type", len=1)]`,
			wantFinalDecision:     "Deny",
		},
		{
			name: "authz1 conditional => deny, authz5 conditional => deny",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			authz2: noOpinion(),
			authz3: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionNoOpinion(),
			},
			authz4: noOpinion(),
			authz5: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectDeny,
				evalDecision:    authorizer.DecisionDeny(),
			},
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), NoOpinion], Conditional(type="test-type", len=1)], NoOpinion, Conditional(type="test-type", len=1)]`,
			wantFinalDecision:     "Deny",
		},

		// === Error handling ===

		{
			name: "authorize error propagated",
			authz1: &evalTestAuthz{
				decision:     authorizer.DecisionAllow(),
				authorizeErr: errors.New("authz error"),
			},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: "Allow",
			wantAuthorizeErr:      true,
			wantFinalDecision:     "Allow",
		},
		{
			name: "evaluate error propagated",
			authz1: &evalTestAuthz{
				conditionEffect: authorizer.ConditionEffectAllow,
				evalDecision:    authorizer.DecisionAllow(),
				evalErr:         errors.New("eval error"),
			},
			authz2:                noOpinion(),
			authz3:                noOpinion(),
			authz4:                noOpinion(),
			authz5:                noOpinion(),
			wantAuthorizeDecision: `ConditionalChain[ConditionalChain[ConditionalChain[Conditional(type="test-type", len=1), NoOpinion], NoOpinion], NoOpinion, NoOpinion]`,
			wantFinalDecision:     "Allow",
			wantFinalErr:          true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			union3 := New(tt.authz4)
			union2 := New(tt.authz1, tt.authz2)
			union1 := New(union2, tt.authz3)
			union0 := New(union1, union3, tt.authz5)

			attrs := authorizer.AttributesRecord{
				User:           &user.DefaultInfo{Name: "testuser"},
				Verb:           "get",
				ConditionsMode: authorizer.ConditionsModeHumanReadable,
			}

			ctx := context.Background()
			authzDecision, authzErr := union0.Authorize(ctx, attrs)

			if (authzErr != nil) != tt.wantAuthorizeErr {
				t.Fatalf("Authorize() error = %v, wantErr %v", authzErr, tt.wantAuthorizeErr)
			}
			if authzDecision.String() != tt.wantAuthorizeDecision {
				t.Errorf("Authorize() = %s, want %s", authzDecision.String(), tt.wantAuthorizeDecision)
			}

			finalDecision, finalErr := union0.EvaluateConditions(ctx, authzDecision, nil)

			if (finalErr != nil) != tt.wantFinalErr {
				t.Fatalf("EvaluateConditions() error = %v, wantErr %v", finalErr, tt.wantFinalErr)
			}
			if finalDecision.String() != tt.wantFinalDecision {
				t.Errorf("EvaluateConditions() = %s, want %s", finalDecision.String(), tt.wantFinalDecision)
			}
		})
	}
}

func TestAuthorizationUnequivocalDeny(t *testing.T) {
	cs := []struct {
		authorizers []authorizer.Authorizer
		decision    authorizer.Decision
	}{
		{
			authorizers: []authorizer.Authorizer{},
			decision:    authorizer.DecisionNoOpinion(""),
		},
		{
			authorizers: []authorizer.Authorizer{
				&mockAuthzHandler{decision: authorizer.DecisionNoOpinion("")},
				&mockAuthzHandler{decision: authorizer.DecisionAllow("")},
				&mockAuthzHandler{decision: authorizer.DecisionDeny("")},
			},
			decision: authorizer.DecisionAllow(""),
		},
		{
			authorizers: []authorizer.Authorizer{
				&mockAuthzHandler{decision: authorizer.DecisionNoOpinion("")},
				&mockAuthzHandler{decision: authorizer.DecisionDeny("")},
				&mockAuthzHandler{decision: authorizer.DecisionAllow("")},
			},
			decision: authorizer.DecisionDeny(""),
		},
		{
			authorizers: []authorizer.Authorizer{
				&mockAuthzHandler{decision: authorizer.DecisionNoOpinion("")},
				&mockAuthzHandler{decision: authorizer.DecisionDeny(""), err: errors.New("webhook failed closed")},
				&mockAuthzHandler{decision: authorizer.DecisionAllow("")},
			},
			decision: authorizer.DecisionDeny(""),
		},
	}
	for i, c := range cs {
		t.Run(fmt.Sprintf("case %v", i), func(t *testing.T) {
			authzHandler := New(c.authorizers...)

			decision, _ := authzHandler.Authorize(context.Background(), nil)
			if !decision.Equal(c.decision) {
				t.Errorf("Unexpected authorization failure: %v, expected: %v", decision, c.decision)
			}
		})
	}
}
