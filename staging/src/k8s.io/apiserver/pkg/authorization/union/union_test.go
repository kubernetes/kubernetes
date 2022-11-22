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

func TestAuthorizationSecondPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionAllow}
	authzHandler := New(handler1, handler2)

	authorized, _, _ := authzHandler.Authorize(context.Background(), nil)
	if authorized != authorizer.DecisionAllow {
		t.Errorf("Unexpected authorization failure")
	}
}

func TestAuthorizationFirstPasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionAllow}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	authzHandler := New(handler1, handler2)

	authorized, _, _ := authzHandler.Authorize(context.Background(), nil)
	if authorized != authorizer.DecisionAllow {
		t.Errorf("Unexpected authorization failure")
	}
}

func TestAuthorizationNonePasses(t *testing.T) {
	handler1 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	handler2 := &mockAuthzHandler{decision: authorizer.DecisionNoOpinion}
	authzHandler := New(handler1, handler2)

	authorized, _, _ := authzHandler.Authorize(context.Background(), nil)
	if authorized == authorizer.DecisionAllow {
		t.Errorf("Expected failed authorization")
	}
}

func TestAuthorizationError(t *testing.T) {
	handler1 := &mockAuthzHandler{err: fmt.Errorf("foo")}
	handler2 := &mockAuthzHandler{err: fmt.Errorf("foo")}
	authzHandler := New(handler1, handler2)

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
			authzHandler := New(c.authorizers...)

			decision, _, _ := authzHandler.Authorize(context.Background(), nil)
			if decision != c.decision {
				t.Errorf("Unexpected authorization failure: %v, expected: %v", decision, c.decision)
			}
		})
	}
}
