/*
Copyright 2024 The Kubernetes Authors.

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

package metrics

import (
	"context"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordAuthorizationDecisionsTotal(t *testing.T) {
	prefix := `
    # HELP apiserver_authorization_decisions_total [ALPHA] Total number of terminal decisions made by an authorizer split by authorizer type, name, and decision.
    # TYPE apiserver_authorization_decisions_total counter`
	metrics := []string{
		namespace + "_" + subsystem + "_decisions_total",
	}

	authorizationDecisionsTotal.Reset()
	RegisterMetrics()

	dummyAuthorizer := &dummyAuthorizer{}
	dummyConditionalAuthorizer := &dummyConditionalAuthorizer{}
	a := InstrumentedAuthorizer("mytype", "myname", dummyAuthorizer)
	ac := InstrumentedAuthorizer("myconditionaltype", "myconditionalname", dummyConditionalAuthorizer)

	// allow
	{
		dummyAuthorizer.decision = authorizer.DecisionAllow
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="allowed",name="myname",type="mytype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// allow (conditional authorizer)
	{
		dummyConditionalAuthorizer.authorizeDecision = authorizer.ConditionsAwareDecisionAllow("", nil)
		_, _, _ = ac.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="allowed",name="myconditionalname",type="myconditionaltype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// deny
	{
		dummyAuthorizer.decision = authorizer.DecisionDeny
		_, _, _ = a.Authorize(context.Background(), nil)
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="denied",name="myname",type="mytype"} 2
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// deny (conditional authorizer)
	{
		dummyConditionalAuthorizer.authorizeDecision = authorizer.ConditionsAwareDecisionDeny("", nil)
		_, _, _ = ac.Authorize(context.Background(), nil)
		_, _, _ = ac.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="denied",name="myconditionalname",type="myconditionaltype"} 2
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// no-opinion emits no metric
	{
		dummyAuthorizer.decision = authorizer.DecisionNoOpinion
		_, _, _ = a.Authorize(context.Background(), nil)
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// no-opinion emits no metric (conditional authorizer)
	{
		dummyConditionalAuthorizer.authorizeDecision = authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		_, _, _ = ac.Authorize(context.Background(), nil)
		_, _, _ = ac.Authorize(context.Background(), nil)
		expectedValue := prefix + `
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}

	// unknown decision emits a metric
	{
		dummyAuthorizer.decision = authorizer.DecisionDeny + 10
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="unknown",name="myname",type="mytype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	}
	// TODO(luxas): Add a test for getting a conditional decision from ConditionsAwareAuthorize, and evaluating a condition, once introduced
}

type dummyAuthorizer struct {
	decision authorizer.Decision
	err      error
}

func (d *dummyAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return d.decision, "", d.err
}

// ConditionsAwareAuthorize is not conditions-aware, converts the Authorize decision.
func (d *dummyAuthorizer) ConditionsAwareAuthorize(ctx context.Context, attrs authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionFromParts(d.Authorize(ctx, attrs))
}

// EvaluateConditions is not supported by this authorizer.
func (*dummyAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

type dummyConditionalAuthorizer struct {
	authorizeDecision authorizer.ConditionsAwareDecision
	evalDecision      authorizer.Decision
	evalErr           error
}

func (d *dummyConditionalAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	switch {
	case d.authorizeDecision.IsAllow():
		return authorizer.DecisionAllow, d.authorizeDecision.Reason(), d.authorizeDecision.Error()
	case d.authorizeDecision.IsNoOpinion():
		return authorizer.DecisionNoOpinion, d.authorizeDecision.Reason(), d.authorizeDecision.Error()
	case d.authorizeDecision.IsDeny():
		return authorizer.DecisionDeny, d.authorizeDecision.Reason(), d.authorizeDecision.Error()
	default: // Conditional case
		return authorizer.DecisionDeny, "failed closed: wanted to return a conditional decision, but called on the conditions-unaware method", nil
	}
}

func (d *dummyConditionalAuthorizer) ConditionsAwareAuthorize(ctx context.Context, attrs authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return d.authorizeDecision
}

func (d *dummyConditionalAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return d.evalDecision, "", d.evalErr
}
