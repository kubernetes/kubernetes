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
	"maps"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRecordAuthorizationDecisionsTotal(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

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

	t.Run("allow", func(t *testing.T) {
		dummyAuthorizer.decision = authorizer.DecisionAllow
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="allowed",name="myname",type="mytype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	})

	t.Run("allow (conditional authorizer)", func(t *testing.T) {
		dummyConditionalAuthorizer.decision = authorizer.ConditionsAwareDecisionAllow("", nil)
		_ = ac.AuthorizeConditionsAware(context.Background(), nil, authorizer.ConditionsEncodingPreferenceOptimized())
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="allowed",name="myconditionalname",type="myconditionaltype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	})

	t.Run("deny", func(t *testing.T) {
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
	})

	t.Run("deny (conditional authorizer)", func(t *testing.T) {
		dummyConditionalAuthorizer.decision = authorizer.ConditionsAwareDecisionDeny("", nil)
		_ = ac.AuthorizeConditionsAware(context.Background(), nil, authorizer.ConditionsEncodingPreferenceOptimized())
		_ = ac.AuthorizeConditionsAware(context.Background(), nil, authorizer.ConditionsEncodingPreferenceOptimized())
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="denied",name="myconditionalname",type="myconditionaltype"} 2
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	})

	t.Run("no-opinion emits no metric", func(t *testing.T) {
		dummyAuthorizer.decision = authorizer.DecisionNoOpinion
		_, _, _ = a.Authorize(context.Background(), nil)
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	})

	t.Run("no-opinion emits no metric (conditional authorizer)", func(t *testing.T) {
		dummyConditionalAuthorizer.decision = authorizer.ConditionsAwareDecisionNoOpinion("", nil)
		_ = ac.AuthorizeConditionsAware(context.Background(), nil, authorizer.ConditionsEncodingPreferenceOptimized())
		_ = ac.AuthorizeConditionsAware(context.Background(), nil, authorizer.ConditionsEncodingPreferenceOptimized())
		expectedValue := prefix + `
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	})

	t.Run("unknown decision emits a metric", func(t *testing.T) {
		dummyAuthorizer.decision = authorizer.DecisionDeny + 10
		_, _, _ = a.Authorize(context.Background(), nil)
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="unknown",name="myname",type="mytype"} 1
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	})

	t.Run("conditional emits a metric (conditional authorizer)", func(t *testing.T) {
		dummyConditionalAuthorizer.decision = authorizer.ConditionsAwareDecisionConditionMap(
			authorizer.ConditionsTargetAdmissionControl,
			authorizer.ConditionType("foo"),
			maps.All(map[string]authorizer.Condition{
				"foo": {
					Condition: "foo",
					Effect:    authorizer.ConditionEffectAllow,
				},
			}),
			"",
			nil,
		)
		_ = ac.AuthorizeConditionsAware(context.Background(), nil, authorizer.ConditionsEncodingPreferenceOptimized())
		_ = ac.AuthorizeConditionsAware(context.Background(), nil, authorizer.ConditionsEncodingPreferenceOptimized())
		expectedValue := prefix + `
			apiserver_authorization_decisions_total{decision="conditional",name="myconditionalname",type="myconditionaltype"} 2
		`
		if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(expectedValue), metrics...); err != nil {
			t.Fatal(err)
		}
		authorizationDecisionsTotal.Reset()
	})
}

type dummyAuthorizer struct {
	decision authorizer.Decision
	err      error
}

func (d *dummyAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return d.decision, "", d.err
}

// AuthorizeConditionsAware is not conditions-aware, converts the Authorize decision.
func (d *dummyAuthorizer) AuthorizeConditionsAware(ctx context.Context, attrs authorizer.Attributes, _ authorizer.ConditionsEncodingPreference) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionFromParts(d.Authorize(ctx, attrs))
}

// EvaluateConditions is not supported by this authorizer.
func (*dummyAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData, _ authorizer.BuiltinConditionsMapEvaluators) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionDeny("", authorizer.ErrorConditionEvaluationNotSupported)
}

type dummyConditionalAuthorizer struct {
	decision authorizer.ConditionsAwareDecision
}

func (d *dummyConditionalAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return authorizer.DecisionPartsFromConditionsAware(d.decision)
}

// AuthorizeConditionsAware is not conditions-aware, converts the Authorize decision.
func (d *dummyConditionalAuthorizer) AuthorizeConditionsAware(ctx context.Context, attrs authorizer.Attributes, _ authorizer.ConditionsEncodingPreference) authorizer.ConditionsAwareDecision {
	return d.decision
}

// EvaluateConditions is not supported by this authorizer.
func (*dummyConditionalAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData, _ authorizer.BuiltinConditionsMapEvaluators) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionDeny("", authorizer.ErrorConditionEvaluationNotSupported)
}
