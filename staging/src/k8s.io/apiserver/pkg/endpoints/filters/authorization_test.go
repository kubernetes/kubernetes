/*
Copyright 2016 The Kubernetes Authors.

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

package filters

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"

	batch "k8s.io/api/batch/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestGetAuthorizerAttributes(t *testing.T) {
	basicLabelRequirement, err := labels.NewRequirement("foo", selection.DoubleEquals, []string{"bar"})
	if err != nil {
		t.Fatal(err)
	}

	testcases := map[string]struct {
		Verb               string
		Path               string
		ExpectedAttributes *authorizer.AttributesRecord
	}{
		"non-resource root": {
			Verb: http.MethodPost,
			Path: "/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "post",
				Path: "/",
			},
		},
		"non-resource api prefix": {
			Verb: http.MethodGet,
			Path: "/api/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "get",
				Path: "/api/",
			},
		},
		"non-resource group api prefix": {
			Verb: http.MethodGet,
			Path: "/apis/extensions/",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb: "get",
				Path: "/apis/extensions/",
			},
		},

		"resource": {
			Verb: http.MethodPost,
			Path: "/api/v1/nodes/mynode",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "create",
				Path:            "/api/v1/nodes/mynode",
				ResourceRequest: true,
				Resource:        "nodes",
				APIVersion:      "v1",
				Name:            "mynode",
			},
		},
		"namespaced resource": {
			Verb: http.MethodPut,
			Path: "/api/v1/namespaces/myns/pods/mypod",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "update",
				Path:            "/api/v1/namespaces/myns/pods/mypod",
				ResourceRequest: true,
				Namespace:       "myns",
				Resource:        "pods",
				APIVersion:      "v1",
				Name:            "mypod",
			},
		},
		"API group resource": {
			Verb: http.MethodGet,
			Path: "/apis/batch/v1/namespaces/myns/jobs",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        batch.GroupName,
				APIVersion:      "v1",
				Namespace:       "myns",
				Resource:        "jobs",
			},
		},
		"disabled, ignore good field selector": {
			Verb: http.MethodGet,
			Path: "/apis/batch/v1/namespaces/myns/jobs?fieldSelector%=foo%3Dbar",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        batch.GroupName,
				APIVersion:      "v1",
				Namespace:       "myns",
				Resource:        "jobs",
			},
		},
		"enabled, good field selector": {
			Verb: http.MethodGet,
			Path: "/apis/batch/v1/namespaces/myns/jobs?fieldSelector=foo%3D%3Dbar",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        batch.GroupName,
				APIVersion:      "v1",
				Namespace:       "myns",
				Resource:        "jobs",
				FieldSelectorRequirements: fields.Requirements{
					fields.OneTermEqualSelector("foo", "bar").Requirements()[0],
				},
			},
		},
		"enabled, bad field selector": {
			Verb: http.MethodGet,
			Path: "/apis/batch/v1/namespaces/myns/jobs?fieldSelector=%2Abar",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:                    "list",
				Path:                    "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest:         true,
				APIGroup:                batch.GroupName,
				APIVersion:              "v1",
				Namespace:               "myns",
				Resource:                "jobs",
				FieldSelectorParsingErr: errors.New("invalid selector: '*bar'; can't understand '*bar'"),
			},
		},
		"disabled, ignore good label selector": {
			Verb: http.MethodGet,
			Path: "/apis/batch/v1/namespaces/myns/jobs?labelSelector%=foo%3Dbar",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        batch.GroupName,
				APIVersion:      "v1",
				Namespace:       "myns",
				Resource:        "jobs",
			},
		},
		"enabled, good label selector": {
			Verb: http.MethodGet,
			Path: "/apis/batch/v1/namespaces/myns/jobs?labelSelector=foo%3D%3Dbar",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:            "list",
				Path:            "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest: true,
				APIGroup:        batch.GroupName,
				APIVersion:      "v1",
				Namespace:       "myns",
				Resource:        "jobs",
				LabelSelectorRequirements: labels.Requirements{
					*basicLabelRequirement,
				},
			},
		},
		"enabled, bad label selector": {
			Verb: http.MethodGet,
			Path: "/apis/batch/v1/namespaces/myns/jobs?labelSelector=%2Abar",
			ExpectedAttributes: &authorizer.AttributesRecord{
				Verb:                    "list",
				Path:                    "/apis/batch/v1/namespaces/myns/jobs",
				ResourceRequest:         true,
				APIGroup:                batch.GroupName,
				APIVersion:              "v1",
				Namespace:               "myns",
				Resource:                "jobs",
				LabelSelectorParsingErr: errors.New("unable to parse requirement: <nil>: Invalid value: \"*bar\": name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')"),
			},
		},
	}

	for k, tc := range testcases {
		t.Run(k, func(t *testing.T) {
			ctx := t.Context()

			req, _ := http.NewRequestWithContext(ctx, tc.Verb, tc.Path, nil)
			req.RemoteAddr = "127.0.0.1"

			var attribs authorizer.Attributes
			var err error
			var handler http.Handler = http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				ctx := req.Context()
				attribs, err = GetAuthorizerAttributes(ctx)
			})
			handler = WithRequestInfo(handler, newTestRequestInfoResolver())
			handler.ServeHTTP(httptest.NewRecorder(), req)

			if err != nil {
				t.Errorf("%s: unexpected error: %v", k, err)
			} else if !reflect.DeepEqual(attribs, tc.ExpectedAttributes) {
				t.Errorf("%s: expected\n\t%#v\ngot\n\t%#v", k, tc.ExpectedAttributes, attribs)
			}
		})
	}
}

type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f fakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}

// ConditionsAwareAuthorize is not conditions-aware, converts the Authorize decision.
func (f fakeAuthorizer) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionFromParts(f.Authorize(ctx, a))
}

// EvaluateConditions is not supported by this authorizer.
func (fakeAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

func TestAuditAnnotation(t *testing.T) {
	testcases := map[string]struct {
		authorizer         fakeAuthorizer
		decisionAnnotation string
		reasonAnnotation   string
	}{
		"decision allow": {
			fakeAuthorizer{
				authorizer.DecisionAllow,
				"RBAC: allowed to patch pod",
				nil,
			},
			"allow",
			"RBAC: allowed to patch pod",
		},
		"decision forbid": {
			fakeAuthorizer{
				authorizer.DecisionDeny,
				"RBAC: not allowed to patch pod",
				nil,
			},
			"forbid",
			"RBAC: not allowed to patch pod",
		},
		"error": {
			fakeAuthorizer{
				authorizer.DecisionNoOpinion,
				"",
				errors.New("can't parse user info"),
			},
			"",
			ReasonError,
		},
	}

	scheme := runtime.NewScheme()
	negotiatedSerializer := serializer.NewCodecFactory(scheme).WithoutConversion()
	for k, tc := range testcases {
		ctx := t.Context()
		handler := WithAuthorization(&fakeHTTPHandler{}, tc.authorizer, negotiatedSerializer)
		// TODO: fake audit injector

		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "/api/v1/namespaces/default/pods", nil)
		req = withTestContext(req, nil, &auditinternal.Event{Level: auditinternal.LevelMetadata})
		ae := audit.AuditContextFrom(req.Context())
		req.RemoteAddr = "127.0.0.1"
		handler.ServeHTTP(httptest.NewRecorder(), req)

		var annotation string
		var ok bool
		if len(tc.decisionAnnotation) > 0 {
			annotation, ok = ae.GetEventAnnotation(DecisionAnnotationKey)
			assert.True(t, ok, k+": decision annotation not found")
			assert.Equal(t, tc.decisionAnnotation, annotation, k+": unexpected decision annotation")
		}

		annotation, ok = ae.GetEventAnnotation(ReasonAnnotationKey)
		assert.True(t, ok, k+": reason annotation not found")
		assert.Equal(t, tc.reasonAnnotation, annotation, k+": unexpected reason annotation")
	}

}

// conditionsAwareFakeAuthorizer allows returning arbitrary ConditionsAwareDecision values.
type conditionsAwareFakeAuthorizer struct {
	makeDecision func() authorizer.ConditionsAwareDecision
}

func (f *conditionsAwareFakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.ConditionsAwareAuthorize(ctx, a).UnconditionalParts(true)
}

func (f *conditionsAwareFakeAuthorizer) ConditionsAwareAuthorize(_ context.Context, _ authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return f.makeDecision()
}

func (f *conditionsAwareFakeAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

func TestWithAuthorization(t *testing.T) {
	scheme := runtime.NewScheme()
	negotiatedSerializer := serializer.NewCodecFactory(scheme).WithoutConversion()

	makeCondMapAllowDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			nil, nil,
			[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c1", Condition: "object.metadata.name == 'foo'", Type: "example.com/cel"}},
		)
	}

	makeCondMapDenyOnlyDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c1", Condition: "object.metadata.name == 'bar'", Type: "example.com/cel"}},
			nil, nil,
		)
	}

	classifierAlwaysTrue := ConditionalAuthorizationRequestClassifier(func(_ authorizer.Attributes) bool { return true })
	classifierAlwaysFalse := ConditionalAuthorizationRequestClassifier(func(_ authorizer.Attributes) bool { return false })

	type expectedOutcome struct {
		statusCode           int
		handlerCalled        bool
		decisionAnnotation   string
		reasonAnnotation     string
		conditionalInContext bool
		// conditionalAnnotation is the expected value of the
		// "authorization.k8s.io/is-conditional-decision" audit annotation. Set to
		// "true" for the conditional-allow filter path; "" means the annotation
		// must not be present.
		conditionalAnnotation string
	}

	tests := []struct {
		name                       string
		authorizer                 authorizer.Authorizer
		conditionalAuthzClassifier ConditionalAuthorizationRequestClassifier
		disabled                   expectedOutcome
		enabled                    expectedOutcome
	}{
		{
			name:       "nil authorizer passes through",
			authorizer: nil,
			disabled:   expectedOutcome{statusCode: http.StatusOK, handlerCalled: true},
			enabled:    expectedOutcome{statusCode: http.StatusOK, handlerCalled: true},
		},
		{
			name:       "allow",
			authorizer: fakeAuthorizer{authorizer.DecisionAllow, "RBAC: allowed", nil},
			disabled:   expectedOutcome{statusCode: http.StatusOK, handlerCalled: true, decisionAnnotation: DecisionAllow, reasonAnnotation: "RBAC: allowed"},
			// gate on: the filter sets the allow decision/reason annotations up front on
			// the unconditional-allow path (mirroring withAuthorization), so downstream
			// handling can rely on them being present even if a later step short-circuits
			// before the AuthorizationConditionsEnforcer plugin runs.
			enabled: expectedOutcome{statusCode: http.StatusOK, handlerCalled: true, decisionAnnotation: DecisionAllow, reasonAnnotation: "RBAC: allowed"},
		},
		{
			name:       "deny",
			authorizer: fakeAuthorizer{authorizer.DecisionDeny, "RBAC: denied", nil},
			disabled:   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "RBAC: denied"},
			enabled:    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "RBAC: denied"},
		},
		{
			name:       "no opinion",
			authorizer: fakeAuthorizer{authorizer.DecisionNoOpinion, "no match", nil},
			disabled:   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "no match"},
			enabled:    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "no match"},
		},
		{
			name:       "no opinion with error",
			authorizer: fakeAuthorizer{authorizer.DecisionNoOpinion, "", errors.New("webhook error")},
			disabled:   expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: ReasonError},
			enabled:    expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: ReasonError},
		},
		{
			name: "no opinion with error (conditions-aware)",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: func() authorizer.ConditionsAwareDecision {
					return authorizer.ConditionsAwareDecisionNoOpinion("", fmt.Errorf("internal issue"))
				},
			},
			disabled: expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: ReasonError},
			enabled:  expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: ReasonError},
		},
		{
			name: "conditional allow + classifier true",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes to NoOpinion (no deny effect) => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: CanBecomeAllowed=true, classifier=true => conditional path. The filter
			// records "is-conditional-decision=true" so downstream audit consumers can tell
			// the request was authorized conditionally even if a later hop errors out before
			// the AuthorizationConditionsEnforcer plugin sets the final decision annotations.
			enabled: expectedOutcome{statusCode: http.StatusOK, handlerCalled: true, conditionalInContext: true, conditionalAnnotation: "true"},
		},
		{
			name: "conditional allow + classifier false",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
			},
			conditionalAuthzClassifier: classifierAlwaysFalse,
			// gate off: condMap constructor fail-closes to NoOpinion => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: classifier rejects, err=nil => forbidden
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
		},
		{
			name: "conditional allow + classifier nil",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
			},
			conditionalAuthzClassifier: nil,
			// gate off: condMap constructor fail-closes to NoOpinion => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: no classifier, err=nil => forbidden
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
		},
		{
			name: "conditional deny-only + classifier true",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapDenyOnlyDecision,
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes to Deny (has deny effect) => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid, reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Deny NoOpinion]) to Deny during unconditional authorization"},
			// gate on: CanBecomeAllowed=false, err=nil => forbidden
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: DecisionForbid},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, mode := range []struct {
				name string
				gate bool
				want expectedOutcome
			}{
				{"disabled", false, tt.disabled},
				{"enabled", true, tt.enabled},
			} {
				t.Run(mode.name, func(t *testing.T) {
					if mode.gate {
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)
					}

					handlerCalled := false
					var gotConditionalDecision bool

					innerHandler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
						handlerCalled = true
						// A conditions-aware decision is always attached to the context when
						// withConditionsAwareAuthorization is used, even for unconditional
						// outcomes. Consider a "conditional" decision propagated only if the
						// decision it carries is not an unconditional Allow/Deny/NoOpinion.
						_, decision, ok := request.ConditionallyAuthorizedDecisionFrom(req.Context())
						gotConditionalDecision = ok && !decision.IsUnconditional()
						w.WriteHeader(http.StatusOK)
					})

					noopMetrics := func(_ context.Context, _ string, _, _ time.Time) {}
					// Mirror the dispatch that the public WithConditionsAwareAuthorization wrapper does:
					// use the conditions-aware handler only when the gate is on and the AuthorizationConditionsEnforcer is enabled.
					// In this test, the admission plugin is always considered present, and thus is the second AND term always true.
					var handler http.Handler
					if mode.gate {
						handler = withConditionsAwareAuthorization(innerHandler, tt.authorizer, negotiatedSerializer, noopMetrics, tt.conditionalAuthzClassifier)
					} else {
						handler = withAuthorization(innerHandler, tt.authorizer, negotiatedSerializer, noopMetrics)
					}

					req, _ := http.NewRequestWithContext(t.Context(), http.MethodGet, "/api/v1/namespaces/default/pods", nil)
					req = withTestContext(req, nil, &auditinternal.Event{Level: auditinternal.LevelMetadata})
					req.RemoteAddr = "127.0.0.1"

					recorder := httptest.NewRecorder()
					handler.ServeHTTP(recorder, req)

					ae := audit.AuditContextFrom(req.Context())
					got := expectedOutcome{
						statusCode:           recorder.Code,
						handlerCalled:        handlerCalled,
						conditionalInContext: gotConditionalDecision,
					}
					got.decisionAnnotation, _ = ae.GetEventAnnotation(DecisionAnnotationKey)
					got.reasonAnnotation, _ = ae.GetEventAnnotation(ReasonAnnotationKey)
					got.conditionalAnnotation, _ = ae.GetEventAnnotation(isConditionalAuthorizationKey)
					if diff := cmp.Diff(mode.want, got, cmp.AllowUnexported(expectedOutcome{})); diff != "" {
						t.Errorf("outcome mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})
	}
}
