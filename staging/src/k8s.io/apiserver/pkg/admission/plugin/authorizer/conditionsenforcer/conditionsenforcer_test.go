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

package conditionsenforcer

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/managedfields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// ---------- scheme setup ----------

var (
	testScheme = runtime.NewScheme()
	testCodecs = serializer.NewCodecFactory(testScheme)
)

func init() {
	metav1.AddToGroupVersion(testScheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(testScheme))
	utilruntime.Must(examplev1.AddToScheme(testScheme))
}

// ---------- fake authorizers ----------

// fakeAuthorizer returns unconditional decisions.
type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f fakeAuthorizer) Authorize(_ context.Context, _ authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}

func (f fakeAuthorizer) ConditionsAwareAuthorize(ctx context.Context, a authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return authorizer.ConditionsAwareDecisionFromParts(f.Authorize(ctx, a))
}

func (fakeAuthorizer) EvaluateConditions(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

// conditionsAwareFakeAuthorizer returns arbitrary ConditionsAwareDecision values and
// supports configurable condition evaluation.
type conditionsAwareFakeAuthorizer struct {
	makeDecision   func() authorizer.ConditionsAwareDecision
	evalConditions func(ctx context.Context, d authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error)
}

func (f *conditionsAwareFakeAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.ConditionsAwareAuthorize(ctx, a).UnconditionalParts(true)
}

func (f *conditionsAwareFakeAuthorizer) ConditionsAwareAuthorize(_ context.Context, _ authorizer.Attributes) authorizer.ConditionsAwareDecision {
	return f.makeDecision()
}

func (f *conditionsAwareFakeAuthorizer) EvaluateConditions(ctx context.Context, d authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
	if f.evalConditions != nil {
		return f.evalConditions(ctx, d, data)
	}
	return authorizer.DecisionDeny, "", authorizer.ErrorConditionEvaluationNotSupported
}

// ---------- fake rest.Updater ----------

type fakeUpdater struct {
	updated bool
}

func (f *fakeUpdater) New() runtime.Object {
	return &example.Pod{}
}

func (f *fakeUpdater) Destroy() {}

func (f *fakeUpdater) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// Simulate an existing object in storage.
	existing := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       "default",
			ResourceVersion: "1",
		},
		Spec: example.PodSpec{
			NodeName: "foo",
		},
	}

	obj, err := objInfo.UpdatedObject(ctx, existing)
	if err != nil {
		return nil, false, err
	}

	if updateValidation != nil {
		if err := updateValidation(ctx, obj, existing); err != nil {
			return nil, false, err
		}
	}

	f.updated = true
	return obj, false, nil
}

// ---------- fake ScopeNamer ----------

type fakeNamer struct {
	namespace string
	name      string
}

func (n *fakeNamer) Namespace(_ *http.Request) (string, error) {
	return n.namespace, nil
}

func (n *fakeNamer) Name(_ *http.Request) (string, string, error) {
	return n.namespace, n.name, nil
}

func (n *fakeNamer) ObjectName(_ runtime.Object) (string, string, error) {
	return n.namespace, n.name, nil
}

// ---------- helpers ----------

func newRequestScope(t *testing.T, auth authorizer.Authorizer) *handlers.RequestScope {
	t.Helper()

	kind := examplev1.SchemeGroupVersion.WithKind("Pod")
	resource := examplev1.SchemeGroupVersion.WithResource("pods")
	hubVersion := example.SchemeGroupVersion
	convertor := runtime.UnsafeObjectConvertor(testScheme)

	fm, err := managedfields.NewDefaultFieldManager(
		managedfields.NewDeducedTypeConverter(),
		convertor,
		testScheme, // defaulter
		testScheme, // creater
		kind,
		hubVersion,
		"",
		nil,
	)
	if err != nil {
		t.Fatalf("failed to create field manager: %v", err)
	}

	return &handlers.RequestScope{
		Namer:                    &fakeNamer{namespace: "default", name: "test-pod"},
		Serializer:               testCodecs,
		Creater:                  testScheme,
		Convertor:                convertor,
		Defaulter:                testScheme,
		Typer:                    testScheme,
		UnsafeConvertor:          convertor,
		Authorizer:               auth,
		Resource:                 resource,
		Kind:                     kind,
		MetaGroupVersion:         schema.GroupVersion{Version: "v1"},
		HubGroupVersion:          hubVersion,
		FieldManager:             fm,
		EquivalentResourceMapper: runtime.NewEquivalentResourceRegistry(),
		MaxRequestBodyBytes:      int64(3 * 1024 * 1024),
	}
}

func newTestRequestInfoResolver() *request.RequestInfoFactory {
	return &request.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}
}

func makeUpdateRequest(t *testing.T) *http.Request {
	t.Helper()

	pod := &examplev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "example.apiserver.k8s.io/v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test-pod",
			Namespace:       "default",
			ResourceVersion: "1",
		},
		Spec: examplev1.PodSpec{
			NodeName: "bar",
		},
	}

	codec := testCodecs.LegacyCodec(examplev1.SchemeGroupVersion)
	body, err := runtime.Encode(codec, pod)
	if err != nil {
		t.Fatalf("failed to encode pod: %v", err)
	}

	req, err := http.NewRequest(http.MethodPut,
		"/apis/example.apiserver.k8s.io/v1/namespaces/default/pods/test-pod",
		bytes.NewReader(body))
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.RemoteAddr = "127.0.0.1"

	return req
}

func setupConditionsEnforcer(t *testing.T) admission.Interface {
	t.Helper()

	plugin := NewConditionalAuthorizationEnforcer(false) // Enablement determined by the feature gate
	plugin.InspectFeatureGates(utilfeature.DefaultFeatureGate)
	if err := plugin.ValidateInitialization(); err != nil {
		t.Fatalf("ValidateInitialization failed: %v", err)
	}
	return plugin
}

func TestConditionsEnforcerEndToEnd(t *testing.T) {
	classifierAlwaysTrue := filters.ConditionalAuthorizationRequestClassifier(func(_ authorizer.Attributes) bool { return true })
	classifierAlwaysFalse := filters.ConditionalAuthorizationRequestClassifier(func(_ authorizer.Attributes) bool { return false })

	makeCondMapAllowDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			nil, nil,
			[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c1", Condition: "object.metadata.name == 'test-pod'", Type: "example.com/cel"}},
		)
	}

	// makeCondMapMixedDecision registers both an allow- and a deny-condition, so
	// PossibleDecisions() == {Allow, Deny, NoOpinion}. Use this when the test needs
	// EvaluateConditions to legitimately return Deny — the enforcer's invariant
	// check rejects a Deny outcome unless Deny was declared possible up-front.
	makeCondMapMixedDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/deny", Condition: "object.metadata.name == 'bad'", Type: "example.com/cel"}},
			nil,
			[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/allow", Condition: "object.metadata.name == 'test-pod'", Type: "example.com/cel"}},
		)
	}

	type expectedOutcome struct {
		statusCode         int
		decisionAnnotation string
		reasonAnnotation   string
		updated            bool // whether the updater was called
	}

	tests := []struct {
		name                       string
		authorizer                 authorizer.Authorizer
		conditionalAuthzClassifier filters.ConditionalAuthorizationRequestClassifier
		disabled                   expectedOutcome
		enabled                    expectedOutcome
	}{
		{
			name:       "unconditional allow passes through",
			authorizer: fakeAuthorizer{authorizer.DecisionAllow, "allowed", nil},
			disabled:   expectedOutcome{statusCode: http.StatusOK, updated: true, decisionAnnotation: "allow", reasonAnnotation: "allowed"},
			enabled:    expectedOutcome{statusCode: http.StatusOK, updated: true, decisionAnnotation: "allow", reasonAnnotation: "allowed"},
		},
		{
			name:       "unconditional deny is rejected at the auth filter",
			authorizer: fakeAuthorizer{authorizer.DecisionDeny, "denied", nil},
			disabled:   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "denied"},
			enabled:    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "denied"},
		},
		{
			name:       "unconditional deny with error yields internal error at the auth filter",
			authorizer: fakeAuthorizer{authorizer.DecisionDeny, "denied", errors.New("unexpected")},
			// TODO(luxas): Is it really expected to not add the decisionAnnotation?
			disabled: expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: "internal error"},
			enabled:  expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: "internal error"},
		},
		{
			name:       "no opinion without error is forbidden at the auth filter",
			authorizer: fakeAuthorizer{authorizer.DecisionNoOpinion, "no match", nil},
			disabled:   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "no match"},
			enabled:    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "no match"},
		},
		{
			name:       "no opinion with error yields internal error at the auth filter",
			authorizer: fakeAuthorizer{authorizer.DecisionNoOpinion, "no match", errors.New("unexpected")},
			disabled:   expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: "internal error"},
			enabled:    expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: "internal error"},
		},
		{
			name: "conditional allow + classifier true + eval allows => update succeeds",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "conditions met", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: auth filter lets through, conditions eval to allow => update succeeds
			enabled: expectedOutcome{statusCode: http.StatusOK, updated: true, decisionAnnotation: "allow", reasonAnnotation: "conditions met"},
		},
		{
			name: "conditional allow + classifier true + eval no opinion => admission rejects",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionNoOpinion, "conditions not met", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes to Deny (deny effect present) => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: auth filter lets through, conditions eval to deny => forbidden from admission
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "conditions not met"},
		},
		{
			name: "conditional allow + classifier true + eval no opinion with error => admission rejects",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionNoOpinion, "", errors.New("unexpected error")
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes to Deny (deny effect present) => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: auth filter lets through, conditions eval to deny => forbidden from admission
			enabled: expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: "internal error"},
		},
		{
			name: "conditional + classifier true + eval denies => admission rejects",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapMixedDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "deny condition matched", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes to Deny (deny effect present) => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Deny Allow NoOpinion]) to Deny during unconditional authorization"},
			// gate on: auth filter lets through, conditions eval to deny => forbidden from admission
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "deny condition matched"},
		},
		{
			name: "conditional + classifier true + eval denies with error => admission rejects",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapMixedDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "failed closed", errors.New("unexpected error")
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes to Deny (deny effect present) => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Deny Allow NoOpinion]) to Deny during unconditional authorization"},
			// gate on: auth filter lets through, conditions eval to deny => forbidden from admission
			enabled: expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: "internal error"},
		},
		{
			name: "conditional allow + classifier true + eval allows (based on versioned object inspection) => update succeeds",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
					new, ok := data.GetObject().(*examplev1.Pod)
					if !ok {
						return authorizer.DecisionDeny, "object not *examplev1.Pod as expected", nil
					}
					if new.Spec.NodeName != "bar" {
						return authorizer.DecisionDeny, "object.nodeName is bar in makeUpdateRequest, should be the same here", nil
					}
					old, ok := data.GetOldObject().(*examplev1.Pod)
					if !ok {
						return authorizer.DecisionDeny, "oldObject not *examplev1.Pod as expected", nil
					}
					if old.Spec.NodeName != "foo" {
						return authorizer.DecisionDeny, "oldObject.nodeName is foo in fakeUpdater.Update, should be the same here", nil
					}
					return authorizer.DecisionAllow, "ok", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: auth filter lets through, conditions eval to allow => update succeeds
			enabled: expectedOutcome{statusCode: http.StatusOK, updated: true, decisionAnnotation: "allow", reasonAnnotation: "ok"},
		},
		{
			name: "conditional allow + classifier false => forbidden at auth filter",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
			},
			conditionalAuthzClassifier: classifierAlwaysFalse,
			disabled:                   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: classifier rejects => forbidden
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
		},
		{
			name: "conditional allow + classifier nil => forbidden at auth filter",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
			},
			conditionalAuthzClassifier: nil,
			disabled:                   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			enabled:                    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
		},
		{
			// Exercises the enforcer's invariant check: an allow-only ConditionsMap has
			// PossibleDecisions == {Allow, NoOpinion}. If EvaluateConditions returns
			// Deny anyway, the enforcer folds the outcome to the FailureDecision
			// (NoOpinion here — no deny effect declared) and surfaces an internal
			// error explaining the invariant violation.
			name: "conditional allow + eval returns impossible Deny => internal error",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "should be ignored", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes to NoOpinion (no deny effect) => forbidden at filter
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed from conditional decision (with possible outcomes [Allow NoOpinion]) to NoOpinion during unconditional authorization"},
			// gate on: auth filter lets through, enforcer's invariant check rejects the
			// out-of-set Deny and returns InternalError with the "internal error"
			// audit reason.
			enabled: expectedOutcome{statusCode: http.StatusInternalServerError, reasonAnnotation: filters.ReasonError},
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

					updater := &fakeUpdater{}
					scope := newRequestScope(t, tt.authorizer)
					admit := setupConditionsEnforcer(t)

					innerHandler := handlers.UpdateResource(updater, scope, admit)

					// Wire up: WithRequestInfo -> WithAuthorization (with conditions support) -> UpdateResource
					var handler http.Handler = innerHandler

					// TODO(luxas): Wire up conditions support for compound authorization for update -> create requests here as well using the real filter.

					// TODO(luxas): Could we use the real BuildHandlerChain here?
					if mode.gate {
						// This test is exercising the conditions-enforcer admission plugin, so mark
						// the enforcer as enabled in the WithConditionsAwareAuthorization dispatch.
						handler = filters.WithConditionsAwareAuthorization(handler, tt.authorizer, testCodecs.WithoutConversion(), true /* conditionsEnforcerEnabled */, tt.conditionalAuthzClassifier)
					} else {
						handler = filters.WithAuthorization(handler, tt.authorizer, testCodecs.WithoutConversion())
					}
					failedHandler := filters.Unauthorized(testCodecs)
					handler = filters.WithAuthentication(handler, authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
						return &authenticator.Response{
							User: &user.DefaultInfo{Name: "test-user"},
						}, true, nil
					}), failedHandler, nil, nil)
					handler = filters.WithRequestInfo(handler, newTestRequestInfoResolver())

					req := makeUpdateRequest(t)
					req = req.WithContext(audit.WithAuditContext(req.Context()))

					recorder := httptest.NewRecorder()
					handler.ServeHTTP(recorder, req)

					if recorder.Code != mode.want.statusCode {
						t.Errorf("status code = %d, want %d; body: %s", recorder.Code, mode.want.statusCode, recorder.Body.String())
					}
					if updater.updated != mode.want.updated {
						t.Errorf("updater called = %v, want %v", updater.updated, mode.want.updated)
					}

					ae := audit.AuditContextFrom(req.Context())
					var gotDecisionAnnotation, gotReasonAnnotation string
					if ae != nil {
						gotDecisionAnnotation, _ = ae.GetEventAnnotation(filters.DecisionAnnotationKey)
						gotReasonAnnotation, _ = ae.GetEventAnnotation(filters.ReasonAnnotationKey)
					}

					if gotDecisionAnnotation != mode.want.decisionAnnotation {
						t.Errorf("decision annotation = %q, want %q", gotDecisionAnnotation, mode.want.decisionAnnotation)
					}
					if gotReasonAnnotation != mode.want.reasonAnnotation {
						t.Errorf("reason annotation = %q, want %q", gotReasonAnnotation, mode.want.reasonAnnotation)
					}
				})
			}
		})
	}
}
