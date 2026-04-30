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
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
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
	return f.ConditionsAwareAuthorize(ctx, a).UnconditionalParts()
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

func setupConditionsEnforcer(t *testing.T, auth authorizer.Authorizer) admission.Interface {
	t.Helper()

	plugin := NewConditionalAuthorizationEnforcer(false) // Enablement determined by the feature gate
	plugin.InspectFeatureGates(utilfeature.DefaultFeatureGate)
	plugin.SetAuthorizer(auth)
	if err := plugin.ValidateInitialization(); err != nil {
		t.Fatalf("ValidateInitialization failed: %v", err)
	}
	return plugin
}

func withCompoundAuthorization(handler http.Handler, compoundAuthorizer authorizer.Authorizer, s runtime.NegotiatedSerializer) http.Handler {
	if compoundAuthorizer == nil {
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		attrs, err := filters.GetAuthorizerAttributes(ctx)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.ConditionalAuthorization) {
			decision, reason, err := compoundAuthorizer.Authorize(ctx, attrs)
			if decision == authorizer.DecisionAllow {
				handler.ServeHTTP(w, req)
				return
			}
			if err != nil {
				responsewriters.InternalError(w, req, err)
				return
			}
			responsewriters.Forbidden(attrs, w, req, reason, s)
			return
		}
		conditionsAwareDecision := compoundAuthorizer.ConditionsAwareAuthorize(ctx, attrs)
		isUnconditionallyAllowed := conditionsAwareDecision.IsAllowed()
		reason := conditionsAwareDecision.Reason()
		err = conditionsAwareDecision.Error()

		if !isUnconditionallyAllowed && conditionsAwareDecision.CanBecomeAllowed() {
			var evalDecision authorizer.Decision
			evalDecision, reason, err = compoundAuthorizer.EvaluateConditions(ctx, conditionsAwareDecision, authorizer.ConditionsData{})
			isUnconditionallyAllowed = evalDecision == authorizer.DecisionAllow
		}

		if isUnconditionallyAllowed {
			// No audit annotation here, as there still one more enforcement to do
			handler.ServeHTTP(w, req)
			return
		}

		if err != nil {
			audit.AddAuditAnnotation(ctx, filters.ReasonAnnotationKey, filters.ReasonError)
			responsewriters.InternalError(w, req, err)
			return
		}

		audit.AddAuditAnnotations(ctx,
			filters.DecisionAnnotationKey, filters.DecisionForbid,
			filters.ReasonAnnotationKey, reason)
		responsewriters.Forbidden(attrs, w, req, reason, s)
		// return
	})
}

// ---------- tests ----------

func TestConditionsEnforcerEndToEnd(t *testing.T) {
	classifierAlwaysTrue := filters.ConditionalAuthorizationRequestClassifier(func(_ authorizer.Attributes) bool { return true })
	classifierAlwaysFalse := filters.ConditionalAuthorizationRequestClassifier(func(_ authorizer.Attributes) bool { return false })

	makeCondMapAllowDecision := func() authorizer.ConditionsAwareDecision {
		return authorizer.ConditionsAwareDecisionConditionsMap(
			authorizer.GenericCondition{ID: "c1", Condition: "object.metadata.name == 'test-pod'", Effect: authorizer.ConditionEffectAllow, Type: "cel"},
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
		compoundAuthorizer         authorizer.Authorizer
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
			name:       "unconditional deny is rejected at auth filter",
			authorizer: fakeAuthorizer{authorizer.DecisionDeny, "denied", nil},
			disabled:   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "denied"},
			enabled:    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "denied"},
		},
		{
			name:       "no opinion without error is forbidden",
			authorizer: fakeAuthorizer{authorizer.DecisionNoOpinion, "no match", nil},
			disabled:   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "no match"},
			enabled:    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "no match"},
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
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			// gate on: auth filter lets through, conditions eval to allow => update succeeds
			enabled: expectedOutcome{statusCode: http.StatusOK, updated: true, decisionAnnotation: "allow", reasonAnnotation: "conditions met"},
		},
		{
			name: "conditional allow + classifier true + eval denies => admission rejects",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "conditions not met", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			disabled:                   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			// gate on: auth filter lets through, conditions eval to deny => forbidden from admission
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "conditions not met"},
		},
		{
			name: "conditional allow + classifier true + eval allows (based on versioned object inspection) => update succeeds",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, data authorizer.ConditionsData) (authorizer.Decision, string, error) {
					new, ok := data.AdmissionControl.GetObject().(*examplev1.Pod)
					if !ok {
						return authorizer.DecisionDeny, "object not *examplev1.Pod as expected", nil
					}
					if new.Spec.NodeName != "bar" {
						return authorizer.DecisionDeny, "object.nodeName is bar in makeUpdateRequest, should be the same here", nil
					}
					old, ok := data.AdmissionControl.GetOldObject().(*examplev1.Pod)
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
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			// gate on: auth filter lets through, conditions eval to allow => update succeeds
			enabled: expectedOutcome{statusCode: http.StatusOK, updated: true, decisionAnnotation: "allow", reasonAnnotation: "ok"},
		},
		{
			name: "conditional allow + classifier false => forbidden at auth filter",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
			},
			conditionalAuthzClassifier: classifierAlwaysFalse,
			disabled:                   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			// gate on: classifier rejects => forbidden
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed: tried to return conditional decision to conditions-unaware authorizer"},
		},
		{
			name: "conditional allow + classifier nil => forbidden at auth filter",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
			},
			conditionalAuthzClassifier: nil,
			disabled:                   expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			enabled:                    expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "failed closed: tried to return conditional decision to conditions-unaware authorizer"},
		},
		// Make sure all registered conditional decisions in the context are enforced
		{
			name: "conditional (=> allow) + compound conditional (=> allow) => update succeeds",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "conditions met", nil
				},
			},
			compoundAuthorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "conditions met", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			// gate on: auth filter lets through, both conditions eval to allow
			enabled: expectedOutcome{statusCode: http.StatusOK, updated: true, decisionAnnotation: "allow", reasonAnnotation: "conditions met"},
		},
		{
			name: "conditional (=> allow) + compound conditional (=> deny) => denied",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "authorization conditions met", nil
				},
			},
			compoundAuthorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "compound conditions not met", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			// gate on: auth filter lets through, but the compound conditions deny the request
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "compound conditions not met"},
		},
		{
			name: "conditional (=> deny) + compound conditional (=> allow) => denied",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionDeny, "authorization conditions not met", nil
				},
			},
			compoundAuthorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: makeCondMapAllowDecision,
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "compound conditions met", nil
				},
			},
			conditionalAuthzClassifier: classifierAlwaysTrue,
			// gate off: condMap constructor fail-closes => forbidden
			disabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorizer tried to return conditional decision, but the ConditionalAuthorization feature gate is disabled"},
			// gate on: auth filter lets through, but the compound conditions deny the request
			enabled: expectedOutcome{statusCode: http.StatusForbidden, decisionAnnotation: "forbid", reasonAnnotation: "authorization conditions not met"},
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
					admit := setupConditionsEnforcer(t, tt.authorizer)

					innerHandler := handlers.UpdateResource(updater, scope, admit)

					// Wire up: WithRequestInfo -> WithAuthorization (with conditions support) -> UpdateResource
					var handler http.Handler = innerHandler

					// Compound authorization only enabled if tt.compoundAuthorizer != nil
					handler = withCompoundAuthorization(handler, tt.compoundAuthorizer, testCodecs)

					if mode.gate {
						handler = filters.WithAuthorizationAndConditionsSupport(handler, tt.authorizer, testCodecs.WithoutConversion(), tt.conditionalAuthzClassifier)
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
