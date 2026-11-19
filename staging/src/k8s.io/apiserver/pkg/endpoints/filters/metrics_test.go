/*
Copyright 2020 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestMetrics(t *testing.T) {
	// Excluding authentication_duration_seconds since it is difficult to predict its values.
	metrics := []string{
		"authenticated_user_requests",
		"authentication_attempts",
	}

	testCases := []struct {
		desc        string
		response    *authenticator.Response
		status      bool
		err         error
		apiAudience authenticator.Audiences
		want        string
	}{
		{
			desc: "auth ok",
			response: &authenticator.Response{
				User: &user.DefaultInfo{Name: "admin"},
			},
			status: true,
			want: `
				# HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="success"} 1
				`,
		},
		{
			desc: "auth failed with error",
			err:  errors.New("some error"),
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="error"} 1
				`,
		},
		{
			desc: "auth failed with status false",
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="failure"} 1
				`,
		},
		{
			desc: "auth failed due to audiences not intersecting",
			response: &authenticator.Response{
				User:      &user.DefaultInfo{Name: "admin"},
				Audiences: authenticator.Audiences{"audience-x"},
			},
			status:      true,
			apiAudience: authenticator.Audiences{"audience-y"},
			want: `
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="error"} 1
				`,
		},
		{
			desc: "audiences not supplied in the response",
			response: &authenticator.Response{
				User: &user.DefaultInfo{Name: "admin"},
			},
			status:      true,
			apiAudience: authenticator.Audiences{"audience-y"},
			want: `
        # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="success"} 1
				`,
		},
		{
			desc: "audiences not supplied to the handler",
			response: &authenticator.Response{
				User:      &user.DefaultInfo{Name: "admin"},
				Audiences: authenticator.Audiences{"audience-x"},
			},
			status: true,
			want: `
        # HELP authenticated_user_requests [ALPHA] Counter of authenticated requests broken out by username.
				# TYPE authenticated_user_requests counter
				authenticated_user_requests{username="admin"} 1
        # HELP authentication_attempts [ALPHA] Counter of authenticated attempts.
        # TYPE authentication_attempts counter
        authentication_attempts{result="success"} 1
				`,
		},
	}

	// Since prometheus' gatherer is global, other tests may have updated metrics already, so
	// we need to reset them prior running this test.
	// This also implies that we can't run this test in parallel with other auth tests.
	authenticatedUserCounter.Reset()
	authenticatedAttemptsCounter.Reset()

	for _, tt := range testCases {
		t.Run(tt.desc, func(t *testing.T) {
			defer authenticatedUserCounter.Reset()
			defer authenticatedAttemptsCounter.Reset()
			done := make(chan struct{})
			auth := WithAuthentication(
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					close(done)
				}),
				authenticator.RequestFunc(func(_ *http.Request) (*authenticator.Response, bool, error) {
					return tt.response, tt.status, tt.err
				}),
				http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
					close(done)
				}),
				tt.apiAudience,
				nil,
			)

			auth.ServeHTTP(httptest.NewRecorder(), &http.Request{})
			<-done

			if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(tt.want), metrics...); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRecordAuthorizationMetricsMetrics(t *testing.T) {
	// Excluding authorization_duration_seconds since it is difficult to predict its values.
	metrics := []string{
		"authorization_attempts_total",
		"authorization_decision_annotations_total",
	}

	// authzVariant is one way of reaching a given authorization result. A nil classifier uses the
	// conditions-unaware WithAuthorization filter, a non-nil one uses WithConditionsAwareAuthorization,
	// where a classifier returning false falls back to the conditions-unaware Authorize, and one
	// returning true uses ConditionsAwareAuthorize.
	type authzVariant struct {
		authorizer authorizer.Authorizer
		classifier ConditionalAuthorizationRequestClassifier
	}
	conditionsUnaware := func(decision authorizer.Decision, reason string, err error) authzVariant {
		return authzVariant{authorizer: fakeAuthorizer{decision, reason, err}}
	}
	// conditionsAware leaves the classifier nil if classifierResult is nil.
	conditionsAware := func(decision authorizer.ConditionsAwareDecision, classifierResult *bool) authzVariant {
		v := authzVariant{authorizer: &conditionsAwareFakeAuthorizer{makeDecision: func() authorizer.ConditionsAwareDecision { return decision }}}
		if classifierResult != nil {
			v.classifier = func(authorizer.Attributes) bool { return *classifierResult }
		}
		return v
	}

	errUserInfo := errors.New("can't parse user info")
	conditionalAllow := authorizer.ConditionsAwareDecisionConditionsMap(
		nil, nil,
		[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/cond", Condition: "maybe", Type: "example.com/foo-type"}},
	)
	conditionalDeny := authorizer.ConditionsAwareDecisionConditionsMap(
		[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/cond", Condition: "maybe-not", Type: "example.com/foo-type"}},
		nil, nil,
	)
	conditionalNoOpinion := authorizer.ConditionsAwareDecisionConditionsMap(
		nil,
		[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/cond", Condition: "maybe", Type: "example.com/foo-type"}},
		nil,
	)

	testCases := []struct {
		wantResult string
		// variants are the authorizer setups that all must produce wantResult.
		variants map[string]authzVariant
	}{
		{
			wantResult: "allowed",
			variants: map[string]authzVariant{
				"conditions-unaware allow":            conditionsUnaware(authorizer.DecisionAllow, "RBAC: allowed to patch pod", nil),
				"conditions-unaware allow with error": conditionsUnaware(authorizer.DecisionAllow, "", errUserInfo),
				"conditions-aware allow":              conditionsAware(authorizer.ConditionsAwareDecisionAllow("", nil), new(true)),
				"conditions-aware allow with error":   conditionsAware(authorizer.ConditionsAwareDecisionAllow("", errUserInfo), new(true)),
			},
		},
		{
			wantResult: "denied",
			variants: map[string]authzVariant{
				"conditions-unaware deny": conditionsUnaware(authorizer.DecisionDeny, "RBAC: not allowed to patch pod", nil),
				"conditions-aware deny":   conditionsAware(authorizer.ConditionsAwareDecisionDeny("", nil), new(true)),
				// Technically the authorizer did not give the Deny response, but because CanBecomeAllowed() == false,
				// we practically deny the response, and thus does this go under the "denied" label (not NoOpinion, as the authorizer did have an opinion).
				"conditional deny, classifier nil":   conditionsAware(conditionalDeny, nil),
				"conditional deny, classifier false": conditionsAware(conditionalDeny, new(false)),
				"conditional deny, classifier true":  conditionsAware(conditionalDeny, new(true)),
			},
		},
		{
			wantResult: "error",
			variants: map[string]authzVariant{
				"conditions-unaware no-opinion with error": conditionsUnaware(authorizer.DecisionNoOpinion, "", errUserInfo),
				"conditions-aware no-opinion with error":   conditionsAware(authorizer.ConditionsAwareDecisionNoOpinion("", errUserInfo), new(true)),
			},
		},
		{
			wantResult: "no-opinion",
			variants: map[string]authzVariant{
				"conditions-unaware no-opinion": conditionsUnaware(authorizer.DecisionNoOpinion, "", nil),
				"conditions-aware no-opinion":   conditionsAware(authorizer.ConditionsAwareDecisionNoOpinion("", nil), new(true)),
				// A ConditionsMap with only NoOpinion conditions folds to NoOpinion already when constructed.
				"conditional no-opinion, classifier true": conditionsAware(conditionalNoOpinion, new(true)),
				// The request is not subject to conditional authorization, so the conditional decision folds to NoOpinion.
				"conditional allow, classifier false": conditionsAware(conditionalAllow, new(false)),
				"conditional allow, classifier nil":   conditionsAware(conditionalAllow, nil),
			},
		},
		{
			wantResult: "conditional",
			variants: map[string]authzVariant{
				"conditional allow, classifier true": conditionsAware(conditionalAllow, new(true)),
			},
		},
	}

	// Since prometheus' gatherer is global, other tests may have updated metrics already, so
	// we need to reset them prior running this test.
	// This also implies that we can't run this test in parallel with other auth tests.
	authorizationAttemptsCounter.Reset()

	scheme := runtime.NewScheme()
	negotiatedSerializer := serializer.NewCodecFactory(scheme).WithoutConversion()

	// Allow creating conditional decisions in the test
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, true)

	for _, tc := range testCases {
		t.Run(tc.wantResult, func(t *testing.T) {
			want := fmt.Sprintf(`
			# HELP authorization_attempts_total [ALPHA] Counter of authorization attempts broken down by result. It can be either 'allowed', 'denied', 'no-opinion', 'conditional' or 'error'.
			# TYPE authorization_attempts_total counter
			authorization_attempts_total{result=%q} 1
			`, tc.wantResult)

			for name, variant := range tc.variants {
				t.Run(name, func(t *testing.T) {
					ctx := t.Context()
					defer authorizationAttemptsCounter.Reset()

					audit := &auditinternal.Event{Level: auditinternal.LevelMetadata}
					// Mirror DefaultBuildHandlerChain: the conditions-aware filter is only used with a non-nil classifier.
					var handler http.Handler
					if variant.classifier != nil {
						var err error
						handler, err = WithConditionsAwareAuthorization(&fakeHTTPHandler{}, variant.authorizer, negotiatedSerializer, variant.classifier)
						if err != nil {
							t.Fatal(err)
						}
					} else {
						handler = WithAuthorization(&fakeHTTPHandler{}, variant.authorizer, negotiatedSerializer)
					}
					// TODO: fake audit injector

					req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "/api/v1/namespaces/default/pods", nil)
					req = withTestContext(req, nil, audit)
					req.RemoteAddr = "127.0.0.1"
					handler.ServeHTTP(httptest.NewRecorder(), req)

					if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(want), metrics...); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}
