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
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// detachContext simulates a handler between authorization and admission that rebuilds the request
// context from scratch, carrying over only what the rest of the request handling needs. Any decision
// registered by the authorization filter is lost.
func detachContext(handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ctx := context.Background()
		if info, ok := request.RequestInfoFrom(req.Context()); ok {
			ctx = request.WithRequestInfo(ctx, info)
		}
		if u, ok := request.UserFrom(req.Context()); ok {
			ctx = request.WithUser(ctx, u)
		}
		handler.ServeHTTP(w, req.WithContext(ctx))
	})
}

// TestConditionsEnforcerContextInvariant verifies the invariant that when the ConditionalAuthorization
// feature gate is on, the authorization filter always registers a decision in the request context, and
// that the AuthorizationConditionsEnforcer fails the request if the decision did not make it to admission.
// This is checked across the four dimensions:
//  1. feature gate on/off
//  2. old WithAuthorization or new WithConditionsAwareAuthorization filter
//  3. AuthorizationConditionsEnforcer admission plugin on/off
//  4. request context forwarded or detached between authorization and admission
//
// The old filter registers an unconditional Allow decision when the feature gate is on, so that it
// keeps working when the enforcer is enabled. The new filter is only used with both the feature gate
// and the enforcer on, as in DefaultBuildHandlerChain.
func TestConditionsEnforcerContextInvariant(t *testing.T) {
	authorizers := []struct {
		name       string
		authorizer authorizer.Authorizer
		// conditional authorizers only allow requests through the new filter.
		conditional bool
	}{
		{
			name:       "unconditional allow",
			authorizer: fakeAuthorizer{authorizer.DecisionAllow, "allowed", nil},
		},
		{
			name: "conditional allow, conditions evaluate to allow",
			authorizer: &conditionsAwareFakeAuthorizer{
				makeDecision: func() authorizer.ConditionsAwareDecision {
					return authorizer.ConditionsAwareDecisionConditionsMap(
						nil, nil,
						[]authorizer.Condition{authorizer.GenericCondition{ID: "example.com/c1", Condition: "object.metadata.name == 'test-pod'", Type: "example.com/cel"}},
					)
				},
				evalConditions: func(_ context.Context, _ authorizer.ConditionsAwareDecision, _ authorizer.ConditionsData) (authorizer.Decision, string, error) {
					return authorizer.DecisionAllow, "conditions met", nil
				},
			},
			conditional: true,
		},
	}
	classifierAlwaysTrue := filters.ConditionalAuthorizationRequestClassifier(func(_ authorizer.Attributes) bool { return true })

	for _, gate := range []bool{false, true} {
		for _, enforcer := range []bool{false, true} {
			for _, newFilter := range []bool{false, true} {
				// DefaultBuildHandlerChain only uses the new filter with the feature gate and the enforcer on.
				if newFilter && (!gate || !enforcer) {
					continue
				}
				for _, detach := range []bool{false, true} {
					for _, authz := range authorizers {
						if authz.conditional && !newFilter {
							continue
						}
						name := fmt.Sprintf("gate=%v/enforcer=%v/newFilter=%v/detach=%v/%s", gate, enforcer, newFilter, detach, authz.name)
						t.Run(name, func(t *testing.T) {
							featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, gate)

							var admit admission.Interface = admission.NewChainHandler()
							if enforcer {
								admit = setupConditionsEnforcer(t)
							}
							updater := &fakeUpdater{}
							var handler http.Handler = handlers.UpdateResource(updater, newRequestScope(t, authz.authorizer), admit)
							if detach {
								handler = detachContext(handler)
							}
							if newFilter {
								var err error
								handler, err = filters.WithConditionsAwareAuthorization(handler, authz.authorizer, testCodecs.WithoutConversion(), classifierAlwaysTrue)
								if err != nil {
									t.Fatal(err)
								}
							} else {
								handler = filters.WithAuthorization(handler, authz.authorizer, testCodecs.WithoutConversion())
							}
							handler = filters.WithAuthentication(handler, authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
								return &authenticator.Response{User: &user.DefaultInfo{Name: "test-user"}}, true, nil
							}), filters.Unauthorized(testCodecs), nil, nil)
							handler = filters.WithRequestInfo(handler, newTestRequestInfoResolver())

							req := makeUpdateRequest(t)
							req = req.WithContext(audit.WithAuditContext(req.Context()))
							recorder := httptest.NewRecorder()
							handler.ServeHTTP(recorder, req)

							// Only the enforcer can notice the detach, and it only runs when the feature gate is on.
							wantDetachCaught := gate && enforcer && detach
							if wantDetachCaught {
								if recorder.Code != http.StatusInternalServerError {
									t.Errorf("status code = %d, want %d; body: %s", recorder.Code, http.StatusInternalServerError, recorder.Body.String())
								}
								if body := recorder.Body.String(); !strings.Contains(body, "a decision must be part of the context") {
									t.Errorf("expected the response to explain the missing decision, got: %s", body)
								}
								if updater.updated {
									t.Error("expected the update to be rejected")
								}
								return
							}
							if recorder.Code != http.StatusOK {
								t.Errorf("status code = %d, want %d; body: %s", recorder.Code, http.StatusOK, recorder.Body.String())
							}
							if !updater.updated {
								t.Error("expected the update to go through")
							}
						})
					}
				}
			}
		}
	}
}
