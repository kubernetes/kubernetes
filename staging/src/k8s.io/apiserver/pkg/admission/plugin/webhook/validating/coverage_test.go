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

package validating

import (
	"net/url"
	"strings"
	"testing"

	registrationv1 "k8s.io/api/admissionregistration/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/coverage"
	webhooktesting "k8s.io/apiserver/pkg/admission/plugin/webhook/testing"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
)

// The coverage semantics are tested against ShouldCallHook in
// webhook/generic. These tests cover what needs the real plugin: dispatch
// order, real matchConditions and failurePolicy, and metrics.

const coverageHookName = "coverage.example.com"

// excludedByMatchConditions would keep the hook from being called if its rules
// matched the request.
func excludedByMatchConditions(h registrationv1.ValidatingWebhook) registrationv1.ValidatingWebhook {
	h.MatchConditions = []registrationv1.MatchCondition{{Name: "never", Expression: "false"}}
	return h
}

func initCoveragePlugin(t *testing.T, wh *Plugin, serverURL *url.URL, hooks []registrationv1.ValidatingWebhook) {
	t.Helper()
	stopCh := make(chan struct{})
	t.Cleanup(func() { close(stopCh) })
	wh.SetDrainedNotification(stopCh)
	client, informer := webhooktesting.NewFakeValidatingDataSource("default", hooks, stopCh)
	wh.SetAuthenticationInfoResolverWrapper(webhooktesting.Wrapper(webhooktesting.NewAuthenticationInfoResolver(new(int32))))
	wh.SetServiceResolver(webhooktesting.NewServiceResolver(*serverURL))
	wh.SetExternalKubeClientSet(client)
	wh.SetExternalKubeInformerFactory(informer)
	if err := wh.ValidateInitialization(); err != nil {
		t.Fatalf("failed to validate initialization: %v", err)
	}
	informer.Start(stopCh)
	informer.WaitForCacheSync(stopCh)
}

func coverageAnnotation(attr *webhooktesting.FakeAttributes) (string, bool) {
	v, ok := attr.GetAnnotations(auditinternal.LevelMetadata)[coverage.AuditAnnotationKey(coverage.WebhookValidating)]
	return v, ok
}

func TestAdmissionCoverage(t *testing.T) {
	testcases := []struct {
		name            string
		noRequirements  bool
		hooks           []registrationv1.ValidatingWebhook
		expectForbidden bool
		expectCalls     int32
	}{
		{
			name:            "violation rejected despite failurePolicy Ignore and excluding matchConditions",
			hooks:           []registrationv1.ValidatingWebhook{excludedByMatchConditions(webhooktesting.NewCoverageWebhook(coverageHookName, "pods"))},
			expectForbidden: true,
		},
		{
			// Proves the matchCondition above would have excluded the request.
			name:  "control: matchConditions exclude a directly matching hook",
			hooks: []registrationv1.ValidatingWebhook{excludedByMatchConditions(webhooktesting.NewCoverageWebhook(coverageHookName, "pods", "pods/dynamic"))},
		},
		{
			// Every validating hook is matched before any is called.
			name: "violation rejects before any webhook is called",
			hooks: []registrationv1.ValidatingWebhook{
				webhooktesting.NewCoverageWebhook("direct.example.com", "pods/*"),
				webhooktesting.NewCoverageWebhook(coverageHookName, "pods"),
			},
			expectForbidden: true,
		},
		{
			name:           "no coverage requirements: unchanged",
			noRequirements: true,
			hooks:          []registrationv1.ValidatingWebhook{webhooktesting.NewCoverageWebhook(coverageHookName, "pods")},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			serverURL, calls := webhooktesting.NewCoverageTestServer(t)
			wh, err := NewValidatingAdmissionWebhook(nil)
			if err != nil {
				t.Fatalf("failed to create validating webhook: %v", err)
			}
			if !tc.noRequirements {
				wh.SetAdmissionCoverage(webhooktesting.NewCoverageRequirements(t))
			}
			initCoveragePlugin(t, wh, serverURL, tc.hooks)
			attrs := webhooktesting.NewPodAttribute(admission.Update, "dynamic", nil)
			before := webhooktesting.CoverageViolations(t, coverageHookName, "validate")

			err = wh.Validate(t.Context(), attrs, webhooktesting.NewObjectInterfacesForTest())

			annotation, annotated := coverageAnnotation(attrs)
			violations := webhooktesting.CoverageViolations(t, coverageHookName, "validate") - before
			if tc.expectForbidden {
				if !apierrors.IsForbidden(err) {
					t.Fatalf("expected 403 Forbidden, got %v", err)
				}
				if !strings.Contains(err.Error(), "intercepts CREATE pods but not UPDATE pods/dynamic") {
					t.Errorf("unexpected message: %v", err)
				}
				want := `[{"name":"coverage.example.com","configuration":"test-webhooks","source":"CREATE pods","target":"UPDATE pods/dynamic"}]`
				if annotation != want {
					t.Errorf("expected audit annotation %s, got %q (present=%v)", want, annotation, annotated)
				}
				if violations != 1 {
					t.Errorf("expected coverage violation metric +1, got %v", violations)
				}
			} else {
				if err != nil {
					t.Fatalf("expected request to be allowed, got %v", err)
				}
				if annotated {
					t.Errorf("expected no coverage audit annotation, got %q", annotation)
				}
				if violations != 0 {
					t.Errorf("expected no coverage violation metric, got +%v", violations)
				}
			}
			if got := calls.Load(); got != tc.expectCalls {
				t.Errorf("expected %d webhook calls, got %d", tc.expectCalls, got)
			}
		})
	}
}
