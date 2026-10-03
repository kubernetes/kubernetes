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

package selfsubjectaccessreview

import (
	"context"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	_ "k8s.io/kubernetes/pkg/apis/authorization/install"
)

type fakeAuthorizer struct {
	decision authorizer.Decision
	reason   string
	err      error
}

func (f *fakeAuthorizer) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	return f.decision, f.reason, f.err
}

// TestCreateIgnoresPostedStatus pins the behavior of k8s 1.37 and earlier: a client
// could post a SelfSubjectAccessReview carrying a bogus status and the request still
// succeeded. The handler clears the status before validating, so status rules that would
// otherwise reject it, here allowed and denied both being set, cannot turn a
// previously-working request into an error. The status that comes back is the
// authorizer's answer rather than what was posted.
//
// It also covers the other half of that pre-validation fixup: the opt-in to
// conditions-awareness is dropped while the feature gate is off, so the request falls
// back to the conditions-unaware Authorize.
func TestCreateIgnoresPostedStatus(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.ConditionalAuthorization, false)

	auth := &fakeAuthorizer{decision: authorizer.DecisionAllow, reason: "myreason"}
	storage := NewREST(auth, legacyscheme.Scheme)

	ctx := genericapirequest.WithUser(
		genericapirequest.WithRequestInfo(
			genericapirequest.NewContext(),
			&genericapirequest.RequestInfo{
				APIGroup:          "authorization.k8s.io",
				APIVersion:        "v1",
				Resource:          "selfsubjectaccessreviews",
				IsResourceRequest: true,
				Verb:              "create",
			},
		),
		&user.DefaultInfo{Name: "bob"},
	)

	ssar := &authorizationapi.SelfSubjectAccessReview{
		Spec: authorizationapi.SelfSubjectAccessReviewSpec{
			ResourceAttributes: &authorizationapi.ResourceAttributes{Verb: "get", Resource: "pods"},
			AuthorizationOptions: &authorizationapi.AuthorizationOptions{
				HandledDecisionTypes: []authorizationapi.ConditionsAwareDecisionType{
					authorizationapi.ConditionsAwareDecisionTypeAllow,
					authorizationapi.ConditionsAwareDecisionTypeDeny,
					authorizationapi.ConditionsAwareDecisionTypeNoOpinion,
					authorizationapi.ConditionsAwareDecisionTypeConditionsMap,
					authorizationapi.ConditionsAwareDecisionTypeUnion,
				},
			},
		},
		// Mutually exclusive, so validating this status would reject the request.
		Status: authorizationapi.SubjectAccessReviewStatus{
			Allowed:         true,
			Denied:          true,
			Reason:          "posted by the client",
			EvaluationError: "posted by the client",
		},
	}

	result, err := storage.Create(ctx, ssar, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("expected the posted status to be ignored, got: %v", err)
	}

	got := result.(*authorizationapi.SelfSubjectAccessReview)
	want := authorizationapi.SubjectAccessReviewStatus{Allowed: true, Reason: "myreason"}
	if !reflect.DeepEqual(got.Status, want) {
		t.Errorf("expected status\n%#v\ngot\n%#v", want, got.Status)
	}
	if got.Spec.AuthorizationOptions != nil {
		t.Errorf("expected the conditions opt-in to be cleared while the feature gate is off, got %#v", got.Spec.AuthorizationOptions)
	}
}
