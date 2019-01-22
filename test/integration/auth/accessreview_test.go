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

package auth

import (
	"errors"
	"net/http"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	api "k8s.io/kubernetes/pkg/apis/core"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/test/integration/framework"
)

// Inject into master an authorizer that uses user info.
// TODO(etune): remove this test once a more comprehensive built-in authorizer is implemented.
type sarAuthorizer struct{}

func (sarAuthorizer) Authorize(a authorizer.Attributes) (authorizer.Decision, string, error) {
	if a.GetUser().GetName() == "dave" {
		return authorizer.DecisionNoOpinion, "no", errors.New("I'm sorry, Dave")
	}

	return authorizer.DecisionAllow, "you're not dave", nil
}

func alwaysAlice(req *http.Request) (*authenticator.Response, bool, error) {
	return &authenticator.Response{
		User: &user.DefaultInfo{
			Name: "alice",
		},
	}, true, nil
}

func TestSubjectAccessReview(t *testing.T) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = authenticator.RequestFunc(alwaysAlice)
	masterConfig.GenericConfig.Authorization.Authorizer = sarAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[api.GroupName].GroupVersion()}})

	tests := []struct {
		name           string
		sar            *authorizationapi.SubjectAccessReview
		expectedError  string
		expectedStatus authorizationapi.SubjectAccessReviewStatus
	}{
		{
			name: "simple allow",
			sar: &authorizationapi.SubjectAccessReview{
				Spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:     "list",
						Group:    api.GroupName,
						Version:  "v1",
						Resource: "pods",
					},
					User: "alice",
				},
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "you're not dave",
			},
		},
		{
			name: "simple deny",
			sar: &authorizationapi.SubjectAccessReview{
				Spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:     "list",
						Group:    api.GroupName,
						Version:  "v1",
						Resource: "pods",
					},
					User: "dave",
				},
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed:         false,
				Reason:          "no",
				EvaluationError: "I'm sorry, Dave",
			},
		},
		{
			name: "simple error",
			sar: &authorizationapi.SubjectAccessReview{
				Spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:     "list",
						Group:    api.GroupName,
						Version:  "v1",
						Resource: "pods",
					},
				},
			},
			expectedError: "at least one of user or group must be specified",
		},
	}

	for _, test := range tests {
		response, err := clientset.Authorization().SubjectAccessReviews().Create(test.sar)
		switch {
		case err == nil && len(test.expectedError) == 0:

		case err != nil && strings.Contains(err.Error(), test.expectedError):
			continue

		case err != nil && len(test.expectedError) != 0:
			t.Errorf("%s: unexpected error: %v", test.name, err)
			continue
		default:
			t.Errorf("%s: expected %v, got %v", test.name, test.expectedError, err)
			continue
		}
		if response.Status != test.expectedStatus {
			t.Errorf("%s: expected %v, got %v", test.name, test.expectedStatus, response.Status)
			continue
		}
	}
}

func TestSelfSubjectAccessReview(t *testing.T) {
	username := "alice"
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = authenticator.RequestFunc(func(req *http.Request) (*authenticator.Response, bool, error) {
		return &authenticator.Response{
			User: &user.DefaultInfo{Name: username},
		}, true, nil
	})
	masterConfig.GenericConfig.Authorization.Authorizer = sarAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[api.GroupName].GroupVersion()}})

	tests := []struct {
		name           string
		username       string
		sar            *authorizationapi.SelfSubjectAccessReview
		expectedError  string
		expectedStatus authorizationapi.SubjectAccessReviewStatus
	}{
		{
			name:     "simple allow",
			username: "alice",
			sar: &authorizationapi.SelfSubjectAccessReview{
				Spec: authorizationapi.SelfSubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:     "list",
						Group:    api.GroupName,
						Version:  "v1",
						Resource: "pods",
					},
				},
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "you're not dave",
			},
		},
		{
			name:     "simple deny",
			username: "dave",
			sar: &authorizationapi.SelfSubjectAccessReview{
				Spec: authorizationapi.SelfSubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:     "list",
						Group:    api.GroupName,
						Version:  "v1",
						Resource: "pods",
					},
				},
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed:         false,
				Reason:          "no",
				EvaluationError: "I'm sorry, Dave",
			},
		},
	}

	for _, test := range tests {
		username = test.username

		response, err := clientset.Authorization().SelfSubjectAccessReviews().Create(test.sar)
		switch {
		case err == nil && len(test.expectedError) == 0:

		case err != nil && strings.Contains(err.Error(), test.expectedError):
			continue

		case err != nil && len(test.expectedError) != 0:
			t.Errorf("%s: unexpected error: %v", test.name, err)
			continue
		default:
			t.Errorf("%s: expected %v, got %v", test.name, test.expectedError, err)
			continue
		}
		if response.Status != test.expectedStatus {
			t.Errorf("%s: expected %v, got %v", test.name, test.expectedStatus, response.Status)
			continue
		}
	}
}

func TestLocalSubjectAccessReview(t *testing.T) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.GenericConfig.Authentication.Authenticator = authenticator.RequestFunc(alwaysAlice)
	masterConfig.GenericConfig.Authorization.Authorizer = sarAuthorizer{}
	_, s, closeFn := framework.RunAMaster(masterConfig)
	defer closeFn()

	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[api.GroupName].GroupVersion()}})

	tests := []struct {
		name           string
		namespace      string
		sar            *authorizationapi.LocalSubjectAccessReview
		expectedError  string
		expectedStatus authorizationapi.SubjectAccessReviewStatus
	}{
		{
			name:      "simple allow",
			namespace: "foo",
			sar: &authorizationapi.LocalSubjectAccessReview{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo"},
				Spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:      "list",
						Group:     api.GroupName,
						Version:   "v1",
						Resource:  "pods",
						Namespace: "foo",
					},
					User: "alice",
				},
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed: true,
				Reason:  "you're not dave",
			},
		},
		{
			name:      "simple deny",
			namespace: "foo",
			sar: &authorizationapi.LocalSubjectAccessReview{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo"},
				Spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:      "list",
						Group:     api.GroupName,
						Version:   "v1",
						Resource:  "pods",
						Namespace: "foo",
					},
					User: "dave",
				},
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed:         false,
				Reason:          "no",
				EvaluationError: "I'm sorry, Dave",
			},
		},
		{
			name:      "conflicting namespace",
			namespace: "foo",
			sar: &authorizationapi.LocalSubjectAccessReview{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo"},
				Spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:      "list",
						Group:     api.GroupName,
						Version:   "v1",
						Resource:  "pods",
						Namespace: "bar",
					},
					User: "dave",
				},
			},
			expectedError: "must match metadata.namespace",
		},
		{
			name:      "missing namespace",
			namespace: "foo",
			sar: &authorizationapi.LocalSubjectAccessReview{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo"},
				Spec: authorizationapi.SubjectAccessReviewSpec{
					ResourceAttributes: &authorizationapi.ResourceAttributes{
						Verb:     "list",
						Group:    api.GroupName,
						Version:  "v1",
						Resource: "pods",
					},
					User: "dave",
				},
			},
			expectedError: "must match metadata.namespace",
		},
	}

	for _, test := range tests {
		response, err := clientset.Authorization().LocalSubjectAccessReviews(test.namespace).Create(test.sar)
		switch {
		case err == nil && len(test.expectedError) == 0:

		case err != nil && strings.Contains(err.Error(), test.expectedError):
			continue

		case err != nil && len(test.expectedError) != 0:
			t.Errorf("%s: unexpected error: %v", test.name, err)
			continue
		default:
			t.Errorf("%s: expected %v, got %v", test.name, test.expectedError, err)
			continue
		}
		if response.Status != test.expectedStatus {
			t.Errorf("%s: expected %#v, got %#v", test.name, test.expectedStatus, response.Status)
			continue
		}
	}
}
