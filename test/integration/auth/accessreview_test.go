// +build integration,!no-etcd

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
	"net/http/httptest"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/auth/user"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
	"k8s.io/kubernetes/test/integration/framework"
)

// Inject into master an authorizer that uses user info.
// TODO(etune): remove this test once a more comprehensive built-in authorizer is implemented.
type sarAuthorizer struct{}

func (sarAuthorizer) Authorize(a authorizer.Attributes) (bool, string, error) {
	if a.GetUser().GetName() == "dave" {
		return false, "no", errors.New("I'm sorry, Dave")
	}

	return true, "you're not dave", nil
}

func alwaysAlice(req *http.Request) (user.Info, bool, error) {
	return &user.DefaultInfo{
		Name: "alice",
	}, true, nil
}

func TestSubjectAccessReview(t *testing.T) {
	// Set up a master
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.Authenticator = authenticator.RequestFunc(alwaysAlice)
	masterConfig.Authorizer = sarAuthorizer{}
	masterConfig.AdmissionControl = admit.NewAlwaysAdmit()
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("error in bringing up the master: %v", err)
	}

	clientset := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

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
