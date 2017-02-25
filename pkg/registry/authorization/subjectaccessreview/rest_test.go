/*
Copyright 2017 The Kubernetes Authors.

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

package subjectaccessreview

import (
	"errors"
	"strings"
	"testing"

	"reflect"

	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

type fakeAuthorizer struct {
	attrs authorizer.Attributes

	ok     bool
	reason string
	err    error
}

func (f *fakeAuthorizer) Authorize(attrs authorizer.Attributes) (bool, string, error) {
	f.attrs = attrs
	return f.ok, f.reason, f.err
}

func TestCreate(t *testing.T) {
	testcases := map[string]struct {
		spec   authorizationapi.SubjectAccessReviewSpec
		ok     bool
		reason string
		err    error

		expectedErr    string
		expectedAttrs  authorizer.Attributes
		expectedStatus authorizationapi.SubjectAccessReviewStatus
	}{
		"empty": {
			expectedErr: "nonResourceAttributes or resourceAttributes",
		},

		"nonresource rejected": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				NonResourceAttributes: &authorizationapi.NonResourceAttributes{Verb: "get", Path: "/mypath"},
			},
			ok:     false,
			reason: "myreason",
			err:    errors.New("myerror"),
			expectedAttrs: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{Name: "bob"},
				Verb:            "get",
				Path:            "/mypath",
				ResourceRequest: false,
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed:         false,
				Reason:          "myreason",
				EvaluationError: "myerror",
			},
		},

		"nonresource allowed": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				NonResourceAttributes: &authorizationapi.NonResourceAttributes{Verb: "get", Path: "/mypath"},
			},
			ok:     true,
			reason: "allowed",
			err:    nil,
			expectedAttrs: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{Name: "bob"},
				Verb:            "get",
				Path:            "/mypath",
				ResourceRequest: false,
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed:         true,
				Reason:          "allowed",
				EvaluationError: "",
			},
		},

		"resource rejected": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				ResourceAttributes: &authorizationapi.ResourceAttributes{
					Namespace:   "myns",
					Verb:        "create",
					Group:       "extensions",
					Version:     "v1beta1",
					Resource:    "deployments",
					Subresource: "scale",
					Name:        "mydeployment",
				},
			},
			ok:     false,
			reason: "myreason",
			err:    errors.New("myerror"),
			expectedAttrs: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{Name: "bob"},
				Namespace:       "myns",
				Verb:            "create",
				APIGroup:        "extensions",
				APIVersion:      "v1beta1",
				Resource:        "deployments",
				Subresource:     "scale",
				Name:            "mydeployment",
				ResourceRequest: true,
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed:         false,
				Reason:          "myreason",
				EvaluationError: "myerror",
			},
		},

		"resource allowed": {
			spec: authorizationapi.SubjectAccessReviewSpec{
				User: "bob",
				ResourceAttributes: &authorizationapi.ResourceAttributes{
					Namespace:   "myns",
					Verb:        "create",
					Group:       "extensions",
					Version:     "v1beta1",
					Resource:    "deployments",
					Subresource: "scale",
					Name:        "mydeployment",
				},
			},
			ok:     true,
			reason: "allowed",
			err:    nil,
			expectedAttrs: authorizer.AttributesRecord{
				User:            &user.DefaultInfo{Name: "bob"},
				Namespace:       "myns",
				Verb:            "create",
				APIGroup:        "extensions",
				APIVersion:      "v1beta1",
				Resource:        "deployments",
				Subresource:     "scale",
				Name:            "mydeployment",
				ResourceRequest: true,
			},
			expectedStatus: authorizationapi.SubjectAccessReviewStatus{
				Allowed:         true,
				Reason:          "allowed",
				EvaluationError: "",
			},
		},
	}

	for k, tc := range testcases {
		auth := &fakeAuthorizer{
			ok:     tc.ok,
			reason: tc.reason,
			err:    tc.err,
		}
		rest := NewREST(auth)

		result, err := rest.Create(genericapirequest.NewContext(), &authorizationapi.SubjectAccessReview{Spec: tc.spec})
		if err != nil {
			if tc.expectedErr != "" {
				if !strings.Contains(err.Error(), tc.expectedErr) {
					t.Errorf("%s: expected %s to contain %q", k, err, tc.expectedErr)
				}
			} else {
				t.Errorf("%s: %v", k, err)
			}
			continue
		}
		if !reflect.DeepEqual(auth.attrs, tc.expectedAttrs) {
			t.Errorf("%s: expected\n%#v\ngot\n%#v", k, tc.expectedAttrs, auth.attrs)
		}
		status := result.(*authorizationapi.SubjectAccessReview).Status
		if !reflect.DeepEqual(status, tc.expectedStatus) {
			t.Errorf("%s: expected\n%#v\ngot\n%#v", k, tc.expectedStatus, status)
		}
	}
}
