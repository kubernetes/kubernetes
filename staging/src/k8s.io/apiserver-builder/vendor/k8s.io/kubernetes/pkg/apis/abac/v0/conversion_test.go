/*
Copyright 2015 The Kubernetes Authors.

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

package v0_test

import (
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	api "k8s.io/kubernetes/pkg/apis/abac"
	"k8s.io/kubernetes/pkg/apis/abac/v0"
)

func TestV0Conversion(t *testing.T) {
	testcases := map[string]struct {
		old      *v0.Policy
		expected *api.Policy
	}{
		// a completely empty policy rule allows everything to all users
		"empty": {
			old:      &v0.Policy{},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated, Readonly: false, NonResourcePath: "*", Namespace: "*", Resource: "*", APIGroup: "*"}},
		},

		// specifying a user is preserved
		"user": {
			old:      &v0.Policy{User: "bob"},
			expected: &api.Policy{Spec: api.PolicySpec{User: "bob", Readonly: false, NonResourcePath: "*", Namespace: "*", Resource: "*", APIGroup: "*"}},
		},

		// specifying a group is preserved (and no longer matches all users)
		"group": {
			old:      &v0.Policy{Group: "mygroup"},
			expected: &api.Policy{Spec: api.PolicySpec{Group: "mygroup", Readonly: false, NonResourcePath: "*", Namespace: "*", Resource: "*", APIGroup: "*"}},
		},

		// specifying * for user or group maps to all authenticated subjects
		"* user": {
			old:      &v0.Policy{User: "*"},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated, Readonly: false, NonResourcePath: "*", Namespace: "*", Resource: "*", APIGroup: "*"}},
		},
		"* group": {
			old:      &v0.Policy{Group: "*"},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated, Readonly: false, NonResourcePath: "*", Namespace: "*", Resource: "*", APIGroup: "*"}},
		},

		// specifying a namespace removes the * match on non-resource path
		"namespace": {
			old:      &v0.Policy{Namespace: "myns"},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated, Readonly: false, NonResourcePath: "", Namespace: "myns", Resource: "*", APIGroup: "*"}},
		},

		// specifying a resource removes the * match on non-resource path
		"resource": {
			old:      &v0.Policy{Resource: "myresource"},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated, Readonly: false, NonResourcePath: "", Namespace: "*", Resource: "myresource", APIGroup: "*"}},
		},

		// specifying a namespace+resource removes the * match on non-resource path
		"namespace+resource": {
			old:      &v0.Policy{Namespace: "myns", Resource: "myresource"},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated, Readonly: false, NonResourcePath: "", Namespace: "myns", Resource: "myresource", APIGroup: "*"}},
		},
	}
	for k, tc := range testcases {
		internal := &api.Policy{}
		if err := api.Scheme.Convert(tc.old, internal, nil); err != nil {
			t.Errorf("%s: unexpected error: %v", k, err)
		}
		if !reflect.DeepEqual(internal, tc.expected) {
			t.Errorf("%s: expected\n\t%#v, got \n\t%#v", k, tc.expected, internal)
		}
	}
}
