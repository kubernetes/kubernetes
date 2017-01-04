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

package v1beta1_test

import (
	"reflect"
	"testing"

	"k8s.io/apiserver/pkg/authentication/user"
	api "k8s.io/kubernetes/pkg/apis/abac"
	"k8s.io/kubernetes/pkg/apis/abac/v1beta1"
)

func TestV1Beta1Conversion(t *testing.T) {
	testcases := map[string]struct {
		old      *v1beta1.Policy
		expected *api.Policy
	}{
		// specifying a user is preserved
		"user": {
			old:      &v1beta1.Policy{Spec: v1beta1.PolicySpec{User: "bob"}},
			expected: &api.Policy{Spec: api.PolicySpec{User: "bob"}},
		},

		// specifying a group is preserved
		"group": {
			old:      &v1beta1.Policy{Spec: v1beta1.PolicySpec{Group: "mygroup"}},
			expected: &api.Policy{Spec: api.PolicySpec{Group: "mygroup"}},
		},

		// specifying * for user or group maps to all authenticated subjects
		"* user": {
			old:      &v1beta1.Policy{Spec: v1beta1.PolicySpec{User: "*"}},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated}},
		},
		"* group": {
			old:      &v1beta1.Policy{Spec: v1beta1.PolicySpec{Group: "*"}},
			expected: &api.Policy{Spec: api.PolicySpec{Group: user.AllAuthenticated}},
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
