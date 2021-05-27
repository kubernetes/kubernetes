/*
Copyright 2021 The Kubernetes Authors.

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

package v1_test

import (
	v1 "k8s.io/api/authentication/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	authentication "k8s.io/kubernetes/pkg/apis/authentication"
	_ "k8s.io/kubernetes/pkg/apis/authentication/install"
	utilpointer "k8s.io/utils/pointer"
	"testing"
)

func TestTokenRequest12Conversion(t *testing.T) {
	testcases := map[string]struct {
		trSpec1 *authentication.TokenRequest
		trSepc2 *v1.TokenRequest
	}{
		"TokenRequest Conversion 1": {
			trSpec1: &authentication.TokenRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: authentication.TokenRequestSpec{
					Audiences:         nil,
					ExpirationSeconds: 10,
					BoundObjectRef:    nil,
				},
			},
			trSepc2: &v1.TokenRequest{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec: v1.TokenRequestSpec{
					Audiences:         nil,
					ExpirationSeconds: utilpointer.Int64Ptr(10),
					BoundObjectRef:    nil,
				},
			},
		},
	}

	for k, tc := range testcases {
		// apps -> appsv1
		internal1 := &v1.TokenRequest{}
		if err := legacyscheme.Scheme.Convert(tc.trSpec1, internal1, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from apps to appsv1", err)
		}

		if !apiequality.Semantic.DeepEqual(internal1, tc.trSepc2) {
			t.Errorf("%q - %q: expected\n\t%#v, got \n\t%#v", k, "from apps to appsv1", tc.trSepc2, internal1)
		}

		// appsv1 -> apps
		internal2 := &authentication.TokenRequest{}
		if err := legacyscheme.Scheme.Convert(tc.trSepc2, internal2, nil); err != nil {
			t.Errorf("%q - %q: unexpected error: %v", k, "from appsv1 to apps", err)
		}
		if !apiequality.Semantic.DeepEqual(internal2, tc.trSpec1) {
			t.Errorf("%q- %q: expected\n\t%#v, got \n\t%#v", k, "from appsv1 to apps", tc.trSpec1, internal2)
		}
	}
}
