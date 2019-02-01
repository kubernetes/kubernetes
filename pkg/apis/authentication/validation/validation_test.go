/*
Copyright 2019 The Kubernetes Authors.

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

// Package validation contains methods to validate kinds in the
// authentication.k8s.io API group.
package validation

import (
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	authentication "k8s.io/kubernetes/pkg/apis/authentication"
)

func TestValidateTokenRequest(t *testing.T) {

	successCase := &authentication.TokenRequest{
		ObjectMeta: metav1.ObjectMeta{Namespace: "a"},
		Spec: authentication.TokenRequestSpec{
			ExpirationSeconds: int64(2 * time.Second),
		},
	}
	if allErrors := ValidateTokenRequest(successCase); len(allErrors) != 0 {
		t.Errorf("expected success: %v", allErrors)
	}
	errorCases := []struct {
		name string
		obj  *authentication.TokenRequest
		msg  string
	}{
		{
			name: "specify larger time ",
			obj: &authentication.TokenRequest{
				ObjectMeta: metav1.ObjectMeta{Namespace: "a"},
				Spec: authentication.TokenRequestSpec{
					ExpirationSeconds: int64(7 * time.Second),
				},
			},
			msg: "may not specify a duration larger than 2^32 seconds",
		},
	}
	for _, c := range errorCases {
		if errs := ValidateTokenRequest(c.obj); len(errs) == 0 {
			t.Errorf("expected failure for %s", c.msg)
		} else if !strings.Contains(errs[0].Error(), c.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], c.msg)
		}
	}

}
