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

package certificates

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	certapi "k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/diff"
)

func TestStrategyCreate(t *testing.T) {
	tests := map[string]struct {
		ctx         api.Context
		obj         runtime.Object
		expectedObj runtime.Object
	}{
		"no user in context, no user in obj": {
			ctx: api.NewContext(),
			obj: &certapi.CertificateSigningRequest{},
			expectedObj: &certapi.CertificateSigningRequest{
				Status: certapi.CertificateSigningRequestStatus{Conditions: []certapi.CertificateSigningRequestCondition{}},
			},
		},
		"user in context, no user in obj": {
			ctx: api.WithUser(
				api.NewContext(),
				&user.DefaultInfo{
					Name:   "bob",
					UID:    "123",
					Groups: []string{"group1"},
					Extra:  map[string][]string{"foo": {"bar"}},
				},
			),
			obj: &certapi.CertificateSigningRequest{},
			expectedObj: &certapi.CertificateSigningRequest{
				Spec: certapi.CertificateSigningRequestSpec{
					Username: "bob",
					UID:      "123",
					Groups:   []string{"group1"},
				},
				Status: certapi.CertificateSigningRequestStatus{Conditions: []certapi.CertificateSigningRequestCondition{}},
			},
		},
		"no user in context, user in obj": {
			ctx: api.NewContext(),
			obj: &certapi.CertificateSigningRequest{
				Spec: certapi.CertificateSigningRequestSpec{
					Username: "bob",
					UID:      "123",
					Groups:   []string{"group1"},
				},
			},
			expectedObj: &certapi.CertificateSigningRequest{
				Status: certapi.CertificateSigningRequestStatus{Conditions: []certapi.CertificateSigningRequestCondition{}},
			},
		},
		"user in context, user in obj": {
			ctx: api.WithUser(
				api.NewContext(),
				&user.DefaultInfo{
					Name: "alice",
					UID:  "234",
				},
			),
			obj: &certapi.CertificateSigningRequest{
				Spec: certapi.CertificateSigningRequestSpec{
					Username: "bob",
					UID:      "123",
					Groups:   []string{"group1"},
				},
			},
			expectedObj: &certapi.CertificateSigningRequest{
				Spec: certapi.CertificateSigningRequestSpec{
					Username: "alice",
					UID:      "234",
					Groups:   nil,
				},
				Status: certapi.CertificateSigningRequestStatus{Conditions: []certapi.CertificateSigningRequestCondition{}},
			},
		},
		"pre-approved status": {
			ctx: api.NewContext(),
			obj: &certapi.CertificateSigningRequest{
				Status: certapi.CertificateSigningRequestStatus{
					Conditions: []certapi.CertificateSigningRequestCondition{
						{Type: certapi.CertificateApproved},
					},
				},
			},
			expectedObj: &certapi.CertificateSigningRequest{
				Status: certapi.CertificateSigningRequestStatus{Conditions: []certapi.CertificateSigningRequestCondition{}},
			}},
	}

	for k, tc := range tests {
		obj := tc.obj
		Strategy.PrepareForCreate(tc.ctx, obj)
		if !reflect.DeepEqual(obj, tc.expectedObj) {
			t.Errorf("%s: object diff: %s", k, diff.ObjectDiff(obj, tc.expectedObj))
		}
	}
}
