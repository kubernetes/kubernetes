/*
Copyright 2018 The Kubernetes Authors.

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

package v1beta1

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	policyv1beta1 "k8s.io/api/policy/v1beta1"
	"k8s.io/utils/pointer"
)

func TestSetDefaults_PodSecurityPolicySpec(t *testing.T) {
	testcases := []struct {
		Name      string
		In        *policyv1beta1.PodSecurityPolicySpec
		ExpectOut *policyv1beta1.PodSecurityPolicySpec
	}{
		{
			Name: "set default PodSecurityPolicySpec.AllowPrivilegeEscalation as true",
			In:   &policyv1beta1.PodSecurityPolicySpec{},
			ExpectOut: &policyv1beta1.PodSecurityPolicySpec{
				AllowPrivilegeEscalation: pointer.Bool(true),
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			SetDefaults_PodSecurityPolicySpec(tc.In)
			if !reflect.DeepEqual(tc.In, tc.ExpectOut) {
				t.Fatalf("unexpected result:\n %s", cmp.Diff(tc.ExpectOut, tc.In))
			}
		})
	}
}
