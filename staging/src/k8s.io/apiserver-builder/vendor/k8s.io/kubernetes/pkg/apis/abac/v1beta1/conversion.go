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

package v1beta1

import (
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	api "k8s.io/kubernetes/pkg/apis/abac"
)

// allAuthenticated matches k8s.io/apiserver/pkg/authentication/user.AllAuthenticated,
// but we don't want an client library (which must include types), depending on a server library
const allAuthenticated = "system:authenticated"

func addConversionFuncs(scheme *runtime.Scheme) error {
	return scheme.AddConversionFuncs(
		func(in *Policy, out *api.Policy, s conversion.Scope) error {
			// Begin by copying all fields
			if err := autoConvert_v1beta1_Policy_To_abac_Policy(in, out, s); err != nil {
				return err
			}

			// In v1beta1, * user or group maps to all authenticated subjects
			if in.Spec.User == "*" || in.Spec.Group == "*" {
				out.Spec.Group = allAuthenticated
				out.Spec.User = ""
			}

			return nil
		},
	)
}
