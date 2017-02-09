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

package v0

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
			out.Spec.User = in.User
			out.Spec.Group = in.Group
			out.Spec.Namespace = in.Namespace
			out.Spec.Resource = in.Resource
			out.Spec.Readonly = in.Readonly

			// In v0, unspecified user and group matches all authenticated subjects
			if len(in.User) == 0 && len(in.Group) == 0 {
				out.Spec.Group = allAuthenticated
			}
			// In v0, user or group of * matches all authenticated subjects
			if in.User == "*" || in.Group == "*" {
				out.Spec.Group = allAuthenticated
				out.Spec.User = ""
			}

			// In v0, leaving namespace empty matches all namespaces
			if len(in.Namespace) == 0 {
				out.Spec.Namespace = "*"
			}
			// In v0, leaving resource empty matches all resources
			if len(in.Resource) == 0 {
				out.Spec.Resource = "*"
			}
			// Any rule in v0 should match all API groups
			out.Spec.APIGroup = "*"

			// In v0, leaving namespace and resource blank allows non-resource paths
			if len(in.Namespace) == 0 && len(in.Resource) == 0 {
				out.Spec.NonResourcePath = "*"
			}

			return nil
		},
	)
}
