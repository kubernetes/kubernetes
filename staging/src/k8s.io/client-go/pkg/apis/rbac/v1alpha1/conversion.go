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

package v1alpha1

import (
	"k8s.io/apimachinery/pkg/conversion"
	api "k8s.io/client-go/pkg/apis/rbac"
)

// allAuthenticated matches k8s.io/apiserver/pkg/authentication/user.AllAuthenticated,
// but we don't want an client library (which must include types), depending on a server library
const allAuthenticated = "system:authenticated"

func Convert_v1alpha1_Subject_To_rbac_Subject(in *Subject, out *api.Subject, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_Subject_To_rbac_Subject(in, out, s); err != nil {
		return err
	}

	// User * in v1alpha1 will only match all authenticated users
	// This is only for compatibility with old RBAC bindings
	// Special treatment for * should not be included in v1beta1
	if out.Kind == UserKind && out.Name == "*" {
		out.Kind = GroupKind
		out.Name = allAuthenticated
	}

	return nil
}
