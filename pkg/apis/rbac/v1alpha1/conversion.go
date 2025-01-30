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
	rbacv1alpha1 "k8s.io/api/rbac/v1alpha1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime/schema"
	api "k8s.io/kubernetes/pkg/apis/rbac"
)

// allAuthenticated matches k8s.io/apiserver/pkg/authentication/user.AllAuthenticated,
// but we don't want an client library (which must include types), depending on a server library
const allAuthenticated = "system:authenticated"

func Convert_v1alpha1_Subject_To_rbac_Subject(in *rbacv1alpha1.Subject, out *api.Subject, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_Subject_To_rbac_Subject(in, out, s); err != nil {
		return err
	}

	// specifically set the APIGroup for the three subjects recognized in v1alpha1
	switch {
	case in.Kind == rbacv1alpha1.ServiceAccountKind:
		out.APIGroup = ""
	case in.Kind == rbacv1alpha1.UserKind:
		out.APIGroup = GroupName
	case in.Kind == rbacv1alpha1.GroupKind:
		out.APIGroup = GroupName
	default:
		// For unrecognized kinds, use the group portion of the APIVersion if we can get it
		if gv, err := schema.ParseGroupVersion(in.APIVersion); err == nil {
			out.APIGroup = gv.Group
		}
	}

	// User * in v1alpha1 will only match all authenticated users
	// This is only for compatibility with old RBAC bindings
	// Special treatment for * should not be included in v1beta1
	if out.Kind == rbacv1alpha1.UserKind && out.APIGroup == GroupName && out.Name == "*" {
		out.Kind = rbacv1alpha1.GroupKind
		out.Name = allAuthenticated
	}

	return nil
}

func Convert_rbac_Subject_To_v1alpha1_Subject(in *api.Subject, out *rbacv1alpha1.Subject, s conversion.Scope) error {
	if err := autoConvert_rbac_Subject_To_v1alpha1_Subject(in, out, s); err != nil {
		return err
	}

	switch {
	case in.Kind == rbacv1alpha1.ServiceAccountKind && in.APIGroup == "":
		// Make service accounts v1
		out.APIVersion = "v1"
	case in.Kind == rbacv1alpha1.UserKind && in.APIGroup == GroupName:
		// users in the rbac API group get v1alpha
		out.APIVersion = SchemeGroupVersion.String()
	case in.Kind == rbacv1alpha1.GroupKind && in.APIGroup == GroupName:
		// groups in the rbac API group get v1alpha
		out.APIVersion = SchemeGroupVersion.String()
	default:
		// otherwise, they get an unspecified version of a group
		out.APIVersion = schema.GroupVersion{Group: in.APIGroup}.String()
	}

	return nil
}
