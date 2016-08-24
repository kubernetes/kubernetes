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

package v1alpha1

import (
	"k8s.io/kubernetes/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return scheme.AddDefaultingFuncs(
		func(obj *RoleBinding) {
			// we don't have a kind set for the ref, default it based on the ref namespace
			// if you specify a kind, you don't need to specify a namespace when binding to roles
			// in the current namespace
			// TODO, this defaulting should die with v1alpha1.  After this, say what you mean.
			if len(obj.RoleRef.Kind) == 0 && len(obj.RoleRef.APIVersion) == 0 {
				// you need the slash so that when you parse the apiVersion as a GroupVersion,
				// you'll end up with a group and not a version
				obj.RoleRef.APIVersion = GroupName + "/"

				if len(obj.RoleRef.Namespace) == 0 {
					obj.RoleRef.Kind = "ClusterRole"
				} else {
					obj.RoleRef.Kind = "Role"
				}
			}
		},
	)
}
