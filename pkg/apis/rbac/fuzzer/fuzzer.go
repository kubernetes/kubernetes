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

package fuzzer

import (
	"sigs.k8s.io/randfill"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// Funcs returns the fuzzer functions for the rbac api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(r *rbac.RoleRef, c randfill.Continue) {
			c.FillNoCustom(r) // fuzz self without calling this function again

			// match defaulter
			if len(r.APIGroup) == 0 {
				r.APIGroup = rbac.GroupName
			}
		},
		func(r *rbac.Subject, c randfill.Continue) {
			switch c.Int31n(3) {
			case 0:
				r.Kind = rbac.ServiceAccountKind
				r.APIGroup = ""
				c.FillNoCustom(&r.Name)
				c.FillNoCustom(&r.Namespace)
			case 1:
				r.Kind = rbac.UserKind
				r.APIGroup = rbac.GroupName
				c.FillNoCustom(&r.Name)
				// user "*" won't round trip because we convert it to the system:authenticated group. try again.
				for r.Name == "*" {
					c.FillNoCustom(&r.Name)
				}
			case 2:
				r.Kind = rbac.GroupKind
				r.APIGroup = rbac.GroupName
				c.FillNoCustom(&r.Name)
			}
		},
	}
}
