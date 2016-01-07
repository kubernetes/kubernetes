/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package api

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

// Group is the API group for abac
const Group = "abac.authorization.kubernetes.io"

// Scheme is the default instance of runtime.Scheme to which types in the abac API group are registered.
var Scheme = runtime.NewScheme()

func init() {
	Scheme.AddInternalGroupVersion(unversioned.GroupVersion{Group: Group, Version: ""})
	Scheme.AddKnownTypes(unversioned.GroupVersion{Group: Group, Version: ""},
		&Policy{},
	)
}

func (obj *Policy) GetObjectKind() unversioned.ObjectKind { return &obj.TypeMeta }
