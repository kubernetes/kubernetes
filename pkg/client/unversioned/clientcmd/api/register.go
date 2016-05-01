/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// Scheme is the default instance of runtime.Scheme to which types in the Kubernetes API are already registered.
var Scheme = runtime.NewScheme()

// SchemeGroupVersion is group version used to register these objects
// TODO this should be in the "kubeconfig" group
var SchemeGroupVersion = unversioned.GroupVersion{Group: "", Version: runtime.APIVersionInternal}

func init() {
	Scheme.AddKnownTypes(SchemeGroupVersion,
		&Config{},
	)
}

func (obj *Config) GetObjectKind() unversioned.ObjectKind { return obj }
func (obj *Config) SetGroupVersionKind(gvk unversioned.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}
func (obj *Config) GroupVersionKind() unversioned.GroupVersionKind {
	return unversioned.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}
