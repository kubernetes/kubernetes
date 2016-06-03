/*
Copyright 2014 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

// SchemeGroupVersion is group version used to register these objects
// TODO this should be in the "kubeconfig" group
var SchemeGroupVersion = unversioned.GroupVersion{Group: "", Version: "v1"}

func init() {
	api.Scheme.AddKnownTypes(SchemeGroupVersion,
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
