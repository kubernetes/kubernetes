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

package v1

import (
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime/schema"
)

// IsAnAPIObject allows clients to preemptively get a reference to an API object and pass it to places that
// intend only to get a reference to that object. This simplifies the event recording interface.
func (obj *ObjectReference) SetGroupVersionKind(gvk schema.GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

func (obj *ObjectReference) GroupVersionKind() schema.GroupVersionKind {
	return schema.FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

func (obj *ObjectReference) GetObjectKind() schema.ObjectKind { return obj }
