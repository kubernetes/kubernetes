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

package unversioned

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = GroupVersion{Group: "", Version: ""}

// Kind takes an unqualified kind and returns back a Group qualified GroupKind
func Kind(kind string) GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// GroupVersionKind supports the runtime.Object interface for any object which embeds
// unversioned.TypeMeta
func (obj TypeMeta) GroupVersionKind() *GroupVersionKind {
	return TypeMetaToGroupVersionKind(obj)
}

func (obj *ListOptions) SetGroupVersionKind(gvk *GroupVersionKind) {
	UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *Status) SetGroupVersionKind(gvk *GroupVersionKind) {
	UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *APIVersions) SetGroupVersionKind(gvk *GroupVersionKind) {
	UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *APIGroupList) SetGroupVersionKind(gvk *GroupVersionKind) {
	UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *APIGroup) SetGroupVersionKind(gvk *GroupVersionKind) {
	UpdateTypeMeta(&obj.TypeMeta, gvk)
}
func (obj *APIResourceList) SetGroupVersionKind(gvk *GroupVersionKind) {
	UpdateTypeMeta(&obj.TypeMeta, gvk)
}
