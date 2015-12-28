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

// SetGroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) SetGroupVersionKind(gvk *GroupVersionKind) {
	obj.APIVersion, obj.Kind = gvk.ToAPIVersionAndKind()
}

// GroupVersionKind satisfies the ObjectKind interface for all objects that embed TypeMeta
func (obj *TypeMeta) GroupVersionKind() *GroupVersionKind {
	return FromAPIVersionAndKind(obj.APIVersion, obj.Kind)
}

func (obj *Status) GetObjectKind() ObjectKind          { return &obj.TypeMeta }
func (obj *APIVersions) GetObjectKind() ObjectKind     { return &obj.TypeMeta }
func (obj *APIGroupList) GetObjectKind() ObjectKind    { return &obj.TypeMeta }
func (obj *APIGroup) GetObjectKind() ObjectKind        { return &obj.TypeMeta }
func (obj *APIResourceList) GetObjectKind() ObjectKind { return &obj.TypeMeta }
func (obj *ExportOptions) GetObjectKind() ObjectKind   { return &obj.TypeMeta }
