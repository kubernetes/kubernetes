/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// AsDefaultVersionedOrOriginal returns the object as a Go object in the external form if possible (matching the
// group version kind of the mapping if provided, a best guess based on serialization if not provided, or obj if it cannot be converted.
// TODO update call sites to specify the scheme they want on their builder.
func AsDefaultVersionedOrOriginal(obj runtime.Object, mapping *meta.RESTMapping) runtime.Object {
	converter := runtime.ObjectConvertor(legacyscheme.Scheme)
	groupVersioner := runtime.GroupVersioner(schema.GroupVersions(legacyscheme.Scheme.PrioritizedVersionsAllGroups()))
	if mapping != nil {
		groupVersioner = mapping.GroupVersionKind.GroupVersion()
	}

	if obj, err := converter.ConvertToVersion(obj, groupVersioner); err == nil {
		return obj
	}
	return obj
}
