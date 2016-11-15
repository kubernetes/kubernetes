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

package generic

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
)

// ObjectMetaFieldsSet returns a fields that represent the ObjectMeta.
func ObjectMetaFieldsSet(objectMeta *api.ObjectMeta, hasNamespaceField bool) fields.Set {
	if !hasNamespaceField {
		return fields.Set{
			"metadata.name": objectMeta.Name,
		}
	}
	return fields.Set{
		"metadata.name":      objectMeta.Name,
		"metadata.namespace": objectMeta.Namespace,
	}
}

// AdObjectMetaField add fields that represent the ObjectMeta to source.
func AddObjectMetaFieldsSet(source fields.Set, objectMeta *api.ObjectMeta, hasNamespaceField bool) fields.Set {
	source["metadata.name"] = objectMeta.Name
	if hasNamespaceField {
		source["metadata.namespace"] = objectMeta.Namespace
	}
	return source
}

// MergeFieldsSets merges a fields'set from fragment into the source.
func MergeFieldsSets(source fields.Set, fragment fields.Set) fields.Set {
	for k, value := range fragment {
		source[k] = value
	}
	return source
}
