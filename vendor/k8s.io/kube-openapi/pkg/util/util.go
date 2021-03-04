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

package util

import (
	"reflect"
	"strings"
)

// [DEPRECATED] ToCanonicalName converts Golang package/type canonical name into REST friendly OpenAPI name.
// This method is deprecated because it has a misleading name. Please use ToRESTFriendlyName
// instead
//
// NOTE: actually the "canonical name" in this method should be named "REST friendly OpenAPI name",
// which is different from "canonical name" defined in GetCanonicalTypeName. The "canonical name" defined
// in GetCanonicalTypeName means Go type names with full package path.
//
// Examples of REST friendly OpenAPI name:
//	Input:  k8s.io/api/core/v1.Pod
//	Output: io.k8s.api.core.v1.Pod
//
//	Input:  k8s.io/api/core/v1
//	Output: io.k8s.api.core.v1
//
//	Input:  csi.storage.k8s.io/v1alpha1.CSINodeInfo
//	Output: io.k8s.storage.csi.v1alpha1.CSINodeInfo
func ToCanonicalName(name string) string {
	return ToRESTFriendlyName(name)
}

// ToRESTFriendlyName converts Golang package/type canonical name into REST friendly OpenAPI name.
//
// Examples of REST friendly OpenAPI name:
//	Input:  k8s.io/api/core/v1.Pod
//	Output: io.k8s.api.core.v1.Pod
//
//	Input:  k8s.io/api/core/v1
//	Output: io.k8s.api.core.v1
//
//	Input:  csi.storage.k8s.io/v1alpha1.CSINodeInfo
//	Output: io.k8s.storage.csi.v1alpha1.CSINodeInfo
func ToRESTFriendlyName(name string) string {
	nameParts := strings.Split(name, "/")
	// Reverse first part. e.g., io.k8s... instead of k8s.io...
	if len(nameParts) > 0 && strings.Contains(nameParts[0], ".") {
		parts := strings.Split(nameParts[0], ".")
		for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {
			parts[i], parts[j] = parts[j], parts[i]
		}
		nameParts[0] = strings.Join(parts, ".")
	}
	return strings.Join(nameParts, ".")
}

// OpenAPICanonicalTypeNamer is an interface for models without Go type to seed model name.
//
// OpenAPI canonical names are Go type names with full package path, for uniquely indentifying
// a model / Go type. If a Go type is vendored from another package, only the path after "/vendor/"
// should be used. For custom resource definition (CRD), the canonical name is expected to be
//     group/version.kind
//
// Examples of canonical name:
//     Go type: k8s.io/kubernetes/pkg/apis/core.Pod
//     CRD:     csi.storage.k8s.io/v1alpha1.CSINodeInfo
//
// Example for vendored Go type:
//     Original full path:  k8s.io/kubernetes/vendor/k8s.io/api/core/v1.Pod
//     Canonical name:      k8s.io/api/core/v1.Pod
type OpenAPICanonicalTypeNamer interface {
	OpenAPICanonicalTypeName() string
}

// GetCanonicalTypeName will find the canonical type name of a sample object, removing
// the "vendor" part of the path
func GetCanonicalTypeName(model interface{}) string {
	if namer, ok := model.(OpenAPICanonicalTypeNamer); ok {
		return namer.OpenAPICanonicalTypeName()
	}
	t := reflect.TypeOf(model)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.PkgPath() == "" {
		return t.Name()
	}
	path := t.PkgPath()
	if strings.Contains(path, "/vendor/") {
		path = path[strings.Index(path, "/vendor/")+len("/vendor/"):]
	}
	return path + "." + t.Name()
}
