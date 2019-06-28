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

// ToCanonicalName converts Golang package/type name into canonical OpenAPI name.
// Examples:
//	Input:  k8s.io/api/core/v1.Pod
//	Output: io.k8s.api.core.v1.Pod
//
//	Input:  k8s.io/api/core/v1
//	Output: io.k8s.api.core.v1
func ToCanonicalName(name string) string {
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

// GetCanonicalTypeName will find the canonical type name of a sample object, removing
// the "vendor" part of the path
func GetCanonicalTypeName(model interface{}) string {
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
