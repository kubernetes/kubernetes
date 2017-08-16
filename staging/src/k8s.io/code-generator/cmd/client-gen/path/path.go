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

package path

import "strings"

// Vendorless removes the longest match of "*/vendor/" from the front of p.
// It is useful if a package locates in vendor/, e.g.,
// k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/apis/meta/v1, because gengo
// indexes the package with its import path, e.g.,
// k8s.io/apimachinery/pkg/apis/meta/v1,
func Vendorless(p string) string {
	if pos := strings.LastIndex(p, "/vendor/"); pos != -1 {
		return p[pos+len("/vendor/"):]
	}
	return p
}
