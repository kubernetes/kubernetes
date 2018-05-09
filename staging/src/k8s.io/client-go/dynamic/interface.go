/*
Copyright 2016 The Kubernetes Authors.

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

package dynamic

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// APIPathResolverFunc knows how to convert a groupVersion to its API path. The Kind field is optional.
// TODO find a better place to move this for existing callers
type APIPathResolverFunc func(kind schema.GroupVersionKind) string

// LegacyAPIPathResolverFunc can resolve paths properly with the legacy API.
// TODO find a better place to move this for existing callers
func LegacyAPIPathResolverFunc(kind schema.GroupVersionKind) string {
	if len(kind.Group) == 0 {
		return "/api"
	}
	return "/apis"
}
