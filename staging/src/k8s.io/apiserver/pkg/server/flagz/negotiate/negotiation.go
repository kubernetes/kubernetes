/*
Copyright 2025 The Kubernetes Authors.

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

package negotiate

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// FlagzEndpointRestrictions implements content negotiation restrictions for the z-pages.
// It is used to validate and restrict which GroupVersionKinds are allowed for structured responses.
type FlagzEndpointRestrictions struct{}

// AllowsMediaTypeTransform checks if the provided GVK is supported for structured z-page responses.
func (FlagzEndpointRestrictions) AllowsMediaTypeTransform(mimeType string, mimeSubType string, gvk *schema.GroupVersionKind) bool {
	if mimeType == "text" && mimeSubType == "plain" {
		return gvk == nil
	}
	return isStructured(gvk)
}

func (FlagzEndpointRestrictions) AllowsServerVersion(string) bool {
	return false
}

func (FlagzEndpointRestrictions) AllowsStreamSchema(s string) bool {
	return false
}

func isStructured(gvk *schema.GroupVersionKind) bool {
	if gvk != nil {
		if gvk.Group == "config.k8s.io" && gvk.Version == "v1alpha1" {
			if gvk.Kind == "Flagz" {
				return true
			}
		}
	}

	return false
}
