/*
Copyright 2015 The Kubernetes Authors.

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

import (
	"k8s.io/apimachinery/pkg/api/validate/content"
)

// IsValidPathSegmentName validates the name can be safely encoded as a path segment
//
// Deprecated: use content.IsPathSegmentName directly.
var IsValidPathSegmentName = content.IsPathSegmentName

// IsValidPathSegmentPrefix validates the name can be used as a prefix for a name which will be encoded as a path segment
// It does not check for exact matches with disallowed names, since an arbitrary suffix might make the name valid
//
// Deprecated: use content.IsPathSegmentPrefix directly.
var IsValidPathSegmentPrefix = content.IsPathSegmentPrefix

// ValidatePathSegmentName validates the name can be safely encoded as a path segment
//
// Deprecated: use a locally defined function.
func ValidatePathSegmentName(name string, prefix bool) []string {
	if prefix {
		return IsValidPathSegmentPrefix(name)
	}
	return IsValidPathSegmentName(name)
}
