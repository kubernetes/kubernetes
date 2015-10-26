/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package validation

import (
	"fmt"
	"strings"
)

// NameMayNotBe specifies strings that cannot be used as names specified as path segments (like the REST API or etcd store)
var NameMayNotBe = []string{".", ".."}

// NameMayNotContain specifies substrings that cannot be used in names specified as path segments (like the REST API or etcd store)
var NameMayNotContain = []string{"/", "%"}

// ValidatePathSegmentName validates the name can be used as a path segment
func ValidatePathSegmentName(name string, prefix bool) (bool, string) {
	// Only check for exact matches if this is the full name (not a prefix)
	if prefix == false {
		for _, illegalName := range NameMayNotBe {
			if name == illegalName {
				return false, fmt.Sprintf(`name may not be %q`, illegalName)
			}
		}
	}

	for _, illegalContent := range NameMayNotContain {
		if strings.Contains(name, illegalContent) {
			return false, fmt.Sprintf(`name may not contain %q`, illegalContent)
		}
	}

	return true, ""
}
