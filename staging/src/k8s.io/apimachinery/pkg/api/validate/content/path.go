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

package content

import (
	"fmt"
	"strings"
)

// Strings that cannot be used as names specified as path segments (like the
// REST API or etcd store).
var pathSegmentNameMayNotBe = []string{".", ".."}

// Substrings that cannot be used in names specified as path segments (like the
// REST API or etcd store).
var pathSegmentNameMayNotContain = []string{"/", "%"}

// IsPathSegmentName validates the name can be safely encoded as a path
// segment.
//
// Note that, for historical reason, this function does not check for
// empty strings or impose a limit on the length of the name.
func IsPathSegmentName(name string) []string {
	for _, illegalName := range pathSegmentNameMayNotBe {
		if name == illegalName {
			return []string{fmt.Sprintf(`may not be '%s'`, illegalName)}
		}
	}

	return IsPathSegmentPrefix(name)
}

// IsPathSegmentPrefix validates the name can be used as a prefix for a
// name which will be encoded as a path segment It does not check for exact
// matches with disallowed names, since an arbitrary suffix might make the name
// valid.
//
// Note that, for historical reason, this function does not check for
// empty strings or impose a limit on the length of the name.
func IsPathSegmentPrefix(name string) []string {
	var errors []string
	for _, illegalContent := range pathSegmentNameMayNotContain {
		if strings.Contains(name, illegalContent) {
			errors = append(errors, fmt.Sprintf(`may not contain '%s'`, illegalContent))
		}
	}

	return errors
}
