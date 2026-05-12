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

package content

import (
	"regexp"
)

const cIdentifierFmt string = "[A-Za-z_][A-Za-z0-9_]*"
const identifierErrMsg string = "a valid C identifier must start with alphabetic character or '_', followed by a string of alphanumeric characters or '_'"

var cIdentifierRegexp = regexp.MustCompile("^" + cIdentifierFmt + "$")

// IsCIdentifier tests for a string that conforms the definition of an identifier
// in C. This checks the format, but not the length.
func IsCIdentifier(value string) []string {
	if !cIdentifierRegexp.MatchString(value) {
		return []string{RegexError(identifierErrMsg, cIdentifierFmt, "my_name", "MY_NAME", "MyName")}
	}
	return nil
}
