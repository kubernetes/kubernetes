/*
Copyright The Kubernetes Authors.

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

package parse

import "strings"

// ParseSELinuxLabel parses a SELinux label string into its components.
// Format: "user:role:type:level" -> [user, role, type, level]
// Missing components are represented as empty strings.
func ParseSELinuxLabel(label string) [4]string {
	var parts [4]string
	if label == "" {
		return parts
	}
	split := strings.SplitN(label, ":", 4)
	copy(parts[:], split)
	return parts
}
