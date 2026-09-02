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

package coverage

// ResourceVersionMode classifies the ResourceVersion string on a
// GetOptions/ListOptions request.
type ResourceVersionMode string

const (
	RVUnset ResourceVersionMode = "RVUnset" // ResourceVersion == ""
	RVZero  ResourceVersionMode = "RVZero"  // ResourceVersion == "0"
	RVExact ResourceVersionMode = "RVExact" // ResourceVersion == any other value
)

func classifyResourceVersionMode(rvm string) ResourceVersionMode {
	switch rvm {
	case "":
		return RVUnset
	case "0":
		return RVZero
	default:
		return RVExact
	}
}
