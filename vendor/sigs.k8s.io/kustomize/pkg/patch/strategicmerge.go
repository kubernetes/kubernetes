/*
Copyright 2018 The Kubernetes Authors.

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

package patch

// StrategicMerge represents a relative path to a
// stategic merge patch with the format
// https://github.com/kubernetes/community/blob/master/contributors/devel/strategic-merge-patch.md
type StrategicMerge string

// Append appends a slice of patch paths to a StrategicMerge slice
func Append(patches []StrategicMerge, paths ...string) []StrategicMerge {
	for _, p := range paths {
		patches = append(patches, StrategicMerge(p))
	}
	return patches
}

// Exist determines if a patch path exists in a slice of StrategicMerge
func Exist(patches []StrategicMerge, path string) bool {
	for _, p := range patches {
		if p == StrategicMerge(path) {
			return true
		}
	}
	return false
}
