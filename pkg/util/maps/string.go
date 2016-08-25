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

package maps

// CopySS makes a shallow copy of a map.
func CopySS(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	copy := make(map[string]string, len(m))
	for k, v := range m {
		copy[k] = v
	}
	return copy
}

// MergeSS returns a new map combining the values of the given maps.
// If there is overlap in the keys, the value from overlay is used.
func MergeSS(base, overlay map[string]string) map[string]string {
	if base == nil {
		return CopySS(overlay)
	}
	merge := CopySS(base)
	for k, v := range overlay {
		merge[k] = v
	}
	return merge
}

// EqualSS tests whether m1 is equal (shallow) to m2. An empty map is considered equivalent to nil.
func EqualSS(m1, m2 map[string]string) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k, v1 := range m1 {
		if v2, ok := m2[k]; !ok || v1 != v2 {
			return false
		}
	}
	return true
}
