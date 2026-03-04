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

package architecture

import (
	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

// compareObjects checks that all expected fields are set as expected.
// The actual object may have additional fields, their values are ignored.
func compareObjects(expected, actual *unstructured.Unstructured) string {
	diff := cmp.Diff(expected.Object, actual.Object,
		// Fields which are not in the expected object can be ignored.
		// Only existing fields need to be compared.
		//
		// A maybe (?) simpler approach would be to trim the actual object,
		// then compare with go-cmp. The advantage of telling go-cmp to
		// ignore fields is that they show up as truncated ("...") in the diff,
		// which is a bit more correct.
		cmp.FilterPath(func(path cmp.Path) bool {
			return fieldIsMissing(expected.Object, path)
		}, cmp.Ignore()),
	)
	return diff
}

// fieldIsMissing returns true if the field identified by the path is not
// present in the object. It works by recursively descending along the
// path and checking the corresponding content of the object along the way.
func fieldIsMissing(obj map[string]any, path cmp.Path) bool {
	// First entry is a NOP.
	missing := fieldIsMissingStep(obj, path[1:])
	// Uncomment for debugging...
	// fmt.Printf("fieldIsMissing: %s %v\n", path.GoString(), missing)
	return missing
}

func fieldIsMissingStep(value any, path []cmp.PathStep) bool {
	if len(path) == 0 {
		// Done, full path was checked.
		return false
	}
	// We only need to descend for certain lookup steps,
	// everything else is treated as "not missing" and thus
	// gets compared.
	switch pathElement := path[0].(type) {
	case cmp.MapIndex:
		key := pathElement.Key().String()
		value, ok := value.(map[string]any)
		if !ok {
			// Type mismatch.
			return false
		}
		entry, found := value[key]
		if !found {
			return true
		}
		return fieldIsMissingStep(entry, path[1:])
	case cmp.SliceIndex:
		key := pathElement.Key()
		value, ok := value.([]any)
		if !ok {
			// Type mismatch.
			return false
		}
		if key < 0 {
			// Not sure why go-cmp uses a negative index, so let's compare it.
			return false
		}
		if key >= len(value) {
			// Slice is smaller -> missing entry.
			return true
		}
		entry := value[key]
		return fieldIsMissingStep(entry, path[1:])
	case cmp.TypeAssertion:
		// Actual value type will be checked when needed,
		// skip the assertion here.
		return fieldIsMissingStep(value, path[1:])
	default:
		return false
	}
}
