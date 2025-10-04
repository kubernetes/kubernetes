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

package diff

import (
	"encoding/json"
	"fmt"

	"github.com/pmezard/go-difflib/difflib"

	"k8s.io/apimachinery/pkg/util/dump"
)

// Diff returns a string representation of the difference between two objects.
// When built without the usegocmp tag, it uses go-difflib/difflib to generate a
// unified diff of the objects. It attempts to use JSON serialization first,
// falling back to an object dump via the dump package if JSON marshaling fails.
func Diff(a, b any) string {

	aStr, aErr := toPrettyJSON(a)
	bStr, bErr := toPrettyJSON(b)
	if aErr != nil || bErr != nil {
		aStr = dump.Pretty(a)
		bStr = dump.Pretty(b)
	}

	diff := difflib.UnifiedDiff{
		A:       difflib.SplitLines(aStr),
		B:       difflib.SplitLines(bStr),
		Context: 3,
	}

	diffstr, err := difflib.GetUnifiedDiffString(diff)
	if err != nil {
		return fmt.Sprintf("error generating diff: %v", err)
	}

	return diffstr
}

// toPrettyJSON converts an object to a pretty-printed JSON string.
func toPrettyJSON(data any) (string, error) {
	jsonData, err := json.MarshalIndent(data, "", " ")
	return string(jsonData), err
}
