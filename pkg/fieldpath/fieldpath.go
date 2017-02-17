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

package fieldpath

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
)

// FormatMap formats map[string]string to a string.
func FormatMap(m map[string]string) (fmtStr string) {
	for key, value := range m {
		fmtStr += fmt.Sprintf("%v=%q\n", key, value)
	}
	fmtStr = strings.TrimSuffix(fmtStr, "\n")

	return
}

// ExtractFieldPathAsString extracts the field from the given object
// and returns it as a string.  The object must be a pointer to an
// API type.
func ExtractFieldPathAsString(obj interface{}, fieldPath string) (string, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", nil
	}

	path, subscript := SplitMaybeSubscriptedPath(fieldPath)

	if len(subscript) > 0 {
		switch path {
		case "metadata.annotations":
			return accessor.GetAnnotations()[subscript], nil
		case "metadata.labels":
			return accessor.GetLabels()[subscript], nil
		default:
			return "", fmt.Errorf("fieldPath %q does not support subscript", fieldPath)
		}
	}

	switch path {
	case "metadata.annotations":
		return FormatMap(accessor.GetAnnotations()), nil
	case "metadata.labels":
		return FormatMap(accessor.GetLabels()), nil
	case "metadata.name":
		return accessor.GetName(), nil
	case "metadata.namespace":
		return accessor.GetNamespace(), nil
	case "metadata.uid":
		return string(accessor.GetUID()), nil
	}

	return "", fmt.Errorf("unsupported fieldPath: %v", fieldPath)
}

// SplitMaybeSubscriptedPath checks whether the specified fieldPath is
// subscripted, and
//  - if yes, this function splits the fieldPath into path and subscript, and
//    returns (path, subscript).
//  - if no, this function returns (fieldPath, "").
//
// Example inputs and outputs:
//  - "metadata.annotations['myKey']" --> ("metadata.annotations", "myKey")
//  - "metadata.annotations['a[b]c']" --> ("metadata.annotations", "a[b]c")
//  - "metadata.labels"               --> ("metadata.labels", "")
//  - "metadata.labels['']"           --> ("metadata.labels", "")
func SplitMaybeSubscriptedPath(fieldPath string) (string, string) {
	if !strings.HasSuffix(fieldPath, "']") {
		return fieldPath, ""
	}
	s := strings.TrimSuffix(fieldPath, "']")
	parts := strings.SplitN(s, "['", 2)
	if len(parts) < 2 {
		return fieldPath, ""
	}
	if len(parts[0]) == 0 || len(parts[1]) == 0 {
		return fieldPath, ""
	}
	return parts[0], parts[1]
}
