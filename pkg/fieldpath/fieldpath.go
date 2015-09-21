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

package fieldpath

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/meta"
)

// formatMap formats map[string]string to a string.
func formatMap(m map[string]string) string {
	var l string
	for key, value := range m {
		l += key + "=" + fmt.Sprintf("%q", value) + "\n"
	}
	return l
}

// ExtractFieldPathAsString extracts the field from the given object
// and returns it as a string.  The object must be a pointer to an
// API type.
//
// Currently, this API is limited to supporting the fieldpaths:
//
// 1.  metadata.name - The name of an API object
// 2.  metadata.namespace - The namespace of an API object
func ExtractFieldPathAsString(obj interface{}, fieldPath string) (string, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", nil
	}

	switch fieldPath {
	case "metadata.annotations":
		return formatMap(accessor.Annotations()), nil
	case "metadata.labels":
		return formatMap(accessor.Labels()), nil
	case "metadata.name":
		return accessor.Name(), nil
	case "metadata.namespace":
		return accessor.Namespace(), nil
	}

	return "", fmt.Errorf("Unsupported fieldPath: %v", fieldPath)
}
