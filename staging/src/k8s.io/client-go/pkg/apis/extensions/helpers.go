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

package extensions

import (
	"strings"
)

// SysctlsFromPodSecurityPolicyAnnotation parses an annotation value of the key
// SysctlsSecurityPolocyAnnotationKey into a slice of sysctls. An empty slice
// is returned if annotation is the empty string.
func SysctlsFromPodSecurityPolicyAnnotation(annotation string) ([]string, error) {
	if len(annotation) == 0 {
		return []string{}, nil
	}

	return strings.Split(annotation, ","), nil
}

// PodAnnotationsFromSysctls creates an annotation value for a slice of Sysctls.
func PodAnnotationsFromSysctls(sysctls []string) string {
	return strings.Join(sysctls, ",")
}
