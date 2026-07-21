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

package api

import (
	"strings"

	resourceapi "k8s.io/api/resource/v1"
)

// ResolveQualifiedName adds the default domain to an unqualified name if and only if needed to form a fully qualified name.
func ResolveQualifiedName(name resourceapi.QualifiedName, defaultDomain string) resourceapi.FullyQualifiedName {
	if strings.Contains(string(name), "/") {
		return resourceapi.FullyQualifiedName(name)
	}
	return resourceapi.FullyQualifiedName(defaultDomain + "/" + string(name))
}
