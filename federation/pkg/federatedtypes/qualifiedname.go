/*
Copyright 2017 The Kubernetes Authors.

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

package federatedtypes

import (
	"fmt"
)

// QualifiedName comprises a resource name with an optional namespace.
// If namespace is provided, a QualifiedName will be rendered as
// "<namespace>/<name>".  If not, it will be rendered as "name".  This
// is intended to allow the FederatedTypeAdapter interface and its
// consumers to operate on both namespaces and namespace-qualified
// resources.

type QualifiedName struct {
	Namespace string
	Name      string
}

// String returns the general purpose string representation
func (n QualifiedName) String() string {
	if len(n.Namespace) == 0 {
		return n.Name
	}
	return fmt.Sprintf("%s/%s", n.Namespace, n.Name)
}
