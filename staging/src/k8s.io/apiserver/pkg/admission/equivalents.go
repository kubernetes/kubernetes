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

package admission

// Equivalent is the request's resource or one of its subresources. Any dynamic admission hook
// that applies to it must also apply to the request. The relation is directional and not
// transitive.
type Equivalent struct {
	// Subresource of the request's resource; "" means the resource itself.
	Subresource string
	// Operations on the equivalent. Empty means the request's operation.
	Operations []Operation
}

// EquivalentsGetter is optionally implemented by ObjectInterfaces. Returning no equivalents
// disables admission-equivalent coverage enforcement for the request.
type EquivalentsGetter interface {
	GetAdmissionEquivalents() []Equivalent
}
