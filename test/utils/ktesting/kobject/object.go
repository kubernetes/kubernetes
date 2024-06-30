/*
Copyright 2023 The Kubernetes Authors.

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

package kobject

// Object is a subset of metav1.Object which provides read-only access
// to name and namespace of an object. The namespace is optional.
type Object interface {
	GetNamespace() string
	GetName() string
}

// NamespacedName combines a resource name with an optional namespace,
// rendered as "[<namespace>/]<name>". It Object and thus can be
// used as function parameter instead of a full API object.
type NamespacedName struct {
	Namespace string
	Name      string
}

var _ Object = NamespacedName{}

// GetNamespace implements [Object.GetNamespace].
func (n NamespacedName) GetNamespace() string {
	return n.Namespace
}

// GetName implements [Object.GetName].
func (n NamespacedName) GetName() string {
	return n.Name
}

// String returns the general purpose string representation
func (n NamespacedName) String() string {
	if n.Namespace != "" {
		return n.Namespace + "/" + n.Name
	}
	return n.Name
}
