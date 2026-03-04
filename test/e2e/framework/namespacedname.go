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

package framework

// NamespacedName comprises a resource name, with a mandatory namespace,
// rendered as "<namespace>/<name>". It implements NamedObject and thus can be
// used as function parameter instead of a full API object.
type NamespacedName struct {
	Namespace string
	Name      string
}

var _ NamedObject = NamespacedName{}

// NamedObject is a subset of metav1.Object which provides read-only access
// to name and namespace of an object.
type NamedObject interface {
	GetNamespace() string
	GetName() string
}

// GetNamespace implements NamedObject.
func (n NamespacedName) GetNamespace() string {
	return n.Namespace
}

// GetName implements NamedObject.
func (n NamespacedName) GetName() string {
	return n.Name
}

// String returns the general purpose string representation
func (n NamespacedName) String() string {
	return n.Namespace + "/" + n.Name
}
