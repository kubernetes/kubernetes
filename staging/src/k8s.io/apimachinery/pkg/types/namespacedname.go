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

package types

// NamespacedName comprises a resource name, with a mandatory namespace,
// rendered as "<namespace>/<name>".  Being a type captures intent and
// helps make sure that UIDs, namespaced names and non-namespaced names
// do not get conflated in code.  For most use cases, namespace and name
// will already have been format validated at the API entry point, so we
// don't do that here.  Where that's not the case (e.g. in testing),
// consider using NamespacedNameOrDie() in testing.go in this package.

type NamespacedName struct {
	Namespace string
	Name      string
}

const (
	Separator = '/'
)

// String returns the general purpose string representation
func (n NamespacedName) String() string {
	return n.Namespace + string(Separator) + n.Name
}

// MarshalLog emits a struct containing required key/value pair
func (n NamespacedName) MarshalLog() interface{} {
	return struct {
		Name, Namespace string
	}{
		Name:      n.Name,
		Namespace: n.Namespace,
	}
}
