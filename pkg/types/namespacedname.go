/*
Copyright 2015 Google Inc. All rights reserved.

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

// NamespacedName comprises a resource name, with a mandatory namespace, rendered as "<namespace>/<name>".  Being a type captures
// intent and helps make sure that UIDs, namespaced names and non-namespaced names do not get conflated in code.
// It also provides a convenient place for utility functions to do things like splitting, joining and rendering.
type NamespacedName struct {
	namespace string
	name      string
}

// NewNamespacedName is the constructor
func NewNamespacedName(namespace, name string) *NamespacedName {
	// TODO: Validate inputs
	return &NamespacedName{namespace, name}
}

// Split returns the namespace and name
func (n *NamespacedName) Split() (namespace, name string) {
	return namespace, name
}

// Name returns the bare name
func (n *NamespacedName) Name() string {
	return n.name
}

// Namespace returns the bare namespace
func (n *NamespacedName) Namespace() string {
	return n.namespace
}

// String returns the general purpose string representation
func (n *NamespacedName) String() string {
	return n.namespace + "/" + n.name
}

// CacheKey returns a key that should be used as the cache key for this NamespacedName
func (n *NamespacedName) CacheKey() *NamespacedName {
	return n
}
