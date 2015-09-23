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

package examples

import "k8s.io/kubernetes/pkg/runtime"

// ExampleBuilder fetches examples and templates for Kubernetes
// API object kinds.
type ExampleBuilder interface {
	// NewExample fetches a new copy of an example object of
	// the given version and kind, prepopulated with common values.
	// If no example is found, it should return false for the boolean,
	// but still return an emtpy object of the correct version and kind.
	NewExample(version, kind string) (runtime.Object, bool, error)
}
