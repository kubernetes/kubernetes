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

package field

import "k8s.io/utils/field"

// Path represents the path from some root to a particular field.
type Path = field.Path

// NewPath creates a root Path object.
func NewPath(name string, moreNames ...string) *Path {
	return field.NewPath(name, moreNames...)
}
