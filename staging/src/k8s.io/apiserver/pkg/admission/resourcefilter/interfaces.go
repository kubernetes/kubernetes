/*
Copyright 2024 The Kubernetes Authors.

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

package resourcefilter

import (
	"k8s.io/apiserver/pkg/admission"
)

// Interface is a resource filter that takes an Attributes and
// check if it should be handled or ignored by the admission plugin.
type Interface interface {
	// ShouldHandle returns true if the admission plugin should handle the request,
	// considering the given Attributes, or false otherwise.
	ShouldHandle(admission.Attributes) bool
}
