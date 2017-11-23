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

package versioned

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
)

// Attributes is a wrapper around the original admission attributes. It allows
// override the internal objects with the versioned ones.
type Attributes struct {
	admission.Attributes
	OldObject runtime.Object
	Object    runtime.Object
}

// GetObject overrides the original GetObjects() and it returns the versioned
// object.
func (v Attributes) GetObject() runtime.Object {
	return v.Object
}

// GetOldObject overrides the original GetOldObjects() and it returns the
// versioned oldObject.
func (v Attributes) GetOldObject() runtime.Object {
	return v.OldObject
}
