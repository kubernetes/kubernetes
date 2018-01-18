/*
Copyright 2014 The Kubernetes Authors.

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

package deny

import (
	"errors"
	"io"

	"k8s.io/apiserver/pkg/admission"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("AlwaysDeny", func(config io.Reader) (admission.Interface, error) {
		return NewAlwaysDeny(), nil
	})
}

// AlwaysDeny is an implementation of admission.Interface which always says no to an admission request.
// It is useful in unit tests to force an operation to be forbidden.
type AlwaysDeny struct{}

var _ admission.MutationInterface = AlwaysDeny{}
var _ admission.ValidationInterface = AlwaysDeny{}

// Admit makes an admission decision based on the request attributes.
func (AlwaysDeny) Admit(a admission.Attributes) (err error) {
	return admission.NewForbidden(a, errors.New("Admission control is denying all modifications"))
}

// Validate makes an admission decision based on the request attributes.  It is NOT allowed to mutate.
func (AlwaysDeny) Validate(a admission.Attributes) (err error) {
	return admission.NewForbidden(a, errors.New("Admission control is denying all modifications"))
}

// Handles returns true if this admission controller can handle the given operation
// where operation can be one of CREATE, UPDATE, DELETE, or CONNECT
func (AlwaysDeny) Handles(operation admission.Operation) bool {
	return true
}

// NewAlwaysDeny creates an always deny admission handler
func NewAlwaysDeny() *AlwaysDeny {
	return new(AlwaysDeny)
}
