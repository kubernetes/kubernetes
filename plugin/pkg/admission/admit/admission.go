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

package admit

import (
	"io"

	"k8s.io/apiserver/pkg/admission"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register("AlwaysAdmit", func(config io.Reader) (admission.Interface, error) {
		return NewAlwaysAdmit(), nil
	})
}

// AlwaysAdmit is an implementation of admission.Interface which always says yes to an admit request.
// It is useful in tests and when using kubernetes in an open manner.
type AlwaysAdmit struct{}

var _ admission.MutationInterface = AlwaysAdmit{}
var _ admission.ValidationInterface = AlwaysAdmit{}

// Admit makes an admission decision based on the request attributes
func (AlwaysAdmit) Admit(a admission.Attributes) (err error) {
	return nil
}

// Validate makes an admission decision based on the request attributes.  It is NOT allowed to mutate.
func (AlwaysAdmit) Validate(a admission.Attributes) (err error) {
	return nil
}

// Handles returns true if this admission controller can handle the given operation
// where operation can be one of CREATE, UPDATE, DELETE, or CONNECT
func (AlwaysAdmit) Handles(operation admission.Operation) bool {
	return true
}

// NewAlwaysAdmit creates a new always admit admission handler
func NewAlwaysAdmit() *AlwaysAdmit {
	return new(AlwaysAdmit)
}
