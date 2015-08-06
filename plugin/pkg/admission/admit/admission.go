/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/client"
)

func init() {
	admission.RegisterPlugin("AlwaysAdmit", func(client client.Interface, config io.Reader) (admission.Interface, error) {
		return NewAlwaysAdmit(), nil
	})
}

// alwaysAdmit is an implementation of admission.Interface which always says yes to an admit request.
// It is useful in tests and when using kubernetes in an open manner.
type alwaysAdmit struct{}

func (alwaysAdmit) Admit(a admission.Attributes) (err error) {
	return nil
}

func (alwaysAdmit) Handles(operation admission.Operation) bool {
	return true
}

// NewAlwaysAdmit creates a new always admit admission handler
func NewAlwaysAdmit() admission.Interface {
	return new(alwaysAdmit)
}
