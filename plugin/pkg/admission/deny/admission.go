/*
Copyright 2014 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"io"
)

func init() {
	admission.RegisterPlugin("AlwaysDeny", func(config io.Reader) (admission.Interface, error) { return NewAlwaysDeny(), nil })
}

// alwaysDeny is an implementation of admission.Interface which always says no to an admission request.
// It is useful in unit tests to force an operation to be forbidden.
type alwaysDeny struct{}

func (alwaysDeny) Admit(a admission.Attributes) (err error) {
	return errors.New("You shall not pass!")
}

func NewAlwaysDeny() admission.Interface {
	return new(alwaysDeny)
}
