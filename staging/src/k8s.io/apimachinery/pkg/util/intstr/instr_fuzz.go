// +build !notest

/*
Copyright 2020 The Kubernetes Authors.

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

package intstr

import (
	fuzz "github.com/google/gofuzz"
)

// Fuzz satisfies fuzz.Interface
func (intstr *IntOrString) Fuzz(c fuzz.Continue) {
	if intstr == nil {
		return
	}
	if c.RandBool() {
		intstr.Type = Int
		c.Fuzz(&intstr.IntVal)
		intstr.StrVal = ""
	} else {
		intstr.Type = String
		intstr.IntVal = 0
		c.Fuzz(&intstr.StrVal)
	}
}

// ensure IntOrString implements fuzz.Interface
var _ fuzz.Interface = &IntOrString{}
