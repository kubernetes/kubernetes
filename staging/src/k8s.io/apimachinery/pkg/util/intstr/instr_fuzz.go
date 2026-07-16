//go:build !notest

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
	"sigs.k8s.io/randfill"
)

// RandFill satisfies randfill.NativeSelfFiller
func (intstr *IntOrString) RandFill(c randfill.Continue) {
	if intstr == nil {
		return
	}
	if c.Bool() {
		intstr.Type = Int
		c.Fill(&intstr.IntVal)
		intstr.StrVal = ""
	} else {
		intstr.Type = String
		intstr.IntVal = 0
		c.Fill(&intstr.StrVal)
	}
}

// ensure IntOrString implements fuzz.Interface
var _ randfill.NativeSelfFiller = &IntOrString{}
