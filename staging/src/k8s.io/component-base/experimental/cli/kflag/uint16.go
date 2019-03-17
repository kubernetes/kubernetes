/*
Copyright 2019 The Kubernetes Authors.

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

// This file is generated. DO NOT EDIT.

package kflag

import(
	"github.com/spf13/pflag"
)

// Uint16Value contains the scratch space for a registered uint16 flag.
// Values can be applied from this scratch space to a target using the Set or Apply methods.
type Uint16Value struct {
	name string
	value uint16
	fs *pflag.FlagSet
}

// Uint16Var registers a flag for type uint16 against the FlagSet, and returns a struct
// of type Uint16Value that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) Uint16Var(name string, def uint16, usage string) *Uint16Value {
	v := &Uint16Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.Uint16Var(&v.value, name, def, usage)
	return v
}

// Set copies the uint16 value to the target if the flag was detected.
func (v *Uint16Value) Set(target *uint16) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the user-provided apply function with the uint16 value if the flag was detected.
func (v *Uint16Value) Apply(apply func(value uint16)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
