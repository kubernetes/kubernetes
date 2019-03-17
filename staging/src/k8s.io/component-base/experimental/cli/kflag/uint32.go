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

// Uint32Value contains the scratch space for a registered uint32 flag.
// Values can be applied from this scratch space to a target using the Set or Apply methods.
type Uint32Value struct {
	name string
	value uint32
	fs *pflag.FlagSet
}

// Uint32Var registers a flag for type uint32 against the FlagSet, and returns a struct
// of type Uint32Value that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) Uint32Var(name string, def uint32, usage string) *Uint32Value {
	v := &Uint32Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.Uint32Var(&v.value, name, def, usage)
	return v
}

// Set copies the uint32 value to the target if the flag was detected.
func (v *Uint32Value) Set(target *uint32) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the user-provided apply function with the uint32 value if the flag was detected.
func (v *Uint32Value) Apply(apply func(value uint32)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
