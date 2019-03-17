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

// Int32Value contains the scratch space for a registered int32 flag.
// Values can be applied from this scratch space to a target using the Set or Apply methods.
type Int32Value struct {
	name string
	value int32
	fs *pflag.FlagSet
}

// Int32Var registers a flag for type int32 against the FlagSet, and returns a struct
// of type Int32Value that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) Int32Var(name string, def int32, usage string) *Int32Value {
	v := &Int32Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.Int32Var(&v.value, name, def, usage)
	return v
}

// Set copies the int32 value to the target if the flag was detected.
func (v *Int32Value) Set(target *int32) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the user-provided apply function with the int32 value if the flag was detected.
func (v *Int32Value) Apply(apply func(value int32)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
