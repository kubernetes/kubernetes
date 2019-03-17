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

// UintValue contains the scratch space for a registered uint flag.
// Values can be applied from this scratch space to a target using the Set or Apply methods.
type UintValue struct {
	name string
	value uint
	fs *pflag.FlagSet
}

// UintVar registers a flag for type uint against the FlagSet, and returns a struct
// of type UintValue that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) UintVar(name string, def uint, usage string) *UintValue {
	v := &UintValue{
		name: name,
		fs: fs.fs,
	}
	fs.fs.UintVar(&v.value, name, def, usage)
	return v
}

// Set copies the uint value to the target if the flag was detected.
func (v *UintValue) Set(target *uint) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the user-provided apply function with the uint value if the flag was detected.
func (v *UintValue) Apply(apply func(value uint)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
