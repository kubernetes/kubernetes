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

package legacyflag

import(
	"github.com/spf13/pflag"
	
)

// Uint16Value is a reference to a registered uint16 flag value.
type Uint16Value struct {
	name string
	value uint16
	fs *pflag.FlagSet
}

// Uint16Var registers a flag for uint16 against the FlagSet, and returns
// a Uint16Value reference to the registered flag value.
func (fs *FlagSet) Uint16Var(name string, def uint16, usage string) *Uint16Value {
	v := &Uint16Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.Uint16Var(&v.value, name, def, usage)
	return v
}

// Set copies the flag value to the target if the flag was set.
func (v *Uint16Value) Set(target *uint16) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the apply func with the flag value if the flag was set.
func (v *Uint16Value) Apply(apply func(value uint16)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
