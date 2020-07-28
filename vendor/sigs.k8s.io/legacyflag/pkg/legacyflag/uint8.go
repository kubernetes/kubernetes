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

// Uint8Value is a reference to a registered uint8 flag value.
type Uint8Value struct {
	name string
	value uint8
	fs *pflag.FlagSet
}

// Uint8Var registers a flag for uint8 against the FlagSet, and returns
// a Uint8Value reference to the registered flag value.
func (fs *FlagSet) Uint8Var(name string, def uint8, usage string) *Uint8Value {
	v := &Uint8Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.Uint8Var(&v.value, name, def, usage)
	return v
}

// Set copies the flag value to the target if the flag was set.
func (v *Uint8Value) Set(target *uint8) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the apply func with the flag value if the flag was set.
func (v *Uint8Value) Apply(apply func(value uint8)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
