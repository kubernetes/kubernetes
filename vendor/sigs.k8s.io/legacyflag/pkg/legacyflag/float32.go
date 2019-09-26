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

// Float32Value is a reference to a registered float32 flag value.
type Float32Value struct {
	name string
	value float32
	fs *pflag.FlagSet
}

// Float32Var registers a flag for float32 against the FlagSet, and returns
// a Float32Value reference to the registered flag value.
func (fs *FlagSet) Float32Var(name string, def float32, usage string) *Float32Value {
	v := &Float32Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.Float32Var(&v.value, name, def, usage)
	return v
}

// Set copies the flag value to the target if the flag was set.
func (v *Float32Value) Set(target *float32) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the apply func with the flag value if the flag was set.
func (v *Float32Value) Apply(apply func(value float32)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
