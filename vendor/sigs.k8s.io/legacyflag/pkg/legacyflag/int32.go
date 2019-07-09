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

// Int32Value is a reference to a registered int32 flag value.
type Int32Value struct {
	name string
	value int32
	fs *pflag.FlagSet
}

// Int32Var registers a flag for int32 against the FlagSet, and returns
// a Int32Value reference to the registered flag value.
func (fs *FlagSet) Int32Var(name string, def int32, usage string) *Int32Value {
	v := &Int32Value{
		name: name,
		fs: fs.fs,
	}
	fs.fs.Int32Var(&v.value, name, def, usage)
	return v
}

// Set copies the flag value to the target if the flag was set.
func (v *Int32Value) Set(target *int32) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the apply func with the flag value if the flag was set.
func (v *Int32Value) Apply(apply func(value int32)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
