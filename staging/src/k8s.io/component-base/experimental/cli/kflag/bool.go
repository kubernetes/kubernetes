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

// BoolValue contains the scratch space for a registered bool flag.
// Values can be applied from this scratch space to a target using the Set or Apply methods.
type BoolValue struct {
	name string
	value bool
	fs *pflag.FlagSet
}

// BoolVar registers a flag for type bool against the FlagSet, and returns a struct
// of type BoolValue that contains the scratch space the flag will be parsed into.
func (fs *FlagSet) BoolVar(name string, def bool, usage string) *BoolValue {
	v := &BoolValue{
		name: name,
		fs: fs.fs,
	}
	fs.fs.BoolVar(&v.value, name, def, usage)
	return v
}

// Set copies the bool value to the target if the flag was detected.
func (v *BoolValue) Set(target *bool) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the user-provided apply function with the bool value if the flag was detected.
func (v *BoolValue) Apply(apply func(value bool)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
