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

package legacyflag

import (
	"github.com/spf13/pflag"
)

// VarValue references a registered Var flag.
type VarValue struct {
	name string
	fs   *pflag.FlagSet
}

// Var registers a flag for a type that implements the pflag.Value interface
// against the FlagSet, and returns a VarValue that references this flag.
func (fs *FlagSet) Var(value pflag.Value, name string, usage string) *VarValue {
	v := &VarValue{
		name: name,
		fs:   fs.fs,
	}
	fs.fs.Var(value, name, usage)
	return v
}

// Value returns the value for the registered flag if the flag was set.
func (v *VarValue) Value() interface{} {
    if v.fs.Changed(v.name) {
        flag := v.fs.Lookup(v.name)
        return flag.Value
    }
    return nil
}

// Apply calls the apply function if the flag associated with VarValue was set.
// Since users supply the scratch-space when constructing the VarValue, they
// must read the scratch space directly in their apply function.
func (v *VarValue) Apply(apply func()) {
	if v.fs.Changed(v.name) {
		apply()
	}
}
