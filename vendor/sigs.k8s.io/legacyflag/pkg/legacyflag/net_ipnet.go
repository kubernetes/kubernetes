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
	"net"
)

// IPNetValue is a reference to a registered net.IPNet flag value.
type IPNetValue struct {
	name string
	value net.IPNet
	fs *pflag.FlagSet
}

// IPNetVar registers a flag for net.IPNet against the FlagSet, and returns
// a IPNetValue reference to the registered flag value.
func (fs *FlagSet) IPNetVar(name string, def net.IPNet, usage string) *IPNetValue {
	v := &IPNetValue{
		name: name,
		fs: fs.fs,
	}
	fs.fs.IPNetVar(&v.value, name, def, usage)
	return v
}

// Set copies the flag value to the target if the flag was set.
func (v *IPNetValue) Set(target *net.IPNet) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the apply func with the flag value if the flag was set.
func (v *IPNetValue) Apply(apply func(value net.IPNet)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
