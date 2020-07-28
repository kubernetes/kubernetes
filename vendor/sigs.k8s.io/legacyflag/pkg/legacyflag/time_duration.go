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
	"time"
)

// DurationValue is a reference to a registered time.Duration flag value.
type DurationValue struct {
	name string
	value time.Duration
	fs *pflag.FlagSet
}

// DurationVar registers a flag for time.Duration against the FlagSet, and returns
// a DurationValue reference to the registered flag value.
func (fs *FlagSet) DurationVar(name string, def time.Duration, usage string) *DurationValue {
	v := &DurationValue{
		name: name,
		fs: fs.fs,
	}
	fs.fs.DurationVar(&v.value, name, def, usage)
	return v
}

// Set copies the flag value to the target if the flag was set.
func (v *DurationValue) Set(target *time.Duration) {
	if v.fs.Changed(v.name) {
		*target = v.value
	}
}

// Apply calls the apply func with the flag value if the flag was set.
func (v *DurationValue) Apply(apply func(value time.Duration)) {
	if v.fs.Changed(v.name) {
		apply(v.value)
	}
}
