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

// FlagSet tracks the registered flags.
type FlagSet struct {
	fs *pflag.FlagSet
}

// NewFlagSet constructs a new FlagSet.
func NewFlagSet(name string) *FlagSet {
	return &FlagSet{
		fs: pflag.NewFlagSet(name, pflag.ContinueOnError),
	}
}

// NewFromPFlagSet creates a new FlagSet given a PFlagSet.
func NewFromPFlagSet(fs *pflag.FlagSet) *FlagSet {
	return &FlagSet{fs: fs}
}

// PflagFlagSet returns the underlying pflag.FlagSet.
func (fs *FlagSet) PflagFlagSet() *pflag.FlagSet {
	return fs.fs
}

// Parse parses the flags.
func (fs *FlagSet) Parse(args []string) error {
	return fs.fs.Parse(args)
}

// MarkDeprecated marks a flag as deprecated.
func (fs *FlagSet) MarkDeprecated(name, message string) error {
	return fs.fs.MarkDeprecated(name, message)
}
