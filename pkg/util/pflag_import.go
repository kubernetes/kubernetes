/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"flag"
	"reflect"
	"strings"

	"github.com/spf13/pflag"
)

// flagValueWrapper implements pflag.Value around a flag.Value.  The main
// difference here is the addition of the Type method that returns a string
// name of the type.  As this is generally unknown, we approximate that with
// reflection.
type flagValueWrapper struct {
	inner    flag.Value
	flagType string
}

func wrapFlagValue(v flag.Value) pflag.Value {
	// If the flag.Value happens to also be a pflag.Value, just use it directly.
	if pv, ok := v.(pflag.Value); ok {
		return pv
	}

	pv := &flagValueWrapper{
		inner: v,
	}

	t := reflect.TypeOf(v)
	if t.Kind() == reflect.Interface || t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	pv.flagType = t.Name()
	pv.flagType = strings.TrimSuffix(pv.flagType, "Value")
	return pv
}

func (v *flagValueWrapper) String() string {
	return v.inner.String()
}

func (v *flagValueWrapper) Set(s string) error {
	return v.inner.Set(s)
}

func (v *flagValueWrapper) Type() string {
	return v.flagType
}

type boolFlag interface {
	flag.Value
	IsBoolFlag() bool
}

func (v *flagValueWrapper) IsBoolFlag() bool {
	if bv, ok := v.inner.(boolFlag); ok {
		return bv.IsBoolFlag()
	}
	return false
}

// Imports a 'flag.Flag' into a 'pflag.FlagSet'.  The "short" option is unset
// and the type is inferred using reflection.
func addFlagToPFlagSet(f *flag.Flag, fs *pflag.FlagSet) {
	if fs.Lookup(f.Name) == nil {
		fs.Var(wrapFlagValue(f.Value), f.Name, f.Usage)
	}
}

// AddFlagSetToPFlagSet adds all of the flags in a 'flag.FlagSet' package flags to a 'pflag.FlagSet'.
func AddFlagSetToPFlagSet(fsIn *flag.FlagSet, fsOut *pflag.FlagSet) {
	fsIn.VisitAll(func(f *flag.Flag) {
		addFlagToPFlagSet(f, fsOut)
	})
}

// AddAllFlagsToPFlags adds all of the top level 'flag' package flags to the top level 'pflag' flags.
func AddAllFlagsToPFlags() {
	AddFlagSetToPFlagSet(flag.CommandLine, pflag.CommandLine)
}

// AddPFlagSetToPFlagSet merges the flags of fsFrom into fsTo.
func AddPFlagSetToPFlagSet(fsFrom *pflag.FlagSet, fsTo *pflag.FlagSet) {
	if fsFrom != nil && fsTo != nil {
		fsFrom.VisitAll(func(f *pflag.Flag) {
			if fsTo.Lookup(f.Name) == nil {
				fsTo.AddFlag(f)
			}
		})
	}
}
