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

package kflag

import (
	"github.com/spf13/pflag"
	utilflag "k8s.io/component-base/cli/flag"
)

// TODO(mtaufen):
// - Figure out how to do codegen.
// - Figure out how to integrate codegen with standard k8s build processes (bazel, scripts, etc).

//go:generate go run gen/gen.go -- kflag.go

// +kflag:bool
// +kflag:[]bool

// +kflag:float32
// +kflag:float64
// +kflag:[]float32
// +kflag:[]float64

// +kflag:int
// +kflag:int8
// +kflag:int16
// +kflag:int32
// +kflag:int64
// +kflag:[]int
// +kflag:[]int8
// +kflag:[]int16
// +kflag:[]int32
// +kflag:[]int64

// +kflag:uint
// +kflag:uint8
// +kflag:uint16
// +kflag:uint32
// +kflag:uint64
// +kflag:[]uint
// +kflag:[]uint8
// +kflag:[]uint16
// +kflag:[]uint32
// +kflag:[]uint64

// +kflag:string
// +kflag:[]string

// +kflag:map[string]string
// +kflag:map[string]bool

// Possible TODO(mtaufen):
// - IP, IPMask, IPNet, etc.
// - Bytes
// - Duration
// - With maps, allow arbitrary specification of separator, e.g. "foo=bar" or "foo<bar" or "foo:bar", whatever you want

// FlagSet tracks the registered flags.
type FlagSet struct {
	fs *pflag.FlagSet
}

// NewFlagSet constructs a new FlagSet. The name argument is the name of the component.
func NewFlagSet(name string) *FlagSet {
	return &FlagSet{
		fs: pflag.NewFlagSet(name, pflag.ContinueOnError),
	}
}

// PflagFlagSet returns the underlying pflag.FlagSet
func (fs *FlagSet) PflagFlagSet() *pflag.FlagSet {
	return fs.fs
}

// Parse the flags
func (fs *FlagSet) Parse(args []string) error {
	return fs.fs.Parse(args)
}

// // Example of a more complicated structure. In this case, we include
// // two helpers, one to override the map completely, and another to
// // merge the map while respecting flag precedence.

// type MapStringBoolValue struct {
// 	name  string
// 	value map[string]bool
// 	fs    *pflag.FlagSet
// }

// // MapStringBoolVar registers an int32 flag against the FlagSet, and returns a MapStringBoolValue that
// // contains the scratch space the flag will be parsed into. This internal value
// // can be applied to a target location with the below helpers.
// func (fs *FlagSet) MapStringBoolVar(name string, def map[string]bool, usage string) *MapStringBoolValue {
// 	val := &MapStringBoolValue{
// 		name:  name,
// 		value: make(map[string]bool),
// 		fs:    fs.fs,
// 	}
// 	for k, v := range def {
// 		val.value[k] = v
// 	}
// 	fs.fs.Var(utilflag.NewMapStringBool(&val.value), name, usage)
// 	return val
// }

// // Set copies the map over the target if the flag was set.
// // It completely overwrites any existing target.
// func (v *MapStringBoolValue) Set(target *map[string]bool) {
// 	if v.fs.Changed(v.name) {
// 		*target = make(map[string]bool)
// 		for k, v := range v.value {
// 			(*target)[k] = v
// 		}
// 	}
// }

// // Merge copies the map keys/values piecewise into the target if the flag was set.
// func (v *MapStringBoolValue) Merge(target *map[string]bool) {
// 	if v.fs.Changed(v.name) {
// 		if *target == nil {
// 			*target = make(map[string]bool)
// 		}
// 		for k, v := range v.value {
// 			(*target)[k] = v
// 		}
// 	}
// }

// // Apply calls the user-provided apply function with the map if the flag was set.
// func (v *MapStringBoolValue) Apply(apply func(value map[string]bool)) {
// 	if v.fs.Changed(v.name) {
// 		apply(v.value)
// 	}
// }

// One way to deal with the generic Var flag. In this case, users provide a pre-defaulted
// scratch space, since we don't know how to construct the underlying type.
// Users must be careful not to mutate the scratch space they pass in.
// The Set helper relies on the generic String() and Set() methods of pflag.Value.
// To avoid too much use of Var, we may also want kflag to implement helpers for
// some of the types currently handled by k8s.io/apimachinery/pkg/util/flag,
// such as MapStringBool (above).

type GenericValue struct {
	name string
	fs   *pflag.FlagSet
}

func (fs *FlagSet) Var(value pflag.Value, name string, usage string) *GenericValue {
	v := &GenericValue{
		name: name,
		fs:   fs.fs,
	}
	fs.fs.Var(value, name, usage)
	return v
}

// Since users pass in the scratch space value, they can close over it with this apply func,
// rather than receiving it as an argument. This can also be more type-safe, since users can
// probably close over the concrete, underlying value of the target, rather than casting from
// pflag.Value to the concrete type. The  added value of GenericValue.Custom is that it only
// calls apply when the flag has been set on the command line.
func (v *GenericValue) Custom(apply func()) {
	if v.fs.Changed(v.name) {
		apply()
	}
}
