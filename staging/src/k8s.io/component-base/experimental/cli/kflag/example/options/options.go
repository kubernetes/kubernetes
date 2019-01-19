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

package options

import (
	"k8s.io/component-base/experimental/cli/kflag"
	"k8s.io/component-base/experimental/cli/kflag/example/config"
)

// Flags that aren't part of the config
type Flags struct {
	OtherNum int32
}

func NewFlags() *Flags {
	return &Flags{
		OtherNum: 4,
	}
}

func NewConfig() *config.Config {
	c := &config.Config{}
	config.Default(c)
	return c
}

// We write an aggregate flag registration function for each structure that flags should be applied to.
// Each aggregrate registration function returns a value application function that
// can apply parsed flag values to any structure of the corresponding type.
// Status quo:
// - flags are registered directly against the structures they target
// - registration functions must be called once per flagset per target structure
// - flags must be re-parsed to re-apply
// This solution:
// - flags are registered against on-demand scratch space
// - registration functions only need to be called once per flagset, because target selection is decoupled from registration
// - flags can be re-applied without re-parsing

// AddConfigFlags is the aggregate flag registration function for flags that target a Config struct.
// It returns a function that can apply parsed flag values to any Config struct without re-parsing the flags.
func AddConfigFlags(fs *kflag.FlagSet) (apply func(c *config.Config)) {
	afs := []func(c *config.Config){}

	// Use the values of a defaulted config as the source of flag defaults.
	// Note that passing in defaults is really just to aid help text generation,
	// if the flag is not set on the command line, no action is taken to apply a
	// value to a target.
	def := NewConfig()

	// Register flags
	{
		v := fs.Int32Var("num", def.Num, "an integer value")
		afs = append(afs, func(c *config.Config) { v.Set(&c.Num) })
	}
	{
		v := fs.MapStringBoolVar("map", def.Map, "a map of strings to boolean values")
		afs = append(afs, func(c *config.Config) { v.Merge(&c.Map) })
	}

	return func(c *config.Config) {
		for _, apply := range afs {
			apply(c)
		}
	}
}

// AddFlags is the aggregate flag registration function for flags that target a Flags struct.
// It returns a function that can apply parsed flag values to any Flags struct without re-parsing the flags.
func AddFlags(fs *kflag.FlagSet) (apply func(c *Flags)) {
	afs := []func(c *Flags){}

	// Get the default flag values
	def := NewFlags()

	// Register flags
	{
		v := fs.Int32Var("other-num", def.OtherNum, "another integer value")
		afs = append(afs, func(f *Flags) { v.Set(&f.OtherNum) })
	}

	return func(c *Flags) {
		for _, apply := range afs {
			apply(c)
		}
	}
}
