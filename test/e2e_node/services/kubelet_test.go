/*
Copyright 2017 The Kubernetes Authors.

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

package services

import (
	"reflect"
	"testing"

	"github.com/spf13/pflag"
)

func TestSplitKnownArgs(t *testing.T) {
	cases := []struct {
		desc        string
		args        []string
		fs          *pflag.FlagSet
		expectKnown []string
		expectOther []string
	}{
		{
			"splits three args:a",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			func() *pflag.FlagSet {
				fs := pflag.NewFlagSet("", pflag.ContinueOnError)
				var a string
				fs.StringVar(&a, "a", a, "")
				return fs
			}(),
			[]string{"--a", "a"},
			[]string{"--b", "b", "--c", "c"},
		},
		{
			"splits three args:b",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			func() *pflag.FlagSet {
				fs := pflag.NewFlagSet("", pflag.ContinueOnError)
				var a string
				fs.StringVar(&a, "b", a, "")
				return fs
			}(),
			[]string{"--b", "b"},
			[]string{"--a", "a", "--c", "c"},
		},
		{
			"splits three args:c",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			func() *pflag.FlagSet {
				fs := pflag.NewFlagSet("", pflag.ContinueOnError)
				var a string
				fs.StringVar(&a, "c", a, "")
				return fs
			}(),
			[]string{"--c", "c"},
			[]string{"--a", "a", "--b", "b"},
		},
		{
			"splits three args:ab",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			func() *pflag.FlagSet {
				fs := pflag.NewFlagSet("", pflag.ContinueOnError)
				var a string
				fs.StringVar(&a, "a", a, "")
				fs.StringVar(&a, "b", a, "")
				return fs
			}(),
			[]string{"--a", "a", "--b", "b"},
			[]string{"--c", "c"},
		},
		{
			"splits three args:bc",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			func() *pflag.FlagSet {
				fs := pflag.NewFlagSet("", pflag.ContinueOnError)
				var a string
				fs.StringVar(&a, "b", a, "")
				fs.StringVar(&a, "c", a, "")
				return fs
			}(),
			[]string{"--b", "b", "--c", "c"},
			[]string{"--a", "a"},
		},
		{
			"splits three args:ac",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			func() *pflag.FlagSet {
				fs := pflag.NewFlagSet("", pflag.ContinueOnError)
				var a string
				fs.StringVar(&a, "a", a, "")
				fs.StringVar(&a, "c", a, "")
				return fs
			}(),
			[]string{"--a", "a", "--c", "c"},
			[]string{"--b", "b"},
		},
		{
			"splits three args:abc",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			func() *pflag.FlagSet {
				fs := pflag.NewFlagSet("", pflag.ContinueOnError)
				var a string
				fs.StringVar(&a, "a", a, "")
				fs.StringVar(&a, "b", a, "")
				fs.StringVar(&a, "c", a, "")
				return fs
			}(),
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			[]string{},
		},
		{
			"splits three args:none",
			[]string{"--a", "a", "--b", "b", "--c", "c"},
			pflag.NewFlagSet("", pflag.ContinueOnError),
			[]string{},
			[]string{"--a", "a", "--b", "b", "--c", "c"},
		},
	}
	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			origArgs := append([]string(nil), c.args...)
			known, other := splitKnownArgs(c.fs, c.args)
			if !reflect.DeepEqual(c.expectKnown, known) {
				t.Errorf("expect known args to be %v, got %v", c.expectKnown, known)
			}
			if !reflect.DeepEqual(c.expectOther, other) {
				t.Errorf("expect other args to be %v, got %v", c.expectOther, other)
			}
			if !reflect.DeepEqual(origArgs, c.args) {
				t.Errorf("args was mutated")
			}
		})
	}
}
