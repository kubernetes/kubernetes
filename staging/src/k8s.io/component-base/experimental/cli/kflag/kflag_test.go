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
	"reflect"
	"testing"

	"k8s.io/component-base/experimental/cli/kflag"
)

func TestInt32VarSet(t *testing.T) {
	cases := []struct {
		name   string
		args   []string
		expect int32
	}{
		{
			name:   "changed",
			args:   []string{"--flag=3"},
			expect: 3,
		},
		{
			name:   "not changed",
			args:   []string{},
			expect: 0,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := kflag.NewFlagSet("")

			var target int32
			v := fs.Int32Var("flag", target, "")

			if err := fs.Parse(c.args); err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			v.Set(&target)

			if target != c.expect {
				t.Errorf("got %v, expect %v", target, c.expect)
			}
		})
	}
}

func TestInt32VarApply(t *testing.T) {
	const def = int32(2)
	cases := []struct {
		name   string
		args   []string
		expect int32
	}{
		{
			name:   "changed",
			args:   []string{"--flag=3"},
			expect: def + 3,
		},
		{
			name:   "not changed",
			args:   []string{},
			expect: def,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := kflag.NewFlagSet("")

			target := def
			v := fs.Int32Var("flag", target, "")

			if err := fs.Parse(c.args); err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			v.Apply(func(value int32) {
				target += value
			})

			if target != c.expect {
				t.Errorf("got %v, expect %v", target, c.expect)
			}
		})
	}
}

func TestMapStringBoolVarSet(t *testing.T) {
	cases := []struct {
		name   string
		args   []string
		expect map[string]bool
	}{
		{
			name: "changed",
			args: []string{"--flag=foo=true,bar=false"},
			expect: map[string]bool{
				"foo": true,
				"bar": false,
			},
		},
		{
			name: "not changed",
			args: []string{},
			expect: map[string]bool{
				"default": true,
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := kflag.NewFlagSet("")

			target := map[string]bool{
				"default": true,
			}
			v := fs.MapStringBoolVar("flag", target, "")

			if err := fs.Parse(c.args); err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			v.Set(&target)

			if !reflect.DeepEqual(target, c.expect) {
				t.Errorf("got %v, expect %v", target, c.expect)
			}
		})
	}
}

func TestMapStringBoolVarMerge(t *testing.T) {
	cases := []struct {
		name   string
		args   []string
		expect map[string]bool
	}{
		{
			name: "changed disjoint",
			args: []string{"--flag=foo=true,bar=false"},
			expect: map[string]bool{
				"foo":     true,
				"bar":     false,
				"default": true,
			},
		},
		{
			name: "changed intersect",
			args: []string{"--flag=foo=true,bar=false,default=false"},
			expect: map[string]bool{
				"foo":     true,
				"bar":     false,
				"default": false,
			},
		},
		{
			name: "not changed",
			args: []string{},
			expect: map[string]bool{
				"default": true,
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := kflag.NewFlagSet("")

			target := map[string]bool{
				"default": true,
			}
			v := fs.MapStringBoolVar("flag", target, "")

			if err := fs.Parse(c.args); err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			v.Merge(&target)

			if !reflect.DeepEqual(target, c.expect) {
				t.Errorf("got %v, expect %v", target, c.expect)
			}
		})
	}
}

func TestMapStringBoolVarApply(t *testing.T) {
	cases := []struct {
		name   string
		args   []string
		expect map[string]bool
	}{
		{
			name: "changed disjoint",
			args: []string{"--flag=foo=true,bar=false"},
			expect: map[string]bool{
				"default": true,
			},
		},
		{
			name: "changed intersect",
			args: []string{"--flag=foo=true,bar=false,default=false"},
			expect: map[string]bool{
				"default": false,
			},
		},
		{
			name: "not changed",
			args: []string{},
			expect: map[string]bool{
				"default": true,
			},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := kflag.NewFlagSet("")

			target := map[string]bool{
				"default": true,
			}
			v := fs.MapStringBoolVar("flag", target, "")

			if err := fs.Parse(c.args); err != nil {
				t.Fatalf("unexpected err: %v", err)
			}

			// Only modify values for keys that already exist
			v.Apply(func(value map[string]bool) {
				for k, v := range value {
					if _, ok := target[k]; ok {
						target[k] = v
					}
				}
			})

			if !reflect.DeepEqual(target, c.expect) {
				t.Errorf("got %v, expect %v", target, c.expect)
			}
		})
	}
}
