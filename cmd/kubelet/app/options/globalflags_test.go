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
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestSideEffect(t *testing.T) {
	// construct an initial flagset
	fs := pflag.NewFlagSet("", pflag.ExitOnError)
	AddGlobalFlags(fs)

	// change some of the values from their defaults and record the new values
	want := map[string]string{}
	fs.VisitAll(func(f *pflag.Flag) {
		// change values for a few basic types
		switch vt := f.Value.Type(); {
		case vt == "string":
			if f.Value.String() == "foo" {
				f.Value.Set("bar")
			} else {
				f.Value.Set("foo")
			}
		case vt == "bool":
			if f.Value.String() == "true" {
				f.Value.Set("false")
			} else {
				f.Value.Set("true")
			}
		case strings.Contains(vt, "int"):
			if f.Value.String() == "1" {
				f.Value.Set("2")
			} else {
				f.Value.Set("1")
			}
		}
		// record the values as set
		want[f.Name] = f.Value.String()
	})

	// construct another flagset
	fs2 := pflag.NewFlagSet("", pflag.ExitOnError)
	AddGlobalFlags(fs2)

	// check if the values in the original flagset still match what we recorded
	got := map[string]string{}
	fs.VisitAll(func(f *pflag.Flag) {
		got[f.Name] = f.Value.String()
	})

	if !reflect.DeepEqual(want, got) {
		t.Fatalf("global flag registration has side effects. Diff:\n%s",
			diff.ObjectDiff(want, got))
	}
}
