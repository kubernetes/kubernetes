/*
Copyright 2014 The Kubernetes Authors.

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

package diff

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"text/tabwriter"

	"github.com/google/go-cmp/cmp" //nolint:depguard
	"k8s.io/apimachinery/pkg/util/dump"
)

func legacyDiff(a, b interface{}) string {
	return cmp.Diff(a, b)
}

// StringDiff diffs a and b and returns a human readable diff.
// DEPRECATED: use github.com/google/go-cmp/cmp.Diff
func StringDiff(a, b string) string {
	return legacyDiff(a, b)
}

// ObjectDiff prints the diff of two go objects and fails if the objects
// contain unhandled unexported fields.
// DEPRECATED: use github.com/google/go-cmp/cmp.Diff
func ObjectDiff(a, b interface{}) string {
	return legacyDiff(a, b)
}

// ObjectGoPrintDiff prints the diff of two go objects and fails if the objects
// contain unhandled unexported fields.
// DEPRECATED: use github.com/google/go-cmp/cmp.Diff
func ObjectGoPrintDiff(a, b interface{}) string {
	return legacyDiff(a, b)
}

// ObjectReflectDiff prints the diff of two go objects and fails if the objects
// contain unhandled unexported fields.
// DEPRECATED: use github.com/google/go-cmp/cmp.Diff
func ObjectReflectDiff(a, b interface{}) string {
	return legacyDiff(a, b)
}

// ObjectGoPrintSideBySide prints a and b as textual dumps side by side,
// enabling easy visual scanning for mismatches.
func ObjectGoPrintSideBySide(a, b interface{}) string {
	sA := dump.Pretty(a)
	sB := dump.Pretty(b)

	linesA := strings.Split(sA, "\n")
	linesB := strings.Split(sB, "\n")
	width := 0
	for _, s := range linesA {
		l := len(s)
		if l > width {
			width = l
		}
	}
	for _, s := range linesB {
		l := len(s)
		if l > width {
			width = l
		}
	}
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, width, 0, 1, ' ', 0)
	max := len(linesA)
	if len(linesB) > max {
		max = len(linesB)
	}
	for i := 0; i < max; i++ {
		var a, b string
		if i < len(linesA) {
			a = linesA[i]
		}
		if i < len(linesB) {
			b = linesB[i]
		}
		fmt.Fprintf(w, "%s\t%s\n", a, b)
	}
	w.Flush()
	return buf.String()
}

// IgnoreUnset is an option that ignores fields that are unset on the right
// hand side of a comparison. This is useful in testing to assert that an
// object is a derivative.
func IgnoreUnset() cmp.Option {
	return cmp.Options{
		// ignore unset fields in v2
		cmp.FilterPath(func(path cmp.Path) bool {
			_, v2 := path.Last().Values()
			switch v2.Kind() {
			case reflect.Slice, reflect.Map:
				if v2.IsNil() || v2.Len() == 0 {
					return true
				}
			case reflect.String:
				if v2.Len() == 0 {
					return true
				}
			case reflect.Interface, reflect.Pointer:
				if v2.IsNil() {
					return true
				}
			}
			return false
		}, cmp.Ignore()),
		// ignore map entries that aren't set in v2
		cmp.FilterPath(func(path cmp.Path) bool {
			switch i := path.Last().(type) {
			case cmp.MapIndex:
				if _, v2 := i.Values(); !v2.IsValid() {
					fmt.Println("E")
					return true
				}
			}
			return false
		}, cmp.Ignore()),
	}
}
