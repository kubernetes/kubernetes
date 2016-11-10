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
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"text/tabwriter"

	"github.com/davecgh/go-spew/spew"

	"k8s.io/kubernetes/pkg/util/validation/field"
)

// StringDiff diffs a and b and returns a human readable diff.
func StringDiff(a, b string) string {
	ba := []byte(a)
	bb := []byte(b)
	out := []byte{}
	i := 0
	for ; i < len(ba) && i < len(bb); i++ {
		if ba[i] != bb[i] {
			break
		}
		out = append(out, ba[i])
	}
	out = append(out, []byte("\n\nA: ")...)
	out = append(out, ba[i:]...)
	out = append(out, []byte("\n\nB: ")...)
	out = append(out, bb[i:]...)
	out = append(out, []byte("\n\n")...)
	return string(out)
}

// ObjectDiff writes the two objects out as JSON and prints out the identical part of
// the objects followed by the remaining part of 'a' and finally the remaining part of 'b'.
// For debugging tests.
func ObjectDiff(a, b interface{}) string {
	ab, err := json.Marshal(a)
	if err != nil {
		panic(fmt.Sprintf("a: %v", err))
	}
	bb, err := json.Marshal(b)
	if err != nil {
		panic(fmt.Sprintf("b: %v", err))
	}
	return StringDiff(string(ab), string(bb))
}

// ObjectGoPrintDiff is like ObjectDiff, but uses go-spew to print the objects,
// which shows absolutely everything by recursing into every single pointer
// (go's %#v formatters OTOH stop at a certain point). This is needed when you
// can't figure out why reflect.DeepEqual is returning false and nothing is
// showing you differences. This will.
func ObjectGoPrintDiff(a, b interface{}) string {
	s := spew.ConfigState{DisableMethods: true}
	return StringDiff(
		s.Sprintf("%#v", a),
		s.Sprintf("%#v", b),
	)
}

func ObjectReflectDiff(a, b interface{}) string {
	vA, vB := reflect.ValueOf(a), reflect.ValueOf(b)
	if vA.Type() != vB.Type() {
		return fmt.Sprintf("type A %T and type B %T do not match", a, b)
	}
	diffs := objectReflectDiff(field.NewPath("object"), vA, vB)
	if len(diffs) == 0 {
		return "<no diffs>"
	}
	out := []string{""}
	for _, d := range diffs {
		out = append(out,
			fmt.Sprintf("%s:", d.path),
			limit(fmt.Sprintf("  a: %#v", d.a), 80),
			limit(fmt.Sprintf("  b: %#v", d.b), 80),
		)
	}
	return strings.Join(out, "\n")
}

func limit(s string, max int) string {
	if len(s) > max {
		return s[:max]
	}
	return s
}

func public(s string) bool {
	if len(s) == 0 {
		return false
	}
	return s[:1] == strings.ToUpper(s[:1])
}

type diff struct {
	path *field.Path
	a, b interface{}
}

type orderedDiffs []diff

func (d orderedDiffs) Len() int      { return len(d) }
func (d orderedDiffs) Swap(i, j int) { d[i], d[j] = d[j], d[i] }
func (d orderedDiffs) Less(i, j int) bool {
	a, b := d[i].path.String(), d[j].path.String()
	if a < b {
		return true
	}
	return false
}

func objectReflectDiff(path *field.Path, a, b reflect.Value) []diff {
	switch a.Type().Kind() {
	case reflect.Struct:
		var changes []diff
		for i := 0; i < a.Type().NumField(); i++ {
			if !public(a.Type().Field(i).Name) {
				if reflect.DeepEqual(a.Interface(), b.Interface()) {
					continue
				}
				return []diff{{path: path, a: fmt.Sprintf("%#v", a), b: fmt.Sprintf("%#v", b)}}
			}
			if sub := objectReflectDiff(path.Child(a.Type().Field(i).Name), a.Field(i), b.Field(i)); len(sub) > 0 {
				changes = append(changes, sub...)
			} else {
				if !reflect.DeepEqual(a.Field(i).Interface(), b.Field(i).Interface()) {
					changes = append(changes, diff{path: path, a: a.Field(i).Interface(), b: b.Field(i).Interface()})
				}
			}
		}
		return changes
	case reflect.Ptr, reflect.Interface:
		if a.IsNil() || b.IsNil() {
			switch {
			case a.IsNil() && b.IsNil():
				return nil
			case a.IsNil():
				return []diff{{path: path, a: nil, b: b.Interface()}}
			default:
				return []diff{{path: path, a: a.Interface(), b: nil}}
			}
		}
		return objectReflectDiff(path, a.Elem(), b.Elem())
	case reflect.Chan:
		if !reflect.DeepEqual(a.Interface(), b.Interface()) {
			return []diff{{path: path, a: a.Interface(), b: b.Interface()}}
		}
		return nil
	case reflect.Slice:
		lA, lB := a.Len(), b.Len()
		l := lA
		if lB < lA {
			l = lB
		}
		if lA == lB && lA == 0 {
			if a.IsNil() != b.IsNil() {
				return []diff{{path: path, a: a.Interface(), b: b.Interface()}}
			}
			return nil
		}
		for i := 0; i < l; i++ {
			if !reflect.DeepEqual(a.Index(i), b.Index(i)) {
				return objectReflectDiff(path.Index(i), a.Index(i), b.Index(i))
			}
		}
		var diffs []diff
		for i := l; i < lA; i++ {
			diffs = append(diffs, diff{path: path.Index(i), a: a.Index(i), b: nil})
		}
		for i := l; i < lB; i++ {
			diffs = append(diffs, diff{path: path.Index(i), a: nil, b: b.Index(i)})
		}
		if len(diffs) == 0 {
			diffs = append(diffs, diff{path: path, a: a, b: b})
		}
		return diffs
	case reflect.Map:
		if reflect.DeepEqual(a.Interface(), b.Interface()) {
			return nil
		}
		aKeys := make(map[interface{}]interface{})
		for _, key := range a.MapKeys() {
			aKeys[key.Interface()] = a.MapIndex(key).Interface()
		}
		var missing []diff
		for _, key := range b.MapKeys() {
			if _, ok := aKeys[key.Interface()]; ok {
				delete(aKeys, key.Interface())
				if reflect.DeepEqual(a.MapIndex(key).Interface(), b.MapIndex(key).Interface()) {
					continue
				}
				missing = append(missing, objectReflectDiff(path.Key(fmt.Sprintf("%s", key.Interface())), a.MapIndex(key), b.MapIndex(key))...)
				continue
			}
			missing = append(missing, diff{path: path.Key(fmt.Sprintf("%s", key.Interface())), a: nil, b: b.MapIndex(key).Interface()})
		}
		for key, value := range aKeys {
			missing = append(missing, diff{path: path.Key(fmt.Sprintf("%s", key)), a: value, b: nil})
		}
		if len(missing) == 0 {
			missing = append(missing, diff{path: path, a: a.Interface(), b: b.Interface()})
		}
		sort.Sort(orderedDiffs(missing))
		return missing
	default:
		if reflect.DeepEqual(a.Interface(), b.Interface()) {
			return nil
		}
		if !a.CanInterface() {
			return []diff{{path: path, a: fmt.Sprintf("%#v", a), b: fmt.Sprintf("%#v", b)}}
		}
		return []diff{{path: path, a: a.Interface(), b: b.Interface()}}
	}
}

// ObjectGoPrintSideBySide prints a and b as textual dumps side by side,
// enabling easy visual scanning for mismatches.
func ObjectGoPrintSideBySide(a, b interface{}) string {
	s := spew.ConfigState{
		Indent: " ",
		// Extra deep spew.
		DisableMethods: true,
	}
	sA := s.Sdump(a)
	sB := s.Sdump(b)

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
