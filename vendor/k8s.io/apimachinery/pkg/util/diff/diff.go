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

	"k8s.io/apimachinery/pkg/util/validation/field"
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

// ObjectReflectDiff returns a multi-line formatted diff between two objects
// of equal type. If an object with private fields is passed you will
// only see string comparison for those fields. Otherwise this presents the
// most human friendly diff of two structs of equal type in this package.
func ObjectReflectDiff(a, b interface{}) string {
	if a == nil && b == nil {
		return "<no diffs>"
	}
	if a == nil {
		return fmt.Sprintf("a is nil and b is not-nil")
	}
	if b == nil {
		return fmt.Sprintf("a is not-nil and b is nil")
	}
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
		elidedA, elidedB := limit(d.a, d.b, 80)
		out = append(out,
			fmt.Sprintf("%s:", d.path),
			fmt.Sprintf("  a: %s", elidedA),
			fmt.Sprintf("  b: %s", elidedB),
		)
	}
	return strings.Join(out, "\n")
}

// limit:
// 1. stringifies aObj and bObj
// 2. elides identical prefixes if either is too long
// 3. elides remaining content from the end if either is too long
func limit(aObj, bObj interface{}, max int) (string, string) {
	elidedPrefix := ""
	elidedASuffix := ""
	elidedBSuffix := ""
	a, b := fmt.Sprintf("%#v", aObj), fmt.Sprintf("%#v", bObj)

	if aObj != nil && bObj != nil {
		if aType, bType := fmt.Sprintf("%T", aObj), fmt.Sprintf("%T", bObj); aType != bType {
			a = fmt.Sprintf("%s (%s)", a, aType)
			b = fmt.Sprintf("%s (%s)", b, bType)
		}
	}

	for {
		switch {
		case len(a) > max && len(a) > 4 && len(b) > 4 && a[:4] == b[:4]:
			// a is too long, b has data, and the first several characters are the same
			elidedPrefix = "..."
			a = a[2:]
			b = b[2:]

		case len(b) > max && len(b) > 4 && len(a) > 4 && a[:4] == b[:4]:
			// b is too long, a has data, and the first several characters are the same
			elidedPrefix = "..."
			a = a[2:]
			b = b[2:]

		case len(a) > max:
			a = a[:max]
			elidedASuffix = "..."

		case len(b) > max:
			b = b[:max]
			elidedBSuffix = "..."

		default:
			// both are short enough
			return elidedPrefix + a + elidedASuffix, elidedPrefix + b + elidedBSuffix
		}
	}
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
		var diffs []diff
		for i := 0; i < l; i++ {
			if !reflect.DeepEqual(a.Index(i), b.Index(i)) {
				diffs = append(diffs, objectReflectDiff(path.Index(i), a.Index(i), b.Index(i))...)
			}
		}
		for i := l; i < lA; i++ {
			diffs = append(diffs, diff{path: path.Index(i), a: a.Index(i), b: nil})
		}
		for i := l; i < lB; i++ {
			diffs = append(diffs, diff{path: path.Index(i), a: nil, b: b.Index(i)})
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
