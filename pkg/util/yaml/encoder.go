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

// Package yaml contains a utility for marshaling an object to yaml, including field comments.
package yaml

import (
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/gengo/parser"
	gengotypes "k8s.io/gengo/types"
)

var durationType = reflect.TypeOf(metav1.Duration{})
var timeType = reflect.TypeOf(metav1.Time{})

type stackItem struct {
	value    reflect.Value
	indent   int
	name     string
	comments string
	inSlice  bool
}

// Marshal writes obj to w as yaml, including field comments.
func Marshal(obj interface{}, w io.Writer) error {
	stack := []stackItem{{value: reflect.ValueOf(obj), indent: 0}}
	var universe gengotypes.Universe
	var err error
	indent := 0

	for {
		if len(stack) == 0 {
			return nil
		}

		// pop off the stack
		current := stack[0]
		currentValue := current.value
		indent = current.indent
		name := current.name
		comments := current.comments
		currentValueType := currentValue.Type()
		stack = stack[1:]

		// only init the universe once
		if universe == nil && currentValueType.PkgPath() != "" {
			b := parser.New()
			if err := b.AddDir(currentValueType.PkgPath()); err != nil {
				return err
			}
			universe, err = b.FindTypes()
			if err != nil {
				return err
			}
		}

		spaces := strings.Repeat(" ", indent)

		// treat metav1.Duration specially
		if currentValueType == durationType {
			// get the underlying time.Duration
			d := currentValue.Interface().(metav1.Duration).Duration
			// reset currentValue to the time.Duration so the switch below can process it correctly
			currentValue = reflect.ValueOf(d)
		}
		// treat metav1.Time specially
		if currentValueType == timeType {
			// get the underlying time.Time
			t := currentValue.Interface().(metav1.Time).Time
			// reset currentValue to the time.Time so the switch below can process it correctly
			currentValue = reflect.ValueOf(t)
		}

		switch currentValue.Kind() {
		case reflect.Ptr:
			value := currentValue.Elem()
			if currentValue.IsNil() {
				// this is so we can print out "null" for the pointer field
				value = reflect.ValueOf("null")
			} else if !value.IsValid() {
				continue
			}
			// if we got a pointer, update the current item's value to be what the pointer points at and
			// reinsert it at the front of the stack.
			current.value = value
			stack = append([]stackItem{current}, stack...)
		case reflect.Interface:
			// if we got an interface, update the current item's value to be what the underlying value and
			// reinsert it at the front of the stack.
			current.value = currentValue.Elem()
			stack = append([]stackItem{current}, stack...)
		case reflect.Struct:
			newIndent := indent
			if current.inSlice {
				// print out the - for the slice element and then make sure all this struct's fields are
				// indented
				fmt.Fprintf(w, "%s-\n", spaces)
				newIndent += 2
			}

			if comments != "" {
				// print out comments for the struct if it has them.
				// we won't have comments for the top-level object, struct elements in a slice, struct
				// values in a map.
				fmt.Fprintf(w, "%s# %s\n", spaces, comments)
			}
			if name != "" {
				// print the struct's name if it has one.
				// we won't have names for the top-level object or for struct elements in a slice
				fmt.Fprintf(w, "%s%s:\n", spaces, name)
				newIndent += 2
			}

			// look up the gengo package & type info so we can get json names & comments for each struct
			// field
			currentValueTypePackage := currentValueType.PkgPath()
			if strings.HasPrefix(currentValueTypePackage, "k8s.io/kubernetes/vendor/") {
				currentValueTypePackage = strings.TrimPrefix(currentValueTypePackage, "k8s.io/kubernetes/vendor/")
			}
			p := universe.Package(currentValueTypePackage)
			t := p.Type(currentValueType.Name())

			// resolve struct fields into stackItems
			var add []stackItem
			for i := 0; i < currentValue.NumField(); i++ {
				// TODO: anonymous structs (like TypeMeta) are currently treated as if they're named, so
				// their members are NOT grouped and sorted with the other fields in the struct to which
				// they belong. This means that e.g. apiVersion and kind are sorted before all the other
				// fields of a struct.
				f := currentValue.Field(i)
				structFieldName := currentValueType.Field(i).Name
				jsonFieldName := ""
				fieldComments := ""

				// iterate through gengo type Members, trying to find the one for the current struct field
				found := false
				validJsonName := false
				for _, m := range t.Members {
					if m.Name == structFieldName {
						jsonFieldName, validJsonName = getJsonName(m.Tags)
						fieldComments = strings.Join(m.CommentLines, " ")
						found = true
						break
					}
				}

				if !validJsonName {
					continue
				}
				if currentValueType.Field(i).Anonymous {
					fieldComments = ""
				}

				if found {
					add = append(add, stackItem{value: f, indent: newIndent, name: jsonFieldName, comments: fieldComments})
				}
			}

			sort.Slice(add, func(i, j int) bool {
				return add[i].name < add[j].name
			})

			// prepend all the fields to the stack
			stack = append(add, stack...)
		case reflect.Slice:
			if indent == -1 {
				indent = 0
			}

			if comments != "" {
				// print out comments for the slice
				fmt.Fprintf(w, "%s# %s\n", spaces, comments)
			}
			if name != "" {
				// print the slice's name
				fmt.Fprintf(w, "%s%s:\n", spaces, name)

			}

			var add []stackItem
			for i := 0; i < currentValue.Len(); i++ {
				add = append(add, stackItem{value: currentValue.Index(i), indent: indent + 2, inSlice: true})
			}

			// prepend all the slice elements to the stack
			stack = append(add, stack...)
		case reflect.Map:
			if comments != "" {
				// print out the map's comments
				fmt.Fprintf(w, "%s# %s\n", spaces, comments)
			}
			if name != "" {
				// print out the map's name
				fmt.Fprintf(w, "%s%s:\n", spaces, name)
			}

			var add []stackItem
			for _, key := range currentValue.MapKeys() {
				value := currentValue.MapIndex(key)
				add = append(add, stackItem{value: value, indent: indent + 2, name: key.Interface().(string)})
			}

			sort.Slice(add, func(i, j int) bool {
				return add[i].name < add[j].name
			})

			// prepend all the map KVs to the stack
			stack = append(add, stack...)
		default:
			// fall-through for values that can be printed directly

			if comments != "" {
				// print out this field's comments
				fmt.Fprintf(w, "%s# %s\n", spaces, comments)
			}

			prefix := ""
			if current.inSlice {
				// if we're a value in a slice, just print out - but don't print any name
				prefix = "- "
			} else if name != "" {
				// otherwise, assuming we have a name, print out "name: "
				prefix = name + ": "
			}

			if currentValueType.Kind() == reflect.String && currentValue.String() == "" {
				// if we have an empty string, we want to print out ""
				currentValue = reflect.ValueOf(`""`)
			}

			// do the actual printing
			fmt.Fprintf(w, "%s%s%v\n", spaces, prefix, currentValue)
		}
	}

	return nil
}

// getJsonName gets the name of the field from the field's struct tags.
func getJsonName(tags string) (string, bool) {
	t := reflect.StructTag(tags)

	jsonTag := t.Get("json")
	if jsonTag == "" {
		return "", false
	}

	parts := strings.Split(jsonTag, ",")

	// json:",inline"
	if parts[0] == "" {
		if len(parts) > 1 && parts[1] == "inline" {
			return "", true
		}
		return "", false
	}
	// json:"-"
	if parts[0] == "-" {
		return "", false
	}

	// json:"foo" or json:"foo,..."
	return parts[0], true
}
