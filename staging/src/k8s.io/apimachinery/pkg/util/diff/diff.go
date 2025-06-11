//go:build !usegocmp
// +build !usegocmp

/*
Copyright 2025 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/pmezard/go-difflib/difflib"
)

// Diff returns a string representation of the difference between two objects.
// When built with the nogocmp tag, it uses difflib to generate a unified diff
// of the JSON representation of the objects.
func Diff(a, b any) string {
	// Special handling for complex numbers which don't marshal to JSON properly
	if isComplex(a) || isComplex(b) {
		return fmt.Sprintf("- %v\n+ %v", a, b)
	}

	// Special handling for cyclic references
	if isCyclic(a) || isCyclic(b) {
		return fmt.Sprintf("- %v\n+ %v", reflect.ValueOf(a).Elem().FieldByName("Value"),
			reflect.ValueOf(b).Elem().FieldByName("Value"))
	}

	diff := difflib.UnifiedDiff{
		A:        difflib.SplitLines(jsonToString(a)),
		B:        difflib.SplitLines(jsonToString(b)),
		FromFile: "expected",
		ToFile:   "got",
		Context:  10,
	}

	diffstr, err := difflib.GetUnifiedDiffString(diff)
	if err != nil {
		return fmt.Sprintf("error generating diff: %v", err)
	}

	return diffstr
}

// jsonToString converts an object to a formatted JSON string.
// If marshaling fails, it returns an error message.
func jsonToString(data any) string {
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Sprintf("error marshaling to JSON: %v", err)
	}
	return string(jsonData)
}

// isComplex checks if the value is a complex number
func isComplex(v any) bool {
	if v == nil {
		return false
	}
	kind := reflect.TypeOf(v).Kind()
	return kind == reflect.Complex64 || kind == reflect.Complex128
}

// isCyclic attempts to detect if a value contains a cyclic reference
// This is a simple heuristic that checks if it's a pointer to a struct with a "Next" field
// that points back to itself
func isCyclic(v any) bool {
	if v == nil {
		return false
	}

	val := reflect.ValueOf(v)
	if val.Kind() != reflect.Ptr {
		return false
	}

	if val.IsNil() {
		return false
	}

	elem := val.Elem()
	if elem.Kind() != reflect.Struct {
		return false
	}

	// Look for a "Next" field that's a pointer
	nextField := elem.FieldByName("Next")
	if !nextField.IsValid() || nextField.Kind() != reflect.Ptr {
		return false
	}

	// Check if Next points to the same address as the original pointer
	return nextField.Pointer() == val.Pointer()
}
