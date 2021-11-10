// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"fmt"
	"sort"
)

// Errors type which contains a list of errors observed during parsing.
type Errors struct {
	errors []Error
	source Source
}

// NewErrors creates a new instance of the Errors type.
func NewErrors(source Source) *Errors {
	return &Errors{
		errors: []Error{},
		source: source}
}

// ReportError records an error at a source location.
func (e *Errors) ReportError(l Location, format string, args ...interface{}) {
	err := Error{
		Location: l,
		Message:  fmt.Sprintf(format, args...),
	}
	e.errors = append(e.errors, err)
}

// GetErrors returns the list of observed errors.
func (e *Errors) GetErrors() []Error {
	return e.errors[:]
}

// Append takes an Errors object as input creates a new Errors object with the current and input
// errors.
func (e *Errors) Append(errs []Error) *Errors {
	return &Errors{
		errors: append(e.errors, errs...),
		source: e.source,
	}
}

// ToDisplayString returns the error set to a newline delimited string.
func (e *Errors) ToDisplayString() string {
	var result = ""
	sort.SliceStable(e.errors, func(i, j int) bool {
		ei := e.errors[i].Location
		ej := e.errors[j].Location
		return ei.Line() < ej.Line() ||
			(ei.Line() == ej.Line() && ei.Column() < ej.Column())
	})
	for i, err := range e.errors {
		if i >= 1 {
			result += "\n"
		}
		result += err.ToDisplayString(e.source)
	}
	return result
}
