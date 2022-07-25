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
	"strings"
)

// Errors type which contains a list of errors observed during parsing.
type Errors struct {
	errors            []Error
	source            Source
	numErrors         int
	maxErrorsToReport int
}

// NewErrors creates a new instance of the Errors type.
func NewErrors(source Source) *Errors {
	return &Errors{
		errors:            []Error{},
		source:            source,
		maxErrorsToReport: 100,
	}
}

// ReportError records an error at a source location.
func (e *Errors) ReportError(l Location, format string, args ...interface{}) {
	e.numErrors++
	if e.numErrors > e.maxErrorsToReport {
		return
	}
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

// Append creates a new Errors object with the current and input errors.
func (e *Errors) Append(errs []Error) *Errors {
	return &Errors{
		errors:            append(e.errors, errs...),
		source:            e.source,
		numErrors:         e.numErrors + len(errs),
		maxErrorsToReport: e.maxErrorsToReport,
	}
}

// ToDisplayString returns the error set to a newline delimited string.
func (e *Errors) ToDisplayString() string {
	errorsInString := e.maxErrorsToReport
	if e.numErrors > e.maxErrorsToReport {
		// add one more error to indicate the number of errors truncated.
		errorsInString++
	} else {
		// otherwise the error set will just contain the number of errors.
		errorsInString = e.numErrors
	}

	result := make([]string, errorsInString)
	sort.SliceStable(e.errors, func(i, j int) bool {
		ei := e.errors[i].Location
		ej := e.errors[j].Location
		return ei.Line() < ej.Line() ||
			(ei.Line() == ej.Line() && ei.Column() < ej.Column())
	})
	for i, err := range e.errors {
		// This can happen during the append of two errors objects
		if i >= e.maxErrorsToReport {
			break
		}
		result[i] = err.ToDisplayString(e.source)
	}
	if e.numErrors > e.maxErrorsToReport {
		result[e.maxErrorsToReport] = fmt.Sprintf("%d more errors were truncated", e.numErrors-e.maxErrorsToReport)
	}
	return strings.Join(result, "\n")
}
