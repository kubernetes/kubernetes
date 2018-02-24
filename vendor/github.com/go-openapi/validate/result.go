// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validate

import (
	"os"

	"github.com/go-openapi/errors"
)

var (
	// Debug is true when the SWAGGER_DEBUG env var is not empty
	Debug = os.Getenv("SWAGGER_DEBUG") != ""
)

type Defaulter interface {
	Apply()
}

type DefaulterFunc func()

func (f DefaulterFunc) Apply() {
	f()
}

// Result represents a validation result
type Result struct {
	Errors     []error
	MatchCount int
	Defaulters []Defaulter
}

// Merge merges this result with the other one, preserving match counts etc
func (r *Result) Merge(other *Result) *Result {
	if other == nil {
		return r
	}
	r.AddErrors(other.Errors...)
	r.MatchCount += other.MatchCount
	r.Defaulters = append(r.Defaulters, other.Defaulters...)
	return r
}

// AddErrors adds errors to this validation result
func (r *Result) AddErrors(errors ...error) {
	// TODO: filter already existing errors
	r.Errors = append(r.Errors, errors...)
}

// IsValid returns true when this result is valid
func (r *Result) IsValid() bool {
	return len(r.Errors) == 0
}

// HasErrors returns true when this result is invalid
func (r *Result) HasErrors() bool {
	return !r.IsValid()
}

// Inc increments the match count
func (r *Result) Inc() {
	r.MatchCount++
}

// AsError renders this result as an error interface
func (r *Result) AsError() error {
	if r.IsValid() {
		return nil
	}
	return errors.CompositeValidationError(r.Errors...)
}

func (r *Result) ApplyDefaults() {
	for _, d := range r.Defaulters {
		d.Apply()
	}
}
