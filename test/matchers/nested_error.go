/*
Copyright 2016 The Kubernetes Authors.

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

package matchers

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/util/errors"

	"github.com/onsi/gomega/types"
)

// A stateful matcher that nests other matchers within it and preserves the error types of the
// nested matcher failures.
type NestingMatcher interface {
	types.GomegaMatcher

	// Returns the failures of nested matchers.
	Failures() []error
}

// An error type for labeling errors on deeply nested matchers.
type NestedError struct {
	Path string
	Err  error
}

func (e *NestedError) Error() string {
	// Indent Errors.
	indented := strings.Replace(e.Err.Error(), "\n", "\n\t", -1)
	return fmt.Sprintf("%s:\n\t%v", e.Path, indented)
}

// Create a NestedError with the given path.
// If err is a NestedError, prepend the path to it.
// If err is an AggregateError, recursively Nest each error.
func Nest(path string, err error) error {
	if ag, ok := err.(errors.Aggregate); ok {
		var errs []error
		for _, e := range ag.Errors() {
			errs = append(errs, Nest(path, e))
		}
		return errors.NewAggregate(errs)
	}
	if ne, ok := err.(*NestedError); ok {
		return &NestedError{
			Path: path + ne.Path,
			Err:  ne.Err,
		}
	}
	return &NestedError{
		Path: path,
		Err:  err,
	}
}
