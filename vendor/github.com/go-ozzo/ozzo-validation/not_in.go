// Copyright 2018 Qiang Xue, Google LLC. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package validation

import (
	"errors"
)

// NotIn returns a validation rule that checks if a value os absent from, the given list of values.
// Note that the value being checked and the possible range of values must be of the same type.
// An empty value is considered valid. Use the Required rule to make sure a value is not empty.
func NotIn(values ...interface{}) *NotInRule {
	return &NotInRule{
		elements: values,
		message:  "must not be in list",
	}
}

type NotInRule struct {
	elements []interface{}
	message  string
}

// Validate checks if the given value is valid or not.
func (r *NotInRule) Validate(value interface{}) error {
	value, isNil := Indirect(value)
	if isNil || IsEmpty(value) {
		return nil
	}

	for _, e := range r.elements {
		if e == value {
			return errors.New(r.message)
		}
	}
	return nil
}

// Error sets the error message for the rule.
func (r *NotInRule) Error(message string) *NotInRule {
	r.message = message
	return r
}
