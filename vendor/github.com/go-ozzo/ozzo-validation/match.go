// Copyright 2016 Qiang Xue. All rights reserved.
// Use of this source code is governed by a MIT-style
// license that can be found in the LICENSE file.

package validation

import (
	"errors"
	"regexp"
)

// Match returns a validation rule that checks if a value matches the specified regular expression.
// This rule should only be used for validating strings and byte slices, or a validation error will be reported.
// An empty value is considered valid. Use the Required rule to make sure a value is not empty.
func Match(re *regexp.Regexp) *MatchRule {
	return &MatchRule{
		re:      re,
		message: "must be in a valid format",
	}
}

type MatchRule struct {
	re      *regexp.Regexp
	message string
}

// Validate checks if the given value is valid or not.
func (v *MatchRule) Validate(value interface{}) error {
	value, isNil := Indirect(value)
	if isNil {
		return nil
	}

	isString, str, isBytes, bs := StringOrBytes(value)
	if isString && (str == "" || v.re.MatchString(str)) {
		return nil
	} else if isBytes && (len(bs) == 0 || v.re.Match(bs)) {
		return nil
	}
	return errors.New(v.message)
}

// Error sets the error message for the rule.
func (v *MatchRule) Error(message string) *MatchRule {
	v.message = message
	return v
}
