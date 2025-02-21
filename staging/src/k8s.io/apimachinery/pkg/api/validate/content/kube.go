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

package content

import (
	"regexp"
	"strings"
	"unicode"
)

// This regex describes the interior of a qualified name's name part, which is
// slightly different than the rules for the first and last characters. For
// better errors, we handle them seperately.
const qualifiedNameInteriorFmt string = "[-A-Za-z0-9_.]+"
const qualifiedNameMaxLength int = 63
const qualifiedNameErrMsg string = "must consist of a name which starts and ends with alphanumeric characters and consist of alphanumeric characters, '-', '_' or '.', and an optional DNS subdomain prefix and '/' (e.g. \"my.name\", \"MyName\", \"example.com/my-name\")"

var qualifedNameRegexp = regexp.MustCompile("^" + qualifiedNameInteriorFmt + "$")

// IsQualifiedName tests whether the value passed is what Kubernetes calls a
// "qualified name".  This is a format used in various places throughout the
// system.  If the value is not valid, a list of error strings is returned.
// Otherwise an empty list (or nil) is returned.
func IsQualifiedName(value string) []string {
	var errs []string

	parts := strings.Split(value, "/")
	var name string
	switch len(parts) {
	case 1:
		name = parts[0]
	case 2:
		var prefix string
		prefix, name = parts[0], parts[1]
		if len(prefix) == 0 {
			errs = append(errs, "prefix part: "+EmptyError())
		} else if msgs := IsDNS1123Subdomain(prefix); len(msgs) != 0 {
			errs = append(errs, prefixEach(msgs, "prefix part: ")...)
		}
	default:
		return append(errs, qualifiedNameErrMsg)
	}

	errs = append(errs, prefixEach(isQualifiedNameName(name), "name part: ")...)
	return errs
}

// isQualifiedNameName verifies just the name part of a qualified name.
func isQualifiedNameName(name string) []string {
	var errs []string

	if len(name) == 0 {
		errs = append(errs, EmptyError())
	} else if len(name) > qualifiedNameMaxLength {
		errs = append(errs, MaxLenError(qualifiedNameMaxLength))
	} else {
		runes := []rune(name)
		if !isAlNum(runes[0]) || !isAlNum(runes[len(runes)-1]) {
			errs = append(errs, "must start and end with alphanumeric characters")
		}
		if len(runes) > 2 && !qualifedNameRegexp.MatchString(string(runes[1:len(runes)-1])) {
			errs = append(errs, "must contain only alphanumeric characters, '-', '_', or '.'")
		}
	}
	return errs
}

func prefixEach(msgs []string, prefix string) []string {
	for i := range msgs {
		msgs[i] = prefix + msgs[i]
	}
	return msgs
}

func isAlNum(r rune) bool {
	if r > unicode.MaxASCII {
		return false
	}
	if unicode.IsLetter(r) {
		return true
	}
	if unicode.IsDigit(r) {
		return true
	}
	return false
}

// LabelValueMaxLength is a Kubernetes label value's max length.
const LabelValueMaxLength int = qualifiedNameMaxLength

// IsLabelValue tests whether the value passed is a valid label value.  If
// the value is not valid, a list of error strings is returned.  Otherwise an
// empty list (or nil) is returned.
func IsLabelValue(value string) []string {
	var errs []string
	if len(value) > 0 {
		errs = append(errs, isQualifiedNameName(value)...)
	}
	return errs
}
