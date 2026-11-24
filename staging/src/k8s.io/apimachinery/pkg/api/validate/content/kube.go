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
	"strings"
)

const labelKeyCharFmt string = "[A-Za-z0-9]"
const labelKeyExtCharFmt string = "[-A-Za-z0-9_.]"
const labelKeyFmt string = "(" + labelKeyCharFmt + labelKeyExtCharFmt + "*)?" + labelKeyCharFmt
const labelKeyErrMsg string = "must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character"
const labelKeyMaxLength int = 63

// matchesLabelName reports whether value matches ^([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]$.
func matchesLabelName(value string) bool {
	n := len(value)
	if n == 0 {
		return false
	}
	if !isASCIIAlnum(value[0]) || !isASCIIAlnum(value[n-1]) {
		return false
	}
	for i := 1; i < n-1; i++ {
		b := value[i]
		if !isASCIIAlnum(b) && b != '-' && b != '_' && b != '.' {
			return false
		}
	}
	return true
}

// IsQualifiedName tests whether the value passed is what Kubernetes calls a
// "qualified name", which is the same as a label key.
//
// Deprecated: use IsLabelKey instead.
var IsQualifiedName = IsLabelKey

// IsLabelKey tests whether the value passed is a valid label key. This format
// is used to validate many fields in the Kubernetes API.
// Label keys consist of an optional prefix and a name, separated by a '/'.
// If the value is not valid, a list of error strings is returned. Otherwise, an
// empty list (or nil) is returned.
func IsLabelKey(value string) []string {
	prefix, name, hasSlash := strings.Cut(value, "/")
	if !hasSlash {
		if len(value) <= labelKeyMaxLength && matchesLabelName(value) {
			return nil
		}
		var errs []string
		if len(value) == 0 {
			errs = append(errs, "name part "+EmptyError())
		} else if len(value) > labelKeyMaxLength {
			errs = append(errs, "name part "+MaxLenError(labelKeyMaxLength))
		}
		if !matchesLabelName(value) {
			errs = append(errs, "name part "+RegexError(labelKeyErrMsg, labelKeyFmt, "MyName", "my.name", "123-abc"))
		}
		return errs
	}

	if strings.IndexByte(name, '/') != -1 {
		return []string{"a valid label key " + RegexError(labelKeyErrMsg, labelKeyFmt, "MyName", "my.name", "123-abc") +
			" with an optional DNS subdomain prefix and '/' (e.g. 'example.com/MyName')"}
	}

	if len(prefix) > 0 && len(prefix) <= DNS1123SubdomainMaxLength && matchesDNS1123Subdomain(prefix, false) &&
		len(name) <= labelKeyMaxLength && matchesLabelName(name) {
		return nil
	}

	var errs []string
	if len(prefix) == 0 {
		errs = append(errs, "prefix part "+EmptyError())
	} else if msgs := IsDNS1123Subdomain(prefix); len(msgs) != 0 {
		errs = append(errs, prefixEach(msgs, "prefix part ")...)
	}

	if len(name) == 0 {
		errs = append(errs, "name part "+EmptyError())
	} else if len(name) > labelKeyMaxLength {
		errs = append(errs, "name part "+MaxLenError(labelKeyMaxLength))
	}
	if !matchesLabelName(name) {
		errs = append(errs, "name part "+RegexError(labelKeyErrMsg, labelKeyFmt, "MyName", "my.name", "123-abc"))
	}
	return errs
}

const labelValueFmt string = "(" + labelKeyFmt + ")?"
const labelValueErrMsg string = "a valid label must be an empty string or consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character"

// LabelValueMaxLength is a label's max length
const LabelValueMaxLength int = 63

// IsLabelValue tests whether the value passed is a valid label value.  If
// the value is not valid, a list of error strings is returned.  Otherwise an
// empty list (or nil) is returned.
func IsLabelValue(value string) []string {
	if len(value) == 0 || (len(value) <= LabelValueMaxLength && matchesLabelName(value)) {
		return nil
	}
	var errs []string
	if len(value) > LabelValueMaxLength {
		errs = append(errs, MaxLenError(LabelValueMaxLength))
	}
	if len(value) > 0 && !matchesLabelName(value) {
		errs = append(errs, RegexError(labelValueErrMsg, labelValueFmt, "MyValue", "my_value", "12345"))
	}
	return errs
}

func prefixEach(msgs []string, prefix string) []string {
	for i := range msgs {
		msgs[i] = prefix + msgs[i]
	}
	return msgs
}

// IsPrefixedLabelKey tests whether the value passed is a valid label key with
// a domain prefix. This allows "example.com/key" but not "key".
// If the value is not valid, a list of error strings is returned. Otherwise,
// an empty list (or nil) is returned.
func IsPrefixedLabelKey(value string) []string {
	if errs := IsLabelKey(value); len(errs) > 0 {
		return errs
	}

	if strings.IndexByte(value, '/') == -1 {
		return []string{"must include a prefix (e.g. 'example.com/key')"}
	}
	return nil
}
