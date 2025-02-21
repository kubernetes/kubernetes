/*
Copyright 2014 The Kubernetes Authors.

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

// This regex describes the interior of a label, which is slightly different
// than the rules for the first and last characters. For better errors, we
// handle them seperately.
const dns1123LabelInteriorFmt string = "[-a-z0-9]+"
const dns1123LabelMaxLength int = 63

// DNS1123LabelMaxLength is a DNS label's max length (RFC 1123).
const DNS1123LabelMaxLength int = dns1123LabelMaxLength

var dnsLabelRegexp = regexp.MustCompile("^" + dns1123LabelInteriorFmt + "$")

// IsDNS1123Label returns error messages if the specified value is not
// a valid DNS "label" (approximately RFC 1123).
func IsDNS1123Label(value string) []string {
	if len(value) > dns1123LabelMaxLength {
		// Don't run further validation if we know it is too long.
		return []string{MaxLenError(dns1123LabelMaxLength)}
	}
	return isDNS1123LabelExceptMaxLength(value)
}

func isAlNumLower(r rune) bool {
	if r > unicode.MaxASCII {
		return false
	}
	if unicode.IsLetter(r) && unicode.IsLower(r) {
		return true
	}
	if unicode.IsDigit(r) {
		return true
	}
	return false
}

// Due to historical reasons, we need to be able to validate the format of a
// label without checking the max length.
func isDNS1123LabelExceptMaxLength(value string) []string {
	if len(value) == 0 {
		// No point in going further.
		return []string{EmptyError()}
	}

	var errs []string

	runes := []rune(value)
	if !isAlNumLower(runes[0]) || !isAlNumLower(runes[len(runes)-1]) {
		errs = append(errs, "must start and end with lower-case alphanumeric characters")
	}
	if len(runes) > 2 && !dnsLabelRegexp.MatchString(string(runes[1:len(runes)-1])) {
		errs = append(errs, "must contain only lower-case alphanumeric characters or '-'")
	}
	return errs
}

const dns1123SubdomainMaxLength int = 253

// DNS1123SubdomainMaxLength is a DNS subdomain's max length (RFC 1123).
const DNS1123SubdomainMaxLength int = dns1123SubdomainMaxLength

// IsDNS1123Subdomain returns error messages if the specified value is not
// a valid DNS "subdomain" (approximately RFC 1123).
// subdomain in DNS (RFC 1123).
func IsDNS1123Subdomain(value string) []string {
	if len(value) > DNS1123SubdomainMaxLength {
		// Don't run further validation if we know it is too long.
		return []string{MaxLenError(dns1123SubdomainMaxLength)}
	}
	if len(value) == 0 {
		// No point in going further.
		return []string{EmptyError()}
	}

	var errs []string

	trimmed := strings.Trim(value, ".")
	if trimmed != value {
		errs = append(errs, "must start and end with lower-case alphanumeric characters")
	}

	value = strings.Trim(value, ".")
	if len(value) > 0 {
		parts := strings.Split(value, ".")
		for _, part := range parts {
			if msgs := isDNS1123LabelExceptMaxLength(part); len(msgs) > 0 {
				// We don't need duplicate errors, but each part might have
				// different errors. Making a set for such little data is expensive.
				for _, msg := range msgs {
					msg := "each part " + msg
					if !msgExists(errs, msg) {
						errs = append(errs, msg)
					}
				}
			}
		}
	}
	return errs
}

func msgExists(haystack []string, needle string) bool {
	for _, msg := range haystack {
		if msg == needle {
			return true
		}
	}
	return false
}
