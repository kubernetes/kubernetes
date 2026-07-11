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
)

const dns1123LabelFmt string = "[a-z0-9]([-a-z0-9]*[a-z0-9])?"

const dns1123LabelErrMsg string = "a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character"

// DNS1123LabelMaxLength is a label's max length in DNS (RFC 1123)
const DNS1123LabelMaxLength int = 63

var dns1123LabelRegexp = regexp.MustCompile("^" + dns1123LabelFmt + "$")

// IsDNS1123Label tests for a string that conforms to the definition of a label in
// DNS (RFC 1123).
func IsDNS1123Label(value string) []string {
	var errs []string
	if len(value) > DNS1123LabelMaxLength {
		errs = append(errs, MaxLenError(DNS1123LabelMaxLength))
	}
	if !dns1123LabelRegexp.MatchString(value) {
		if dns1123SubdomainRegexp.MatchString(value) {
			// It was a valid subdomain and not a valid label.  Since we
			// already checked length, it must be dots.
			errs = append(errs, "must not contain dots")
		} else {
			errs = append(errs, RegexError(dns1123LabelErrMsg, dns1123LabelFmt, "my-name", "123-abc"))
		}
	}
	return errs
}

const dns1123SubdomainFmt string = dns1123LabelFmt + "(\\." + dns1123LabelFmt + ")*"
const dns1123SubdomainFmtCaseless string = "(?i)" + dns1123SubdomainFmt
const dns1123SubdomainErrorMsg string = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character"
const dns1123SubdomainCaselessErrorMsg string = "an RFC 1123 subdomain must consist of alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character"

// DNS1123SubdomainMaxLength is a subdomain's max length in DNS (RFC 1123)
const DNS1123SubdomainMaxLength int = 253

var dns1123SubdomainRegexp = regexp.MustCompile("^" + dns1123SubdomainFmt + "$")
var dns1123SubdomainCaselessRegexp = regexp.MustCompile("^" + dns1123SubdomainFmtCaseless + "$")

// IsDNS1123Subdomain tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1123) lowercase.
func IsDNS1123Subdomain(value string) []string {
	return isDNS1123Subdomain(value, false)
}

// IsDNS1123SubdomainCaseless tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1123).
//
// Deprecated: API validation should never be caseless. Caseless validation is a vector
// for bugs and failed uniqueness assumptions. For example, names like "foo.com" and
// "FOO.COM" are both accepted as valid, but they are typically not treated as equal by
// consumers (e.g. CSI and DRA driver names). This fails the "least surprise" principle and
// can cause inconsistent behaviors.
//
// Note: This allows uppercase names but is not caseless — uppercase and lowercase are
// treated as different values. Use IsDNS1123Subdomain for strict, lowercase validation
// instead.
func IsDNS1123SubdomainCaseless(value string) []string {
	return isDNS1123Subdomain(value, true)
}

func isDNS1123Subdomain(value string, caseless bool) []string {
	var errs []string
	if len(value) > DNS1123SubdomainMaxLength {
		errs = append(errs, MaxLenError(DNS1123SubdomainMaxLength))
	}
	errorMsg := dns1123SubdomainErrorMsg
	example := "example.com"
	regexp := dns1123SubdomainRegexp
	if caseless {
		errorMsg = dns1123SubdomainCaselessErrorMsg
		example = "Example.com"
		regexp = dns1123SubdomainCaselessRegexp
	}
	if !regexp.MatchString(value) {
		errs = append(errs, RegexError(errorMsg, dns1123SubdomainFmt, example))
	}
	return errs
}
