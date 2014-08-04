/*
Copyright 2014 Google Inc. All rights reserved.

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

package util

import (
	"regexp"
)

const dnsLabelFmt string = "[a-z0-9]([-a-z0-9]*[a-z0-9])?"

var dnsLabelRegexp = regexp.MustCompile("^" + dnsLabelFmt + "$")

const dnsLabelMaxLength int = 63

// IsDNSLabel tests for a string that conforms to the definition of a label in
// DNS (RFC 1035/1123).
func IsDNSLabel(value string) bool {
	return len(value) <= dnsLabelMaxLength && dnsLabelRegexp.MatchString(value)
}

const dnsSubdomainFmt string = dnsLabelFmt + "(\\." + dnsLabelFmt + ")*"

var dnsSubdomainRegexp = regexp.MustCompile("^" + dnsSubdomainFmt + "$")

const dnsSubdomainMaxLength int = 253

// IsDNSSubdomain tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1035/1123).
func IsDNSSubdomain(value string) bool {
	return len(value) <= dnsSubdomainMaxLength && dnsSubdomainRegexp.MatchString(value)
}

const cIdentifierFmt string = "[A-Za-z_][A-Za-z0-9_]*"

var cIdentifierRegexp = regexp.MustCompile("^" + cIdentifierFmt + "$")

// IsCIdentifier tests for a string that conforms the definition of an identifier
// in C. This checks the format, but not the length.
func IsCIdentifier(value string) bool {
	return cIdentifierRegexp.MatchString(value)
}

// IsValidPortNum tests that the argument is a valid, non-zero port number.
func IsValidPortNum(port int) bool {
	return 0 < port && port < 65536
}

const dns952IdentifierFmt string = "[a-z]([-a-z0-9]*[a-z0-9])?"

var dns952Regexp = regexp.MustCompile("^" + dns952IdentifierFmt + "$")

const dns952MaxLength = 24

func IsDNS952Label(value string) bool {
	return len(value) <= dns952MaxLength && dns952Regexp.MatchString(value)
}
