/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"net"
	"regexp"
	"strings"
)

const qnameCharFmt string = "[A-Za-z0-9]"
const qnameExtCharFmt string = "[-A-Za-z0-9_.]"
const QualifiedNameFmt string = "(" + qnameCharFmt + qnameExtCharFmt + "*)?" + qnameCharFmt
const QualifiedNameMaxLength int = 63

var qualifiedNameRegexp = regexp.MustCompile("^" + QualifiedNameFmt + "$")

func IsQualifiedName(value string) bool {
	parts := strings.Split(value, "/")
	var name string
	switch len(parts) {
	case 1:
		name = parts[0]
	case 2:
		var prefix string
		prefix, name = parts[0], parts[1]
		if prefix == "" || !IsDNS1123Subdomain(prefix) {
			return false
		}
	default:
		return false
	}

	return name != "" && len(name) <= QualifiedNameMaxLength && qualifiedNameRegexp.MatchString(name)
}

const LabelValueFmt string = "(" + QualifiedNameFmt + ")?"
const LabelValueMaxLength int = 63

var labelValueRegexp = regexp.MustCompile("^" + LabelValueFmt + "$")

func IsValidLabelValue(value string) bool {
	return (len(value) <= LabelValueMaxLength && labelValueRegexp.MatchString(value))
}

const DNS1123LabelFmt string = "[a-z0-9]([-a-z0-9]*[a-z0-9])?"
const DNS1123LabelMaxLength int = 63

var dns1123LabelRegexp = regexp.MustCompile("^" + DNS1123LabelFmt + "$")

// IsDNS1123Label tests for a string that conforms to the definition of a label in
// DNS (RFC 1123).
func IsDNS1123Label(value string) bool {
	return len(value) <= DNS1123LabelMaxLength && dns1123LabelRegexp.MatchString(value)
}

const DNS1123SubdomainFmt string = DNS1123LabelFmt + "(\\." + DNS1123LabelFmt + ")*"
const DNS1123SubdomainMaxLength int = 253

var dns1123SubdomainRegexp = regexp.MustCompile("^" + DNS1123SubdomainFmt + "$")

// IsDNS1123Subdomain tests for a string that conforms to the definition of a
// subdomain in DNS (RFC 1123).
func IsDNS1123Subdomain(value string) bool {
	return len(value) <= DNS1123SubdomainMaxLength && dns1123SubdomainRegexp.MatchString(value)
}

const DNS952LabelFmt string = "[a-z]([-a-z0-9]*[a-z0-9])?"
const DNS952LabelMaxLength int = 24

var dns952LabelRegexp = regexp.MustCompile("^" + DNS952LabelFmt + "$")

// IsDNS952Label tests for a string that conforms to the definition of a label in
// DNS (RFC 952).
func IsDNS952Label(value string) bool {
	return len(value) <= DNS952LabelMaxLength && dns952LabelRegexp.MatchString(value)
}

const CIdentifierFmt string = "[A-Za-z_][A-Za-z0-9_]*"

var cIdentifierRegexp = regexp.MustCompile("^" + CIdentifierFmt + "$")

// IsCIdentifier tests for a string that conforms the definition of an identifier
// in C. This checks the format, but not the length.
func IsCIdentifier(value string) bool {
	return cIdentifierRegexp.MatchString(value)
}

// IsValidPortNum tests that the argument is a valid, non-zero port number.
func IsValidPortNum(port int) bool {
	return 0 < port && port < 65536
}

const doubleHyphensFmt string = ".*(--).*"

var doubleHyphensRegexp = regexp.MustCompile("^" + doubleHyphensFmt + "$")

const IdentifierNoHyphensBeginEndFmt string = "[a-z0-9]([a-z0-9-]*[a-z0-9])*"

var identifierNoHyphensBeginEndRegexp = regexp.MustCompile("^" + IdentifierNoHyphensBeginEndFmt + "$")

const atLeastOneLetterFmt string = ".*[a-z].*"

var atLeastOneLetterRegexp = regexp.MustCompile("^" + atLeastOneLetterFmt + "$")

// IsValidPortName check that the argument is valid syntax. It must be non empty and no more than 15 characters long
// It must contains at least one letter [a-z] and it must contains only [a-z0-9-].
// Hypens ('-') cannot be leading or trailing character of the string and cannot be adjacent to other hyphens.
// Although RFC 6335 allows upper and lower case characters but case is ignored for comparison purposes: (HTTP
// and http denote the same service).
func IsValidPortName(port string) bool {
	if len(port) < 1 || len(port) > 15 {
		return false
	}
	if doubleHyphensRegexp.MatchString(port) {
		return false
	}
	if identifierNoHyphensBeginEndRegexp.MatchString(port) && atLeastOneLetterRegexp.MatchString(port) {
		return true
	}
	return false
}

// IsValidIPv4 tests that the argument is a valid IPv4 address.
func IsValidIPv4(value string) bool {
	return net.ParseIP(value) != nil && net.ParseIP(value).To4() != nil
}

const percentFmt string = "[0-9]+%"

var percentRegexp = regexp.MustCompile("^" + percentFmt + "$")

func IsValidPercent(percent string) bool {
	return percentRegexp.MatchString(percent)
}
