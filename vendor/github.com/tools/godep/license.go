package main

import (
	"strings"
)

// LicenseFilePrefix is a list of filename prefixes that indicate it
//  might contain a software license
var LicenseFilePrefix = []string{
	"licence", // UK spelling
	"license", // US spelling
	"copying",
	"unlicense",
	"copyright",
	"copyleft",
	"authors",
	"contributors",
}

// LegalFileSubstring are substrings that indicate the file is likely
// to contain some type of legal declaration.  "legal" is often used
// that it might moved to LicenseFilePrefix
var LegalFileSubstring = []string{
	"legal",
	"notice",
	"disclaimer",
	"patent",
	"third-party",
	"thirdparty",
}

// IsLicenseFile returns true if the filename might be contain a
// software license
func IsLicenseFile(filename string) bool {
	lowerfile := strings.ToLower(filename)
	for _, prefix := range LicenseFilePrefix {
		if strings.HasPrefix(lowerfile, prefix) {
			return true
		}
	}
	return false
}

// IsLegalFile returns true if the file is likely to contain some type
// of of legal declaration or licensing information
func IsLegalFile(filename string) bool {
	lowerfile := strings.ToLower(filename)
	for _, prefix := range LicenseFilePrefix {
		if strings.HasPrefix(lowerfile, prefix) {
			return true
		}
	}
	for _, substring := range LegalFileSubstring {
		if strings.Contains(lowerfile, substring) {
			return true
		}
	}
	return false
}
