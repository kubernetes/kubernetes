package dns

import (
	"regexp"
)

// IPLocalhost is a regex pattern for IPv4 or IPv6 loopback range.
const IPLocalhost = `((127\.([0-9]{1,3}\.){2}[0-9]{1,3})|(::1)$)`

// IPv4Localhost is a regex pattern for IPv4 localhost address range.
const IPv4Localhost = `(127\.([0-9]{1,3}\.){2}[0-9]{1,3})`

var localhostIPRegexp = regexp.MustCompile(IPLocalhost)
var localhostIPv4Regexp = regexp.MustCompile(IPv4Localhost)

// IsLocalhost returns true if ip matches the localhost IP regular expression.
// Used for determining if nameserver settings are being passed which are
// localhost addresses
func IsLocalhost(ip string) bool {
	return localhostIPRegexp.MatchString(ip)
}

// IsIPv4Localhost returns true if ip matches the IPv4 localhost regular expression.
func IsIPv4Localhost(ip string) bool {
	return localhostIPv4Regexp.MatchString(ip)
}
