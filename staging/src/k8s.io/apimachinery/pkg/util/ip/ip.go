/*
Copyright 2023 The Kubernetes Authors.

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

package ip

import (
	"fmt"
	"net/netip"
	"strings"

	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

var invalidIP netip.Addr

// ParseIP parses a valid IPv4 or IPv6 address. Note that ParseIP's concept of "valid" is
// stricter than current API validation. (FIXME add new validation function.) In
// particular:
//
//  1. IPv4 IPs with leading "0"s (e.g. "001.002.003.004") are not allowed.
//  2. IPv6-wrapped IPv4 IPs (e.g. "::ffff:1.2.3.4") are not allowed.
//
// Contrast ParseLegacyIP and ParseCanonicalIP.
func ParseIP(ipStr string) (netip.Addr, error) {
	ip, err := netip.ParseAddr(ipStr)
	if err == nil {
		// netip.ParseAddr accepts IPv6 addresses with zone specifiers (eg
		// "fe80::1234%eth0") but this is not valid for most Kubernetes use cases.
		if ip.Zone() != "" {
			return invalidIP, fmt.Errorf("IP address %q with zone value is not allowed", ipStr)
		}

		if ip.Is4In6() {
			return invalidIP, fmt.Errorf("IPv4-in-IPv6 address %q is not allowed", ipStr)
		}
	}

	return ip, err
}

// ParseLegacyIP parses an IPv4 or IPv6 address in a legacy API field, relaxing the
// additional restrictions imposed by ParseIP(). The required context value should
// indicate where the legacy address value is coming from (e.g., a field or annotation
// name). This function MUST NOT be used for parsing IPs that are not from legacy API
// objects. (It will log a warning if it finds an "invalid" IP, so it should not be used
// when parsing a string where one of the "invalid" formats is actually allowed; use
// `netip.ParseAddr` directly if you need to parse other formats.)
//
// The first return value contains the result of parsing the passed-in ipStr in the way
// that legacy `net.ParseIP` would have parsed it (ignoring leading "0"s, but not fixing
// up IPv4-in-IPv6). The second return value contains the same IP value, but with
// IPv4-in-IPv6 converted to plain IPv4. You should generally use the second value if the
// first value is not required for backward-compatibility.
//
// Note that if you have an API field containing a legacy IP value then it is not safe to
// pass the raw value of that field directly to external APIs (including command-line
// APIs) because most C-based code will treat IPv4 addresses with leading "0"s as octal
// rather than decimal, causing them to interpret it as a different address than
// ParseLegacyIP() does, potentially creating security issues. You should instead always
// call ParseLegacyIP() on it first, then convert the returned value to a string, and pass
// that to external APIs.
//
// Contrast ParseIP and ParseCanonicalIP.
func ParseLegacyIP(ipStr string, context LegacyIPStringContext) (netip.Addr, netip.Addr, error) {
	ip, err := netip.ParseAddr(ipStr)
	if err == nil {
		if ip.Zone() != "" {
			return invalidIP, invalidIP, fmt.Errorf("IP address %q with zone value is not allowed", ipStr)
		}
		if ip.Is4In6() {
			klog.InfoS("Parsed invalid legacy IP address", "context", context, "ip", ipStr)
			return ip, ip.Unmap(), nil
		}

		return ip, ip, nil
	}

	badIP := netutils.ParseIPSloppy(ipStr)
	if badIP != nil {
		klog.InfoS("Parsed invalid legacy IP address", "context", context, "ip", ipStr)
		// Recursively call ourselves in case it has other problems too!
		return ParseLegacyIP(badIP.String(), context)
	}

	return invalidIP, invalidIP, err
}

// ParseCanonicalIP parses a valid IPv4 or IPv6 address and confirms that it was in
// canonical form (i.e., the form that `.String()` would return). For IPv4, any valid IPv4
// address is also canonical. For IPv6, canonical addresses have no unnecessary leading
// "0"s, use "::" in the first possible place, and use lowercase letters.
func ParseCanonicalIP(ipStr string) (netip.Addr, error) {
	ip, err := ParseIP(ipStr)
	if err != nil {
		return invalidIP, err
	}

	if ip.String() != ipStr {
		return invalidIP, fmt.Errorf("not accepting IP address %q which is not in canonical form", ipStr)
	}
	return ip, nil
}

// ParseIPList parses a comma-separated list of IPs as with ParseIP. If ipStrList is the
// empty string, this will return an empty list of IPs.
func ParseIPList(ipStrList string) ([]netip.Addr, error) {
	var err error

	if ipStrList == "" {
		return []netip.Addr{}, nil
	}

	ipStrs := strings.Split(ipStrList, ",")
	ips := make([]netip.Addr, len(ipStrs))
	for i := range ipStrs {
		ips[i], err = ParseIP(ipStrs[i])
		if err != nil {
			return nil, err
		}
	}
	return ips, nil
}

// ParseLegacyIPList parses a comma-separated list of IPs as with ParseLegacyIP (returning
// two lists of parsed IPs corresponding to the two ParseLegacyIP return values). If
// ipStrList is the empty string, this will return an empty list of IPs.
func ParseLegacyIPList(ipStrList string, context LegacyIPStringContext) ([]netip.Addr, []netip.Addr, error) {
	var err error

	if ipStrList == "" {
		return []netip.Addr{}, []netip.Addr{}, nil
	}

	ipStrs := strings.Split(ipStrList, ",")
	ips := make([]netip.Addr, len(ipStrs))
	cleanIPs := make([]netip.Addr, len(ipStrs))
	for i := range ipStrs {
		ips[i], cleanIPs[i], err = ParseLegacyIP(ipStrs[i], context)
		if err != nil {
			return nil, nil, err
		}
	}
	return ips, cleanIPs, nil
}

// UnparseIPList takes an array of IPs and returns a comma-separated list of IP strings.
func UnparseIPList(ips []netip.Addr) string {
	var b strings.Builder

	for i := range ips {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(ips[i].String())
	}
	return b.String()
}

// MustParseIP parses an IPv4 or IPv6 string in canonical format, and panics on failure.
// This can be used for test cases or compile-time constants.
func MustParseIP(ipStr string) netip.Addr {
	ip, err := ParseCanonicalIP(ipStr)
	if err != nil {
		panic(err)
	}
	return ip
}

// MustParseIPs parses an array of IPv4 or IPv6 strings in canonical format, and panics on
// failure. This can be used for test cases or compile-time constants.
func MustParseIPs(ipStrs []string) []netip.Addr {
	ips := make([]netip.Addr, len(ipStrs))
	for i := range ipStrs {
		ips[i] = MustParseIP(ipStrs[i])
	}
	return ips
}
