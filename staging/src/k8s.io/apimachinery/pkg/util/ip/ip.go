/*
Copyright 2024 The Kubernetes Authors.

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

	netutils "k8s.io/utils/net"
)

// ParseValidIP parses an IPv4 or IPv6 address that is valid according to current
// Kubernetes validation rules. In particular:
//
//  1. IPv4 IPs must not have any leading "0"s in octets (e.g. "010.002.003.004").
//     Historically, net.ParseIP (and later netutils.ParseIPSloppy) simply ignored the
//     leading "0"s, but some libc-based software treats 0-prefixed octets as octal,
//     meaning different software would interpret the same string as a different IP.
//     (Current net.ParseIP and netip.ParseAddr reject inputs with leading "0"s.)
//
//     (More generally, IPv4 IPs must use the ordinary decimal dotted quad notation, not
//     any of the weirder formats historically supported by inet_aton(), which have never
//     been supported by net.ParseIP/netip.ParseAddr.)
//
//  2. IPv4-mapped IPv6 IPs (e.g. "::ffff:1.2.3.4") are not allowed, because they may be
//     treated as IPv4 by some software and IPv6 by other software, again potentially
//     leading to different interpretations of the same values. (net.ParseIP and
//     netip.ParseAddr both allow these, but there are no use cases for representing IPv4
//     addresses as IPv4-mapped IPv6 addresses in Kubernetes.)
//
//  3. IPs may not have a trailing zone identifier (e.g. "fe80::1234%eth0"). This syntax
//     is accepted by netip.ParseAddr, but not by net.ParseIP/netutils.ParseIPSloppy.
//
// Any IP accepted by ParseValidIP would also be parsed identically by net.ParseIP,
// netutils.ParseIPSloppy, netip.ParseAddr, and pretty much any external IP-parsing API.
//
// If you are parsing an IP that has already been declared valid by other code (the
// apiserver, the command-line-argument parser, etc), then you should generally use
// ParseIP rather than ParseValidIP, so that you don't accidentally impose stricter
// validation standards on data that may have historically been less-strictly validated.
func ParseValidIP(ipStr string) (netip.Addr, error) {
	if ipStr == "" {
		return netip.Addr{}, fmt.Errorf("expected an IP address")
	}

	ip, err := netip.ParseAddr(ipStr)
	if err != nil {
		// netip.ParseAddr returns very technical/low-level error messages
		// so we ignore err itself.

		if ip := netutils.ParseIPSloppy(ipStr); ip != nil {
			return netip.Addr{}, fmt.Errorf("IP address should not have leading 0s")
		}
		if _, err := netip.ParsePrefix(ipStr); err == nil {
			return netip.Addr{}, fmt.Errorf("expected IP address, got CIDR value")
		}

		return netip.Addr{}, fmt.Errorf("not a valid IP address")
	}
	if ip.Zone() != "" {
		return netip.Addr{}, fmt.Errorf("IP address with zone value is not allowed")
	}
	if ip.Is4In6() {
		return netip.Addr{}, fmt.Errorf("IPv4-mapped IPv6 address is not allowed")
	}

	return ip, nil
}

// ParseIP parses an IPv4 or IPv6 address, accepting both fully-valid IP addresses and
// irregular forms that were accepted by older versions of Kubernetes. Any value which
// would be accepted by ParseValidIP will always be accepted and parsed identically by
// ParseIP, so there is generally no need to use ParseValidIP on data that has already
// been validated.
//
// Note that if you have an object field, command line flag, etc, containing a "legacy" IP
// address value that needs to be parsed with ParseIP rather than ParseValidIP, then you
// should not pass the raw string value of that field to other APIs (inside or outside of
// Kubernetes) because other software may parse it to a different IP value (or refuse to
// parse it at all). You should instead always call ParseIP on it first and then convert
// the parsed value back to a string, and pass that canonicalized string value to other
// APIs.
func ParseIP(ipStr string) (netip.Addr, error) {
	// Note: if we want to get rid of netutils.ParseIPSloppy (and its forked copy of
	// golang 1.16's net.ParseIP), we should be able to use some invocation of
	// regexp.ReplaceAllString to get rid of leading 0s in ipStr.

	ip := netutils.ParseIPSloppy(ipStr)
	if ip == nil {
		// If netutils.ParseIPSloppy() rejected it then ParseValidIP is sure to
		// reject it as well (either it's invalid, or it contains an IPv6 zone).
		// So use that to get an error message.
		return ParseValidIP(ipStr)
	}

	return AddrFromIP(ip), nil
}

// ParseCanonicalIP parses a valid IPv4 or IPv6 address and confirms that it was in
// canonical form (that is, it confirms that calling `.String()` on the output value
// yields the input value). For IPv4, any IPv4 address accepted by ParseValidIP is also
// canonical, but for IPv6, ParseCanonicalIP requires the address to be in the canonical
// form specified by RFC 5952. (In particular, it can have no extra "0"s, must use the
// "::" in the correct place, and must use lowercase letters.)
//
// ParseCanonicalIP is preferred to ParseValidIP for validating fields in new APIs,
// because there is exactly 1 canonical representation for any IP, meaning
// canonical-IP-valued fields can just be treated as strings when checking for equality or
// uniqueness.
func ParseCanonicalIP(ipStr string) (netip.Addr, error) {
	ip, err := ParseValidIP(ipStr)
	if err != nil {
		return ip, err
	}

	if ip.Is6() {
		canonical := ip.String()
		if ipStr != canonical {
			return netip.Addr{}, fmt.Errorf("not accepting IP address %q which is not in canonical form (%q)", ipStr, canonical)
		}
	}

	return ip, nil
}

// MustParseIP parses an IPv4 or IPv6 string (which must be in canonical format), and
// panics on failure. This can be used for test cases or compile-time constants.
func MustParseIP(ipStr string) netip.Addr {
	ip, err := ParseCanonicalIP(ipStr)
	if err != nil {
		panic(err)
	}
	return ip
}

// MustParseIPs parses a list of IPv4 or IPv6 strings (which must be in canonical
// format), and panics on failure. This can be used for test cases or compile-time
// constants.
func MustParseIPs(ipStrs ...string) []netip.Addr {
	ips := make([]netip.Addr, len(ipStrs))
	for i := range ipStrs {
		ips[i] = MustParseIP(ipStrs[i])
	}
	return ips
}
