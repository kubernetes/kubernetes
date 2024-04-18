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
	"strings"

	netutils "k8s.io/utils/net"
)

// ParseValidCIDR parses an IPv4 or IPv6 CIDR string that is valid according to current
// Kubernetes validation rules. In particular:
//
//  1. The IP part of the CIDR string must be valid according to ParseValidIP.
//
//  2. The IP part of the CIDR must not have any bits set beyond the prefix length (e.g.
//     an IPv4 CIDR with a prefix length of "24" must have "0" for its final octet).
//     net.ParseCIDR and netip.ParsePrefix both accept CIDR values such as
//     "192.168.1.5/24", because such strings are often used to describe the IPs assigned
//     to network interfaces. But there are no Kubernetes API objects that expect IP
//     addresses represented in that form, and passing values in that form to an external
//     API that was expecting a subnet or a mask value would have undefined results.
//     (E.g., "192.168.1.5/24" might be treated as meaning either "192.168.1.0/24" or
//     "192.168.1.5/32" depending on how it gets processed.)
//
//  3. The prefix length must consist only of numbers, and not have any leading "0"s.
//     (net.ParseCIDR and netutils.ParseCIDRSloppy allow leading "0s" (e.g.
//     "192.168.0.0/016"), and net.ParsePrefix in golang 1.21 and earlier allows both
//     leading "0"s and a leading "+".)
//
// Any CIDR string accepted by ParseValidCIDR would also be parsed identically by
// net.ParseCIDR, netutils.ParseCIDRSloppy, netip.ParsePrefix, and pretty much any
// external CIDR-parsing API.
//
// If you are parsing a CIDR string that has already been declared valid by other code
// (the apiserver, the command-line-argument parser, etc), then you should generally use
// ParseCIDR rather than ParseValidCIDR, so that you don't accidentally impose stricter
// validation standards on data that may have historically been less-strictly validated.
func ParseValidCIDR(cidrStr string) (netip.Prefix, error) {
	if cidrStr == "" {
		return netip.Prefix{}, fmt.Errorf("expected a CIDR value")
	}

	cidr, err := netip.ParsePrefix(cidrStr)
	if err != nil {
		// netip.ParsePrefix returns very technical/low-level error messages
		// so we ignore err itself.

		if _, _, err := netutils.ParseCIDRSloppy(cidrStr); err == nil {
			return netip.Prefix{}, fmt.Errorf("IP address in CIDR should not have leading 0s")
		}
		if _, err := netip.ParseAddr(cidrStr); err == nil {
			return netip.Prefix{}, fmt.Errorf("expected CIDR value, got plain IP address")
		}

		return netip.Prefix{}, fmt.Errorf("not a valid CIDR value")
	}

	// (Unlike netip.ParseAddr, netip.ParsePrefix doesn't allow IPv6 zones, so
	// we don't have to check for that.)

	if cidr.Addr().Is4In6() {
		return netip.Prefix{}, fmt.Errorf("IP address in CIDR should not be an IPv4-mapped IPv6 address")
	} else if cidr != cidr.Masked() {
		return netip.Prefix{}, fmt.Errorf("CIDR value should not have any bits set beyond the prefix length")
	}

	// Up through golang 1.21, ParsePrefix just parses the prefix length with
	// strconv.Atoi and so ends up allowing things like "1.2.3.0/+24".
	// https://github.com/golang/go/issues/63850
	if strings.Contains(cidrStr, "/+") {
		return netip.Prefix{}, fmt.Errorf("not a valid CIDR value")
	}
	// Both net.ParseCIDR and netip.ParsePrefix-up-to-1.21 allow "1.2.3.0/024".
	if strings.Contains(cidrStr, "/0") && !strings.HasSuffix(cidrStr, "/0") {
		return netip.Prefix{}, fmt.Errorf("prefix length in CIDR should not have leading 0s")
	}

	return cidr, nil
}

// ParseCIDR parses an IPv4 or IPv6 CIDR string, accepting both fully-valid CIDR values
// and irregular values that were accepted by older versions of Kubernetes. Any value
// which would be accepted by ParseValidCIDR will always be accepted and parsed
// identically by ParseCIDR, so there is generally no need to use ParseValidCIDR on data
// that has already been validated.
//
// Note that if you have an object field, command line flag, etc, containing a "legacy"
// CIDR value that needs to be parsed with ParseCIDR rather than ParseValidCIDR, then you
// should not pass the raw string value of that field to other APIs (inside or outside of
// Kubernetes) because other software may parse it to a different CIDR value (or refuse to
// parse it at all). You should instead always call ParseCIDR on it first and then convert
// the parsed value back to a string, and pass that canonicalized string value to other
// APIs.
func ParseCIDR(cidrStr string) (netip.Prefix, error) {
	// Note: if we want to get rid of netutils.ParseCIDRSloppy (and its forked copy of
	// golang 1.16's net.ParseCIDR), we should be able to use some invocation of
	// regexp.ReplaceAllString to get rid of leading 0s in cidrStr.

	_, cidr, _ := netutils.ParseCIDRSloppy(cidrStr)
	if cidr == nil {
		// If netutils.ParseCIDRSloppy() rejected it then ParseValidCIDR is sure
		// to reject it as well. So use that to get a (better) error message.
		return ParseValidCIDR(cidrStr)
	}

	return PrefixFromIPNet(cidr).Masked(), nil
}

// ParseCanonicalCIDR parses a valid IPv4 or IPv6 CIDR string and confirms that it was in
// canonical form (that is, it confirms that calling `.String()` on the output value
// yields the input value). For IPv4, any CIDR value accepted by ParseValidCIDR is also
// canonical, but for IPv6, the IP part of the string must also be in the canonical form
// specified by RFC 5952. (In particular, it can have no leading "0"s, must use the "::"
// in the correct place, and must use lowercase letters.)
//
// ParseCanonicalCIDR is preferred to ParseValidCIDR for validating fields in new APIs,
// because there is exactly 1 canonical representation for any CIDR value, meaning
// canonical-CIDR-valued fields can just be treated as strings when checking for equality
// or uniqueness.
func ParseCanonicalCIDR(cidrStr string) (netip.Prefix, error) {
	cidr, err := ParseValidCIDR(cidrStr)
	if err != nil {
		return cidr, err
	}

	canonical := cidr.String()
	if cidrStr != canonical {
		return netip.Prefix{}, fmt.Errorf("not accepting CIDR string %q which is not in canonical form (%q)", cidrStr, canonical)
	}

	return cidr, err
}

// MustParseCIDR parses an IPv4 or IPv6 CIDR string (which must be in canonical format),
// and panics on failure. This can be used for test cases or compile-time constants.
func MustParseCIDR(cidrStr string) netip.Prefix {
	cidr, err := ParseCanonicalCIDR(cidrStr)
	if err != nil {
		panic(err)
	}
	return cidr
}

// MustParseCIDRs parses a list of IPv4 or IPv6 CIDR strings (which must be in canonical
// format), and panics on failure. This can be used for test cases or compile-time
// constants.
func MustParseCIDRs(cidrStrs ...string) []netip.Prefix {
	cidrs := make([]netip.Prefix, len(cidrStrs))
	for i := range cidrStrs {
		cidrs[i] = MustParseCIDR(cidrStrs[i])
	}
	return cidrs
}
