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

var invalidCIDR netip.Prefix

// ParseCIDR parses a valid IPv4 or IPv6 CIDR string. Note that ParseCIDR's concept of
// "valid" is stricter than current API validation. (FIXME add new validation function.)
// In particular:
//
//  1. IPv4 IPs with leading "0"s (e.g. "001.002.003.000/24") are not allowed.
//  2. IPv6-wrapped IPv4 IPs (e.g. "::ffff:1.2.3.0/24") are not allowed.
//  3. CIDRs with non-0 bits after the prefix length (e.g. "1.2.3.4/24") are not allowed.
//
// Contrast ParseLegacyCIDR and ParseCanonicalCIDR.
func ParseCIDR(cidrStr string) (netip.Prefix, error) {
	cidr, err := netip.ParsePrefix(cidrStr)
	if err == nil {
		// (Unlike netip.ParseAddr, netip.ParsePrefix doesn't allow IPv6 zones, so
		// we don't have to check for that.)

		if cidr.Addr().Is4In6() {
			return invalidCIDR, fmt.Errorf("IPv4-in-IPv6 address %q is not allowed", cidrStr)
		}
		if cidr != cidr.Masked() {
			return invalidCIDR, fmt.Errorf("invalid CIDR value %q; should not have any bits set beyond the prefix length", cidrStr)
		}
	}

	return cidr, err
}

// ParseLegacyCIDR parses an IPv4 or IPv6 address in a legacy API field, relaxing the
// additional restrictions imposed by ParseCIDR().The required context value should
// indicate where the legacy address value is coming from (e.g., a field or annotation
// name). This function MUST NOT be used for parsing CIDRs that are not from legacy API
// objects. (It will log a warning if it finds an "invalid" CIDR, so it should not be used
// when parsing a string where one of the "invalid" formats is actually allowed; use
// `netip.ParsePrefix` directly if you need to parse other formats, or if you are parsing
// an "IP with subnet length" value like "1.2.3.4/24".)
//
// The first return value contains the result of parsing the passed-in cidrStr in the way
// that legacy `net.ParseCIDR` would have parsed it (ignoring leading "0"s and not caring
// about prefix lengths, but not fixing up IPv4-in-IPv6). The second return value contains
// the CIDR value with all bits after the prefix length set to 0, and with IPv4-in-IPv6
// converted to plain IPv4. You should generally use the second value if the first value
// is not required for backward-compatibility.
//
// Note that if you have an API field containing a legacy CIDR value then it is not safe
// to pass the raw value of that field directly to external APIs (including command-line
// APIs) because:
//
//  1. Most C-based code will treat IPv4 addresses with leading "0"s as octal rather than
//     decimal, causing them to interpret it as a different address than
//     ParseLegacyCIDR() does, potentially creating security issues.
//
//  2. Different APIs/tools differ in how they interpret CIDR addresses with "incorrect"
//     prefix lengths. E.g., some tools may interpret "1.2.3.4/24" as meaning
//     "1.2.3.0/24", while others may interpret it as "1.2.3.4/32".
//
// Contrast ParseCIDR and ParseCanonicalCIDR.
func ParseLegacyCIDR(cidrStr string, context LegacyIPStringContext) (netip.Prefix, netip.Prefix, error) {
	cidr, err := netip.ParsePrefix(cidrStr)
	if err == nil {
		// (Unlike netip.ParseAddr, netip.ParsePrefix doesn't allow IPv6 zones, so
		// we don't have to check for that.)

		if cidr.Addr().Is4In6() || cidr != cidr.Masked() {
			klog.InfoS("Parsed invalid legacy CIDR address", "context", context, "cidr", cidrStr)
			// Note that ParsePrefix only accepts IPv4-in-IPv6 CIDRs where the
			// prefix length is relative to the IPv4 address, but calling
			// .Masked() on such a CIDR will mask the IPv6 value, not the IPv4
			// value. So we have to Unmap first, then mask.
			cleanCIDR, _ := netip.ParsePrefix(fmt.Sprintf("%s/%d", cidr.Addr().Unmap().String(), cidr.Bits()))
			cleanCIDR = cleanCIDR.Masked()
			return cidr, cleanCIDR, nil

		}

		return cidr, cidr, err
	}

	_, badCIDR, _ := netutils.ParseCIDRSloppy(cidrStr)
	if badCIDR != nil {
		klog.InfoS("Parsed invalid legacy CIDR address", "context", context, "cidr", cidrStr)
		// Recursively call ourselves in case it has other problems too!
		return ParseLegacyCIDR(badCIDR.String(), context)
	}

	return invalidCIDR, invalidCIDR, err
}

// ParseCanonicalCIDR parses a valid IPv4 or IPv6 CIDR string and confirms that it was in
// canonical form (i.e., the form that `.String()` would return). For IPv4, any valid IPv4
// CIDR string is also canonical. For IPv6, canonical CIDR strings have no unnecessary
// leading "0"s, use "::" in the first possible place, and use lowercase letters.
func ParseCanonicalCIDR(cidrStr string) (netip.Prefix, error) {
	cidr, err := ParseCIDR(cidrStr)
	if err != nil {
		return invalidCIDR, err
	}

	if cidr.String() != cidrStr {
		return invalidCIDR, fmt.Errorf("not accepting CIDR string %q which is not in canonical form", cidrStr)
	}
	return cidr, nil
}

// ParseCIDRList parses a comma-separated list of CIDRs as with ParseCIDR.
func ParseCIDRList(cidrStrList string) ([]netip.Prefix, error) {
	var err error

	if cidrStrList == "" {
		return []netip.Prefix{}, nil
	}

	cidrStrs := strings.Split(cidrStrList, ",")
	cidrs := make([]netip.Prefix, len(cidrStrs))
	for i := range cidrStrs {
		cidrs[i], err = ParseCIDR(cidrStrs[i])
		if err != nil {
			return nil, err
		}
	}
	return cidrs, nil
}

// ParseLegacyCIDRList parses a comma-separated list of CIDRs as with ParseLegacyCIDR
// (returning two lists of parsed CIDRs corresponding to the two ParseLegacyCIDR return
// values).
func ParseLegacyCIDRList(cidrStrList string, context LegacyIPStringContext) ([]netip.Prefix, []netip.Prefix, error) {
	var err error

	if cidrStrList == "" {
		return []netip.Prefix{}, []netip.Prefix{}, nil
	}

	cidrStrs := strings.Split(cidrStrList, ",")
	cidrs := make([]netip.Prefix, len(cidrStrs))
	cleanCIDRs := make([]netip.Prefix, len(cidrStrs))
	for i := range cidrStrs {
		cidrs[i], cleanCIDRs[i], err = ParseLegacyCIDR(cidrStrs[i], context)
		if err != nil {
			return nil, nil, err
		}
	}
	return cidrs, cleanCIDRs, nil
}

// UnparseCIDRList takes an array of CIDRs and returns a comma-separated list of CIDR strings.
func UnparseCIDRList(cidrs []netip.Prefix) string {
	var b strings.Builder

	for i := range cidrs {
		if i > 0 {
			b.WriteString(",")
		}
		b.WriteString(cidrs[i].String())
	}
	return b.String()
}

// MustParseCIDR parses an IPv4 or IPv6 CIDR string in canonical format, and panics on
// failure. This can be used for test cases or compile-time constants.
func MustParseCIDR(cidrStr string) netip.Prefix {
	cidr, err := ParseCanonicalCIDR(cidrStr)
	if err != nil {
		panic(err)
	}
	return cidr
}

// MustParseCIDRs parses an array of IPv4 or IPv6 CIDR strings in canonical format, and
// panics on failure. This can be used for test cases or compile-time constants.
func MustParseCIDRs(cidrStrs []string) []netip.Prefix {
	cidrs := make([]netip.Prefix, len(cidrStrs))
	for i := range cidrStrs {
		cidrs[i] = MustParseCIDR(cidrStrs[i])
	}
	return cidrs
}
