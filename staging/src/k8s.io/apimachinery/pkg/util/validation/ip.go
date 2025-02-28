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

package validation

import (
	"fmt"
	"net"
	"net/netip"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

func parseIP(fldPath *field.Path, value string, strictValidation bool) (net.IP, field.ErrorList) {
	var allErrors field.ErrorList

	ip := netutils.ParseIPSloppy(value)
	if ip == nil {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be a valid IP address, (e.g. 10.9.8.7 or 2001:db8::ffff)"))
		return nil, allErrors
	}

	if strictValidation {
		addr, err := netip.ParseAddr(value)
		if err != nil {
			// If netutils.ParseIPSloppy parsed it, but netip.ParseAddr
			// doesn't, then it must have illegal leading 0s.
			allErrors = append(allErrors, field.Invalid(fldPath, value, "must not have leading 0s"))
		}
		if addr.Is4In6() {
			allErrors = append(allErrors, field.Invalid(fldPath, value, "must not be an IPv4-mapped IPv6 address"))
		}
	}

	return ip, allErrors
}

// IsValidIPForLegacyField tests that the argument is a valid IP address for a "legacy"
// API field that predates strict IP validation. In particular, this allows IPs that are
// not in canonical form (e.g., "FE80:0:0:0:0:0:0:0abc" instead of "fe80::abc").
//
// If strictValidation is false, this also allows IPs in certain invalid or ambiguous
// formats:
//
//  1. IPv4 IPs are allowed to have leading "0"s in octets (e.g. "010.002.003.004").
//     Historically, net.ParseIP (and later netutils.ParseIPSloppy) simply ignored leading
//     "0"s in IPv4 addresses, but most libc-based software treats 0-prefixed IPv4 octets
//     as octal, meaning different software might interpret the same string as two
//     different IPs, potentially leading to security issues. (Current net.ParseIP and
//     netip.ParseAddr simply reject inputs with leading "0"s.)
//
//  2. IPv4-mapped IPv6 IPs (e.g. "::ffff:1.2.3.4") are allowed. These can also lead to
//     different software interpreting the value in different ways, because they may be
//     treated as IPv4 by some software and IPv6 by other software. (net.ParseIP and
//     netip.ParseAddr both allow these, but there are no use cases for representing IPv4
//     addresses as IPv4-mapped IPv6 addresses in Kubernetes.)
//
// This function should only be used to validate the existing fields that were
// historically validated in this way, and strictValidation should be true unless the
// StrictIPCIDRValidation feature gate is disabled. Use IsValidIP for parsing new fields.
func IsValidIPForLegacyField(fldPath *field.Path, value string, strictValidation bool) field.ErrorList {
	_, allErrors := parseIP(fldPath, value, strictValidation)
	return allErrors.WithOrigin("format=ip-sloppy")
}

// IsValidIP tests that the argument is a valid IP address, according to current
// Kubernetes standards for IP address validation.
func IsValidIP(fldPath *field.Path, value string) field.ErrorList {
	ip, allErrors := parseIP(fldPath, value, true)
	if len(allErrors) != 0 {
		return allErrors.WithOrigin("format=ip-strict")
	}

	if value != ip.String() {
		allErrors = append(allErrors, field.Invalid(fldPath, value, fmt.Sprintf("must be in canonical form (%q)", ip.String())))
	}
	return allErrors.WithOrigin("format=ip-strict")
}

// GetWarningsForIP returns warnings for IP address values in non-standard forms. This
// should only be used with fields that are validated with IsValidIPForLegacyField().
func GetWarningsForIP(fldPath *field.Path, value string) []string {
	ip := netutils.ParseIPSloppy(value)
	if ip == nil {
		klog.ErrorS(nil, "GetWarningsForIP called on value that was not validated with IsValidIPForLegacyField", "field", fldPath, "value", value)
		return nil
	}

	addr, _ := netip.ParseAddr(value)
	if !addr.IsValid() || addr.Is4In6() {
		// This catches 2 cases: leading 0s (if ParseIPSloppy() accepted it but
		// ParseAddr() doesn't) or IPv4-mapped IPv6 (.Is4In6()). Either way,
		// re-stringifying the net.IP value will give the preferred form.
		return []string{
			fmt.Sprintf("%s: non-standard IP address %q will be considered invalid in a future Kubernetes release: use %q", fldPath, value, ip.String()),
		}
	}

	// If ParseIPSloppy() and ParseAddr() both accept it then it's fully valid, though
	// it may be non-canonical.
	if addr.Is6() && addr.String() != value {
		return []string{
			fmt.Sprintf("%s: IPv6 address %q should be in RFC 5952 canonical format (%q)", fldPath, value, addr.String()),
		}
	}

	return nil
}

func parseCIDR(fldPath *field.Path, value string, strictValidation bool) (*net.IPNet, field.ErrorList) {
	var allErrors field.ErrorList

	_, ipnet, err := netutils.ParseCIDRSloppy(value)
	if err != nil {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be a valid CIDR value, (e.g. 10.9.8.0/24 or 2001:db8::/64)"))
		return nil, allErrors
	}

	if strictValidation {
		prefix, err := netip.ParsePrefix(value)
		if err != nil {
			// If netutils.ParseCIDRSloppy parsed it, but netip.ParsePrefix
			// doesn't, then it must have illegal leading 0s (either in the
			// IP part or the prefix).
			allErrors = append(allErrors, field.Invalid(fldPath, value, "must not have leading 0s in IP or prefix length"))
		} else if prefix.Addr().Is4In6() {
			allErrors = append(allErrors, field.Invalid(fldPath, value, "must not have an IPv4-mapped IPv6 address"))
		} else if prefix.Addr() != prefix.Masked().Addr() {
			allErrors = append(allErrors, field.Invalid(fldPath, value, "must not have bits set beyond the prefix length"))
		}
	}

	return ipnet, allErrors
}

// IsValidCIDRForLegacyField tests that the argument is a valid CIDR value for a "legacy"
// API field that predates strict IP validation. In particular, this allows IPs that are
// not in canonical form (e.g., "FE80:0abc:0:0:0:0:0:0/64" instead of "fe80:abc::/64").
//
// If strictValidation is false, this also allows CIDR values in certain invalid or
// ambiguous formats:
//
//  1. The IP part of the CIDR value is parsed as with IsValidIPForLegacyField with
//     strictValidation=false.
//
//  2. The CIDR value is allowed to be either a "subnet"/"mask" (with the lower bits after
//     the prefix length all being 0), or an "interface address" as with `ip addr` (with a
//     complete IP address and associated subnet length). With strict validation, the
//     value is required to be in "subnet"/"mask" form.
//
//  3. The prefix length is allowed to have leading 0s.
//
// This function should only be used to validate the existing fields that were
// historically validated in this way, and strictValidation should be true unless the
// StrictIPCIDRValidation feature gate is disabled. Use IsValidCIDR or
// IsValidInterfaceAddress for parsing new fields.
func IsValidCIDRForLegacyField(fldPath *field.Path, value string, strictValidation bool) field.ErrorList {
	_, allErrors := parseCIDR(fldPath, value, strictValidation)
	return allErrors
}

// IsValidCIDR tests that the argument is a valid CIDR value, according to current
// Kubernetes standards for CIDR validation. This function is only for
// "subnet"/"mask"-style CIDR values (e.g., "192.168.1.0/24", with no bits set beyond the
// prefix length). Use IsValidInterfaceAddress for "ifaddr"-style CIDR values.
func IsValidCIDR(fldPath *field.Path, value string) field.ErrorList {
	ipnet, allErrors := parseCIDR(fldPath, value, true)
	if len(allErrors) != 0 {
		return allErrors
	}

	if value != ipnet.String() {
		allErrors = append(allErrors, field.Invalid(fldPath, value, fmt.Sprintf("must be in canonical form (%q)", ipnet.String())))
	}
	return allErrors
}

// GetWarningsForCIDR returns warnings for CIDR values in non-standard forms. This should
// only be used with fields that are validated with IsValidCIDRForLegacyField().
func GetWarningsForCIDR(fldPath *field.Path, value string) []string {
	ip, ipnet, err := netutils.ParseCIDRSloppy(value)
	if err != nil {
		klog.ErrorS(err, "GetWarningsForCIDR called on value that was not validated with IsValidCIDRForLegacyField", "field", fldPath, "value", value)
		return nil
	}

	var warnings []string

	// Check for bits set after prefix length
	if !ip.Equal(ipnet.IP) {
		_, addrlen := ipnet.Mask.Size()
		singleIPCIDR := fmt.Sprintf("%s/%d", ip.String(), addrlen)
		warnings = append(warnings,
			fmt.Sprintf("%s: CIDR value %q is ambiguous in this context (should be %q or %q?)", fldPath, value, ipnet.String(), singleIPCIDR),
		)
	}

	prefix, _ := netip.ParsePrefix(value)
	addr := prefix.Addr()
	if !prefix.IsValid() || addr.Is4In6() {
		// This catches 2 cases: leading 0s (if ParseCIDRSloppy() accepted it but
		// ParsePrefix() doesn't) or IPv4-mapped IPv6 (.Is4In6()). Either way,
		// re-stringifying the net.IPNet value will give the preferred form.
		warnings = append(warnings,
			fmt.Sprintf("%s: non-standard CIDR value %q will be considered invalid in a future Kubernetes release: use %q", fldPath, value, ipnet.String()),
		)
	}

	// If ParseCIDRSloppy() and ParsePrefix() both accept it then it's fully valid,
	// though it may be non-canonical. But only check this if there are no other
	// warnings, since either of the other warnings would also cause a round-trip
	// failure.
	if len(warnings) == 0 && addr.Is6() && prefix.String() != value {
		warnings = append(warnings,
			fmt.Sprintf("%s: IPv6 CIDR value %q should be in RFC 5952 canonical format (%q)", fldPath, value, prefix.String()),
		)
	}

	return warnings
}

// IsValidInterfaceAddress tests that the argument is a valid "ifaddr"-style CIDR value in
// canonical form (e.g., "192.168.1.5/24", with a complete IP address and associated
// subnet length). Use IsValidCIDR for "subnet"/"mask"-style CIDR values (e.g.,
// "192.168.1.0/24").
func IsValidInterfaceAddress(fldPath *field.Path, value string) field.ErrorList {
	var allErrors field.ErrorList
	ip, ipnet, err := netutils.ParseCIDRSloppy(value)
	if err != nil {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be a valid address in CIDR form, (e.g. 10.9.8.7/24 or 2001:db8::1/64)"))
		return allErrors
	}

	// The canonical form of `value` is not `ipnet.String()`, because `ipnet` doesn't
	// include the bits after the prefix. We need to construct the canonical form
	// ourselves from `ip` and `ipnet.Mask`.
	maskSize, _ := ipnet.Mask.Size()
	if netutils.IsIPv4(ip) && maskSize > net.IPv4len*8 {
		// "::ffff:192.168.0.1/120" -> "192.168.0.1/24"
		maskSize -= (net.IPv6len - net.IPv4len) * 8
	}
	canonical := fmt.Sprintf("%s/%d", ip.String(), maskSize)
	if value != canonical {
		allErrors = append(allErrors, field.Invalid(fldPath, value, fmt.Sprintf("must be in canonical form (%q)", canonical)))
	}
	return allErrors
}
