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
	"net/netip"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

// IPValidationFlag is an option for IsValidIP
type IPValidationFlag string

const (
	// IPIsIPv4 validates that the IP is IPv4
	IPIsIPv4 IPValidationFlag = "IPIsIPv4"

	// IPIsIPv6 validates that the IP is IPv6
	IPIsIPv6 IPValidationFlag = "IPIsIPv6"

	// IPIsCanonical validates that the IP is in canonical form.
	IPIsCanonical IPValidationFlag = "IPIsCanonical"

	// IPLegacyValidation validates the IP according to legacy validation rules
	IPLegacyValidation IPValidationFlag = "IPLegacyValidation"
)

// IsValidIP tests that the argument is a valid IP address according to flags.
//
// By default, IsValidIP uses "strict" validation. If you pass the IPLegacyValidation
// flag, it will instead use the traditional Kubernetes IP validation, which allows some
// invalid or ambiguous formats:
//
//  1. Legacy validation allows IPv4 IPs to have leading "0"s in octets (e.g.
//     "010.002.003.004"). Historically, net.ParseIP (and later netutils.ParseIPSloppy)
//     simply ignored leading "0"s in IPv4 addresses, but most libc-based software treats
//     0-prefixed IPv4 octets as octal, meaning different software might interpret the
//     same string as two different IPs, potentially leading to security issues. (Current
//     net.ParseIP and netip.ParseAddr simply reject inputs with leading "0"s.)
//
//  2. Legacy validation allows IPv4-mapped IPv6 IPs (e.g. "::ffff:1.2.3.4"). These can
//     also lead to different software interpreting the value in different ways, because
//     they may be treated as IPv4 by some software and IPv6 by other software, again
//     potentially leading to different interpretations of the same values. (net.ParseIP
//     and netip.ParseAddr both allow these, but there are no use cases for representing
//     IPv4 addresses as IPv4-mapped IPv6 addresses in Kubernetes.)
//
// Legacy validation should *only* be used for the existing API fields that have
// traditionally been validated that way (and only when the StrictIPCIDRValidation feature
// gate is disabled). All new API fields should use strict validation, to ensure that the
// IP address will be accepted, and interpreted identically, by net.ParseIP,
// netutils.ParseIPSloppy, netip.ParseAddr, and equivalent APIs in other languages.
func IsValidIP(fldPath *field.Path, value string, flags ...IPValidationFlag) field.ErrorList {
	var allErrors field.ErrorList

	ip := netutils.ParseIPSloppy(value)
	if ip == nil {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be a valid IP address, (e.g. 10.9.8.7 or 2001:db8::ffff)"))
		return allErrors
	}

	flagSet := sets.New(flags...)

	if !flagSet.Has(IPLegacyValidation) {
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

	// Skip checking IPIsCanonical if we got one of the above strict-parsing errors,
	// to avoid complaining about the same problem twice.
	if flagSet.Has(IPIsCanonical) && len(allErrors) == 0 {
		if value != ip.String() {
			allErrors = append(allErrors, field.Invalid(fldPath, value, fmt.Sprintf("must be in canonical form (%q)", ip.String())))
		}
	}

	if flagSet.Has(IPIsIPv4) && !netutils.IsIPv4(ip) {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be an IPv4 address"))
	}
	if flagSet.Has(IPIsIPv6) && !netutils.IsIPv6(ip) {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be an IPv6 address"))
	}

	return allErrors
}

// GetWarningsForIP returns warnings for IP address values in non-standard forms. This
// should only be used with fields that are validated with IsValidIP().
func GetWarningsForIP(fldPath *field.Path, value string) []string {
	ip := netutils.ParseIPSloppy(value)
	if ip == nil {
		klog.ErrorS(nil, "GetWarningsForIP called on value that was not validated with IsValidIP", "field", fldPath, "value", value)
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

// CIDRValidationFlag is an option for IsValidCIDR
type CIDRValidationFlag string

const (
	// CIDRIsIPv4 validates that the CIDR is IPv4
	CIDRIsIPv4 CIDRValidationFlag = "CIDRIsIPv4"

	// CIDRIsIPv6 validates that the CIDR is IPv6
	CIDRIsIPv6 CIDRValidationFlag = "CIDRIsIPv6"

	// CIDRIsCanonical validates that the CIDR is in canonical form.
	CIDRIsCanonical CIDRValidationFlag = "CIDRIsCanonical"

	// CIDRIsAddress says that a CIDR value is expected to be an interface address
	// (with bits set beyond the prefix length) rather than a subnet/mask (with no
	// bits set beyond the prefix length). With legacy parsing, either form is
	// allowed, but with strict parsing, only the subnet/mask form is accepted unless
	// you pass this flag.
	CIDRIsAddress CIDRValidationFlag = "CIDRIsAddress"

	// CIDRLegacyValidation validates the CIDR according to legacy validation rules
	CIDRLegacyValidation CIDRValidationFlag = "CIDRLegacyValidation"
)

// IsValidCIDR tests that the argument is a valid CIDR value according to flags.
//
// By default, IsValidCIDR uses "strict" validation. If you pass the CIDRLegacyValidation
// flag, it will instead use the traditional Kubernetes CIDR validation, which allows some
// invalid or ambiguous formats:
//
//  1. The IP part of the CIDR value is parsed as with IsValidIP's IPLegacyValidation.
//
//  2. The CIDR value is allowed to be either a "subnet"/"mask" (with the lower bits after
//     the prefix length all being 0), or an "interface address" as with `ip addr` (with a
//     complete IP address and associated subnet length). With strict parsing, the value
//     is required to be in "subnet"/"mask" form, unless you pass CIDRIsAddress.
//
//  3. The prefix length is allowed to have leading 0s.
//
// Legacy validation should *only* be used for the existing API fields that have
// traditionally been validated that way (and only when the StrictIPCIDRValidation feature
// gate is disabled). All new API fields should use strict validation, to ensure that the
// CIDR value will be accepted, and interpreted identically, by net.ParseCIDR,
// netutils.ParseCIDRSloppy, netip.ParsePrefix, and equivalent APIs in other languages.
func IsValidCIDR(fldPath *field.Path, value string, flags ...CIDRValidationFlag) field.ErrorList {
	var allErrors field.ErrorList
	ip, ipnet, err := netutils.ParseCIDRSloppy(value)
	if err != nil {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be a valid CIDR value, (e.g. 10.9.8.0/24 or 2001:db8::/64)"))
		return allErrors
	}

	flagSet := sets.New(flags...)

	if !flagSet.Has(CIDRLegacyValidation) {
		prefix, err := netip.ParsePrefix(value)
		if err != nil {
			// If netutils.ParseCIDRSloppy parsed it, but netip.ParsePrefix
			// doesn't, then it must have illegal leading 0s (either in the
			// IP part or the prefix).
			allErrors = append(allErrors, field.Invalid(fldPath, value, "must not have leading 0s in IP or prefix length"))
		}
		if prefix.Addr().Is4In6() {
			allErrors = append(allErrors, field.Invalid(fldPath, value, "must not have an IPv4-mapped IPv6 address"))
		}
	}

	// Skip checking CIDRIsCanonical if we got one of the above strict-parsing errors,
	// to avoid complaining about the same problem twice.
	if flagSet.Has(CIDRIsCanonical) && len(allErrors) == 0 {
		// (For subnet/mask CIDRs, we could just check "value != ipnet.String()",
		// but this code does the right thing with ifaddr CIDRs too.)
		maskSize, _ := ipnet.Mask.Size()
		canonical := fmt.Sprintf("%s/%d", ip.String(), maskSize)
		if value != canonical {
			allErrors = append(allErrors, field.Invalid(fldPath, value, fmt.Sprintf("must be in canonical form (%q)", canonical)))
		}
	}

	if !flagSet.Has(CIDRLegacyValidation) && !flagSet.Has(CIDRIsAddress) && !ip.Equal(ipnet.IP) {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must not have bits set beyond the prefix length"))
	}

	if flagSet.Has(CIDRIsIPv4) && !netutils.IsIPv4(ip) {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be an IPv4 CIDR"))
	}
	if flagSet.Has(CIDRIsIPv6) && !netutils.IsIPv6(ip) {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be an IPv6 CIDR"))
	}

	return allErrors
}

// GetWarningsForCIDR returns warnings for CIDR values in non-standard forms. This should
// only be used with fields that are validated with IsValidCIDR().
func GetWarningsForCIDR(fldPath *field.Path, value string) []string {
	ip, ipnet, err := netutils.ParseCIDRSloppy(value)
	if err != nil {
		klog.ErrorS(err, "GetWarningsForCIDR called on value that was not validated with IsValidCIDR", "field", fldPath, "value", value)
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
