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
	"net/netip"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	netutils "k8s.io/utils/net"
)

// IsValidIP tests that the argument is a valid IP address according to current Kubernetes
// validation rules. In particular:
//
//  1. IPv4 IPs must not have any leading "0"s in octets (e.g. "010.002.003.004"). (More
//     generally, they must use the ordinary "decimal dotted quad notation", not any of
//     the weirder IPv4 formats historically supported by inet_aton().) Historically,
//     net.ParseIP (and later netutils.ParseIPSloppy) simply ignored leading "0"s in IPv4
//     addresses, but most libc-based software treats 0-prefixed IPv4 octets as octal,
//     meaning different software might interpret the same string as two different IPs,
//     potentially leading to security issues. (Current net.ParseIP and netip.ParseAddr
//     simply reject inputs with leading "0"s.)
//
//  2. IPv4-mapped IPv6 IPs (e.g. "::ffff:1.2.3.4") are not allowed, because they may be
//     treated as IPv4 by some software and IPv6 by other software, again potentially
//     leading to different interpretations of the same values. (net.ParseIP and
//     netip.ParseAddr both allow these, but there are no use cases for representing IPv4
//     addresses as IPv4-mapped IPv6 addresses in Kubernetes.)
//
//  3. IPs may not have a trailing zone identifier (e.g. "fe80::1234%eth0"). (This syntax
//     is accepted by netip.ParseAddr, but not by net.ParseIP/netutils.ParseIPSloppy.)
//
// Any address that is valid according to IsValidIP will be accepted, and interpreted
// identically, by net.ParseIP, netutils.ParseIPSloppy, netip.ParseAddr, and equivalent
// APIs in other languages.
//
// Note that objects created before Kubernetes 1.32 may contain IP addresses that are not
// considered valid according to all of these rules. When validating an update of such an
// object, you must not require old IP-valued fields to revalidate if they are not being
// changed.
func IsValidIP(fldPath *field.Path, value string) field.ErrorList {
	_, allErrors := isValidIPInternal(fldPath, value)
	return allErrors
}

func isValidIPInternal(fldPath *field.Path, value string) (netip.Addr, field.ErrorList) {
	var allErrors field.ErrorList
	if value == "" {
		allErrors = append(allErrors, field.Required(fldPath, "expected an IP address"))
		return netip.Addr{}, allErrors
	}

	ip, err := netip.ParseAddr(value)
	if err != nil {
		// netip.ParseAddr returns very technical/low-level error messages so we
		// ignore err itself.

		if ip := netutils.ParseIPSloppy(value); ip != nil {
			allErrors = append(allErrors, field.Invalid(fldPath, value, "IP address cannot have leading 0s"))
			return netip.Addr{}, allErrors
		}
		if _, err := netip.ParsePrefix(value); err == nil {
			allErrors = append(allErrors, field.Invalid(fldPath, value, "expected an IP address (e.g. 10.9.8.7 or 2001:db8::ffff), got CIDR value"))
			return netip.Addr{}, allErrors
		}

		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be a valid IP address (e.g. 10.9.8.7 or 2001:db8::ffff)"))
		return netip.Addr{}, allErrors
	}

	if ip.Zone() != "" {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "IP address with zone specifier is not allowed"))
	}
	if ip.Is4In6() {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "IPv4-mapped IPv6 address is not allowed"))
	}

	return ip, allErrors
}

// IsValidIPv4Address tests that the argument is a valid IPv4 address.
func IsValidIPv4Address(fldPath *field.Path, value string) field.ErrorList {
	ip, allErrors := isValidIPInternal(fldPath, value)
	// Only check this if there were no other errors, to avoid false positives
	if len(allErrors) == 0 && !ip.Is4() {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be an IPv4 address"))
	}
	return allErrors
}

// IsValidIPv6Address tests that the argument is a valid IPv6 address.
func IsValidIPv6Address(fldPath *field.Path, value string) field.ErrorList {
	ip, allErrors := isValidIPInternal(fldPath, value)
	// Only check this if there were no other errors, to avoid false positives
	if len(allErrors) == 0 && !ip.Is6() {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be an IPv6 address"))
	}
	return allErrors
}

// IsValidImmutableIPUpdate returns true if we should allow changing an "immutable" IP
// address field from oldValue to newValue. This is allowed so that objects with fields
// that historically allowed ambiguous/unsafe IP address syntax can be fixed to use
// proper syntax instead.
//
// So, for example, this would allow updating:
//
//   - "010.001.002.003" to "10.1.2.3"
//   - "::ffff:192.168.0.3" to "192.168.0.3"
func IsValidImmutableIPUpdate(oldValue, newValue string) bool {
	// We allow "changing" a value to itself
	if oldValue == newValue {
		return true
	}

	// Otherwise, oldValue must be invalid, and newValue must be valid
	if errs := IsValidIP(nil, oldValue); len(errs) == 0 {
		return false
	}
	if errs := IsValidIP(nil, newValue); len(errs) != 0 {
		return false
	}

	// oldValue must round-trip to newValue
	oldIP := netutils.ParseIPSloppy(oldValue)
	return oldIP != nil && oldIP.String() == newValue
}

// IsValidCIDR tests that the argument is a valid CIDR string according to current
// Kubernetes validation rules. In particular:
//
//  1. The IP part of the CIDR string must be valid according to IsValidIP.
//
//  2. The CIDR value must describe a "subnet" or "mask" (with the lower bits after the
//     prefix length all set to 0), not an "address". net.ParseCIDR and netip.ParsePrefix
//     both accept strings such as "192.168.1.5/24", because such strings are often used
//     to describe the IPs assigned to network interfaces (where "192.168.1.5/24" means
//     "the IP 192.168.1.5 on the subnet 192.168.1.0/24"). But there are no Kubernetes API
//     objects that expect IP addresses represented in that form, and passing values in
//     that form to an external API that was expecting a subnet or a mask value would have
//     undefined results. (E.g., "192.168.1.5/24" might be treated as meaning either
//     "192.168.1.0/24" or "192.168.1.5/32" depending on how it gets processed.)
//
//  3. The prefix length must consist only of numbers, and not have any leading "0"s.
//     (net.ParseCIDR and netutils.ParseCIDRSloppy allow leading "0s" (e.g.
//     "192.168.0.0/016"), and net.ParsePrefix in golang 1.21 and earlier allowed both
//     leading "0"s and a leading "+".)
//
// Any CIDR value that is valid according to IsValidCIDR will be accepted, and interpreted
// identically, by net.ParseCIDR, netutils.ParseCIDRSloppy, netip.ParsePrefix, and
// equivalent APIs in other languages.
//
// Note that objects created before Kubernetes 1.32 may contain CIDR values that are not
// considered valid according to all of these rules. When validating an update of such an
// object, you must not require old CIDR-valued fields to revalidate if they are not being
// changed.
func IsValidCIDR(fldPath *field.Path, value string) field.ErrorList {
	var allErrors field.ErrorList
	if value == "" {
		allErrors = append(allErrors, field.Required(fldPath, "expected a CIDR value"))
		return allErrors
	}

	cidr, err := netip.ParsePrefix(value)
	if err != nil {
		// netip.ParsePrefix returns very technical/low-level error messages
		// so we ignore err itself.

		if _, _, err := netutils.ParseCIDRSloppy(value); err == nil {
			// If ParseCIDRSloppy accepted it but ParsePrefix did not
			// then the problem must be leading 0s.
			if strings.Contains(value, "/0") && !strings.HasSuffix(value, "/0") {
				allErrors = append(allErrors, field.Invalid(fldPath, value, "prefix length in CIDR cannot have leading 0s"))
			} else {
				allErrors = append(allErrors, field.Invalid(fldPath, value, "IP address in CIDR cannot have leading 0s"))
			}
			return allErrors
		}
		if _, err := netip.ParseAddr(value); err == nil {
			allErrors = append(allErrors, field.Invalid(fldPath, value, "expected a valid CIDR value (e.g. 10.9.8.0/24 or 2001:db8::/64), got IP address"))
			return allErrors
		}

		allErrors = append(allErrors, field.Invalid(fldPath, value, "must be a valid CIDR value (e.g. 10.9.8.0/24 or 2001:db8::/64)"))
		return allErrors
	}

	// (Unlike netip.ParseAddr, netip.ParsePrefix doesn't allow IPv6 zones, so
	// we don't have to check for that.)

	if cidr.Addr().Is4In6() {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "CIDR containing IPv4-mapped IPv6 address is not allowed"))
	} else if cidr != cidr.Masked() {
		allErrors = append(allErrors, field.Invalid(fldPath, value, "CIDR value should not have any bits set beyond the prefix length"))
	}

	return allErrors
}

// IsValidImmutableCIDRUpdate returns true if we should allow changing an "immutable" CIDR
// field from oldValue to newValue. This is allowed so that objects with fields that
// historically allowed ambiguous/unsafe CIDR syntax can be fixed to use proper syntax
// instead.
//
// So, for example, this would allow updating:
//
//   - "010.001.002.000/24" to "10.1.2.0/24"
//   - "::ffff:192.168.0.0/112" to "192.168.0.0/16"
//   - "1.2.3.4/24" to "1.2.3.0/24"
//   - "2001:db8::/064" to "2001:db8::/64"
func IsValidImmutableCIDRUpdate(oldValue, newValue string) bool {
	// We allow "changing" a value to itself
	if oldValue == newValue {
		return true
	}

	// Otherwise, oldValue must be invalid, and newValue must be valid
	if errs := IsValidCIDR(nil, oldValue); len(errs) == 0 {
		return false
	}
	if errs := IsValidCIDR(nil, newValue); len(errs) != 0 {
		return false
	}

	// oldValue must round-trip to newValue
	_, oldCIDR, _ := netutils.ParseCIDRSloppy(oldValue)
	return oldCIDR != nil && oldCIDR.String() == newValue
}
