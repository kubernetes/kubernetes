// Package dnsutil contains higher-level methods useful with the dns
// package.  While package dns implements the DNS protocols itself,
// these functions are related but not directly required for protocol
// processing.  They are often useful in preparing input/output of the
// functions in package dns.
package dnsutil

import (
	"strings"

	"github.com/miekg/dns"
)

// AddDomain adds origin to s if s is not already a FQDN.
// Note that the result may not be a FQDN.  If origin does not end
// with a ".", the result won't either.
// This implements the zonefile convention (specified in RFC 1035,
// Section "5.1. Format") that "@" represents the
// apex (bare) domain. i.e. AddOrigin("@", "foo.com.") returns "foo.com.".
func AddOrigin(s, origin string) string {
	// ("foo.", "origin.") -> "foo." (already a FQDN)
	// ("foo", "origin.") -> "foo.origin."
	// ("foo"), "origin" -> "foo.origin"
	// ("@", "origin.") -> "origin." (@ represents the apex (bare) domain)
	// ("", "origin.") -> "origin." (not obvious)
	// ("foo", "") -> "foo" (not obvious)

	if dns.IsFqdn(s) {
		return s // s is already a FQDN, no need to mess with it.
	}
	if len(origin) == 0 {
		return s // Nothing to append.
	}
	if s == "@" || len(s) == 0 {
		return origin // Expand apex.
	}

	if origin == "." {
		return s + origin // AddOrigin(s, ".") is an expensive way to add a ".".
	}

	return s + "." + origin // The simple case.
}

// TrimDomainName trims origin from s if s is a subdomain.
// This function will never return "", but returns "@" instead (@ represents the apex (bare) domain).
func TrimDomainName(s, origin string) string {
	// An apex (bare) domain is always returned as "@".
	// If the return value ends in a ".", the domain was not the suffix.
	// origin can end in "." or not. Either way the results should be the same.

	if len(s) == 0 {
		return "@" // Return the apex (@) rather than "".
	}
	// Someone is using TrimDomainName(s, ".") to remove a dot if it exists.
	if origin == "." {
		return strings.TrimSuffix(s, origin)
	}

	// Dude, you aren't even if the right subdomain!
	if !dns.IsSubDomain(origin, s) {
		return s
	}

	slabels := dns.Split(s)
	olabels := dns.Split(origin)
	m := dns.CompareDomainName(s, origin)
	if len(olabels) == m {
		if len(olabels) == len(slabels) {
			return "@" // origin == s
		}
		if (s[0] == '.') && (len(slabels) == (len(olabels) + 1)) {
			return "@" // TrimDomainName(".foo.", "foo.")
		}
	}

	// Return the first (len-m) labels:
	return s[:slabels[len(slabels)-m]-1]
}
