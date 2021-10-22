// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen.go

// Package publicsuffix provides a public suffix list based on data from
// https://publicsuffix.org/
//
// A public suffix is one under which Internet users can directly register
// names. It is related to, but different from, a TLD (top level domain).
//
// "com" is a TLD (top level domain). Top level means it has no dots.
//
// "com" is also a public suffix. Amazon and Google have registered different
// siblings under that domain: "amazon.com" and "google.com".
//
// "au" is another TLD, again because it has no dots. But it's not "amazon.au".
// Instead, it's "amazon.com.au".
//
// "com.au" isn't an actual TLD, because it's not at the top level (it has
// dots). But it is an eTLD (effective TLD), because that's the branching point
// for domain name registrars.
//
// Another name for "an eTLD" is "a public suffix". Often, what's more of
// interest is the eTLD+1, or one more label than the public suffix. For
// example, browsers partition read/write access to HTTP cookies according to
// the eTLD+1. Web pages served from "amazon.com.au" can't read cookies from
// "google.com.au", but web pages served from "maps.google.com" can share
// cookies from "www.google.com", so you don't have to sign into Google Maps
// separately from signing into Google Web Search. Note that all four of those
// domains have 3 labels and 2 dots. The first two domains are each an eTLD+1,
// the last two are not (but share the same eTLD+1: "google.com").
//
// All of these domains have the same eTLD+1:
//  - "www.books.amazon.co.uk"
//  - "books.amazon.co.uk"
//  - "amazon.co.uk"
// Specifically, the eTLD+1 is "amazon.co.uk", because the eTLD is "co.uk".
//
// There is no closed form algorithm to calculate the eTLD of a domain.
// Instead, the calculation is data driven. This package provides a
// pre-compiled snapshot of Mozilla's PSL (Public Suffix List) data at
// https://publicsuffix.org/
package publicsuffix // import "golang.org/x/net/publicsuffix"

// TODO: specify case sensitivity and leading/trailing dot behavior for
// func PublicSuffix and func EffectiveTLDPlusOne.

import (
	"fmt"
	"net/http/cookiejar"
	"strings"
)

// List implements the cookiejar.PublicSuffixList interface by calling the
// PublicSuffix function.
var List cookiejar.PublicSuffixList = list{}

type list struct{}

func (list) PublicSuffix(domain string) string {
	ps, _ := PublicSuffix(domain)
	return ps
}

func (list) String() string {
	return version
}

// PublicSuffix returns the public suffix of the domain using a copy of the
// publicsuffix.org database compiled into the library.
//
// icann is whether the public suffix is managed by the Internet Corporation
// for Assigned Names and Numbers. If not, the public suffix is either a
// privately managed domain (and in practice, not a top level domain) or an
// unmanaged top level domain (and not explicitly mentioned in the
// publicsuffix.org list). For example, "foo.org" and "foo.co.uk" are ICANN
// domains, "foo.dyndns.org" and "foo.blogspot.co.uk" are private domains and
// "cromulent" is an unmanaged top level domain.
//
// Use cases for distinguishing ICANN domains like "foo.com" from private
// domains like "foo.appspot.com" can be found at
// https://wiki.mozilla.org/Public_Suffix_List/Use_Cases
func PublicSuffix(domain string) (publicSuffix string, icann bool) {
	lo, hi := uint32(0), uint32(numTLD)
	s, suffix, icannNode, wildcard := domain, len(domain), false, false
loop:
	for {
		dot := strings.LastIndex(s, ".")
		if wildcard {
			icann = icannNode
			suffix = 1 + dot
		}
		if lo == hi {
			break
		}
		f := find(s[1+dot:], lo, hi)
		if f == notFound {
			break
		}

		u := nodes[f] >> (nodesBitsTextOffset + nodesBitsTextLength)
		icannNode = u&(1<<nodesBitsICANN-1) != 0
		u >>= nodesBitsICANN
		u = children[u&(1<<nodesBitsChildren-1)]
		lo = u & (1<<childrenBitsLo - 1)
		u >>= childrenBitsLo
		hi = u & (1<<childrenBitsHi - 1)
		u >>= childrenBitsHi
		switch u & (1<<childrenBitsNodeType - 1) {
		case nodeTypeNormal:
			suffix = 1 + dot
		case nodeTypeException:
			suffix = 1 + len(s)
			break loop
		}
		u >>= childrenBitsNodeType
		wildcard = u&(1<<childrenBitsWildcard-1) != 0
		if !wildcard {
			icann = icannNode
		}

		if dot == -1 {
			break
		}
		s = s[:dot]
	}
	if suffix == len(domain) {
		// If no rules match, the prevailing rule is "*".
		return domain[1+strings.LastIndex(domain, "."):], icann
	}
	return domain[suffix:], icann
}

const notFound uint32 = 1<<32 - 1

// find returns the index of the node in the range [lo, hi) whose label equals
// label, or notFound if there is no such node. The range is assumed to be in
// strictly increasing node label order.
func find(label string, lo, hi uint32) uint32 {
	for lo < hi {
		mid := lo + (hi-lo)/2
		s := nodeLabel(mid)
		if s < label {
			lo = mid + 1
		} else if s == label {
			return mid
		} else {
			hi = mid
		}
	}
	return notFound
}

// nodeLabel returns the label for the i'th node.
func nodeLabel(i uint32) string {
	x := nodes[i]
	length := x & (1<<nodesBitsTextLength - 1)
	x >>= nodesBitsTextLength
	offset := x & (1<<nodesBitsTextOffset - 1)
	return text[offset : offset+length]
}

// EffectiveTLDPlusOne returns the effective top level domain plus one more
// label. For example, the eTLD+1 for "foo.bar.golang.org" is "golang.org".
func EffectiveTLDPlusOne(domain string) (string, error) {
	if strings.HasPrefix(domain, ".") || strings.HasSuffix(domain, ".") || strings.Contains(domain, "..") {
		return "", fmt.Errorf("publicsuffix: empty label in domain %q", domain)
	}

	suffix, _ := PublicSuffix(domain)
	if len(domain) <= len(suffix) {
		return "", fmt.Errorf("publicsuffix: cannot derive eTLD+1 for domain %q", domain)
	}
	i := len(domain) - len(suffix) - 1
	if domain[i] != '.' {
		return "", fmt.Errorf("publicsuffix: invalid public suffix %q for domain %q", suffix, domain)
	}
	return domain[1+strings.LastIndex(domain[:i], "."):], nil
}
