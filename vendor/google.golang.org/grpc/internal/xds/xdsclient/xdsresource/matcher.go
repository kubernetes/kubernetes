/*
 *
 * Copyright 2020 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package xdsresource

import (
	"fmt"
	rand "math/rand/v2"
	"strings"

	"google.golang.org/grpc/internal/grpcutil"
	iresolver "google.golang.org/grpc/internal/resolver"
	"google.golang.org/grpc/internal/xds/matcher"
	"google.golang.org/grpc/metadata"
)

// RouteToMatcher converts a route to a Matcher to match incoming RPC's against.
//
// Only expected to be called on a Route that passed validation checks by the
// xDS client.
func RouteToMatcher(r *Route) *CompositeMatcher {
	var pm pathMatcher
	switch {
	case r.Regex != nil:
		pm = newPathRegexMatcher(r.Regex)
	case r.Path != nil:
		pm = newPathExactMatcher(*r.Path, r.CaseInsensitive)
	case r.Prefix != nil:
		pm = newPathPrefixMatcher(*r.Prefix, r.CaseInsensitive)
	default:
		panic("illegal route: missing path_matcher")
	}

	headerMatchers := make([]matcher.HeaderMatcher, 0, len(r.Headers))
	for _, h := range r.Headers {
		var matcherT matcher.HeaderMatcher
		invert := h.InvertMatch != nil && *h.InvertMatch
		switch {
		case h.ExactMatch != nil && *h.ExactMatch != "":
			matcherT = matcher.NewHeaderExactMatcher(h.Name, *h.ExactMatch, invert)
		case h.RegexMatch != nil:
			matcherT = matcher.NewHeaderRegexMatcher(h.Name, h.RegexMatch, invert)
		case h.PrefixMatch != nil && *h.PrefixMatch != "":
			matcherT = matcher.NewHeaderPrefixMatcher(h.Name, *h.PrefixMatch, invert)
		case h.SuffixMatch != nil && *h.SuffixMatch != "":
			matcherT = matcher.NewHeaderSuffixMatcher(h.Name, *h.SuffixMatch, invert)
		case h.RangeMatch != nil:
			matcherT = matcher.NewHeaderRangeMatcher(h.Name, h.RangeMatch.Start, h.RangeMatch.End, invert)
		case h.PresentMatch != nil:
			matcherT = matcher.NewHeaderPresentMatcher(h.Name, *h.PresentMatch, invert)
		case h.StringMatch != nil:
			matcherT = matcher.NewHeaderStringMatcher(h.Name, *h.StringMatch, invert)
		default:
			panic("illegal route: missing header_match_specifier")
		}
		headerMatchers = append(headerMatchers, matcherT)
	}

	var fractionMatcher *fractionMatcher
	if r.Fraction != nil {
		fractionMatcher = newFractionMatcher(*r.Fraction)
	}
	return newCompositeMatcher(pm, headerMatchers, fractionMatcher)
}

// CompositeMatcher is a matcher that holds onto many matchers and aggregates
// the matching results.
type CompositeMatcher struct {
	pm  pathMatcher
	hms []matcher.HeaderMatcher
	fm  *fractionMatcher
}

func newCompositeMatcher(pm pathMatcher, hms []matcher.HeaderMatcher, fm *fractionMatcher) *CompositeMatcher {
	return &CompositeMatcher{pm: pm, hms: hms, fm: fm}
}

// Match returns true if all matchers return true.
func (a *CompositeMatcher) Match(info iresolver.RPCInfo) bool {
	if a.pm != nil && !a.pm.match(info.Method) {
		return false
	}

	// Call headerMatchers even if md is nil, because routes may match
	// non-presence of some headers.
	var md metadata.MD
	if info.Context != nil {
		md, _ = metadata.FromOutgoingContext(info.Context)
		if extraMD, ok := grpcutil.ExtraMetadata(info.Context); ok {
			md = metadata.Join(md, extraMD)
			// Remove all binary headers. They are hard to match with. May need
			// to add back if asked by users.
			for k := range md {
				if strings.HasSuffix(k, "-bin") {
					delete(md, k)
				}
			}
		}
	}
	for _, m := range a.hms {
		if !m.Match(md) {
			return false
		}
	}

	if a.fm != nil && !a.fm.match() {
		return false
	}
	return true
}

func (a *CompositeMatcher) String() string {
	var ret string
	if a.pm != nil {
		ret += a.pm.String()
	}
	for _, m := range a.hms {
		ret += m.String()
	}
	if a.fm != nil {
		ret += a.fm.String()
	}
	return ret
}

type fractionMatcher struct {
	fraction int64 // real fraction is fraction/1,000,000.
}

func newFractionMatcher(fraction uint32) *fractionMatcher {
	return &fractionMatcher{fraction: int64(fraction)}
}

// RandInt64n overwrites rand for control in tests.
var RandInt64n = rand.Int64N

func (fm *fractionMatcher) match() bool {
	t := RandInt64n(1000000)
	return t <= fm.fraction
}

func (fm *fractionMatcher) String() string {
	return fmt.Sprintf("fraction:%v", fm.fraction)
}

type domainMatchType int

const (
	domainMatchTypeInvalid domainMatchType = iota
	domainMatchTypeUniversal
	domainMatchTypePrefix
	domainMatchTypeSuffix
	domainMatchTypeExact
)

// Exact > Suffix > Prefix > Universal > Invalid.
func (t domainMatchType) betterThan(b domainMatchType) bool {
	return t > b
}

func matchTypeForDomain(d string) domainMatchType {
	if d == "" {
		return domainMatchTypeInvalid
	}
	if d == "*" {
		return domainMatchTypeUniversal
	}
	if strings.HasPrefix(d, "*") {
		return domainMatchTypeSuffix
	}
	if strings.HasSuffix(d, "*") {
		return domainMatchTypePrefix
	}
	if strings.Contains(d, "*") {
		return domainMatchTypeInvalid
	}
	return domainMatchTypeExact
}

func match(domain, host string) (domainMatchType, bool) {
	switch typ := matchTypeForDomain(domain); typ {
	case domainMatchTypeInvalid:
		return typ, false
	case domainMatchTypeUniversal:
		return typ, true
	case domainMatchTypePrefix:
		// abc.*
		return typ, strings.HasPrefix(host, strings.TrimSuffix(domain, "*"))
	case domainMatchTypeSuffix:
		// *.123
		return typ, strings.HasSuffix(host, strings.TrimPrefix(domain, "*"))
	case domainMatchTypeExact:
		return typ, domain == host
	default:
		return domainMatchTypeInvalid, false
	}
}

// FindBestMatchingVirtualHost returns the virtual host whose domains field best
// matches host
//
//	The domains field support 4 different matching pattern types:
//
//	- Exact match
//	- Suffix match (e.g. “*ABC”)
//	- Prefix match (e.g. “ABC*)
//	- Universal match (e.g. “*”)
//
//	The best match is defined as:
//	- A match is better if it’s matching pattern type is better.
//	  * Exact match > suffix match > prefix match > universal match.
//
//	- If two matches are of the same pattern type, the longer match is
//	  better.
//	  * This is to compare the length of the matching pattern, e.g. “*ABCDE” >
//	    “*ABC”
func FindBestMatchingVirtualHost(host string, vHosts []*VirtualHost) *VirtualHost { // Maybe move this crap to client
	var (
		matchVh   *VirtualHost
		matchType = domainMatchTypeInvalid
		matchLen  int
	)
	for _, vh := range vHosts {
		for _, domain := range vh.Domains {
			typ, matched := match(domain, host)
			if typ == domainMatchTypeInvalid {
				// The rds response is invalid.
				return nil
			}
			if matchType.betterThan(typ) || matchType == typ && matchLen >= len(domain) || !matched {
				// The previous match has better type, or the previous match has
				// better length, or this domain isn't a match.
				continue
			}
			matchVh = vh
			matchType = typ
			matchLen = len(domain)
		}
	}
	return matchVh
}

// FindBestMatchingVirtualHostServer returns the virtual host whose domains field best
// matches authority.
func FindBestMatchingVirtualHostServer(authority string, vHosts []VirtualHostWithInterceptors) *VirtualHostWithInterceptors {
	var (
		matchVh   *VirtualHostWithInterceptors
		matchType = domainMatchTypeInvalid
		matchLen  int
	)
	for _, vh := range vHosts {
		for _, domain := range vh.Domains {
			typ, matched := match(domain, authority)
			if typ == domainMatchTypeInvalid {
				// The rds response is invalid.
				return nil
			}
			if matchType.betterThan(typ) || matchType == typ && matchLen >= len(domain) || !matched {
				// The previous match has better type, or the previous match has
				// better length, or this domain isn't a match.
				continue
			}
			matchVh = &vh
			matchType = typ
			matchLen = len(domain)
		}
	}
	return matchVh
}
