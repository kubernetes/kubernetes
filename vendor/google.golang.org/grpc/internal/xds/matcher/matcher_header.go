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
 *
 */

package matcher

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"google.golang.org/grpc/internal/grpcutil"
	"google.golang.org/grpc/metadata"
)

// HeaderMatcher is an interface for header matchers. These are
// documented in (EnvoyProxy link here?). These matchers will match on different
// aspects of HTTP header name/value pairs.
type HeaderMatcher interface {
	Match(metadata.MD) bool
	String() string
}

// valueFromMD retrieves metadata from context. If there are
// multiple values, the values are concatenated with "," (comma and no space).
//
// All header matchers only match against the comma-concatenated string.
func valueFromMD(md metadata.MD, key string) (string, bool) {
	vs, ok := md[key]
	if !ok {
		return "", false
	}
	return strings.Join(vs, ","), true
}

// HeaderExactMatcher matches on an exact match of the value of the header.
type HeaderExactMatcher struct {
	key    string
	exact  string
	invert bool
}

// NewHeaderExactMatcher returns a new HeaderExactMatcher.
func NewHeaderExactMatcher(key, exact string, invert bool) *HeaderExactMatcher {
	return &HeaderExactMatcher{key: key, exact: exact, invert: invert}
}

// Match returns whether the passed in HTTP Headers match according to the
// HeaderExactMatcher.
func (hem *HeaderExactMatcher) Match(md metadata.MD) bool {
	v, ok := valueFromMD(md, hem.key)
	if !ok {
		return false
	}
	return (v == hem.exact) != hem.invert
}

func (hem *HeaderExactMatcher) String() string {
	return fmt.Sprintf("headerExact:%v:%v", hem.key, hem.exact)
}

// HeaderRegexMatcher matches on whether the entire request header value matches
// the regex.
type HeaderRegexMatcher struct {
	key    string
	re     *regexp.Regexp
	invert bool
}

// NewHeaderRegexMatcher returns a new HeaderRegexMatcher.
func NewHeaderRegexMatcher(key string, re *regexp.Regexp, invert bool) *HeaderRegexMatcher {
	return &HeaderRegexMatcher{key: key, re: re, invert: invert}
}

// Match returns whether the passed in HTTP Headers match according to the
// HeaderRegexMatcher.
func (hrm *HeaderRegexMatcher) Match(md metadata.MD) bool {
	v, ok := valueFromMD(md, hrm.key)
	if !ok {
		return false
	}
	return grpcutil.FullMatchWithRegex(hrm.re, v) != hrm.invert
}

func (hrm *HeaderRegexMatcher) String() string {
	return fmt.Sprintf("headerRegex:%v:%v", hrm.key, hrm.re.String())
}

// HeaderRangeMatcher matches on whether the request header value is within the
// range. The header value must be an integer in base 10 notation.
type HeaderRangeMatcher struct {
	key        string
	start, end int64 // represents [start, end).
	invert     bool
}

// NewHeaderRangeMatcher returns a new HeaderRangeMatcher.
func NewHeaderRangeMatcher(key string, start, end int64, invert bool) *HeaderRangeMatcher {
	return &HeaderRangeMatcher{key: key, start: start, end: end, invert: invert}
}

// Match returns whether the passed in HTTP Headers match according to the
// HeaderRangeMatcher.
func (hrm *HeaderRangeMatcher) Match(md metadata.MD) bool {
	v, ok := valueFromMD(md, hrm.key)
	if !ok {
		return false
	}
	if i, err := strconv.ParseInt(v, 10, 64); err == nil && i >= hrm.start && i < hrm.end {
		return !hrm.invert
	}
	return hrm.invert
}

func (hrm *HeaderRangeMatcher) String() string {
	return fmt.Sprintf("headerRange:%v:[%d,%d)", hrm.key, hrm.start, hrm.end)
}

// HeaderPresentMatcher will match based on whether the header is present in the
// whole request.
type HeaderPresentMatcher struct {
	key     string
	present bool
}

// NewHeaderPresentMatcher returns a new HeaderPresentMatcher.
func NewHeaderPresentMatcher(key string, present bool, invert bool) *HeaderPresentMatcher {
	if invert {
		present = !present
	}
	return &HeaderPresentMatcher{key: key, present: present}
}

// Match returns whether the passed in HTTP Headers match according to the
// HeaderPresentMatcher.
func (hpm *HeaderPresentMatcher) Match(md metadata.MD) bool {
	vs, ok := valueFromMD(md, hpm.key)
	present := ok && len(vs) > 0 // TODO: Are we sure we need this len(vs) > 0?
	return present == hpm.present
}

func (hpm *HeaderPresentMatcher) String() string {
	return fmt.Sprintf("headerPresent:%v:%v", hpm.key, hpm.present)
}

// HeaderPrefixMatcher matches on whether the prefix of the header value matches
// the prefix passed into this struct.
type HeaderPrefixMatcher struct {
	key    string
	prefix string
	invert bool
}

// NewHeaderPrefixMatcher returns a new HeaderPrefixMatcher.
func NewHeaderPrefixMatcher(key string, prefix string, invert bool) *HeaderPrefixMatcher {
	return &HeaderPrefixMatcher{key: key, prefix: prefix, invert: invert}
}

// Match returns whether the passed in HTTP Headers match according to the
// HeaderPrefixMatcher.
func (hpm *HeaderPrefixMatcher) Match(md metadata.MD) bool {
	v, ok := valueFromMD(md, hpm.key)
	if !ok {
		return false
	}
	return strings.HasPrefix(v, hpm.prefix) != hpm.invert
}

func (hpm *HeaderPrefixMatcher) String() string {
	return fmt.Sprintf("headerPrefix:%v:%v", hpm.key, hpm.prefix)
}

// HeaderSuffixMatcher matches on whether the suffix of the header value matches
// the suffix passed into this struct.
type HeaderSuffixMatcher struct {
	key    string
	suffix string
	invert bool
}

// NewHeaderSuffixMatcher returns a new HeaderSuffixMatcher.
func NewHeaderSuffixMatcher(key string, suffix string, invert bool) *HeaderSuffixMatcher {
	return &HeaderSuffixMatcher{key: key, suffix: suffix, invert: invert}
}

// Match returns whether the passed in HTTP Headers match according to the
// HeaderSuffixMatcher.
func (hsm *HeaderSuffixMatcher) Match(md metadata.MD) bool {
	v, ok := valueFromMD(md, hsm.key)
	if !ok {
		return false
	}
	return strings.HasSuffix(v, hsm.suffix) != hsm.invert
}

func (hsm *HeaderSuffixMatcher) String() string {
	return fmt.Sprintf("headerSuffix:%v:%v", hsm.key, hsm.suffix)
}

// HeaderContainsMatcher matches on whether the header value contains the
// value passed into this struct.
type HeaderContainsMatcher struct {
	key      string
	contains string
	invert   bool
}

// NewHeaderContainsMatcher returns a new HeaderContainsMatcher. key is the HTTP
// Header key to match on, and contains is the value that the header should
// contain for a successful match. An empty contains string does not
// work, use HeaderPresentMatcher in that case.
func NewHeaderContainsMatcher(key string, contains string, invert bool) *HeaderContainsMatcher {
	return &HeaderContainsMatcher{key: key, contains: contains, invert: invert}
}

// Match returns whether the passed in HTTP Headers match according to the
// HeaderContainsMatcher.
func (hcm *HeaderContainsMatcher) Match(md metadata.MD) bool {
	v, ok := valueFromMD(md, hcm.key)
	if !ok {
		return false
	}
	return strings.Contains(v, hcm.contains) != hcm.invert
}

func (hcm *HeaderContainsMatcher) String() string {
	return fmt.Sprintf("headerContains:%v%v", hcm.key, hcm.contains)
}

// HeaderStringMatcher matches on whether the header value matches against the
// StringMatcher specified.
type HeaderStringMatcher struct {
	key           string
	stringMatcher StringMatcher
	invert        bool
}

// NewHeaderStringMatcher returns a new HeaderStringMatcher.
func NewHeaderStringMatcher(key string, sm StringMatcher, invert bool) *HeaderStringMatcher {
	return &HeaderStringMatcher{
		key:           key,
		stringMatcher: sm,
		invert:        invert,
	}
}

// Match returns whether the passed in HTTP Headers match according to the
// specified StringMatcher.
func (hsm *HeaderStringMatcher) Match(md metadata.MD) bool {
	v, ok := valueFromMD(md, hsm.key)
	if !ok {
		return false
	}
	return hsm.stringMatcher.Match(v) != hsm.invert
}

func (hsm *HeaderStringMatcher) String() string {
	return fmt.Sprintf("headerString:%v:%v", hsm.key, hsm.stringMatcher)
}
