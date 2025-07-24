// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"iter"
	"maps"
	"net/http"
	"net/textproto"
	"strconv"
	"strings"
	"time"
)

// RawTime is a string that represents a time in HTTP date format.
type RawTime string

// Value returns the time and a boolean indicating whether the result is valid.
func (r RawTime) Value() (t time.Time, valid bool) {
	if r == "" {
		return
	}
	parsedTime, err := http.ParseTime(string(r))
	if err != nil {
		return
	}
	return parsedTime, true
}

// RawDeltaSeconds is a string that represents a delta time in seconds,
// as defined in §1.2.2 of RFC 9111.
//
// This implementation supports values up to the maximum range of int64
// (9223372036854775807 seconds). Values exceeding 2147483648 (2^31) are
// valid and will not be capped, as allowed by the RFC, which permits
// using the greatest positive integer the implementation can represent.
type RawDeltaSeconds string

func (r RawDeltaSeconds) Value() (dur time.Duration, valid bool) {
	if len(r) == 0 || r[0] == '-' {
		return
	}
	seconds, err := strconv.ParseInt(string(r), 10, 64)
	if err != nil {
		return
	}

	return time.Duration(seconds) * time.Second, true
}

// RawCSVSeq is a string that represents a sequence of comma-separated values.
type RawCSVSeq string

// Value returns an iterator over the raw comma-separated string and a boolean indicating
// whether the result is valid.
func (s RawCSVSeq) Value() (seq iter.Seq[string], valid bool) {
	if len(s) == 0 {
		return
	}
	return TrimmedCSVSeq(string(s)), true
}

// directivesSeq2 returns an iterator over all key-value pairs in a string of
// cache directives (as specified in 9111, §5.2.1 and 5.2.2). The
// iterator yields the key (token) and value (argument) of each directive.
//
// It guarantees that the key is always non-empty, and if a value is not
// present, it yields an empty string as the value.
func directivesSeq2(s string) iter.Seq2[string, string] {
	return func(yield func(string, string) bool) {
		for part := range TrimmedCSVSeq(s) {
			key, value, found := strings.Cut(part, "=")
			if !found {
				key = textproto.TrimString(part)
				value = ""
			} else {
				// value = textproto.TrimString(ParseQuotedString(value))
				value = textproto.TrimString(value)
			}
			if len(key) == 0 {
				continue
			}
			if !yield(key, value) {
				return
			}
		}
	}
}

// parseDirectives parses a string of cache directives and returns a map
// where the keys are the directive names and the values are the arguments.
func parseDirectives(s string) map[string]string {
	return maps.Collect(directivesSeq2(s))
}

func hasToken(d map[string]string, token string) bool {
	_, ok := d[token]
	return ok
}

func getDurationDirective(d map[string]string, token string) (dur time.Duration, valid bool) {
	if v, ok := d[token]; ok {
		return RawDeltaSeconds(v).Value()
	}
	return
}

// CCRequestDirectives is a map of request directives from the Cache-Control
// header field as defined in RFC 9111, §5.2.1. The keys are the directive tokens,
// and the values are the arguments (if any) as strings.
//
// This implementation does not perform any transformations on the request,
// hence we ignore the "no-transform" directive.
type CCRequestDirectives map[string]string

func ParseCCRequestDirectives(header http.Header) CCRequestDirectives {
	value := header.Get("Cache-Control")
	if value == "" {
		return nil
	}
	return parseDirectives(value)
}

// MaxAge parses the "max-age" request directive as defined in RFC 9111, §5.2.1.1.
func (d CCRequestDirectives) MaxAge() (dur time.Duration, valid bool) {
	return getDurationDirective(d, "max-age")
}

// MaxStale parses the "max-stale" request directive as defined in RFC 9111, §5.2.1.2.
func (d CCRequestDirectives) MaxStale() (dur RawDeltaSeconds, valid bool) {
	if v, ok := d["max-stale"]; ok {
		return RawDeltaSeconds(v), true
	}
	return
}

// MinFresh parses the "min-fresh" request directive as defined in RFC 9111, §5.2.1.3.
func (d CCRequestDirectives) MinFresh() (dur time.Duration, valid bool) {
	return getDurationDirective(d, "min-fresh")
}

// NoCache reports the presence of the "no-cache" request directive as defined in RFC 9111, §5.2.1.4.
func (d CCRequestDirectives) NoCache() bool {
	return hasToken(d, "no-cache")
}

// NoStore reports the presence of the "no-store" request directive as defined in RFC 9111, §5.2.1.5.
func (d CCRequestDirectives) NoStore() bool {
	return hasToken(d, "no-store")
}

// OnlyIfCached reports the presence of the "only-if-cached" request directive as defined in RFC 9111, §5.2.1.7.
func (d CCRequestDirectives) OnlyIfCached() bool {
	return hasToken(d, "only-if-cached")
}

// StaleIfError parses the "stale-if-error" request directive (extension) as defined in RFC 5861, §4.
func (d CCRequestDirectives) StaleIfError() (dur time.Duration, valid bool) {
	return getDurationDirective(d, "stale-if-error")
}

// CCResponseDirectives is a map of response directives from the Cache-Control
// header field. The keys are the directive tokens, and the values are the arguments (if any)
// as strings.
//
// The following directives per RFC 9111, §5.2.2 are not applicable to private caches:
//   - "private" (§5.2.2.7)
//   - "proxy-revalidate" (§5.2.2.8)
//   - "s-maxage" (§5.2.2.10)
//
// This implementation does not perform any transformations on the request,
// hence we ignore the "no-transform" directive.
type CCResponseDirectives map[string]string

func ParseCCResponseDirectives(header http.Header) CCResponseDirectives {
	value := header.Get("Cache-Control")
	if value == "" {
		return nil
	}
	return parseDirectives(value)
}

// MaxAge parses the "max-age" response directive as defined in RFC 9111, §5.2.2.1.
func (d CCResponseDirectives) MaxAge() (dur time.Duration, valid bool) {
	return getDurationDirective(d, "max-age")
}

// MaxAgePresent reports the presence of the "max-age" response directive as defined in RFC 9111, §5.2.2.1.
func (d CCResponseDirectives) MaxAgePresent() bool {
	return hasToken(d, "max-age")
}

// MustRevalidate reports the presence of the "must-revalidate" response directive as defined in RFC 9111, §5.2.2.2.
func (d CCResponseDirectives) MustRevalidate() bool {
	return hasToken(d, "must-revalidate")
}

// MustUnderstand reports the presence of the "must-understand" response directive as defined in RFC 9111, §5.2.2.3.
func (d CCResponseDirectives) MustUnderstand() bool {
	return hasToken(d, "must-understand")
}

// NoCache parses the "no-cache" response directive as defined in RFC 9111, §5.2.2.4.
func (d CCResponseDirectives) NoCache() (fields RawCSVSeq, present bool) {
	v, ok := d["no-cache"]
	if !ok {
		return
	}
	return RawCSVSeq(ParseQuotedString(v)), true
}

// NoStore reports the presence of the "no-store" response directive as defined in RFC 9111, §5.2.2.5.
func (d CCResponseDirectives) NoStore() bool {
	return hasToken(d, "no-store")
}

// Public reports the presence of the "public" response directive as defined in RFC 9111, §5.2.2.9.
func (d CCResponseDirectives) Public() bool {
	return hasToken(d, "public")
}

// StaleIfError parses the "stale-if-error" response directive (extension) as defined in RFC 5861, §4.
func (d CCResponseDirectives) StaleIfError() (dur time.Duration, valid bool) {
	return getDurationDirective(d, "stale-if-error")
}

// StaleWhileRevalidate parses the "stale-while-revalidate" response directive (extension) as defined in RFC 5861, §3.
func (d CCResponseDirectives) StaleWhileRevalidate() (dur time.Duration, valid bool) {
	return getDurationDirective(d, "stale-while-revalidate")
}

// Immutable reports the presence of the "immutable" response directive (extension) as defined in RFC 8246, §2.
func (d CCResponseDirectives) Immutable() bool {
	return hasToken(d, "immutable")
}
