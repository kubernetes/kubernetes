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
	"cmp"
	"hash/fnv"
	"iter"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"unique"
)

// normalizationHeader interns the sets of header fields that require specific normalization.
var normalizationHeader struct {
	sync.Once
	byQValue,
	byEncoding, byTimeInsensitive,
	byOrderInsensitive, byCaseInsensitive map[string]struct{}
}

func initNormalizationHeader() {
	normalizationHeader.Do(func() {
		normalizationHeader.byQValue = make(map[string]struct{})
		for _, field := range []string{
			"Accept",
			"Accept-Charset",
			"Accept-Language",
		} {
			normalizationHeader.byQValue[field] = struct{}{}
		}

		normalizationHeader.byEncoding = make(map[string]struct{})
		for _, field := range []string{
			"Content-Encoding",
			// The below can also accept quality values, see RFC 9110
			"Accept-Encoding",
			"TE",
		} {
			normalizationHeader.byEncoding[field] = struct{}{}
		}

		normalizationHeader.byTimeInsensitive = make(map[string]struct{})
		for _, field := range []string{
			"If-Modified-Since",
			"If-Unmodified-Since",
			"Date",
		} {
			normalizationHeader.byTimeInsensitive[field] = struct{}{}
		}

		normalizationHeader.byOrderInsensitive = make(map[string]struct{})
		for _, field := range []string{
			"Cache-Control",
			"Connection",
			"Content-Language",
			"Expect",
			"Pragma",
			"Upgrade",
			"Vary",
			"Via",
		} {
			normalizationHeader.byOrderInsensitive[field] = struct{}{}
		}

		normalizationHeader.byCaseInsensitive = make(map[string]struct{})
		for _, field := range []string{
			"Content-Type",
			"Content-Disposition",
			"Host",
			"Referer",
			"User-Agent",
			"Server",
			"Origin",
		} {
			normalizationHeader.byCaseInsensitive[field] = struct{}{}
		}
	})
}

func hasNormalizationHeader(m map[string]struct{}, field string) bool {
	_, ok := m[field]
	return ok
}

// normalizeHeaderValue normalizes a header value according to the rules defined
// in RFC 9111 §4.1. Assumes a canonicalized header field name.
func normalizeHeaderValue(field, value string) string {
	if value == "" {
		return ""
	}

	initNormalizationHeader()
	switch {
	case hasNormalizationHeader(normalizationHeader.byEncoding, field):
		value = normalizeEncodingHeader(value)
		if field == "Content-Encoding" {
			// No quality values.
			return value
		}
		fallthrough

	case hasNormalizationHeader(normalizationHeader.byQValue, field):
		return normalizeOrderInsensitiveWithQValues(value)

	case hasNormalizationHeader(normalizationHeader.byOrderInsensitive, field):
		return normalizeOrderInsensitive(value)

	case hasNormalizationHeader(normalizationHeader.byCaseInsensitive, field):
		return strings.ToLower(value)

	case hasNormalizationHeader(normalizationHeader.byTimeInsensitive, field):
		return strings.TrimSpace(value)

	case field == "Authorization":
		parts := strings.SplitN(value, " ", 2)
		if len(parts) == 2 {
			return strings.ToLower(parts[0]) + " " + parts[1]
		}
		return value

	default:
		return value
	}
}

// normalizeOrderInsensitive normalizes comma-separated values where order doesn't matter.
func normalizeOrderInsensitive(value string) string {
	parts := slices.Sorted(TrimmedCSVSeq(value))
	return strings.Join(parts, ",")
}

// Sentinel quality values for normalization.
var (
	maxQValue = unique.Make(1.0)
	minQValue = unique.Make(0.001)
)

// normalizeOrderInsensitiveWithQValues handles headers with quality values.
//
// References:
//   - https://www.rfc-editor.org/rfc/rfc9110.html#name-content-negotiation-field-f
//   - https://www.rfc-editor.org/rfc/rfc9110.html#name-content-negotiation-fields
func normalizeOrderInsensitiveWithQValues(value string) string {
	type qualityValue struct {
		main   string                 // the main value (e.g., "text/html")
		q      unique.Handle[float64] // the quality value (q=0.8); default is 1.0
		params []string               // any additional parameters (sorted)
	}
	parts := slices.Collect(TrimmedCSVSeq(value))
	qualityParts := make([]qualityValue, 0, len(parts))
outer:
	for i := range parts {
		part := parts[i]
		main, allParamsRaw, found := strings.Cut(part, ";")
		q := maxQValue
		params := make([]string, 0, 2)
		if found {
			for param := range strings.SplitSeq(allParamsRaw, ";") {
				param = strings.TrimSpace(param)
				switch {
				// q is case insensitive
				case len(param) > 2 && strings.EqualFold(param[:2], "q="):
					qRaw := param[2:]
					if qRaw == "0" || qRaw == "0.0" {
						continue outer // skip this part, as it has q=0
					}
					if qVal, err := strconv.ParseFloat(qRaw, 64); err == nil {
						q = unique.Make(
							min(max(qVal, minQValue.Value()), maxQValue.Value()),
						)
					}
				case param != "":
					params = append(params, param)
				}
			}
			slices.Sort(params)
		}
		qualityParts = append(qualityParts, qualityValue{
			main:   main,
			q:      q,
			params: params,
		})
	}

	// Sort by quality value, then by number of wildcards in main,
	// then lexicographically by main, then by number of parameters,
	// then lexicographically by parameters.
	slices.SortFunc(qualityParts, func(a, b qualityValue) int {
		return cmp.Or(
			cmp.Compare(b.q.Value(), a.q.Value()),
			cmp.Compare(strings.Count(a.main, "*"), strings.Count(b.main, "*")),
			cmp.Compare(a.main, b.main),
			cmp.Compare(len(b.params), len(a.params)),
			slices.Compare(a.params, b.params),
		)
	})

	// Keep first (highest ranked) for each main value
	qualityParts = slices.CompactFunc(qualityParts, func(a, b qualityValue) bool {
		return a.main == b.main
	})

	// Reconstruct
	var s strings.Builder
	for i, qp := range qualityParts {
		if i > 0 {
			s.WriteByte(',')
		}
		s.WriteString(qp.main)
		for _, p := range qp.params {
			s.WriteByte(';')
			s.WriteString(p)
		}
		if qp.q != maxQValue {
			s.WriteString(";q=")
			s.WriteString(formatQValue(qp.q.Value()))
		}
	}
	return s.String()
}

func formatQValue(q float64) string {
	str := strconv.FormatFloat(q, 'f', 3, 64)
	str = strings.TrimRight(str, "0")
	return strings.TrimSuffix(str, ".")
}

// Aliases for encoding names (see RFC 9110 §16.6.1).
//
// References:
//   - https://www.rfc-editor.org/rfc/rfc9110.html#name-content-coding
//   - https://datatracker.ietf.org/doc/html/rfc9110#name-te
//   - https://www.rfc-editor.org/rfc/rfc9112#section-7
var encodingReplacer = strings.NewReplacer(
	"x-gzip", "gzip",
	"x-compress", "compress",
)

// normalizeEncodingHeader handles special cases for encoding headers.
func normalizeEncodingHeader(value string) string {
	value = encodingReplacer.Replace(value)
	return normalizeOrderInsensitive(value)
}

// VaryHeaderNormalizer describes the interface implemented by types that can
// normalize request header field values given a Vary header field value, as per
// RFC 9111 §4.1.
type VaryHeaderNormalizer interface {
	NormalizeVaryHeader(vary string, reqHeader http.Header) iter.Seq2[string, string]
}

type VaryHeaderNormalizerFunc func(vary string, reqHeader http.Header) iter.Seq2[string, string]

func (f VaryHeaderNormalizerFunc) NormalizeVaryHeader(
	vary string,
	reqHeader http.Header,
) iter.Seq2[string, string] {
	return f(vary, reqHeader)
}
func NewVaryHeaderNormalizer() VaryHeaderNormalizer {
	return VaryHeaderNormalizerFunc(normalizeVaryHeaderSeq2)
}

func normalizeVaryHeaderSeq2(vary string, reqHeader http.Header) iter.Seq2[string, string] {
	return func(yield func(string, string) bool) {
		for name := range TrimmedCSVCanonicalSeq(vary) {
			values := reqHeader[name]
			value := ""
			// an empty value is valid and means "no variation"
			if len(values) > 0 {
				// NOTE: The policy of this cache is to use just the first header line
				value = normalizeHeaderValue(name, values[0])
			}
			if !yield(name, value) {
				return
			}
		}
	}
}

type HeaderValueNormalizer interface {
	NormalizeHeaderValue(field, value string) string
}

type HeaderValueNormalizerFunc func(field, value string) string

func (f HeaderValueNormalizerFunc) NormalizeHeaderValue(field, value string) string {
	return f(field, value)
}

func NewHeaderValueNormalizer() HeaderValueNormalizer {
	return HeaderValueNormalizerFunc(normalizeHeaderValue)
}

// VaryKeyer describes the interface implemented by types that can generate a
// unique key for a cached response based on the URL and Vary headers, according
// to RFC 9111 §4.1.
type VaryKeyer interface {
	VaryKey(urlKey string, varyHeaders map[string]string) string
}

type VaryKeyerFunc func(urlKey string, varyHeaders map[string]string) string

func (f VaryKeyerFunc) VaryKey(urlKey string, varyHeaders map[string]string) string {
	return f(urlKey, varyHeaders)
}

func NewVaryKeyer() VaryKeyer { return VaryKeyerFunc(makeVaryKey) }

const noVaryHash = "0"

func makeVaryKey(urlKey string, varyHeaders map[string]string) string {
	if len(varyHeaders) == 0 {
		return urlKey + "#" + noVaryHash // No Vary headers, so no variations
	}
	varyHash := makeVaryHash(varyHeaders)
	return urlKey + "#" + strconv.FormatUint(varyHash, 10)
}

func makeVaryHash(vary map[string]string) uint64 {
	h := fnv.New64a()
	keys := make([]string, 0, len(vary))
	keys = slices.AppendSeq(keys, maps.Keys(vary))
	slices.Sort(keys)
	for _, k := range keys {
		h.Write([]byte(k))
		h.Write([]byte(vary[k]))
	}
	return h.Sum64()
}
