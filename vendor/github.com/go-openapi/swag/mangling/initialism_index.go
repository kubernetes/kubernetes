// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package mangling

import (
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

// DefaultInitialisms returns all the initialisms configured by default for this package.
//
// # Motivation
//
// Common initialisms are acronyms for which the ordinary camel-casing rules are altered and
// for which we retain the original case.
//
// This is largely specific to the go naming conventions enforced by golint (now revive).
//
// # Example
//
// In go, "id" is a good-looking identifier, but "Id" is not and "ID" is preferred
// (notice that this stems only from conventions: the go compiler accepts all of these).
//
// Similarly, we may use "http", but not "Http". In this case, "HTTP" is preferred.
//
// # Reference and customization
//
// The default list of these casing-style exceptions is taken from the [github.com/mgechev/revive] linter for go:
// https://github.com/mgechev/revive/blob/master/lint/name.go#L93
//
// There are a few additions to the original list, such as IPv4, IPv6 and OAI ("OpenAPI").
//
// For these additions, "IPv4" would be preferred to "Ipv4" or "IPV4", and "OAI" to "Oai"
//
// You may redefine this list entirely using the mangler option [WithInitialisms], or simply add extra definitions
// using [WithAdditionalInitialisms].
//
// # Mixed-case and plurals
//
// Notice that initialisms are not necessarily fully upper-cased: a mixed-case initialism indicates the preferred casing.
//
// Obviously, lower-case only initialisms do not make a lot of sense: if lower-case only initialisms are added,
// they will be considered fully capitalized.
//
// Plural forms use mixed case like "IDs". And so do values like "IPv4" or "IPv6".
//
// The [NameMangler] automatically detects simple plurals for words such as "IDs" or "APIs",
// so you don't need to configure these variants.
//
// At this moment, it doesn't support pluralization of terms that ends with an 's' (or 'S'), since there is
// no clear consensus on whether a word like DNS should be pluralized as DNSes or remain invariant.
// The [NameMangler] consider those invariant. Therefore DNSs or DNSes are not recognized as plurals for DNS.
//
// Besids, we don't want to support pluralization of terms which would otherwise conflict with another one,
// like "HTTPs" vs "HTTPS". All these should be considered invariant. Hence: "Https" matches "HTTPS" and
// "HTTPSS" is "HTTPS" followed by "S".
func DefaultInitialisms() []string {
	return []string{
		"ACL",
		"API",
		"ASCII",
		"CPU",
		"CSS",
		"DNS",
		"EOF",
		"GUID",
		"HTML",
		"HTTPS",
		"HTTP",
		"ID",
		"IP",
		"IPv4", // prefer the mixed case outcome IPv4 over the capitalized IPV4
		"IPv6", // prefer the mixed case outcome IPv6 over the capitalized IPV6
		"JSON",
		"LHS",
		"OAI",
		"QPS",
		"RAM",
		"RHS",
		"RPC",
		"SLA",
		"SMTP",
		"SQL",
		"SSH",
		"TCP",
		"TLS",
		"TTL",
		"UDP",
		"UI",
		"UID",
		"UUID",
		"URI",
		"URL",
		"UTF8",
		"VM",
		"XML",
		"XMPP",
		"XSRF",
		"XSS",
	}
}

type indexOfInitialisms struct {
	initialismsCache

	index map[string]struct{}
}

func newIndexOfInitialisms() *indexOfInitialisms {
	return &indexOfInitialisms{
		index: make(map[string]struct{}),
	}
}

func (m *indexOfInitialisms) add(words ...string) *indexOfInitialisms {
	for _, word := range words {
		// sanitization of injected words: trimmed from blanks, and must start with a letter
		trimmed := strings.TrimSpace(word)

		firstRune, _ := utf8.DecodeRuneInString(trimmed)
		if !unicode.IsLetter(firstRune) {
			continue
		}

		// Initialisms are case-sensitive. This means that we support mixed-case words.
		// However, if specified as a lower-case string, the initialism should be fully capitalized.
		if trimmed == strings.ToLower(trimmed) {
			m.index[strings.ToUpper(trimmed)] = struct{}{}

			continue
		}

		m.index[trimmed] = struct{}{}
	}
	return m
}

func (m *indexOfInitialisms) sorted() []string {
	result := make([]string, 0, len(m.index))
	for k := range m.index {
		result = append(result, k)
	}
	sort.Sort(sort.Reverse(byInitialism(result)))
	return result
}

func (m *indexOfInitialisms) buildCache() {
	m.build(m.sorted(), m.pluralForm)
}

// initialismsCache caches all needed pre-computed and converted initialism entries,
// in the desired resolution order.
type initialismsCache struct {
	initialisms           []string
	initialismsRunes      [][]rune
	initialismsUpperCased [][]rune // initialisms cached in their trimmed, upper-cased version
	initialismsPluralForm []pluralForm
}

func (c *initialismsCache) build(in []string, pluralfunc func(string) pluralForm) {
	c.initialisms = in
	c.initialismsRunes = asRunes(c.initialisms)
	c.initialismsUpperCased = asUpperCased(c.initialisms)
	c.initialismsPluralForm = asPluralForms(c.initialisms, pluralfunc)
}

// pluralForm denotes the kind of pluralization to be used for initialisms.
//
// At this moment, initialisms are either invariant or follow a simple plural form with an
// extra (lower case) "s".
type pluralForm uint8

const (
	notPlural pluralForm = iota
	invariantPlural
	simplePlural
)

func (f pluralForm) String() string {
	switch f {
	case notPlural:
		return "notPlural"
	case invariantPlural:
		return "invariantPlural"
	case simplePlural:
		return "simplePlural"
	default:
		return "<unknown>"
	}
}

// pluralForm indicates how we want to pluralize a given initialism.
//
// Besides configured invariant forms (like HTTP and HTTPS),
// an initialism is normally pluralized by adding a single 's', like in IDs.
//
// Initialisms ending with an 'S' or an 's' are configured as invariant (we don't
// support plural forms like CSSes or DNSes, however the mechanism could be extended to
// do just that).
func (m *indexOfInitialisms) pluralForm(key string) pluralForm {
	if _, ok := m.index[key]; !ok {
		return notPlural
	}

	if strings.HasSuffix(strings.ToUpper(key), "S") {
		return invariantPlural
	}

	if _, ok := m.index[key+"s"]; ok {
		return invariantPlural
	}

	if _, ok := m.index[key+"S"]; ok {
		return invariantPlural
	}

	return simplePlural
}

type byInitialism []string

func (s byInitialism) Len() int {
	return len(s)
}
func (s byInitialism) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Less specifies the order in which initialisms are prioritized:
// 1. match longest first
// 2. when equal length, match in reverse lexicographical order, lower case match comes first
func (s byInitialism) Less(i, j int) bool {
	if len(s[i]) != len(s[j]) {
		return len(s[i]) < len(s[j])
	}

	return s[i] < s[j]
}

func asRunes(in []string) [][]rune {
	out := make([][]rune, len(in))
	for i, initialism := range in {
		out[i] = []rune(initialism)
	}

	return out
}

func asUpperCased(in []string) [][]rune {
	out := make([][]rune, len(in))

	for i, initialism := range in {
		out[i] = []rune(upper(trim(initialism)))
	}

	return out
}

// asPluralForms bakes an index of pluralization support.
func asPluralForms(in []string, pluralFunc func(string) pluralForm) []pluralForm {
	out := make([]pluralForm, len(in))
	for i, initialism := range in {
		out[i] = pluralFunc(initialism)
	}

	return out
}
