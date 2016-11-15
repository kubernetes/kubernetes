// Package language defines languages that implement CLDR pluralization.
package language

import (
	"fmt"
	"strings"
)

// Language is a written human language.
type Language struct {
	// Tag uniquely identifies the language as defined by RFC 5646.
	//
	// Most language tags are a two character language code (ISO 639-1)
	// optionally followed by a dash and a two character country code (ISO 3166-1).
	// (e.g. en, pt-br)
	Tag string
	*PluralSpec
}

func (l *Language) String() string {
	return l.Tag
}

// MatchingTags returns the set of language tags that map to this Language.
// e.g. "zh-hans-cn" yields {"zh", "zh-hans", "zh-hans-cn"}
// BUG: This should be computed once and stored as a field on Language for efficiency,
//      but this would require changing how Languages are constructed.
func (l *Language) MatchingTags() []string {
	parts := strings.Split(l.Tag, "-")
	var prefix, matches []string
	for _, part := range parts {
		prefix = append(prefix, part)
		match := strings.Join(prefix, "-")
		matches = append(matches, match)
	}
	return matches
}

// Parse returns a slice of supported languages found in src or nil if none are found.
// It can parse language tags and Accept-Language headers.
func Parse(src string) []*Language {
	var langs []*Language
	start := 0
	for end, chr := range src {
		switch chr {
		case ',', ';', '.':
			tag := strings.TrimSpace(src[start:end])
			if spec := getPluralSpec(tag); spec != nil {
				langs = append(langs, &Language{NormalizeTag(tag), spec})
			}
			start = end + 1
		}
	}
	if start > 0 {
		tag := strings.TrimSpace(src[start:])
		if spec := getPluralSpec(tag); spec != nil {
			langs = append(langs, &Language{NormalizeTag(tag), spec})
		}
		return dedupe(langs)
	}
	if spec := getPluralSpec(src); spec != nil {
		langs = append(langs, &Language{NormalizeTag(src), spec})
	}
	return langs
}

func dedupe(langs []*Language) []*Language {
	found := make(map[string]struct{}, len(langs))
	deduped := make([]*Language, 0, len(langs))
	for _, lang := range langs {
		if _, ok := found[lang.Tag]; !ok {
			found[lang.Tag] = struct{}{}
			deduped = append(deduped, lang)
		}
	}
	return deduped
}

// MustParse is similar to Parse except it panics instead of retuning a nil Language.
func MustParse(src string) []*Language {
	langs := Parse(src)
	if len(langs) == 0 {
		panic(fmt.Errorf("unable to parse language from %q", src))
	}
	return langs
}

// Add adds support for a new language.
func Add(l *Language) {
	tag := NormalizeTag(l.Tag)
	pluralSpecs[tag] = l.PluralSpec
}

// NormalizeTag returns a language tag with all lower-case characters
// and dashes "-" instead of underscores "_"
func NormalizeTag(tag string) string {
	tag = strings.ToLower(tag)
	return strings.Replace(tag, "_", "-", -1)
}
