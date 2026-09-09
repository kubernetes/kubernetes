// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package mangling

import (
	"strings"
	"unicode"
)

// NameMangler knows how to transform sentences or words into
// identifiers that are a better fit in contexts such as:
//
//   - unexported or exported go variable identifiers
//   - file names
//   - camel cased identifiers
//   - ...
//
// The [NameMangler] is safe for concurrent use, save for its [NameMangler.AddInitialisms] method,
// which is not.
//
// # Known limitations
//
// At this moment, the [NameMangler] doesn't play well with "all caps" text:
//
// unless every single upper-cased word is declared as an initialism, capitalized words would generally
// not be transformed with the expected result, e.g.
//
//	ToFileName("THIS_IS_ALL_CAPS")
//
// yields the weird outcome
//
//	"t_h_i_s_i_s_a_l_l_c_a_p_s"
type NameMangler struct {
	options

	index *indexOfInitialisms

	splitter              splitter
	splitterWithPostSplit splitter

	_ struct{}
}

// NewNameMangler builds a name mangler ready to convert strings.
//
// The default name mangler is configured with default common initialisms and all default options.
func NewNameMangler(opts ...Option) NameMangler {
	m := NameMangler{
		options: optionsWithDefaults(opts),
		index:   newIndexOfInitialisms(),
	}
	m.addInitialisms(m.commonInitialisms...)

	// a splitter that returns matches lexemes as ready-to-assemble strings:
	// details of the lexemes are redeemed.
	m.splitter = newSplitter(
		withInitialismsCache(&m.index.initialismsCache),
		withReplaceFunc(m.replaceFunc),
	)

	// a splitter that returns matches lexemes ready for post-processing
	m.splitterWithPostSplit = newSplitter(
		withInitialismsCache(&m.index.initialismsCache),
		withReplaceFunc(m.replaceFunc),
		withPostSplitInitialismCheck,
	)

	return m
}

// AddInitialisms declares extra initialisms to the mangler.
//
// It declares extra words as "initialisms" (i.e. words that won't be camel cased or titled cased),
// on top of the existing list of common initialisms (such as ID, HTTP...).
//
// Added words must start with a (unicode) letter. If some don't, they are ignored.
// Added words are either fully capitalized or mixed-cased. Lower-case only words are considered capitalized.
//
// It is typically used just after initializing the [NameMangler].
//
// When all initialisms are known at the time the mangler is initialized, it is preferable to
// use [NewNameMangler] with the option [WithAdditionalInitialisms].
//
// Adding initialisms mutates the mangler and should not be carried out concurrently with other calls to the mangler.
func (m *NameMangler) AddInitialisms(words ...string) {
	m.addInitialisms(words...)
}

// Initialisms renders the list of initialisms supported by this mangler.
func (m *NameMangler) Initialisms() []string {
	return m.index.initialisms
}

// Camelize a single word.
//
// Example:
//
//   - "HELLO" and "hello" become "Hello".
func (m NameMangler) Camelize(word string) string {
	ru := []rune(word)

	switch len(ru) {
	case 0:
		return ""
	case 1:
		return string(unicode.ToUpper(ru[0]))
	default:
		camelized := poolOfBuffers.BorrowBuffer(len(word))
		camelized.Grow(len(word))
		defer func() {
			poolOfBuffers.RedeemBuffer(camelized)
		}()

		camelized.WriteRune(unicode.ToUpper(ru[0]))
		for _, ru := range ru[1:] {
			camelized.WriteRune(unicode.ToLower(ru))
		}

		return camelized.String()
	}
}

// ToFileName generates a suitable snake-case file name from a sentence.
//
// It lower-cases everything with underscore (_) as a word separator.
//
// Examples:
//
//   - "Hello, Swagger" becomes "hello_swagger"
//   - "HelloSwagger" becomes "hello_swagger"
func (m NameMangler) ToFileName(name string) string {
	inptr := m.split(name)
	in := *inptr
	out := make([]string, 0, len(in))

	for _, w := range in {
		out = append(out, lower(w))
	}
	poolOfStrings.RedeemStrings(inptr)

	return strings.Join(out, "_")
}

// ToCommandName generates a suitable CLI command name from a sentence.
//
// It lower-cases everything with dash (-) as a word separator.
//
// Examples:
//
//   - "Hello, Swagger" becomes "hello-swagger"
//   - "HelloSwagger" becomes "hello-swagger"
func (m NameMangler) ToCommandName(name string) string {
	inptr := m.split(name)
	in := *inptr
	out := make([]string, 0, len(in))

	for _, w := range in {
		out = append(out, lower(w))
	}
	poolOfStrings.RedeemStrings(inptr)

	return strings.Join(out, "-")
}

// ToHumanNameLower represents a code name as a human-readable series of words.
//
// It lower-cases everything with blank space as a word separator.
//
// NOTE: parts recognized as initialisms just keep their original casing.
//
// Examples:
//
//   - "Hello, Swagger" becomes "hello swagger"
//   - "HelloSwagger" or "Hello-Swagger" become "hello swagger"
func (m NameMangler) ToHumanNameLower(name string) string {
	s := m.splitterWithPostSplit
	in := s.split(name)
	out := make([]string, 0, len(*in))

	for _, w := range *in {
		if !w.IsInitialism() {
			out = append(out, lower(w.GetOriginal()))
		} else {
			out = append(out, trim(w.GetOriginal()))
		}
	}

	poolOfLexems.RedeemLexems(in)

	return strings.Join(out, " ")
}

// ToHumanNameTitle represents a code name as a human-readable series of titleized words.
//
// It titleizes every word with blank space as a word separator.
//
// Examples:
//
//   - "hello, Swagger" becomes "Hello Swagger"
//   - "helloSwagger" becomes "Hello Swagger"
func (m NameMangler) ToHumanNameTitle(name string) string {
	s := m.splitterWithPostSplit
	in := s.split(name)

	out := make([]string, 0, len(*in))
	for _, w := range *in {
		original := trim(w.GetOriginal())
		if !w.IsInitialism() {
			out = append(out, m.Camelize(original))
		} else {
			out = append(out, original)
		}
	}
	poolOfLexems.RedeemLexems(in)

	return strings.Join(out, " ")
}

// ToJSONName generates a camelized single-word version of a sentence.
//
// The output assembles every camelized word, but for the first word, which
// is lower-cased.
//
// Example:
//
//   - "Hello_swagger" becomes "helloSwagger"
func (m NameMangler) ToJSONName(name string) string {
	inptr := m.split(name)
	in := *inptr
	out := make([]string, 0, len(in))

	for i, w := range in {
		if i == 0 {
			out = append(out, lower(w))
			continue
		}
		out = append(out, m.Camelize(trim(w)))
	}

	poolOfStrings.RedeemStrings(inptr)

	return strings.Join(out, "")
}

// ToVarName generates a legit unexported go variable name from a sentence.
//
// The generated name plays well with linters (see also [NameMangler.ToGoName]).
//
// Examples:
//
//   - "Hello_swagger" becomes "helloSwagger"
//   - "Http_server" becomes "httpServer"
//
// This name applies the same rules as [NameMangler.ToGoName] (legit exported variable), save the
// capitalization of the initial rune.
//
// Special case: when the initial part is a recognized as an initialism (like in the example above),
// the full part is lower-cased.
func (m NameMangler) ToVarName(name string) string {
	return m.goIdentifier(name, false)
}

// ToGoName generates a legit exported go variable name from a sentence.
//
// The generated name plays well with most linters.
//
// ToGoName abides by the go "exported" symbol rule starting with an upper-case letter.
//
// Examples:
//
//   - "hello_swagger" becomes "HelloSwagger"
//   - "Http_server" becomes "HTTPServer"
//
// # Edge cases
//
// Whenever the first rune is not eligible to upper case, a special prefix is prepended to the resulting name.
// By default this is simply "X" and you may customize this behavior using the [WithGoNamePrefixFunc] option.
//
// This happens when the first rune is not a letter, e.g. a digit, or a symbol that has no word transliteration
// (see also [WithReplaceFunc] about symbol transliterations),
// as well as for most East Asian or Devanagari runes, for which there is no such concept as upper-case.
//
// # Linting
//
// [revive], the successor of golint is the reference linter.
//
// This means that [NameMangler.ToGoName] supports the initialisms that revive checks (see also [DefaultInitialisms]).
//
// At this moment, there is no attempt to transliterate unicode into ascii, meaning that some linters
// (e.g. asciicheck, gosmopolitan) may croak on go identifiers generated from unicode input.
//
// [revive]: https://github.com/mgechev/revive
func (m NameMangler) ToGoName(name string) string {
	return m.goIdentifier(name, true)
}

func (m NameMangler) goIdentifier(name string, exported bool) string {
	s := m.splitterWithPostSplit
	lexems := s.split(name)
	defer func() {
		poolOfLexems.RedeemLexems(lexems)
	}()
	lexemes := *lexems

	if len(lexemes) == 0 {
		return ""
	}

	result := poolOfBuffers.BorrowBuffer(len(name))
	defer func() {
		poolOfBuffers.RedeemBuffer(result)
	}()

	firstPart := lexemes[0]
	if !exported {
		if ok := firstPart.WriteLower(result, true); !ok {
			// NOTE: an initialism as the first part is lower-cased: no longer generates stuff like hTTPxyz.
			//
			// same prefixing rule applied to unexported variable as to an exported one, so that we have consistent
			// names, whether the generated identifier is exported or not.
			result.WriteString(strings.ToLower(m.prefixFunc()(name)))
			result.WriteString(lexemes[0].GetOriginal())
		}
	} else {
		if ok := firstPart.WriteTitleized(result, true); !ok {
			// "repairs" a lexeme that doesn't start with a letter to become
			// the start a legit go name. The current strategy is very crude and simply adds a fixed prefix,
			// e.g. "X".
			// For instance "1_sesame_street" would be split into lexemes ["1", "sesame", "street"] and
			// the first one ("1") would result in something like "X1" (with the default prefix function).
			//
			// NOTE: no longer forcing the first part to be fully upper-cased
			result.WriteString(m.prefixFunc()(name))
			result.WriteString(lexemes[0].GetOriginal())
		}
	}

	for _, lexem := range lexemes[1:] {
		// NOTE: no longer forcing initialism parts to be fully upper-cased:
		// * pluralized initialism preserve their trailing "s"
		// * mixed-cased initialisms, such as IPv4, are preserved
		if ok := lexem.WriteTitleized(result, false); !ok {
			// it's not titleized: perhaps it's too short, perhaps the first rune is not a letter.
			// write anyway
			result.WriteString(lexem.GetOriginal())
		}
	}

	return result.String()
}

func (m *NameMangler) addInitialisms(words ...string) {
	m.index.add(words...)
	m.index.buildCache()
}

// split calls the inner splitter.
func (m NameMangler) split(str string) *[]string {
	s := m.splitter
	lexems := s.split(str)
	result := poolOfStrings.BorrowStrings()

	for _, lexem := range *lexems {
		*result = append(*result, lexem.GetOriginal())
	}
	poolOfLexems.RedeemLexems(lexems)

	return result
}
