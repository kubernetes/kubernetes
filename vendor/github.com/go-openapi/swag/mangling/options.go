// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package mangling

type (
	// PrefixFunc defines a safeguard rule (that may depend on the input string), to prefix
	// a generated go name (in [NameMangler.ToGoName] and [NameMangler.ToVarName]).
	//
	// See [NameMangler.ToGoName] for more about which edge cases the prefix function covers.
	PrefixFunc func(string) string

	// ReplaceFunc is a transliteration function to replace special runes by a word.
	ReplaceFunc func(r rune) (string, bool)

	// Option to configure a [NameMangler].
	Option func(*options)

	options struct {
		commonInitialisms []string

		goNamePrefixFunc    PrefixFunc
		goNamePrefixFuncPtr *PrefixFunc
		replaceFunc         func(r rune) (string, bool)
	}
)

func (o *options) prefixFunc() PrefixFunc {
	if o.goNamePrefixFuncPtr != nil && *o.goNamePrefixFuncPtr != nil {
		return *o.goNamePrefixFuncPtr
	}

	return o.goNamePrefixFunc
}

// WithGoNamePrefixFunc overrides the default prefix rule to safeguard generated go names.
//
// Example:
//
// This helps convert "123" into "{prefix}123" (a very crude strategy indeed, but it works).
//
// See [github.com/go-swagger/go-swagger/generator.DefaultFuncMap] for an example.
//
// The prefix function is assumed to return a string that starts with an upper case letter.
//
// The default is to prefix with "X".
//
// See [NameMangler.ToGoName] for more about which edge cases the prefix function covers.
func WithGoNamePrefixFunc(fn PrefixFunc) Option {
	return func(o *options) {
		o.goNamePrefixFunc = fn
	}
}

// WithGoNamePrefixFuncPtr is like [WithGoNamePrefixFunc] but it specifies a pointer to a function.
//
// [WithGoNamePrefixFunc] should be preferred in most situations. This option should only serve the
// purpose of handling special situations where the prefix function is not an internal variable
// (e.g. an exported package global).
//
// [WithGoNamePrefixFuncPtr] supersedes [WithGoNamePrefixFunc] if it also specified.
//
// If the provided pointer is nil or points to a nil value, this option has no effect.
//
// The caller should ensure that no undesirable concurrent changes are applied to the function pointed to.
func WithGoNamePrefixFuncPtr(ptr *PrefixFunc) Option {
	return func(o *options) {
		o.goNamePrefixFuncPtr = ptr
	}
}

// WithInitialisms declares the initialisms this mangler supports.
//
// This supersedes any pre-loaded defaults (see [DefaultInitialisms] for more about what initialisms are).
//
// It declares words to be recognized as "initialisms" (i.e. words that won't be camel cased or titled cased).
//
// Words must start with a (unicode) letter. If some don't, they are ignored.
// Words are either fully capitalized or mixed-cased. Lower-case only words are considered capitalized.
func WithInitialisms(words ...string) Option {
	return func(o *options) {
		o.commonInitialisms = words
	}
}

// WithAdditionalInitialisms adds new initialisms to the currently supported list (see [DefaultInitialisms]).
//
// The same sanitization rules apply as those described for [WithInitialisms].
func WithAdditionalInitialisms(words ...string) Option {
	return func(o *options) {
		o.commonInitialisms = append(o.commonInitialisms, words...)
	}
}

// WithReplaceFunc specifies a custom transliteration function instead of the default.
//
// The default translates the following characters into words as follows:
//
//   - '@' -> 'At'
//   - '&' -> 'And'
//   - '|' -> 'Pipe'
//   - '$' -> 'Dollar'
//   - '!' -> 'Bang'
//
// Notice that the outcome of a transliteration should always be titleized.
func WithReplaceFunc(fn ReplaceFunc) Option {
	return func(o *options) {
		o.replaceFunc = fn
	}
}

func defaultPrefixFunc(_ string) string {
	return "X"
}

// defaultReplaceTable finds a word representation for special characters.
func defaultReplaceTable(r rune) (string, bool) {
	switch r {
	case '@':
		return "At ", true
	case '&':
		return "And ", true
	case '|':
		return "Pipe ", true
	case '$':
		return "Dollar ", true
	case '!':
		return "Bang ", true
	case '-':
		return "", true
	case '_':
		return "", true
	default:
		return "", false
	}
}

func optionsWithDefaults(opts []Option) options {
	o := options{
		commonInitialisms: DefaultInitialisms(),
		goNamePrefixFunc:  defaultPrefixFunc,
		replaceFunc:       defaultReplaceTable,
	}

	for _, apply := range opts {
		apply(&o)
	}

	return o
}
