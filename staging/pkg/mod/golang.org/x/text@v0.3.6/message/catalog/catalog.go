// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package catalog defines collections of translated format strings.
//
// This package mostly defines types for populating catalogs with messages. The
// catmsg package contains further definitions for creating custom message and
// dictionary types as well as packages that use Catalogs.
//
// Package catalog defines various interfaces: Dictionary, Loader, and Message.
// A Dictionary maintains a set of translations of format strings for a single
// language. The Loader interface defines a source of dictionaries. A
// translation of a format string is represented by a Message.
//
//
// Catalogs
//
// A Catalog defines a programmatic interface for setting message translations.
// It maintains a set of per-language dictionaries with translations for a set
// of keys. For message translation to function properly, a translation should
// be defined for each key for each supported language. A dictionary may be
// underspecified, though, if there is a parent language that already defines
// the key. For example, a Dictionary for "en-GB" could leave out entries that
// are identical to those in a dictionary for "en".
//
//
// Messages
//
// A Message is a format string which varies on the value of substitution
// variables. For instance, to indicate the number of results one could want "no
// results" if there are none, "1 result" if there is 1, and "%d results" for
// any other number. Catalog is agnostic to the kind of format strings that are
// used: for instance, messages can follow either the printf-style substitution
// from package fmt or use templates.
//
// A Message does not substitute arguments in the format string. This job is
// reserved for packages that render strings, such as message, that use Catalogs
// to selected string. This separation of concerns allows Catalog to be used to
// store any kind of formatting strings.
//
//
// Selecting messages based on linguistic features of substitution arguments
//
// Messages may vary based on any linguistic features of the argument values.
// The most common one is plural form, but others exist.
//
// Selection messages are provided in packages that provide support for a
// specific linguistic feature. The following snippet uses plural.Selectf:
//
//   catalog.Set(language.English, "You are %d minute(s) late.",
//       plural.Selectf(1, "",
//           plural.One, "You are 1 minute late.",
//           plural.Other, "You are %d minutes late."))
//
// In this example, a message is stored in the Catalog where one of two messages
// is selected based on the first argument, a number. The first message is
// selected if the argument is singular (identified by the selector "one") and
// the second message is selected in all other cases. The selectors are defined
// by the plural rules defined in CLDR. The selector "other" is special and will
// always match. Each language always defines one of the linguistic categories
// to be "other." For English, singular is "one" and plural is "other".
//
// Selects can be nested. This allows selecting sentences based on features of
// multiple arguments or multiple linguistic properties of a single argument.
//
//
// String interpolation
//
// There is often a lot of commonality between the possible variants of a
// message. For instance, in the example above the word "minute" varies based on
// the plural catogory of the argument, but the rest of the sentence is
// identical. Using interpolation the above message can be rewritten as:
//
//   catalog.Set(language.English, "You are %d minute(s) late.",
//       catalog.Var("minutes",
//           plural.Selectf(1, "", plural.One, "minute", plural.Other, "minutes")),
//       catalog.String("You are %[1]d ${minutes} late."))
//
// Var is defined to return the variable name if the message does not yield a
// match. This allows us to further simplify this snippet to
//
//   catalog.Set(language.English, "You are %d minute(s) late.",
//       catalog.Var("minutes", plural.Selectf(1, "", plural.One, "minute")),
//       catalog.String("You are %d ${minutes} late."))
//
// Overall this is still only a minor improvement, but things can get a lot more
// unwieldy if more than one linguistic feature is used to determine a message
// variant. Consider the following example:
//
//   // argument 1: list of hosts, argument 2: list of guests
//   catalog.Set(language.English, "%[1]v invite(s) %[2]v to their party.",
//     catalog.Var("their",
//         plural.Selectf(1, ""
//             plural.One, gender.Select(1, "female", "her", "other", "his"))),
//     catalog.Var("invites", plural.Selectf(1, "", plural.One, "invite"))
//     catalog.String("%[1]v ${invites} %[2]v to ${their} party.")),
//
// Without variable substitution, this would have to be written as
//
//   // argument 1: list of hosts, argument 2: list of guests
//   catalog.Set(language.English, "%[1]v invite(s) %[2]v to their party.",
//     plural.Selectf(1, "",
//         plural.One, gender.Select(1,
//             "female", "%[1]v invites %[2]v to her party."
//             "other", "%[1]v invites %[2]v to his party."),
//         plural.Other, "%[1]v invites %[2]v to their party.")
//
// Not necessarily shorter, but using variables there is less duplication and
// the messages are more maintenance friendly. Moreover, languages may have up
// to six plural forms. This makes the use of variables more welcome.
//
// Different messages using the same inflections can reuse variables by moving
// them to macros. Using macros we can rewrite the message as:
//
//   // argument 1: list of hosts, argument 2: list of guests
//   catalog.SetString(language.English, "%[1]v invite(s) %[2]v to their party.",
//       "%[1]v ${invites(1)} %[2]v to ${their(1)} party.")
//
// Where the following macros were defined separately.
//
//   catalog.SetMacro(language.English, "invites", plural.Selectf(1, "",
//      plural.One, "invite"))
//   catalog.SetMacro(language.English, "their", plural.Selectf(1, "",
//      plural.One, gender.Select(1, "female", "her", "other", "his"))),
//
// Placeholders use parentheses and the arguments to invoke a macro.
//
//
// Looking up messages
//
// Message lookup using Catalogs is typically only done by specialized packages
// and is not something the user should be concerned with. For instance, to
// express the tardiness of a user using the related message we defined earlier,
// the user may use the package message like so:
//
//   p := message.NewPrinter(language.English)
//   p.Printf("You are %d minute(s) late.", 5)
//
// Which would print:
//   You are 5 minutes late.
//
//
// This package is UNDER CONSTRUCTION and its API may change.
package catalog // import "golang.org/x/text/message/catalog"

// TODO:
// Some way to freeze a catalog.
// - Locking on each lockup turns out to be about 50% of the total running time
//   for some of the benchmarks in the message package.
// Consider these:
// - Sequence type to support sequences in user-defined messages.
// - Garbage collection: Remove dictionaries that can no longer be reached
//   as other dictionaries have been added that cover all possible keys.

import (
	"errors"
	"fmt"

	"golang.org/x/text/internal"

	"golang.org/x/text/internal/catmsg"
	"golang.org/x/text/language"
)

// A Catalog allows lookup of translated messages.
type Catalog interface {
	// Languages returns all languages for which the Catalog contains variants.
	Languages() []language.Tag

	// Matcher returns a Matcher for languages from this Catalog.
	Matcher() language.Matcher

	// A Context is used for evaluating Messages.
	Context(tag language.Tag, r catmsg.Renderer) *Context

	// This method also makes Catalog a private interface.
	lookup(tag language.Tag, key string) (data string, ok bool)
}

// NewFromMap creates a Catalog from the given map. If a Dictionary is
// underspecified the entry is retrieved from a parent language.
func NewFromMap(dictionaries map[string]Dictionary, opts ...Option) (Catalog, error) {
	options := options{}
	for _, o := range opts {
		o(&options)
	}
	c := &catalog{
		dicts: map[language.Tag]Dictionary{},
	}
	_, hasFallback := dictionaries[options.fallback.String()]
	if hasFallback {
		// TODO: Should it be okay to not have a fallback language?
		// Catalog generators could enforce there is always a fallback.
		c.langs = append(c.langs, options.fallback)
	}
	for lang, dict := range dictionaries {
		tag, err := language.Parse(lang)
		if err != nil {
			return nil, fmt.Errorf("catalog: invalid language tag %q", lang)
		}
		if _, ok := c.dicts[tag]; ok {
			return nil, fmt.Errorf("catalog: duplicate entry for tag %q after normalization", tag)
		}
		c.dicts[tag] = dict
		if !hasFallback || tag != options.fallback {
			c.langs = append(c.langs, tag)
		}
	}
	if hasFallback {
		internal.SortTags(c.langs[1:])
	} else {
		internal.SortTags(c.langs)
	}
	c.matcher = language.NewMatcher(c.langs)
	return c, nil
}

// A Dictionary is a source of translations for a single language.
type Dictionary interface {
	// Lookup returns a message compiled with catmsg.Compile for the given key.
	// It returns false for ok if such a message could not be found.
	Lookup(key string) (data string, ok bool)
}

type catalog struct {
	langs   []language.Tag
	dicts   map[language.Tag]Dictionary
	macros  store
	matcher language.Matcher
}

func (c *catalog) Languages() []language.Tag { return c.langs }
func (c *catalog) Matcher() language.Matcher { return c.matcher }

func (c *catalog) lookup(tag language.Tag, key string) (data string, ok bool) {
	for ; ; tag = tag.Parent() {
		if dict, ok := c.dicts[tag]; ok {
			if data, ok := dict.Lookup(key); ok {
				return data, true
			}
		}
		if tag == language.Und {
			break
		}
	}
	return "", false
}

// Context returns a Context for formatting messages.
// Only one Message may be formatted per context at any given time.
func (c *catalog) Context(tag language.Tag, r catmsg.Renderer) *Context {
	return &Context{
		cat: c,
		tag: tag,
		dec: catmsg.NewDecoder(tag, r, &dict{&c.macros, tag}),
	}
}

// A Builder allows building a Catalog programmatically.
type Builder struct {
	options
	matcher language.Matcher

	index  store
	macros store
}

type options struct {
	fallback language.Tag
}

// An Option configures Catalog behavior.
type Option func(*options)

// Fallback specifies the default fallback language. The default is Und.
func Fallback(tag language.Tag) Option {
	return func(o *options) { o.fallback = tag }
}

// TODO:
// // Catalogs specifies one or more sources for a Catalog.
// // Lookups are in order.
// // This can be changed inserting a Catalog used for setting, which implements
// // Loader, used for setting in the chain.
// func Catalogs(d ...Loader) Option {
// 	return nil
// }
//
// func Delims(start, end string) Option {}
//
// func Dict(tag language.Tag, d ...Dictionary) Option

// NewBuilder returns an empty mutable Catalog.
func NewBuilder(opts ...Option) *Builder {
	c := &Builder{}
	for _, o := range opts {
		o(&c.options)
	}
	return c
}

// SetString is shorthand for Set(tag, key, String(msg)).
func (c *Builder) SetString(tag language.Tag, key string, msg string) error {
	return c.set(tag, key, &c.index, String(msg))
}

// Set sets the translation for the given language and key.
//
// When evaluation this message, the first Message in the sequence to msgs to
// evaluate to a string will be the message returned.
func (c *Builder) Set(tag language.Tag, key string, msg ...Message) error {
	return c.set(tag, key, &c.index, msg...)
}

// SetMacro defines a Message that may be substituted in another message.
// The arguments to a macro Message are passed as arguments in the
// placeholder the form "${foo(arg1, arg2)}".
func (c *Builder) SetMacro(tag language.Tag, name string, msg ...Message) error {
	return c.set(tag, name, &c.macros, msg...)
}

// ErrNotFound indicates there was no message for the given key.
var ErrNotFound = errors.New("catalog: message not found")

// String specifies a plain message string. It can be used as fallback if no
// other strings match or as a simple standalone message.
//
// It is an error to pass more than one String in a message sequence.
func String(name string) Message {
	return catmsg.String(name)
}

// Var sets a variable that may be substituted in formatting patterns using
// named substitution of the form "${name}". The name argument is used as a
// fallback if the statements do not produce a match. The statement sequence may
// not contain any Var calls.
//
// The name passed to a Var must be unique within message sequence.
func Var(name string, msg ...Message) Message {
	return &catmsg.Var{Name: name, Message: firstInSequence(msg)}
}

// Context returns a Context for formatting messages.
// Only one Message may be formatted per context at any given time.
func (b *Builder) Context(tag language.Tag, r catmsg.Renderer) *Context {
	return &Context{
		cat: b,
		tag: tag,
		dec: catmsg.NewDecoder(tag, r, &dict{&b.macros, tag}),
	}
}

// A Context is used for evaluating Messages.
// Only one Message may be formatted per context at any given time.
type Context struct {
	cat Catalog
	tag language.Tag // TODO: use compact index.
	dec *catmsg.Decoder
}

// Execute looks up and executes the message with the given key.
// It returns ErrNotFound if no message could be found in the index.
func (c *Context) Execute(key string) error {
	data, ok := c.cat.lookup(c.tag, key)
	if !ok {
		return ErrNotFound
	}
	return c.dec.Execute(data)
}
