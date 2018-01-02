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
// specific linguistic feature. The following snippet uses plural.Select:
//
//   catalog.Set(language.English, "You are %d minute(s) late.",
//       plural.Select(1,
//           "one", "You are 1 minute late.",
//           "other", "You are %d minutes late."))
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
//           plural.Select(1, "one", "minute", "other", "minutes")),
//       catalog.String("You are %[1]d ${minutes} late."))
//
// Var is defined to return the variable name if the message does not yield a
// match. This allows us to further simplify this snippet to
//
//   catalog.Set(language.English, "You are %d minute(s) late.",
//       catalog.Var("minutes", plural.Select(1, "one", "minute")),
//       catalog.String("You are %d ${minutes} late."))
//
// Overall this is still only a minor improvement, but things can get a lot more
// unwieldy if more than one linguistic feature is used to determine a message
// variant. Consider the following example:
//
//   // argument 1: list of hosts, argument 2: list of guests
//   catalog.Set(language.English, "%[1]v invite(s) %[2]v to their party.",
//     catalog.Var("their",
//         plural.Select(1,
//             "one", gender.Select(1, "female", "her", "other", "his"))),
//     catalog.Var("invites", plural.Select(1, "one", "invite"))
//     catalog.String("%[1]v ${invites} %[2]v to ${their} party.")),
//
// Without variable substitution, this would have to be written as
//
//   // argument 1: list of hosts, argument 2: list of guests
//   catalog.Set(language.English, "%[1]v invite(s) %[2]v to their party.",
//     plural.Select(1,
//         "one", gender.Select(1,
//             "female", "%[1]v invites %[2]v to her party."
//             "other", "%[1]v invites %[2]v to his party."),
//         "other", "%[1]v invites %[2]v to their party.")
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
//   catalog.SetMacro(language.English, "invites", plural.Select(1, "one", "invite"))
//   catalog.SetMacro(language.English, "their", plural.Select(1,
//      "one", gender.Select(1, "female", "her", "other", "his"))),
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

	"golang.org/x/text/internal/catmsg"
	"golang.org/x/text/language"
)

// A Catalog holds translations for messages for supported languages.
type Catalog struct {
	options

	index  store
	macros store
}

type options struct{}

// An Option configures Catalog behavior.
type Option func(*options)

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

// New returns a new Catalog.
func New(opts ...Option) *Catalog {
	c := &Catalog{}
	for _, o := range opts {
		o(&c.options)
	}
	return c
}

// Languages returns all languages for which the Catalog contains variants.
func (c *Catalog) Languages() []language.Tag {
	return c.index.languages()
}

// SetString is shorthand for Set(tag, key, String(msg)).
func (c *Catalog) SetString(tag language.Tag, key string, msg string) error {
	return c.set(tag, key, &c.index, String(msg))
}

// Set sets the translation for the given language and key.
//
// When evaluation this message, the first Message in the sequence to msgs to
// evaluate to a string will be the message returned.
func (c *Catalog) Set(tag language.Tag, key string, msg ...Message) error {
	return c.set(tag, key, &c.index, msg...)
}

// SetMacro defines a Message that may be substituted in another message.
// The arguments to a macro Message are passed as arguments in the
// placeholder the form "${foo(arg1, arg2)}".
func (c *Catalog) SetMacro(tag language.Tag, name string, msg ...Message) error {
	return c.set(tag, name, &c.macros, msg...)
}

// ErrNotFound indicates there was no message for the given key.
var ErrNotFound = errors.New("catalog: message not found")

// A Message holds a collection of translations for the same phrase that may
// vary based on the values of substitution arguments.
type Message interface {
	catmsg.Message
}

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

// firstInSequence is a message type that prints the first message in the
// sequence that resolves to a match for the given substitution arguments.
type firstInSequence []Message

func (s firstInSequence) Compile(e *catmsg.Encoder) error {
	e.EncodeMessageType(catmsg.First)
	err := catmsg.ErrIncomplete
	for i, m := range s {
		if err == nil {
			return fmt.Errorf("catalog: message argument %d is complete and blocks subsequent messages", i-1)
		}
		err = e.EncodeMessage(m)
	}
	return err
}

// Context returns a Context for formatting messages.
// Only one Message may be formatted per context at any given time.
func (c *Catalog) Context(tag language.Tag, r catmsg.Renderer) *Context {
	return &Context{
		cat: c,
		tag: tag,
		dec: catmsg.NewDecoder(tag, r, &dict{&c.macros, tag}),
	}
}

// A Context is used for evaluating Messages.
// Only one Message may be formatted per context at any given time.
type Context struct {
	cat *Catalog
	tag language.Tag
	dec *catmsg.Decoder
}

// Execute looks up and executes the message with the given key.
// It returns ErrNotFound if no message could be found in the index.
func (c *Context) Execute(key string) error {
	data, ok := c.cat.index.lookup(c.tag, key)
	if !ok {
		return ErrNotFound
	}
	return c.dec.Execute(data)
}
