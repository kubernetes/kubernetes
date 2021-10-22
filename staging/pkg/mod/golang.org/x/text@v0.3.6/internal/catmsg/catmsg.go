// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package catmsg contains support types for package x/text/message/catalog.
//
// This package contains the low-level implementations of Message used by the
// catalog package and provides primitives for other packages to implement their
// own. For instance, the plural package provides functionality for selecting
// translation strings based on the plural category of substitution arguments.
//
//
// Encoding and Decoding
//
// Catalogs store Messages encoded as a single string. Compiling a message into
// a string both results in compacter representation and speeds up evaluation.
//
// A Message must implement a Compile method to convert its arbitrary
// representation to a string. The Compile method takes an Encoder which
// facilitates serializing the message. Encoders also provide more context of
// the messages's creation (such as for which language the message is intended),
// which may not be known at the time of the creation of the message.
//
// Each message type must also have an accompanying decoder registered to decode
// the message. This decoder takes a Decoder argument which provides the
// counterparts for the decoding.
//
//
// Renderers
//
// A Decoder must be initialized with a Renderer implementation. These
// implementations must be provided by packages that use Catalogs, typically
// formatting packages such as x/text/message. A typical user will not need to
// worry about this type; it is only relevant to packages that do string
// formatting and want to use the catalog package to handle localized strings.
//
// A package that uses catalogs for selecting strings receives selection results
// as sequence of substrings passed to the Renderer. The following snippet shows
// how to express the above example using the message package.
//
//   message.Set(language.English, "You are %d minute(s) late.",
//       catalog.Var("minutes", plural.Select(1, "one", "minute")),
//       catalog.String("You are %[1]d ${minutes} late."))
//
//   p := message.NewPrinter(language.English)
//   p.Printf("You are %d minute(s) late.", 5) // always 5 minutes late.
//
// To evaluate the Printf, package message wraps the arguments in a Renderer
// that is passed to the catalog for message decoding. The call sequence that
// results from evaluating the above message, assuming the person is rather
// tardy, is:
//
//   Render("You are %[1]d ")
//   Arg(1)
//   Render("minutes")
//   Render(" late.")
//
// The calls to Arg is caused by the plural.Select execution, which evaluates
// the argument to determine whether the singular or plural message form should
// be selected. The calls to Render reports the partial results to the message
// package for further evaluation.
package catmsg

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/text/language"
)

// A Handle refers to a registered message type.
type Handle int

// A Handler decodes and evaluates data compiled by a Message and sends the
// result to the Decoder. The output may depend on the value of the substitution
// arguments, accessible by the Decoder's Arg method. The Handler returns false
// if there is no translation for the given substitution arguments.
type Handler func(d *Decoder) bool

// Register records the existence of a message type and returns a Handle that
// can be used in the Encoder's EncodeMessageType method to create such
// messages. The prefix of the name should be the package path followed by
// an optional disambiguating string.
// Register will panic if a handle for the same name was already registered.
func Register(name string, handler Handler) Handle {
	mutex.Lock()
	defer mutex.Unlock()

	if _, ok := names[name]; ok {
		panic(fmt.Errorf("catmsg: handler for %q already exists", name))
	}
	h := Handle(len(handlers))
	names[name] = h
	handlers = append(handlers, handler)
	return h
}

// These handlers require fixed positions in the handlers slice.
const (
	msgVars Handle = iota
	msgFirst
	msgRaw
	msgString
	msgAffix
	// Leave some arbitrary room for future expansion: 20 should suffice.
	numInternal = 20
)

const prefix = "golang.org/x/text/internal/catmsg."

var (
	// TODO: find a more stable way to link handles to message types.
	mutex sync.Mutex
	names = map[string]Handle{
		prefix + "Vars":   msgVars,
		prefix + "First":  msgFirst,
		prefix + "Raw":    msgRaw,
		prefix + "String": msgString,
		prefix + "Affix":  msgAffix,
	}
	handlers = make([]Handler, numInternal)
)

func init() {
	// This handler is a message type wrapper that initializes a decoder
	// with a variable block. This message type, if present, is always at the
	// start of an encoded message.
	handlers[msgVars] = func(d *Decoder) bool {
		blockSize := int(d.DecodeUint())
		d.vars = d.data[:blockSize]
		d.data = d.data[blockSize:]
		return d.executeMessage()
	}

	// First takes the first message in a sequence that results in a match for
	// the given substitution arguments.
	handlers[msgFirst] = func(d *Decoder) bool {
		for !d.Done() {
			if d.ExecuteMessage() {
				return true
			}
		}
		return false
	}

	handlers[msgRaw] = func(d *Decoder) bool {
		d.Render(d.data)
		return true
	}

	// A String message alternates between a string constant and a variable
	// substitution.
	handlers[msgString] = func(d *Decoder) bool {
		for !d.Done() {
			if str := d.DecodeString(); str != "" {
				d.Render(str)
			}
			if d.Done() {
				break
			}
			d.ExecuteSubstitution()
		}
		return true
	}

	handlers[msgAffix] = func(d *Decoder) bool {
		// TODO: use an alternative method for common cases.
		prefix := d.DecodeString()
		suffix := d.DecodeString()
		if prefix != "" {
			d.Render(prefix)
		}
		ret := d.ExecuteMessage()
		if suffix != "" {
			d.Render(suffix)
		}
		return ret
	}
}

var (
	// ErrIncomplete indicates a compiled message does not define translations
	// for all possible argument values. If this message is returned, evaluating
	// a message may result in the ErrNoMatch error.
	ErrIncomplete = errors.New("catmsg: incomplete message; may not give result for all inputs")

	// ErrNoMatch indicates no translation message matched the given input
	// parameters when evaluating a message.
	ErrNoMatch = errors.New("catmsg: no translation for inputs")
)

// A Message holds a collection of translations for the same phrase that may
// vary based on the values of substitution arguments.
type Message interface {
	// Compile encodes the format string(s) of the message as a string for later
	// evaluation.
	//
	// The first call Compile makes on the encoder must be EncodeMessageType.
	// The handle passed to this call may either be a handle returned by
	// Register to encode a single custom message, or HandleFirst followed by
	// a sequence of calls to EncodeMessage.
	//
	// Compile must return ErrIncomplete if it is possible for evaluation to
	// not match any translation for a given set of formatting parameters.
	// For example, selecting a translation based on plural form may not yield
	// a match if the form "Other" is not one of the selectors.
	//
	// Compile may return any other application-specific error. For backwards
	// compatibility with package like fmt, which often do not do sanity
	// checking of format strings ahead of time, Compile should still make an
	// effort to have some sensible fallback in case of an error.
	Compile(e *Encoder) error
}

// Compile converts a Message to a data string that can be stored in a Catalog.
// The resulting string can subsequently be decoded by passing to the Execute
// method of a Decoder.
func Compile(tag language.Tag, macros Dictionary, m Message) (data string, err error) {
	// TODO: pass macros so they can be used for validation.
	v := &Encoder{inBody: true} // encoder for variables
	v.root = v
	e := &Encoder{root: v, parent: v, tag: tag} // encoder for messages
	err = m.Compile(e)
	// This package serves te message package, which in turn is meant to be a
	// drop-in replacement for fmt.  With the fmt package, format strings are
	// evaluated lazily and errors are handled by substituting strings in the
	// result, rather then returning an error. Dealing with multiple languages
	// makes it more important to check errors ahead of time. We chose to be
	// consistent and compatible and allow graceful degradation in case of
	// errors.
	buf := e.buf[stripPrefix(e.buf):]
	if len(v.buf) > 0 {
		// Prepend variable block.
		b := make([]byte, 1+maxVarintBytes+len(v.buf)+len(buf))
		b[0] = byte(msgVars)
		b = b[:1+encodeUint(b[1:], uint64(len(v.buf)))]
		b = append(b, v.buf...)
		b = append(b, buf...)
		buf = b
	}
	if err == nil {
		err = v.err
	}
	return string(buf), err
}

// FirstOf is a message type that prints the first message in the sequence that
// resolves to a match for the given substitution arguments.
type FirstOf []Message

// Compile implements Message.
func (s FirstOf) Compile(e *Encoder) error {
	e.EncodeMessageType(msgFirst)
	err := ErrIncomplete
	for i, m := range s {
		if err == nil {
			return fmt.Errorf("catalog: message argument %d is complete and blocks subsequent messages", i-1)
		}
		err = e.EncodeMessage(m)
	}
	return err
}

// Var defines a message that can be substituted for a placeholder of the same
// name. If an expression does not result in a string after evaluation, Name is
// used as the substitution. For example:
//    Var{
//      Name:    "minutes",
//      Message: plural.Select(1, "one", "minute"),
//    }
// will resolve to minute for singular and minutes for plural forms.
type Var struct {
	Name    string
	Message Message
}

var errIsVar = errors.New("catmsg: variable used as message")

// Compile implements Message.
//
// Note that this method merely registers a variable; it does not create an
// encoded message.
func (v *Var) Compile(e *Encoder) error {
	if err := e.addVar(v.Name, v.Message); err != nil {
		return err
	}
	// Using a Var by itself is an error. If it is in a sequence followed by
	// other messages referring to it, this error will be ignored.
	return errIsVar
}

// Raw is a message consisting of a single format string that is passed as is
// to the Renderer.
//
// Note that a Renderer may still do its own variable substitution.
type Raw string

// Compile implements Message.
func (r Raw) Compile(e *Encoder) (err error) {
	e.EncodeMessageType(msgRaw)
	// Special case: raw strings don't have a size encoding and so don't use
	// EncodeString.
	e.buf = append(e.buf, r...)
	return nil
}

// String is a message consisting of a single format string which contains
// placeholders that may be substituted with variables.
//
// Variable substitutions are marked with placeholders and a variable name of
// the form ${name}. Any other substitutions such as Go templates or
// printf-style substitutions are left to be done by the Renderer.
//
// When evaluation a string interpolation, a Renderer will receive separate
// calls for each placeholder and interstitial string. For example, for the
// message: "%[1]v ${invites} %[2]v to ${their} party." The sequence of calls
// is:
//   d.Render("%[1]v ")
//   d.Arg(1)
//   d.Render(resultOfInvites)
//   d.Render(" %[2]v to ")
//   d.Arg(2)
//   d.Render(resultOfTheir)
//   d.Render(" party.")
// where the messages for "invites" and "their" both use a plural.Select
// referring to the first argument.
//
// Strings may also invoke macros. Macros are essentially variables that can be
// reused. Macros may, for instance, be used to make selections between
// different conjugations of a verb. See the catalog package description for an
// overview of macros.
type String string

// Compile implements Message. It parses the placeholder formats and returns
// any error.
func (s String) Compile(e *Encoder) (err error) {
	msg := string(s)
	const subStart = "${"
	hasHeader := false
	p := 0
	b := []byte{}
	for {
		i := strings.Index(msg[p:], subStart)
		if i == -1 {
			break
		}
		b = append(b, msg[p:p+i]...)
		p += i + len(subStart)
		if i = strings.IndexByte(msg[p:], '}'); i == -1 {
			b = append(b, "$!(MISSINGBRACE)"...)
			err = fmt.Errorf("catmsg: missing '}'")
			p = len(msg)
			break
		}
		name := strings.TrimSpace(msg[p : p+i])
		if q := strings.IndexByte(name, '('); q == -1 {
			if !hasHeader {
				hasHeader = true
				e.EncodeMessageType(msgString)
			}
			e.EncodeString(string(b))
			e.EncodeSubstitution(name)
			b = b[:0]
		} else if j := strings.IndexByte(name[q:], ')'); j == -1 {
			// TODO: what should the error be?
			b = append(b, "$!(MISSINGPAREN)"...)
			err = fmt.Errorf("catmsg: missing ')'")
		} else if x, sErr := strconv.ParseUint(strings.TrimSpace(name[q+1:q+j]), 10, 32); sErr != nil {
			// TODO: handle more than one argument
			b = append(b, "$!(BADNUM)"...)
			err = fmt.Errorf("catmsg: invalid number %q", strings.TrimSpace(name[q+1:q+j]))
		} else {
			if !hasHeader {
				hasHeader = true
				e.EncodeMessageType(msgString)
			}
			e.EncodeString(string(b))
			e.EncodeSubstitution(name[:q], int(x))
			b = b[:0]
		}
		p += i + 1
	}
	b = append(b, msg[p:]...)
	if !hasHeader {
		// Simplify string to a raw string.
		Raw(string(b)).Compile(e)
	} else if len(b) > 0 {
		e.EncodeString(string(b))
	}
	return err
}

// Affix is a message that adds a prefix and suffix to another message.
// This is mostly used add back whitespace to a translation that was stripped
// before sending it out.
type Affix struct {
	Message Message
	Prefix  string
	Suffix  string
}

// Compile implements Message.
func (a Affix) Compile(e *Encoder) (err error) {
	// TODO: consider adding a special message type that just adds a single
	// return. This is probably common enough to handle the majority of cases.
	// Get some stats first, though.
	e.EncodeMessageType(msgAffix)
	e.EncodeString(a.Prefix)
	e.EncodeString(a.Suffix)
	e.EncodeMessage(a.Message)
	return nil
}
