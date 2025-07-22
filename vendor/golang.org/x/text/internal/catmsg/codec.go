// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package catmsg

import (
	"errors"
	"fmt"

	"golang.org/x/text/language"
)

// A Renderer renders a Message.
type Renderer interface {
	// Render renders the given string. The given string may be interpreted as a
	// format string, such as the one used by the fmt package or a template.
	Render(s string)

	// Arg returns the i-th argument passed to format a message. This method
	// should return nil if there is no such argument. Messages need access to
	// arguments to allow selecting a message based on linguistic features of
	// those arguments.
	Arg(i int) interface{}
}

// A Dictionary specifies a source of messages, including variables or macros.
type Dictionary interface {
	// Lookup returns the message for the given key. It returns false for ok if
	// such a message could not be found.
	Lookup(key string) (data string, ok bool)

	// TODO: consider returning an interface, instead of a string. This will
	// allow implementations to do their own message type decoding.
}

// An Encoder serializes a Message to a string.
type Encoder struct {
	// The root encoder is used for storing encoded variables.
	root *Encoder
	// The parent encoder provides the surrounding scopes for resolving variable
	// names.
	parent *Encoder

	tag language.Tag

	// buf holds the encoded message so far. After a message completes encoding,
	// the contents of buf, prefixed by the encoded length, are flushed to the
	// parent buffer.
	buf []byte

	// vars is the lookup table of variables in the current scope.
	vars []keyVal

	err    error
	inBody bool // if false next call must be EncodeMessageType
}

type keyVal struct {
	key    string
	offset int
}

// Language reports the language for which the encoded message will be stored
// in the Catalog.
func (e *Encoder) Language() language.Tag { return e.tag }

func (e *Encoder) setError(err error) {
	if e.root.err == nil {
		e.root.err = err
	}
}

// EncodeUint encodes x.
func (e *Encoder) EncodeUint(x uint64) {
	e.checkInBody()
	var buf [maxVarintBytes]byte
	n := encodeUint(buf[:], x)
	e.buf = append(e.buf, buf[:n]...)
}

// EncodeString encodes s.
func (e *Encoder) EncodeString(s string) {
	e.checkInBody()
	e.EncodeUint(uint64(len(s)))
	e.buf = append(e.buf, s...)
}

// EncodeMessageType marks the current message to be of type h.
//
// It must be the first call of a Message's Compile method.
func (e *Encoder) EncodeMessageType(h Handle) {
	if e.inBody {
		panic("catmsg: EncodeMessageType not the first method called")
	}
	e.inBody = true
	e.EncodeUint(uint64(h))
}

// EncodeMessage serializes the given message inline at the current position.
func (e *Encoder) EncodeMessage(m Message) error {
	e = &Encoder{root: e.root, parent: e, tag: e.tag}
	err := m.Compile(e)
	if _, ok := m.(*Var); !ok {
		e.flushTo(e.parent)
	}
	return err
}

func (e *Encoder) checkInBody() {
	if !e.inBody {
		panic("catmsg: expected prior call to EncodeMessageType")
	}
}

// stripPrefix indicates the number of prefix bytes that must be stripped to
// turn a single-element sequence into a message that is just this single member
// without its size prefix. If the message can be stripped, b[1:n] contains the
// size prefix.
func stripPrefix(b []byte) (n int) {
	if len(b) > 0 && Handle(b[0]) == msgFirst {
		x, n, _ := decodeUint(b[1:])
		if 1+n+int(x) == len(b) {
			return 1 + n
		}
	}
	return 0
}

func (e *Encoder) flushTo(dst *Encoder) {
	data := e.buf
	p := stripPrefix(data)
	if p > 0 {
		data = data[1:]
	} else {
		// Prefix the size.
		dst.EncodeUint(uint64(len(data)))
	}
	dst.buf = append(dst.buf, data...)
}

func (e *Encoder) addVar(key string, m Message) error {
	for _, v := range e.parent.vars {
		if v.key == key {
			err := fmt.Errorf("catmsg: duplicate variable %q", key)
			e.setError(err)
			return err
		}
	}
	scope := e.parent
	// If a variable message is Incomplete, and does not evaluate to a message
	// during execution, we fall back to the variable name. We encode this by
	// appending the variable name if the message reports it's incomplete.

	err := m.Compile(e)
	if err != ErrIncomplete {
		e.setError(err)
	}
	switch {
	case len(e.buf) == 1 && Handle(e.buf[0]) == msgFirst: // empty sequence
		e.buf = e.buf[:0]
		e.inBody = false
		fallthrough
	case len(e.buf) == 0:
		// Empty message.
		if err := String(key).Compile(e); err != nil {
			e.setError(err)
		}
	case err == ErrIncomplete:
		if Handle(e.buf[0]) != msgFirst {
			seq := &Encoder{root: e.root, parent: e}
			seq.EncodeMessageType(msgFirst)
			e.flushTo(seq)
			e = seq
		}
		// e contains a sequence; append the fallback string.
		e.EncodeMessage(String(key))
	}

	// Flush result to variable heap.
	offset := len(e.root.buf)
	e.flushTo(e.root)
	e.buf = e.buf[:0]

	// Record variable offset in current scope.
	scope.vars = append(scope.vars, keyVal{key: key, offset: offset})
	return err
}

const (
	substituteVar = iota
	substituteMacro
	substituteError
)

// EncodeSubstitution inserts a resolved reference to a variable or macro.
//
// This call must be matched with a call to ExecuteSubstitution at decoding
// time.
func (e *Encoder) EncodeSubstitution(name string, arguments ...int) {
	if arity := len(arguments); arity > 0 {
		// TODO: also resolve macros.
		e.EncodeUint(substituteMacro)
		e.EncodeString(name)
		for _, a := range arguments {
			e.EncodeUint(uint64(a))
		}
		return
	}
	for scope := e; scope != nil; scope = scope.parent {
		for _, v := range scope.vars {
			if v.key != name {
				continue
			}
			e.EncodeUint(substituteVar) // TODO: support arity > 0
			e.EncodeUint(uint64(v.offset))
			return
		}
	}
	// TODO: refer to dictionary-wide scoped variables.
	e.EncodeUint(substituteError)
	e.EncodeString(name)
	e.setError(fmt.Errorf("catmsg: unknown var %q", name))
}

// A Decoder deserializes and evaluates messages that are encoded by an encoder.
type Decoder struct {
	tag    language.Tag
	dst    Renderer
	macros Dictionary

	err  error
	vars string
	data string

	macroArg int // TODO: allow more than one argument
}

// NewDecoder returns a new Decoder.
//
// Decoders are designed to be reused for multiple invocations of Execute.
// Only one goroutine may call Execute concurrently.
func NewDecoder(tag language.Tag, r Renderer, macros Dictionary) *Decoder {
	return &Decoder{
		tag:    tag,
		dst:    r,
		macros: macros,
	}
}

func (d *Decoder) setError(err error) {
	if d.err == nil {
		d.err = err
	}
}

// Language returns the language in which the message is being rendered.
//
// The destination language may be a child language of the language used for
// encoding. For instance, a decoding language of "pt-PT" is consistent with an
// encoding language of "pt".
func (d *Decoder) Language() language.Tag { return d.tag }

// Done reports whether there are more bytes to process in this message.
func (d *Decoder) Done() bool { return len(d.data) == 0 }

// Render implements Renderer.
func (d *Decoder) Render(s string) { d.dst.Render(s) }

// Arg implements Renderer.
//
// During evaluation of macros, the argument positions may be mapped to
// arguments that differ from the original call.
func (d *Decoder) Arg(i int) interface{} {
	if d.macroArg != 0 {
		if i != 1 {
			panic("catmsg: only macros with single argument supported")
		}
		i = d.macroArg
	}
	return d.dst.Arg(i)
}

// DecodeUint decodes a number that was encoded with EncodeUint and advances the
// position.
func (d *Decoder) DecodeUint() uint64 {
	x, n, err := decodeUintString(d.data)
	d.data = d.data[n:]
	if err != nil {
		d.setError(err)
	}
	return x
}

// DecodeString decodes a string that was encoded with EncodeString and advances
// the position.
func (d *Decoder) DecodeString() string {
	size := d.DecodeUint()
	s := d.data[:size]
	d.data = d.data[size:]
	return s
}

// SkipMessage skips the message at the current location and advances the
// position.
func (d *Decoder) SkipMessage() {
	n := int(d.DecodeUint())
	d.data = d.data[n:]
}

// Execute decodes and evaluates msg.
//
// Only one goroutine may call execute.
func (d *Decoder) Execute(msg string) error {
	d.err = nil
	if !d.execute(msg) {
		return ErrNoMatch
	}
	return d.err
}

func (d *Decoder) execute(msg string) bool {
	saved := d.data
	d.data = msg
	ok := d.executeMessage()
	d.data = saved
	return ok
}

// executeMessageFromData is like execute, but also decodes a leading message
// size and clips the given string accordingly.
//
// It reports the number of bytes consumed and whether a message was selected.
func (d *Decoder) executeMessageFromData(s string) (n int, ok bool) {
	saved := d.data
	d.data = s
	size := int(d.DecodeUint())
	n = len(s) - len(d.data)
	// Sanitize the setting. This allows skipping a size argument for
	// RawString and method Done.
	d.data = d.data[:size]
	ok = d.executeMessage()
	n += size - len(d.data)
	d.data = saved
	return n, ok
}

var errUnknownHandler = errors.New("catmsg: string contains unsupported handler")

// executeMessage reads the handle id, initializes the decoder and executes the
// message. It is assumed that all of d.data[d.p:] is the single message.
func (d *Decoder) executeMessage() bool {
	if d.Done() {
		// We interpret no data as a valid empty message.
		return true
	}
	handle := d.DecodeUint()

	var fn Handler
	mutex.Lock()
	if int(handle) < len(handlers) {
		fn = handlers[handle]
	}
	mutex.Unlock()
	if fn == nil {
		d.setError(errUnknownHandler)
		d.execute(fmt.Sprintf("\x02$!(UNKNOWNMSGHANDLER=%#x)", handle))
		return true
	}
	return fn(d)
}

// ExecuteMessage decodes and executes the message at the current position.
func (d *Decoder) ExecuteMessage() bool {
	n, ok := d.executeMessageFromData(d.data)
	d.data = d.data[n:]
	return ok
}

// ExecuteSubstitution executes the message corresponding to the substitution
// as encoded by EncodeSubstitution.
func (d *Decoder) ExecuteSubstitution() {
	switch x := d.DecodeUint(); x {
	case substituteVar:
		offset := d.DecodeUint()
		d.executeMessageFromData(d.vars[offset:])
	case substituteMacro:
		name := d.DecodeString()
		data, ok := d.macros.Lookup(name)
		old := d.macroArg
		// TODO: support macros of arity other than 1.
		d.macroArg = int(d.DecodeUint())
		switch {
		case !ok:
			// TODO: detect this at creation time.
			d.setError(fmt.Errorf("catmsg: undefined macro %q", name))
			fallthrough
		case !d.execute(data):
			d.dst.Render(name) // fall back to macro name.
		}
		d.macroArg = old
	case substituteError:
		d.dst.Render(d.DecodeString())
	default:
		panic("catmsg: unreachable")
	}
}
