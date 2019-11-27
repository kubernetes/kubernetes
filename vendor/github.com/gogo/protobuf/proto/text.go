// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Go support for Protocol Buffers - Google's data interchange format
//
// Copyright 2010 The Go Authors.  All rights reserved.
// https://github.com/golang/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package proto

// Functions for writing the text protocol buffer format.

import (
	"bufio"
	"bytes"
	"encoding"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"reflect"
	"sort"
	"strings"
	"sync"
	"time"
)

var (
	newline         = []byte("\n")
	spaces          = []byte("                                        ")
	endBraceNewline = []byte("}\n")
	backslashN      = []byte{'\\', 'n'}
	backslashR      = []byte{'\\', 'r'}
	backslashT      = []byte{'\\', 't'}
	backslashDQ     = []byte{'\\', '"'}
	backslashBS     = []byte{'\\', '\\'}
	posInf          = []byte("inf")
	negInf          = []byte("-inf")
	nan             = []byte("nan")
)

type writer interface {
	io.Writer
	WriteByte(byte) error
}

// textWriter is an io.Writer that tracks its indentation level.
type textWriter struct {
	ind      int
	complete bool // if the current position is a complete line
	compact  bool // whether to write out as a one-liner
	w        writer
}

func (w *textWriter) WriteString(s string) (n int, err error) {
	if !strings.Contains(s, "\n") {
		if !w.compact && w.complete {
			w.writeIndent()
		}
		w.complete = false
		return io.WriteString(w.w, s)
	}
	// WriteString is typically called without newlines, so this
	// codepath and its copy are rare.  We copy to avoid
	// duplicating all of Write's logic here.
	return w.Write([]byte(s))
}

func (w *textWriter) Write(p []byte) (n int, err error) {
	newlines := bytes.Count(p, newline)
	if newlines == 0 {
		if !w.compact && w.complete {
			w.writeIndent()
		}
		n, err = w.w.Write(p)
		w.complete = false
		return n, err
	}

	frags := bytes.SplitN(p, newline, newlines+1)
	if w.compact {
		for i, frag := range frags {
			if i > 0 {
				if err := w.w.WriteByte(' '); err != nil {
					return n, err
				}
				n++
			}
			nn, err := w.w.Write(frag)
			n += nn
			if err != nil {
				return n, err
			}
		}
		return n, nil
	}

	for i, frag := range frags {
		if w.complete {
			w.writeIndent()
		}
		nn, err := w.w.Write(frag)
		n += nn
		if err != nil {
			return n, err
		}
		if i+1 < len(frags) {
			if err := w.w.WriteByte('\n'); err != nil {
				return n, err
			}
			n++
		}
	}
	w.complete = len(frags[len(frags)-1]) == 0
	return n, nil
}

func (w *textWriter) WriteByte(c byte) error {
	if w.compact && c == '\n' {
		c = ' '
	}
	if !w.compact && w.complete {
		w.writeIndent()
	}
	err := w.w.WriteByte(c)
	w.complete = c == '\n'
	return err
}

func (w *textWriter) indent() { w.ind++ }

func (w *textWriter) unindent() {
	if w.ind == 0 {
		log.Print("proto: textWriter unindented too far")
		return
	}
	w.ind--
}

func writeName(w *textWriter, props *Properties) error {
	if _, err := w.WriteString(props.OrigName); err != nil {
		return err
	}
	if props.Wire != "group" {
		return w.WriteByte(':')
	}
	return nil
}

func requiresQuotes(u string) bool {
	// When type URL contains any characters except [0-9A-Za-z./\-]*, it must be quoted.
	for _, ch := range u {
		switch {
		case ch == '.' || ch == '/' || ch == '_':
			continue
		case '0' <= ch && ch <= '9':
			continue
		case 'A' <= ch && ch <= 'Z':
			continue
		case 'a' <= ch && ch <= 'z':
			continue
		default:
			return true
		}
	}
	return false
}

// isAny reports whether sv is a google.protobuf.Any message
func isAny(sv reflect.Value) bool {
	type wkt interface {
		XXX_WellKnownType() string
	}
	t, ok := sv.Addr().Interface().(wkt)
	return ok && t.XXX_WellKnownType() == "Any"
}

// writeProto3Any writes an expanded google.protobuf.Any message.
//
// It returns (false, nil) if sv value can't be unmarshaled (e.g. because
// required messages are not linked in).
//
// It returns (true, error) when sv was written in expanded format or an error
// was encountered.
func (tm *TextMarshaler) writeProto3Any(w *textWriter, sv reflect.Value) (bool, error) {
	turl := sv.FieldByName("TypeUrl")
	val := sv.FieldByName("Value")
	if !turl.IsValid() || !val.IsValid() {
		return true, errors.New("proto: invalid google.protobuf.Any message")
	}

	b, ok := val.Interface().([]byte)
	if !ok {
		return true, errors.New("proto: invalid google.protobuf.Any message")
	}

	parts := strings.Split(turl.String(), "/")
	mt := MessageType(parts[len(parts)-1])
	if mt == nil {
		return false, nil
	}
	m := reflect.New(mt.Elem())
	if err := Unmarshal(b, m.Interface().(Message)); err != nil {
		return false, nil
	}
	w.Write([]byte("["))
	u := turl.String()
	if requiresQuotes(u) {
		writeString(w, u)
	} else {
		w.Write([]byte(u))
	}
	if w.compact {
		w.Write([]byte("]:<"))
	} else {
		w.Write([]byte("]: <\n"))
		w.ind++
	}
	if err := tm.writeStruct(w, m.Elem()); err != nil {
		return true, err
	}
	if w.compact {
		w.Write([]byte("> "))
	} else {
		w.ind--
		w.Write([]byte(">\n"))
	}
	return true, nil
}

func (tm *TextMarshaler) writeStruct(w *textWriter, sv reflect.Value) error {
	if tm.ExpandAny && isAny(sv) {
		if canExpand, err := tm.writeProto3Any(w, sv); canExpand {
			return err
		}
	}
	st := sv.Type()
	sprops := GetProperties(st)
	for i := 0; i < sv.NumField(); i++ {
		fv := sv.Field(i)
		props := sprops.Prop[i]
		name := st.Field(i).Name

		if name == "XXX_NoUnkeyedLiteral" {
			continue
		}

		if strings.HasPrefix(name, "XXX_") {
			// There are two XXX_ fields:
			//   XXX_unrecognized []byte
			//   XXX_extensions   map[int32]proto.Extension
			// The first is handled here;
			// the second is handled at the bottom of this function.
			if name == "XXX_unrecognized" && !fv.IsNil() {
				if err := writeUnknownStruct(w, fv.Interface().([]byte)); err != nil {
					return err
				}
			}
			continue
		}
		if fv.Kind() == reflect.Ptr && fv.IsNil() {
			// Field not filled in. This could be an optional field or
			// a required field that wasn't filled in. Either way, there
			// isn't anything we can show for it.
			continue
		}
		if fv.Kind() == reflect.Slice && fv.IsNil() {
			// Repeated field that is empty, or a bytes field that is unused.
			continue
		}

		if props.Repeated && fv.Kind() == reflect.Slice {
			// Repeated field.
			for j := 0; j < fv.Len(); j++ {
				if err := writeName(w, props); err != nil {
					return err
				}
				if !w.compact {
					if err := w.WriteByte(' '); err != nil {
						return err
					}
				}
				v := fv.Index(j)
				if v.Kind() == reflect.Ptr && v.IsNil() {
					// A nil message in a repeated field is not valid,
					// but we can handle that more gracefully than panicking.
					if _, err := w.Write([]byte("<nil>\n")); err != nil {
						return err
					}
					continue
				}
				if len(props.Enum) > 0 {
					if err := tm.writeEnum(w, v, props); err != nil {
						return err
					}
				} else if err := tm.writeAny(w, v, props); err != nil {
					return err
				}
				if err := w.WriteByte('\n'); err != nil {
					return err
				}
			}
			continue
		}
		if fv.Kind() == reflect.Map {
			// Map fields are rendered as a repeated struct with key/value fields.
			keys := fv.MapKeys()
			sort.Sort(mapKeys(keys))
			for _, key := range keys {
				val := fv.MapIndex(key)
				if err := writeName(w, props); err != nil {
					return err
				}
				if !w.compact {
					if err := w.WriteByte(' '); err != nil {
						return err
					}
				}
				// open struct
				if err := w.WriteByte('<'); err != nil {
					return err
				}
				if !w.compact {
					if err := w.WriteByte('\n'); err != nil {
						return err
					}
				}
				w.indent()
				// key
				if _, err := w.WriteString("key:"); err != nil {
					return err
				}
				if !w.compact {
					if err := w.WriteByte(' '); err != nil {
						return err
					}
				}
				if err := tm.writeAny(w, key, props.MapKeyProp); err != nil {
					return err
				}
				if err := w.WriteByte('\n'); err != nil {
					return err
				}
				// nil values aren't legal, but we can avoid panicking because of them.
				if val.Kind() != reflect.Ptr || !val.IsNil() {
					// value
					if _, err := w.WriteString("value:"); err != nil {
						return err
					}
					if !w.compact {
						if err := w.WriteByte(' '); err != nil {
							return err
						}
					}
					if err := tm.writeAny(w, val, props.MapValProp); err != nil {
						return err
					}
					if err := w.WriteByte('\n'); err != nil {
						return err
					}
				}
				// close struct
				w.unindent()
				if err := w.WriteByte('>'); err != nil {
					return err
				}
				if err := w.WriteByte('\n'); err != nil {
					return err
				}
			}
			continue
		}
		if props.proto3 && fv.Kind() == reflect.Slice && fv.Len() == 0 {
			// empty bytes field
			continue
		}
		if props.proto3 && fv.Kind() != reflect.Ptr && fv.Kind() != reflect.Slice {
			// proto3 non-repeated scalar field; skip if zero value
			if isProto3Zero(fv) {
				continue
			}
		}

		if fv.Kind() == reflect.Interface {
			// Check if it is a oneof.
			if st.Field(i).Tag.Get("protobuf_oneof") != "" {
				// fv is nil, or holds a pointer to generated struct.
				// That generated struct has exactly one field,
				// which has a protobuf struct tag.
				if fv.IsNil() {
					continue
				}
				inner := fv.Elem().Elem() // interface -> *T -> T
				tag := inner.Type().Field(0).Tag.Get("protobuf")
				props = new(Properties) // Overwrite the outer props var, but not its pointee.
				props.Parse(tag)
				// Write the value in the oneof, not the oneof itself.
				fv = inner.Field(0)

				// Special case to cope with malformed messages gracefully:
				// If the value in the oneof is a nil pointer, don't panic
				// in writeAny.
				if fv.Kind() == reflect.Ptr && fv.IsNil() {
					// Use errors.New so writeAny won't render quotes.
					msg := errors.New("/* nil */")
					fv = reflect.ValueOf(&msg).Elem()
				}
			}
		}

		if err := writeName(w, props); err != nil {
			return err
		}
		if !w.compact {
			if err := w.WriteByte(' '); err != nil {
				return err
			}
		}

		if len(props.Enum) > 0 {
			if err := tm.writeEnum(w, fv, props); err != nil {
				return err
			}
		} else if err := tm.writeAny(w, fv, props); err != nil {
			return err
		}

		if err := w.WriteByte('\n'); err != nil {
			return err
		}
	}

	// Extensions (the XXX_extensions field).
	pv := sv
	if pv.CanAddr() {
		pv = sv.Addr()
	} else {
		pv = reflect.New(sv.Type())
		pv.Elem().Set(sv)
	}
	if _, err := extendable(pv.Interface()); err == nil {
		if err := tm.writeExtensions(w, pv); err != nil {
			return err
		}
	}

	return nil
}

// writeAny writes an arbitrary field.
func (tm *TextMarshaler) writeAny(w *textWriter, v reflect.Value, props *Properties) error {
	v = reflect.Indirect(v)

	if props != nil {
		if len(props.CustomType) > 0 {
			custom, ok := v.Interface().(Marshaler)
			if ok {
				data, err := custom.Marshal()
				if err != nil {
					return err
				}
				if err := writeString(w, string(data)); err != nil {
					return err
				}
				return nil
			}
		} else if len(props.CastType) > 0 {
			if _, ok := v.Interface().(interface {
				String() string
			}); ok {
				switch v.Kind() {
				case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
					reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
					_, err := fmt.Fprintf(w, "%d", v.Interface())
					return err
				}
			}
		} else if props.StdTime {
			t, ok := v.Interface().(time.Time)
			if !ok {
				return fmt.Errorf("stdtime is not time.Time, but %T", v.Interface())
			}
			tproto, err := timestampProto(t)
			if err != nil {
				return err
			}
			propsCopy := *props // Make a copy so that this is goroutine-safe
			propsCopy.StdTime = false
			err = tm.writeAny(w, reflect.ValueOf(tproto), &propsCopy)
			return err
		} else if props.StdDuration {
			d, ok := v.Interface().(time.Duration)
			if !ok {
				return fmt.Errorf("stdtime is not time.Duration, but %T", v.Interface())
			}
			dproto := durationProto(d)
			propsCopy := *props // Make a copy so that this is goroutine-safe
			propsCopy.StdDuration = false
			err := tm.writeAny(w, reflect.ValueOf(dproto), &propsCopy)
			return err
		}
	}

	// Floats have special cases.
	if v.Kind() == reflect.Float32 || v.Kind() == reflect.Float64 {
		x := v.Float()
		var b []byte
		switch {
		case math.IsInf(x, 1):
			b = posInf
		case math.IsInf(x, -1):
			b = negInf
		case math.IsNaN(x):
			b = nan
		}
		if b != nil {
			_, err := w.Write(b)
			return err
		}
		// Other values are handled below.
	}

	// We don't attempt to serialise every possible value type; only those
	// that can occur in protocol buffers.
	switch v.Kind() {
	case reflect.Slice:
		// Should only be a []byte; repeated fields are handled in writeStruct.
		if err := writeString(w, string(v.Bytes())); err != nil {
			return err
		}
	case reflect.String:
		if err := writeString(w, v.String()); err != nil {
			return err
		}
	case reflect.Struct:
		// Required/optional group/message.
		var bra, ket byte = '<', '>'
		if props != nil && props.Wire == "group" {
			bra, ket = '{', '}'
		}
		if err := w.WriteByte(bra); err != nil {
			return err
		}
		if !w.compact {
			if err := w.WriteByte('\n'); err != nil {
				return err
			}
		}
		w.indent()
		if v.CanAddr() {
			// Calling v.Interface on a struct causes the reflect package to
			// copy the entire struct. This is racy with the new Marshaler
			// since we atomically update the XXX_sizecache.
			//
			// Thus, we retrieve a pointer to the struct if possible to avoid
			// a race since v.Interface on the pointer doesn't copy the struct.
			//
			// If v is not addressable, then we are not worried about a race
			// since it implies that the binary Marshaler cannot possibly be
			// mutating this value.
			v = v.Addr()
		}
		if etm, ok := v.Interface().(encoding.TextMarshaler); ok {
			text, err := etm.MarshalText()
			if err != nil {
				return err
			}
			if _, err = w.Write(text); err != nil {
				return err
			}
		} else {
			if v.Kind() == reflect.Ptr {
				v = v.Elem()
			}
			if err := tm.writeStruct(w, v); err != nil {
				return err
			}
		}
		w.unindent()
		if err := w.WriteByte(ket); err != nil {
			return err
		}
	default:
		_, err := fmt.Fprint(w, v.Interface())
		return err
	}
	return nil
}

// equivalent to C's isprint.
func isprint(c byte) bool {
	return c >= 0x20 && c < 0x7f
}

// writeString writes a string in the protocol buffer text format.
// It is similar to strconv.Quote except we don't use Go escape sequences,
// we treat the string as a byte sequence, and we use octal escapes.
// These differences are to maintain interoperability with the other
// languages' implementations of the text format.
func writeString(w *textWriter, s string) error {
	// use WriteByte here to get any needed indent
	if err := w.WriteByte('"'); err != nil {
		return err
	}
	// Loop over the bytes, not the runes.
	for i := 0; i < len(s); i++ {
		var err error
		// Divergence from C++: we don't escape apostrophes.
		// There's no need to escape them, and the C++ parser
		// copes with a naked apostrophe.
		switch c := s[i]; c {
		case '\n':
			_, err = w.w.Write(backslashN)
		case '\r':
			_, err = w.w.Write(backslashR)
		case '\t':
			_, err = w.w.Write(backslashT)
		case '"':
			_, err = w.w.Write(backslashDQ)
		case '\\':
			_, err = w.w.Write(backslashBS)
		default:
			if isprint(c) {
				err = w.w.WriteByte(c)
			} else {
				_, err = fmt.Fprintf(w.w, "\\%03o", c)
			}
		}
		if err != nil {
			return err
		}
	}
	return w.WriteByte('"')
}

func writeUnknownStruct(w *textWriter, data []byte) (err error) {
	if !w.compact {
		if _, err := fmt.Fprintf(w, "/* %d unknown bytes */\n", len(data)); err != nil {
			return err
		}
	}
	b := NewBuffer(data)
	for b.index < len(b.buf) {
		x, err := b.DecodeVarint()
		if err != nil {
			_, ferr := fmt.Fprintf(w, "/* %v */\n", err)
			return ferr
		}
		wire, tag := x&7, x>>3
		if wire == WireEndGroup {
			w.unindent()
			if _, werr := w.Write(endBraceNewline); werr != nil {
				return werr
			}
			continue
		}
		if _, ferr := fmt.Fprint(w, tag); ferr != nil {
			return ferr
		}
		if wire != WireStartGroup {
			if err = w.WriteByte(':'); err != nil {
				return err
			}
		}
		if !w.compact || wire == WireStartGroup {
			if err = w.WriteByte(' '); err != nil {
				return err
			}
		}
		switch wire {
		case WireBytes:
			buf, e := b.DecodeRawBytes(false)
			if e == nil {
				_, err = fmt.Fprintf(w, "%q", buf)
			} else {
				_, err = fmt.Fprintf(w, "/* %v */", e)
			}
		case WireFixed32:
			x, err = b.DecodeFixed32()
			err = writeUnknownInt(w, x, err)
		case WireFixed64:
			x, err = b.DecodeFixed64()
			err = writeUnknownInt(w, x, err)
		case WireStartGroup:
			err = w.WriteByte('{')
			w.indent()
		case WireVarint:
			x, err = b.DecodeVarint()
			err = writeUnknownInt(w, x, err)
		default:
			_, err = fmt.Fprintf(w, "/* unknown wire type %d */", wire)
		}
		if err != nil {
			return err
		}
		if err := w.WriteByte('\n'); err != nil {
			return err
		}
	}
	return nil
}

func writeUnknownInt(w *textWriter, x uint64, err error) error {
	if err == nil {
		_, err = fmt.Fprint(w, x)
	} else {
		_, err = fmt.Fprintf(w, "/* %v */", err)
	}
	return err
}

type int32Slice []int32

func (s int32Slice) Len() int           { return len(s) }
func (s int32Slice) Less(i, j int) bool { return s[i] < s[j] }
func (s int32Slice) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// writeExtensions writes all the extensions in pv.
// pv is assumed to be a pointer to a protocol message struct that is extendable.
func (tm *TextMarshaler) writeExtensions(w *textWriter, pv reflect.Value) error {
	emap := extensionMaps[pv.Type().Elem()]
	e := pv.Interface().(Message)

	var m map[int32]Extension
	var mu sync.Locker
	if em, ok := e.(extensionsBytes); ok {
		eb := em.GetExtensions()
		var err error
		m, err = BytesToExtensionsMap(*eb)
		if err != nil {
			return err
		}
		mu = notLocker{}
	} else if _, ok := e.(extendableProto); ok {
		ep, _ := extendable(e)
		m, mu = ep.extensionsRead()
		if m == nil {
			return nil
		}
	}

	// Order the extensions by ID.
	// This isn't strictly necessary, but it will give us
	// canonical output, which will also make testing easier.

	mu.Lock()
	ids := make([]int32, 0, len(m))
	for id := range m {
		ids = append(ids, id)
	}
	sort.Sort(int32Slice(ids))
	mu.Unlock()

	for _, extNum := range ids {
		ext := m[extNum]
		var desc *ExtensionDesc
		if emap != nil {
			desc = emap[extNum]
		}
		if desc == nil {
			// Unknown extension.
			if err := writeUnknownStruct(w, ext.enc); err != nil {
				return err
			}
			continue
		}

		pb, err := GetExtension(e, desc)
		if err != nil {
			return fmt.Errorf("failed getting extension: %v", err)
		}

		// Repeated extensions will appear as a slice.
		if !desc.repeated() {
			if err := tm.writeExtension(w, desc.Name, pb); err != nil {
				return err
			}
		} else {
			v := reflect.ValueOf(pb)
			for i := 0; i < v.Len(); i++ {
				if err := tm.writeExtension(w, desc.Name, v.Index(i).Interface()); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (tm *TextMarshaler) writeExtension(w *textWriter, name string, pb interface{}) error {
	if _, err := fmt.Fprintf(w, "[%s]:", name); err != nil {
		return err
	}
	if !w.compact {
		if err := w.WriteByte(' '); err != nil {
			return err
		}
	}
	if err := tm.writeAny(w, reflect.ValueOf(pb), nil); err != nil {
		return err
	}
	if err := w.WriteByte('\n'); err != nil {
		return err
	}
	return nil
}

func (w *textWriter) writeIndent() {
	if !w.complete {
		return
	}
	remain := w.ind * 2
	for remain > 0 {
		n := remain
		if n > len(spaces) {
			n = len(spaces)
		}
		w.w.Write(spaces[:n])
		remain -= n
	}
	w.complete = false
}

// TextMarshaler is a configurable text format marshaler.
type TextMarshaler struct {
	Compact   bool // use compact text format (one line).
	ExpandAny bool // expand google.protobuf.Any messages of known types
}

// Marshal writes a given protocol buffer in text format.
// The only errors returned are from w.
func (tm *TextMarshaler) Marshal(w io.Writer, pb Message) error {
	val := reflect.ValueOf(pb)
	if pb == nil || val.IsNil() {
		w.Write([]byte("<nil>"))
		return nil
	}
	var bw *bufio.Writer
	ww, ok := w.(writer)
	if !ok {
		bw = bufio.NewWriter(w)
		ww = bw
	}
	aw := &textWriter{
		w:        ww,
		complete: true,
		compact:  tm.Compact,
	}

	if etm, ok := pb.(encoding.TextMarshaler); ok {
		text, err := etm.MarshalText()
		if err != nil {
			return err
		}
		if _, err = aw.Write(text); err != nil {
			return err
		}
		if bw != nil {
			return bw.Flush()
		}
		return nil
	}
	// Dereference the received pointer so we don't have outer < and >.
	v := reflect.Indirect(val)
	if err := tm.writeStruct(aw, v); err != nil {
		return err
	}
	if bw != nil {
		return bw.Flush()
	}
	return nil
}

// Text is the same as Marshal, but returns the string directly.
func (tm *TextMarshaler) Text(pb Message) string {
	var buf bytes.Buffer
	tm.Marshal(&buf, pb)
	return buf.String()
}

var (
	defaultTextMarshaler = TextMarshaler{}
	compactTextMarshaler = TextMarshaler{Compact: true}
)

// TODO: consider removing some of the Marshal functions below.

// MarshalText writes a given protocol buffer in text format.
// The only errors returned are from w.
func MarshalText(w io.Writer, pb Message) error { return defaultTextMarshaler.Marshal(w, pb) }

// MarshalTextString is the same as MarshalText, but returns the string directly.
func MarshalTextString(pb Message) string { return defaultTextMarshaler.Text(pb) }

// CompactText writes a given protocol buffer in compact text format (one line).
func CompactText(w io.Writer, pb Message) error { return compactTextMarshaler.Marshal(w, pb) }

// CompactTextString is the same as CompactText, but returns the string directly.
func CompactTextString(pb Message) string { return compactTextMarshaler.Text(pb) }
