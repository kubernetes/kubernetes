// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"bytes"
	"encoding"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

const wrapTextMarshalV2 = false

// TextMarshaler is a configurable text format marshaler.
type TextMarshaler struct {
	Compact   bool // use compact text format (one line)
	ExpandAny bool // expand google.protobuf.Any messages of known types
}

// Marshal writes the proto text format of m to w.
func (tm *TextMarshaler) Marshal(w io.Writer, m Message) error {
	b, err := tm.marshal(m)
	if len(b) > 0 {
		if _, err := w.Write(b); err != nil {
			return err
		}
	}
	return err
}

// Text returns a proto text formatted string of m.
func (tm *TextMarshaler) Text(m Message) string {
	b, _ := tm.marshal(m)
	return string(b)
}

func (tm *TextMarshaler) marshal(m Message) ([]byte, error) {
	mr := MessageReflect(m)
	if mr == nil || !mr.IsValid() {
		return []byte("<nil>"), nil
	}

	if wrapTextMarshalV2 {
		if m, ok := m.(encoding.TextMarshaler); ok {
			return m.MarshalText()
		}

		opts := prototext.MarshalOptions{
			AllowPartial: true,
			EmitUnknown:  true,
		}
		if !tm.Compact {
			opts.Indent = "  "
		}
		if !tm.ExpandAny {
			opts.Resolver = (*protoregistry.Types)(nil)
		}
		return opts.Marshal(mr.Interface())
	} else {
		w := &textWriter{
			compact:   tm.Compact,
			expandAny: tm.ExpandAny,
			complete:  true,
		}

		if m, ok := m.(encoding.TextMarshaler); ok {
			b, err := m.MarshalText()
			if err != nil {
				return nil, err
			}
			w.Write(b)
			return w.buf, nil
		}

		err := w.writeMessage(mr)
		return w.buf, err
	}
}

var (
	defaultTextMarshaler = TextMarshaler{}
	compactTextMarshaler = TextMarshaler{Compact: true}
)

// MarshalText writes the proto text format of m to w.
func MarshalText(w io.Writer, m Message) error { return defaultTextMarshaler.Marshal(w, m) }

// MarshalTextString returns a proto text formatted string of m.
func MarshalTextString(m Message) string { return defaultTextMarshaler.Text(m) }

// CompactText writes the compact proto text format of m to w.
func CompactText(w io.Writer, m Message) error { return compactTextMarshaler.Marshal(w, m) }

// CompactTextString returns a compact proto text formatted string of m.
func CompactTextString(m Message) string { return compactTextMarshaler.Text(m) }

var (
	newline         = []byte("\n")
	endBraceNewline = []byte("}\n")
	posInf          = []byte("inf")
	negInf          = []byte("-inf")
	nan             = []byte("nan")
)

// textWriter is an io.Writer that tracks its indentation level.
type textWriter struct {
	compact   bool // same as TextMarshaler.Compact
	expandAny bool // same as TextMarshaler.ExpandAny
	complete  bool // whether the current position is a complete line
	indent    int  // indentation level; never negative
	buf       []byte
}

func (w *textWriter) Write(p []byte) (n int, _ error) {
	newlines := bytes.Count(p, newline)
	if newlines == 0 {
		if !w.compact && w.complete {
			w.writeIndent()
		}
		w.buf = append(w.buf, p...)
		w.complete = false
		return len(p), nil
	}

	frags := bytes.SplitN(p, newline, newlines+1)
	if w.compact {
		for i, frag := range frags {
			if i > 0 {
				w.buf = append(w.buf, ' ')
				n++
			}
			w.buf = append(w.buf, frag...)
			n += len(frag)
		}
		return n, nil
	}

	for i, frag := range frags {
		if w.complete {
			w.writeIndent()
		}
		w.buf = append(w.buf, frag...)
		n += len(frag)
		if i+1 < len(frags) {
			w.buf = append(w.buf, '\n')
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
	w.buf = append(w.buf, c)
	w.complete = c == '\n'
	return nil
}

func (w *textWriter) writeName(fd protoreflect.FieldDescriptor) {
	if !w.compact && w.complete {
		w.writeIndent()
	}
	w.complete = false

	if fd.Kind() != protoreflect.GroupKind {
		w.buf = append(w.buf, fd.Name()...)
		w.WriteByte(':')
	} else {
		// Use message type name for group field name.
		w.buf = append(w.buf, fd.Message().Name()...)
	}

	if !w.compact {
		w.WriteByte(' ')
	}
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

// writeProto3Any writes an expanded google.protobuf.Any message.
//
// It returns (false, nil) if sv value can't be unmarshaled (e.g. because
// required messages are not linked in).
//
// It returns (true, error) when sv was written in expanded format or an error
// was encountered.
func (w *textWriter) writeProto3Any(m protoreflect.Message) (bool, error) {
	md := m.Descriptor()
	fdURL := md.Fields().ByName("type_url")
	fdVal := md.Fields().ByName("value")

	url := m.Get(fdURL).String()
	mt, err := protoregistry.GlobalTypes.FindMessageByURL(url)
	if err != nil {
		return false, nil
	}

	b := m.Get(fdVal).Bytes()
	m2 := mt.New()
	if err := proto.Unmarshal(b, m2.Interface()); err != nil {
		return false, nil
	}
	w.Write([]byte("["))
	if requiresQuotes(url) {
		w.writeQuotedString(url)
	} else {
		w.Write([]byte(url))
	}
	if w.compact {
		w.Write([]byte("]:<"))
	} else {
		w.Write([]byte("]: <\n"))
		w.indent++
	}
	if err := w.writeMessage(m2); err != nil {
		return true, err
	}
	if w.compact {
		w.Write([]byte("> "))
	} else {
		w.indent--
		w.Write([]byte(">\n"))
	}
	return true, nil
}

func (w *textWriter) writeMessage(m protoreflect.Message) error {
	md := m.Descriptor()
	if w.expandAny && md.FullName() == "google.protobuf.Any" {
		if canExpand, err := w.writeProto3Any(m); canExpand {
			return err
		}
	}

	fds := md.Fields()
	for i := 0; i < fds.Len(); {
		fd := fds.Get(i)
		if od := fd.ContainingOneof(); od != nil {
			fd = m.WhichOneof(od)
			i += od.Fields().Len()
		} else {
			i++
		}
		if fd == nil || !m.Has(fd) {
			continue
		}

		switch {
		case fd.IsList():
			lv := m.Get(fd).List()
			for j := 0; j < lv.Len(); j++ {
				w.writeName(fd)
				v := lv.Get(j)
				if err := w.writeSingularValue(v, fd); err != nil {
					return err
				}
				w.WriteByte('\n')
			}
		case fd.IsMap():
			kfd := fd.MapKey()
			vfd := fd.MapValue()
			mv := m.Get(fd).Map()

			type entry struct{ key, val protoreflect.Value }
			var entries []entry
			mv.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
				entries = append(entries, entry{k.Value(), v})
				return true
			})
			sort.Slice(entries, func(i, j int) bool {
				switch kfd.Kind() {
				case protoreflect.BoolKind:
					return !entries[i].key.Bool() && entries[j].key.Bool()
				case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind, protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
					return entries[i].key.Int() < entries[j].key.Int()
				case protoreflect.Uint32Kind, protoreflect.Fixed32Kind, protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
					return entries[i].key.Uint() < entries[j].key.Uint()
				case protoreflect.StringKind:
					return entries[i].key.String() < entries[j].key.String()
				default:
					panic("invalid kind")
				}
			})
			for _, entry := range entries {
				w.writeName(fd)
				w.WriteByte('<')
				if !w.compact {
					w.WriteByte('\n')
				}
				w.indent++
				w.writeName(kfd)
				if err := w.writeSingularValue(entry.key, kfd); err != nil {
					return err
				}
				w.WriteByte('\n')
				w.writeName(vfd)
				if err := w.writeSingularValue(entry.val, vfd); err != nil {
					return err
				}
				w.WriteByte('\n')
				w.indent--
				w.WriteByte('>')
				w.WriteByte('\n')
			}
		default:
			w.writeName(fd)
			if err := w.writeSingularValue(m.Get(fd), fd); err != nil {
				return err
			}
			w.WriteByte('\n')
		}
	}

	if b := m.GetUnknown(); len(b) > 0 {
		w.writeUnknownFields(b)
	}
	return w.writeExtensions(m)
}

func (w *textWriter) writeSingularValue(v protoreflect.Value, fd protoreflect.FieldDescriptor) error {
	switch fd.Kind() {
	case protoreflect.FloatKind, protoreflect.DoubleKind:
		switch vf := v.Float(); {
		case math.IsInf(vf, +1):
			w.Write(posInf)
		case math.IsInf(vf, -1):
			w.Write(negInf)
		case math.IsNaN(vf):
			w.Write(nan)
		default:
			fmt.Fprint(w, v.Interface())
		}
	case protoreflect.StringKind:
		// NOTE: This does not validate UTF-8 for historical reasons.
		w.writeQuotedString(string(v.String()))
	case protoreflect.BytesKind:
		w.writeQuotedString(string(v.Bytes()))
	case protoreflect.MessageKind, protoreflect.GroupKind:
		var bra, ket byte = '<', '>'
		if fd.Kind() == protoreflect.GroupKind {
			bra, ket = '{', '}'
		}
		w.WriteByte(bra)
		if !w.compact {
			w.WriteByte('\n')
		}
		w.indent++
		m := v.Message()
		if m2, ok := m.Interface().(encoding.TextMarshaler); ok {
			b, err := m2.MarshalText()
			if err != nil {
				return err
			}
			w.Write(b)
		} else {
			w.writeMessage(m)
		}
		w.indent--
		w.WriteByte(ket)
	case protoreflect.EnumKind:
		if ev := fd.Enum().Values().ByNumber(v.Enum()); ev != nil {
			fmt.Fprint(w, ev.Name())
		} else {
			fmt.Fprint(w, v.Enum())
		}
	default:
		fmt.Fprint(w, v.Interface())
	}
	return nil
}

// writeQuotedString writes a quoted string in the protocol buffer text format.
func (w *textWriter) writeQuotedString(s string) {
	w.WriteByte('"')
	for i := 0; i < len(s); i++ {
		switch c := s[i]; c {
		case '\n':
			w.buf = append(w.buf, `\n`...)
		case '\r':
			w.buf = append(w.buf, `\r`...)
		case '\t':
			w.buf = append(w.buf, `\t`...)
		case '"':
			w.buf = append(w.buf, `\"`...)
		case '\\':
			w.buf = append(w.buf, `\\`...)
		default:
			if isPrint := c >= 0x20 && c < 0x7f; isPrint {
				w.buf = append(w.buf, c)
			} else {
				w.buf = append(w.buf, fmt.Sprintf(`\%03o`, c)...)
			}
		}
	}
	w.WriteByte('"')
}

func (w *textWriter) writeUnknownFields(b []byte) {
	if !w.compact {
		fmt.Fprintf(w, "/* %d unknown bytes */\n", len(b))
	}

	for len(b) > 0 {
		num, wtyp, n := protowire.ConsumeTag(b)
		if n < 0 {
			return
		}
		b = b[n:]

		if wtyp == protowire.EndGroupType {
			w.indent--
			w.Write(endBraceNewline)
			continue
		}
		fmt.Fprint(w, num)
		if wtyp != protowire.StartGroupType {
			w.WriteByte(':')
		}
		if !w.compact || wtyp == protowire.StartGroupType {
			w.WriteByte(' ')
		}
		switch wtyp {
		case protowire.VarintType:
			v, n := protowire.ConsumeVarint(b)
			if n < 0 {
				return
			}
			b = b[n:]
			fmt.Fprint(w, v)
		case protowire.Fixed32Type:
			v, n := protowire.ConsumeFixed32(b)
			if n < 0 {
				return
			}
			b = b[n:]
			fmt.Fprint(w, v)
		case protowire.Fixed64Type:
			v, n := protowire.ConsumeFixed64(b)
			if n < 0 {
				return
			}
			b = b[n:]
			fmt.Fprint(w, v)
		case protowire.BytesType:
			v, n := protowire.ConsumeBytes(b)
			if n < 0 {
				return
			}
			b = b[n:]
			fmt.Fprintf(w, "%q", v)
		case protowire.StartGroupType:
			w.WriteByte('{')
			w.indent++
		default:
			fmt.Fprintf(w, "/* unknown wire type %d */", wtyp)
		}
		w.WriteByte('\n')
	}
}

// writeExtensions writes all the extensions in m.
func (w *textWriter) writeExtensions(m protoreflect.Message) error {
	md := m.Descriptor()
	if md.ExtensionRanges().Len() == 0 {
		return nil
	}

	type ext struct {
		desc protoreflect.FieldDescriptor
		val  protoreflect.Value
	}
	var exts []ext
	m.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		if fd.IsExtension() {
			exts = append(exts, ext{fd, v})
		}
		return true
	})
	sort.Slice(exts, func(i, j int) bool {
		return exts[i].desc.Number() < exts[j].desc.Number()
	})

	for _, ext := range exts {
		// For message set, use the name of the message as the extension name.
		name := string(ext.desc.FullName())
		if isMessageSet(ext.desc.ContainingMessage()) {
			name = strings.TrimSuffix(name, ".message_set_extension")
		}

		if !ext.desc.IsList() {
			if err := w.writeSingularExtension(name, ext.val, ext.desc); err != nil {
				return err
			}
		} else {
			lv := ext.val.List()
			for i := 0; i < lv.Len(); i++ {
				if err := w.writeSingularExtension(name, lv.Get(i), ext.desc); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (w *textWriter) writeSingularExtension(name string, v protoreflect.Value, fd protoreflect.FieldDescriptor) error {
	fmt.Fprintf(w, "[%s]:", name)
	if !w.compact {
		w.WriteByte(' ')
	}
	if err := w.writeSingularValue(v, fd); err != nil {
		return err
	}
	w.WriteByte('\n')
	return nil
}

func (w *textWriter) writeIndent() {
	if !w.complete {
		return
	}
	for i := 0; i < w.indent*2; i++ {
		w.buf = append(w.buf, ' ')
	}
	w.complete = false
}
