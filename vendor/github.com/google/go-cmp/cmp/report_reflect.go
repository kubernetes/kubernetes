// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

package cmp

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"unicode"

	"github.com/google/go-cmp/cmp/internal/flags"
	"github.com/google/go-cmp/cmp/internal/value"
)

type formatValueOptions struct {
	// AvoidStringer controls whether to avoid calling custom stringer
	// methods like error.Error or fmt.Stringer.String.
	AvoidStringer bool

	// ShallowPointers controls whether to avoid descending into pointers.
	// Useful when printing map keys, where pointer comparison is performed
	// on the pointer address rather than the pointed-at value.
	ShallowPointers bool

	// PrintAddresses controls whether to print the address of all pointers,
	// slice elements, and maps.
	PrintAddresses bool
}

// FormatType prints the type as if it were wrapping s.
// This may return s as-is depending on the current type and TypeMode mode.
func (opts formatOptions) FormatType(t reflect.Type, s textNode) textNode {
	// Check whether to emit the type or not.
	switch opts.TypeMode {
	case autoType:
		switch t.Kind() {
		case reflect.Struct, reflect.Slice, reflect.Array, reflect.Map:
			if s.Equal(textNil) {
				return s
			}
		default:
			return s
		}
	case elideType:
		return s
	}

	// Determine the type label, applying special handling for unnamed types.
	typeName := t.String()
	if t.Name() == "" {
		// According to Go grammar, certain type literals contain symbols that
		// do not strongly bind to the next lexicographical token (e.g., *T).
		switch t.Kind() {
		case reflect.Chan, reflect.Func, reflect.Ptr:
			typeName = "(" + typeName + ")"
		}
		typeName = strings.Replace(typeName, "struct {", "struct{", -1)
		typeName = strings.Replace(typeName, "interface {", "interface{", -1)
	}

	// Avoid wrap the value in parenthesis if unnecessary.
	if s, ok := s.(textWrap); ok {
		hasParens := strings.HasPrefix(s.Prefix, "(") && strings.HasSuffix(s.Suffix, ")")
		hasBraces := strings.HasPrefix(s.Prefix, "{") && strings.HasSuffix(s.Suffix, "}")
		if hasParens || hasBraces {
			return textWrap{typeName, s, ""}
		}
	}
	return textWrap{typeName + "(", s, ")"}
}

// FormatValue prints the reflect.Value, taking extra care to avoid descending
// into pointers already in m. As pointers are visited, m is also updated.
func (opts formatOptions) FormatValue(v reflect.Value, m visitedPointers) (out textNode) {
	if !v.IsValid() {
		return nil
	}
	t := v.Type()

	// Check whether there is an Error or String method to call.
	if !opts.AvoidStringer && v.CanInterface() {
		// Avoid calling Error or String methods on nil receivers since many
		// implementations crash when doing so.
		if (t.Kind() != reflect.Ptr && t.Kind() != reflect.Interface) || !v.IsNil() {
			switch v := v.Interface().(type) {
			case error:
				return textLine("e" + formatString(v.Error()))
			case fmt.Stringer:
				return textLine("s" + formatString(v.String()))
			}
		}
	}

	// Check whether to explicitly wrap the result with the type.
	var skipType bool
	defer func() {
		if !skipType {
			out = opts.FormatType(t, out)
		}
	}()

	var ptr string
	switch t.Kind() {
	case reflect.Bool:
		return textLine(fmt.Sprint(v.Bool()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return textLine(fmt.Sprint(v.Int()))
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		// Unnamed uints are usually bytes or words, so use hexadecimal.
		if t.PkgPath() == "" || t.Kind() == reflect.Uintptr {
			return textLine(formatHex(v.Uint()))
		}
		return textLine(fmt.Sprint(v.Uint()))
	case reflect.Float32, reflect.Float64:
		return textLine(fmt.Sprint(v.Float()))
	case reflect.Complex64, reflect.Complex128:
		return textLine(fmt.Sprint(v.Complex()))
	case reflect.String:
		return textLine(formatString(v.String()))
	case reflect.UnsafePointer, reflect.Chan, reflect.Func:
		return textLine(formatPointer(v))
	case reflect.Struct:
		var list textList
		for i := 0; i < v.NumField(); i++ {
			vv := v.Field(i)
			if value.IsZero(vv) {
				continue // Elide fields with zero values
			}
			s := opts.WithTypeMode(autoType).FormatValue(vv, m)
			list = append(list, textRecord{Key: t.Field(i).Name, Value: s})
		}
		return textWrap{"{", list, "}"}
	case reflect.Slice:
		if v.IsNil() {
			return textNil
		}
		if opts.PrintAddresses {
			ptr = formatPointer(v)
		}
		fallthrough
	case reflect.Array:
		var list textList
		for i := 0; i < v.Len(); i++ {
			vi := v.Index(i)
			if vi.CanAddr() { // Check for cyclic elements
				p := vi.Addr()
				if m.Visit(p) {
					var out textNode
					out = textLine(formatPointer(p))
					out = opts.WithTypeMode(emitType).FormatType(p.Type(), out)
					out = textWrap{"*", out, ""}
					list = append(list, textRecord{Value: out})
					continue
				}
			}
			s := opts.WithTypeMode(elideType).FormatValue(vi, m)
			list = append(list, textRecord{Value: s})
		}
		return textWrap{ptr + "{", list, "}"}
	case reflect.Map:
		if v.IsNil() {
			return textNil
		}
		if m.Visit(v) {
			return textLine(formatPointer(v))
		}

		var list textList
		for _, k := range value.SortKeys(v.MapKeys()) {
			sk := formatMapKey(k)
			sv := opts.WithTypeMode(elideType).FormatValue(v.MapIndex(k), m)
			list = append(list, textRecord{Key: sk, Value: sv})
		}
		if opts.PrintAddresses {
			ptr = formatPointer(v)
		}
		return textWrap{ptr + "{", list, "}"}
	case reflect.Ptr:
		if v.IsNil() {
			return textNil
		}
		if m.Visit(v) || opts.ShallowPointers {
			return textLine(formatPointer(v))
		}
		if opts.PrintAddresses {
			ptr = formatPointer(v)
		}
		skipType = true // Let the underlying value print the type instead
		return textWrap{"&" + ptr, opts.FormatValue(v.Elem(), m), ""}
	case reflect.Interface:
		if v.IsNil() {
			return textNil
		}
		// Interfaces accept different concrete types,
		// so configure the underlying value to explicitly print the type.
		skipType = true // Print the concrete type instead
		return opts.WithTypeMode(emitType).FormatValue(v.Elem(), m)
	default:
		panic(fmt.Sprintf("%v kind not handled", v.Kind()))
	}
}

// formatMapKey formats v as if it were a map key.
// The result is guaranteed to be a single line.
func formatMapKey(v reflect.Value) string {
	var opts formatOptions
	opts.TypeMode = elideType
	opts.AvoidStringer = true
	opts.ShallowPointers = true
	s := opts.FormatValue(v, visitedPointers{}).String()
	return strings.TrimSpace(s)
}

// formatString prints s as a double-quoted or backtick-quoted string.
func formatString(s string) string {
	// Use quoted string if it the same length as a raw string literal.
	// Otherwise, attempt to use the raw string form.
	qs := strconv.Quote(s)
	if len(qs) == 1+len(s)+1 {
		return qs
	}

	// Disallow newlines to ensure output is a single line.
	// Only allow printable runes for readability purposes.
	rawInvalid := func(r rune) bool {
		return r == '`' || r == '\n' || !(unicode.IsPrint(r) || r == '\t')
	}
	if strings.IndexFunc(s, rawInvalid) < 0 {
		return "`" + s + "`"
	}
	return qs
}

// formatHex prints u as a hexadecimal integer in Go notation.
func formatHex(u uint64) string {
	var f string
	switch {
	case u <= 0xff:
		f = "0x%02x"
	case u <= 0xffff:
		f = "0x%04x"
	case u <= 0xffffff:
		f = "0x%06x"
	case u <= 0xffffffff:
		f = "0x%08x"
	case u <= 0xffffffffff:
		f = "0x%010x"
	case u <= 0xffffffffffff:
		f = "0x%012x"
	case u <= 0xffffffffffffff:
		f = "0x%014x"
	case u <= 0xffffffffffffffff:
		f = "0x%016x"
	}
	return fmt.Sprintf(f, u)
}

// formatPointer prints the address of the pointer.
func formatPointer(v reflect.Value) string {
	p := v.Pointer()
	if flags.Deterministic {
		p = 0xdeadf00f // Only used for stable testing purposes
	}
	return fmt.Sprintf("⟪0x%x⟫", p)
}

type visitedPointers map[value.Pointer]struct{}

// Visit inserts pointer v into the visited map and reports whether it had
// already been visited before.
func (m visitedPointers) Visit(v reflect.Value) bool {
	p := value.PointerOf(v)
	_, visited := m[p]
	m[p] = struct{}{}
	return visited
}
