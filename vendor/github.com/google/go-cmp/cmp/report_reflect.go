// Copyright 2019, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmp

import (
	"bytes"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/google/go-cmp/cmp/internal/value"
)

type formatValueOptions struct {
	// AvoidStringer controls whether to avoid calling custom stringer
	// methods like error.Error or fmt.Stringer.String.
	AvoidStringer bool

	// PrintAddresses controls whether to print the address of all pointers,
	// slice elements, and maps.
	PrintAddresses bool

	// QualifiedNames controls whether FormatType uses the fully qualified name
	// (including the full package path as opposed to just the package name).
	QualifiedNames bool

	// VerbosityLevel controls the amount of output to produce.
	// A higher value produces more output. A value of zero or lower produces
	// no output (represented using an ellipsis).
	// If LimitVerbosity is false, then the level is treated as infinite.
	VerbosityLevel int

	// LimitVerbosity specifies that formatting should respect VerbosityLevel.
	LimitVerbosity bool
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
		if opts.DiffMode == diffIdentical {
			return s // elide type for identical nodes
		}
	case elideType:
		return s
	}

	// Determine the type label, applying special handling for unnamed types.
	typeName := value.TypeString(t, opts.QualifiedNames)
	if t.Name() == "" {
		// According to Go grammar, certain type literals contain symbols that
		// do not strongly bind to the next lexicographical token (e.g., *T).
		switch t.Kind() {
		case reflect.Chan, reflect.Func, reflect.Ptr:
			typeName = "(" + typeName + ")"
		}
	}
	return &textWrap{Prefix: typeName, Value: wrapParens(s)}
}

// wrapParens wraps s with a set of parenthesis, but avoids it if the
// wrapped node itself is already surrounded by a pair of parenthesis or braces.
// It handles unwrapping one level of pointer-reference nodes.
func wrapParens(s textNode) textNode {
	var refNode *textWrap
	if s2, ok := s.(*textWrap); ok {
		// Unwrap a single pointer reference node.
		switch s2.Metadata.(type) {
		case leafReference, trunkReference, trunkReferences:
			refNode = s2
			if s3, ok := refNode.Value.(*textWrap); ok {
				s2 = s3
			}
		}

		// Already has delimiters that make parenthesis unnecessary.
		hasParens := strings.HasPrefix(s2.Prefix, "(") && strings.HasSuffix(s2.Suffix, ")")
		hasBraces := strings.HasPrefix(s2.Prefix, "{") && strings.HasSuffix(s2.Suffix, "}")
		if hasParens || hasBraces {
			return s
		}
	}
	if refNode != nil {
		refNode.Value = &textWrap{Prefix: "(", Value: refNode.Value, Suffix: ")"}
		return s
	}
	return &textWrap{Prefix: "(", Value: s, Suffix: ")"}
}

// FormatValue prints the reflect.Value, taking extra care to avoid descending
// into pointers already in ptrs. As pointers are visited, ptrs is also updated.
func (opts formatOptions) FormatValue(v reflect.Value, parentKind reflect.Kind, ptrs *pointerReferences) (out textNode) {
	if !v.IsValid() {
		return nil
	}
	t := v.Type()

	// Check slice element for cycles.
	if parentKind == reflect.Slice {
		ptrRef, visited := ptrs.Push(v.Addr())
		if visited {
			return makeLeafReference(ptrRef, false)
		}
		defer ptrs.Pop()
		defer func() { out = wrapTrunkReference(ptrRef, false, out) }()
	}

	// Check whether there is an Error or String method to call.
	if !opts.AvoidStringer && v.CanInterface() {
		// Avoid calling Error or String methods on nil receivers since many
		// implementations crash when doing so.
		if (t.Kind() != reflect.Ptr && t.Kind() != reflect.Interface) || !v.IsNil() {
			var prefix, strVal string
			func() {
				// Swallow and ignore any panics from String or Error.
				defer func() { recover() }()
				switch v := v.Interface().(type) {
				case error:
					strVal = v.Error()
					prefix = "e"
				case fmt.Stringer:
					strVal = v.String()
					prefix = "s"
				}
			}()
			if prefix != "" {
				return opts.formatString(prefix, strVal)
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

	switch t.Kind() {
	case reflect.Bool:
		return textLine(fmt.Sprint(v.Bool()))
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return textLine(fmt.Sprint(v.Int()))
	case reflect.Uint, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return textLine(fmt.Sprint(v.Uint()))
	case reflect.Uint8:
		if parentKind == reflect.Slice || parentKind == reflect.Array {
			return textLine(formatHex(v.Uint()))
		}
		return textLine(fmt.Sprint(v.Uint()))
	case reflect.Uintptr:
		return textLine(formatHex(v.Uint()))
	case reflect.Float32, reflect.Float64:
		return textLine(fmt.Sprint(v.Float()))
	case reflect.Complex64, reflect.Complex128:
		return textLine(fmt.Sprint(v.Complex()))
	case reflect.String:
		return opts.formatString("", v.String())
	case reflect.UnsafePointer, reflect.Chan, reflect.Func:
		return textLine(formatPointer(value.PointerOf(v), true))
	case reflect.Struct:
		var list textList
		v := makeAddressable(v) // needed for retrieveUnexportedField
		maxLen := v.NumField()
		if opts.LimitVerbosity {
			maxLen = ((1 << opts.verbosity()) >> 1) << 2 // 0, 4, 8, 16, 32, etc...
			opts.VerbosityLevel--
		}
		for i := 0; i < v.NumField(); i++ {
			vv := v.Field(i)
			if value.IsZero(vv) {
				continue // Elide fields with zero values
			}
			if len(list) == maxLen {
				list.AppendEllipsis(diffStats{})
				break
			}
			sf := t.Field(i)
			if supportExporters && !isExported(sf.Name) {
				vv = retrieveUnexportedField(v, sf, true)
			}
			s := opts.WithTypeMode(autoType).FormatValue(vv, t.Kind(), ptrs)
			list = append(list, textRecord{Key: sf.Name, Value: s})
		}
		return &textWrap{Prefix: "{", Value: list, Suffix: "}"}
	case reflect.Slice:
		if v.IsNil() {
			return textNil
		}

		// Check whether this is a []byte of text data.
		if t.Elem() == reflect.TypeOf(byte(0)) {
			b := v.Bytes()
			isPrintSpace := func(r rune) bool { return unicode.IsPrint(r) && unicode.IsSpace(r) }
			if len(b) > 0 && utf8.Valid(b) && len(bytes.TrimFunc(b, isPrintSpace)) == 0 {
				out = opts.formatString("", string(b))
				return opts.WithTypeMode(emitType).FormatType(t, out)
			}
		}

		fallthrough
	case reflect.Array:
		maxLen := v.Len()
		if opts.LimitVerbosity {
			maxLen = ((1 << opts.verbosity()) >> 1) << 2 // 0, 4, 8, 16, 32, etc...
			opts.VerbosityLevel--
		}
		var list textList
		for i := 0; i < v.Len(); i++ {
			if len(list) == maxLen {
				list.AppendEllipsis(diffStats{})
				break
			}
			s := opts.WithTypeMode(elideType).FormatValue(v.Index(i), t.Kind(), ptrs)
			list = append(list, textRecord{Value: s})
		}

		out = &textWrap{Prefix: "{", Value: list, Suffix: "}"}
		if t.Kind() == reflect.Slice && opts.PrintAddresses {
			header := fmt.Sprintf("ptr:%v, len:%d, cap:%d", formatPointer(value.PointerOf(v), false), v.Len(), v.Cap())
			out = &textWrap{Prefix: pointerDelimPrefix + header + pointerDelimSuffix, Value: out}
		}
		return out
	case reflect.Map:
		if v.IsNil() {
			return textNil
		}

		// Check pointer for cycles.
		ptrRef, visited := ptrs.Push(v)
		if visited {
			return makeLeafReference(ptrRef, opts.PrintAddresses)
		}
		defer ptrs.Pop()

		maxLen := v.Len()
		if opts.LimitVerbosity {
			maxLen = ((1 << opts.verbosity()) >> 1) << 2 // 0, 4, 8, 16, 32, etc...
			opts.VerbosityLevel--
		}
		var list textList
		for _, k := range value.SortKeys(v.MapKeys()) {
			if len(list) == maxLen {
				list.AppendEllipsis(diffStats{})
				break
			}
			sk := formatMapKey(k, false, ptrs)
			sv := opts.WithTypeMode(elideType).FormatValue(v.MapIndex(k), t.Kind(), ptrs)
			list = append(list, textRecord{Key: sk, Value: sv})
		}

		out = &textWrap{Prefix: "{", Value: list, Suffix: "}"}
		out = wrapTrunkReference(ptrRef, opts.PrintAddresses, out)
		return out
	case reflect.Ptr:
		if v.IsNil() {
			return textNil
		}

		// Check pointer for cycles.
		ptrRef, visited := ptrs.Push(v)
		if visited {
			out = makeLeafReference(ptrRef, opts.PrintAddresses)
			return &textWrap{Prefix: "&", Value: out}
		}
		defer ptrs.Pop()

		skipType = true // Let the underlying value print the type instead
		out = opts.FormatValue(v.Elem(), t.Kind(), ptrs)
		out = wrapTrunkReference(ptrRef, opts.PrintAddresses, out)
		out = &textWrap{Prefix: "&", Value: out}
		return out
	case reflect.Interface:
		if v.IsNil() {
			return textNil
		}
		// Interfaces accept different concrete types,
		// so configure the underlying value to explicitly print the type.
		skipType = true // Print the concrete type instead
		return opts.WithTypeMode(emitType).FormatValue(v.Elem(), t.Kind(), ptrs)
	default:
		panic(fmt.Sprintf("%v kind not handled", v.Kind()))
	}
}

func (opts formatOptions) formatString(prefix, s string) textNode {
	maxLen := len(s)
	maxLines := strings.Count(s, "\n") + 1
	if opts.LimitVerbosity {
		maxLen = (1 << opts.verbosity()) << 5   // 32, 64, 128, 256, etc...
		maxLines = (1 << opts.verbosity()) << 2 //  4, 8, 16, 32, 64, etc...
	}

	// For multiline strings, use the triple-quote syntax,
	// but only use it when printing removed or inserted nodes since
	// we only want the extra verbosity for those cases.
	lines := strings.Split(strings.TrimSuffix(s, "\n"), "\n")
	isTripleQuoted := len(lines) >= 4 && (opts.DiffMode == '-' || opts.DiffMode == '+')
	for i := 0; i < len(lines) && isTripleQuoted; i++ {
		lines[i] = strings.TrimPrefix(strings.TrimSuffix(lines[i], "\r"), "\r") // trim leading/trailing carriage returns for legacy Windows endline support
		isPrintable := func(r rune) bool {
			return unicode.IsPrint(r) || r == '\t' // specially treat tab as printable
		}
		line := lines[i]
		isTripleQuoted = !strings.HasPrefix(strings.TrimPrefix(line, prefix), `"""`) && !strings.HasPrefix(line, "...") && strings.TrimFunc(line, isPrintable) == "" && len(line) <= maxLen
	}
	if isTripleQuoted {
		var list textList
		list = append(list, textRecord{Diff: opts.DiffMode, Value: textLine(prefix + `"""`), ElideComma: true})
		for i, line := range lines {
			if numElided := len(lines) - i; i == maxLines-1 && numElided > 1 {
				comment := commentString(fmt.Sprintf("%d elided lines", numElided))
				list = append(list, textRecord{Diff: opts.DiffMode, Value: textEllipsis, ElideComma: true, Comment: comment})
				break
			}
			list = append(list, textRecord{Diff: opts.DiffMode, Value: textLine(line), ElideComma: true})
		}
		list = append(list, textRecord{Diff: opts.DiffMode, Value: textLine(prefix + `"""`), ElideComma: true})
		return &textWrap{Prefix: "(", Value: list, Suffix: ")"}
	}

	// Format the string as a single-line quoted string.
	if len(s) > maxLen+len(textEllipsis) {
		return textLine(prefix + formatString(s[:maxLen]) + string(textEllipsis))
	}
	return textLine(prefix + formatString(s))
}

// formatMapKey formats v as if it were a map key.
// The result is guaranteed to be a single line.
func formatMapKey(v reflect.Value, disambiguate bool, ptrs *pointerReferences) string {
	var opts formatOptions
	opts.DiffMode = diffIdentical
	opts.TypeMode = elideType
	opts.PrintAddresses = disambiguate
	opts.AvoidStringer = disambiguate
	opts.QualifiedNames = disambiguate
	opts.VerbosityLevel = maxVerbosityPreset
	opts.LimitVerbosity = true
	s := opts.FormatValue(v, reflect.Map, ptrs).String()
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
	if utf8.ValidString(s) && strings.IndexFunc(s, rawInvalid) < 0 {
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
