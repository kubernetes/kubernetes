// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"encoding/base32"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"sync"
)

// optimizeCommon specifies whether to use optimizations targeted for certain
// common patterns, rather than using the slower, but more general logic.
// All tests should pass regardless of whether this is true or not.
const optimizeCommon = true

var (
	// Most natural Go type that correspond with each JSON type.
	anyType          = reflect.TypeOf((*any)(nil)).Elem()            // JSON value
	boolType         = reflect.TypeOf((*bool)(nil)).Elem()           // JSON bool
	stringType       = reflect.TypeOf((*string)(nil)).Elem()         // JSON string
	float64Type      = reflect.TypeOf((*float64)(nil)).Elem()        // JSON number
	mapStringAnyType = reflect.TypeOf((*map[string]any)(nil)).Elem() // JSON object
	sliceAnyType     = reflect.TypeOf((*[]any)(nil)).Elem()          // JSON array

	bytesType       = reflect.TypeOf((*[]byte)(nil)).Elem()
	emptyStructType = reflect.TypeOf((*struct{})(nil)).Elem()
)

const startDetectingCyclesAfter = 1000

type seenPointers map[typedPointer]struct{}

type typedPointer struct {
	typ reflect.Type
	ptr any // always stores unsafe.Pointer, but avoids depending on unsafe
}

// visit visits pointer p of type t, reporting an error if seen before.
// If successfully visited, then the caller must eventually call leave.
func (m *seenPointers) visit(v reflect.Value) error {
	p := typedPointer{v.Type(), v.UnsafePointer()}
	if _, ok := (*m)[p]; ok {
		return &SemanticError{action: "marshal", GoType: p.typ, Err: errors.New("encountered a cycle")}
	}
	if *m == nil {
		*m = make(map[typedPointer]struct{})
	}
	(*m)[p] = struct{}{}
	return nil
}
func (m *seenPointers) leave(v reflect.Value) {
	p := typedPointer{v.Type(), v.UnsafePointer()}
	delete(*m, p)
}

func makeDefaultArshaler(t reflect.Type) *arshaler {
	switch t.Kind() {
	case reflect.Bool:
		return makeBoolArshaler(t)
	case reflect.String:
		return makeStringArshaler(t)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return makeIntArshaler(t)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return makeUintArshaler(t)
	case reflect.Float32, reflect.Float64:
		return makeFloatArshaler(t)
	case reflect.Map:
		return makeMapArshaler(t)
	case reflect.Struct:
		return makeStructArshaler(t)
	case reflect.Slice:
		fncs := makeSliceArshaler(t)
		if t.AssignableTo(bytesType) {
			return makeBytesArshaler(t, fncs)
		}
		return fncs
	case reflect.Array:
		fncs := makeArrayArshaler(t)
		if reflect.SliceOf(t.Elem()).AssignableTo(bytesType) {
			return makeBytesArshaler(t, fncs)
		}
		return fncs
	case reflect.Pointer:
		return makePointerArshaler(t)
	case reflect.Interface:
		return makeInterfaceArshaler(t)
	default:
		return makeInvalidArshaler(t)
	}
}

func makeBoolArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			return newInvalidFormatError("marshal", t, mo.format)
		}

		// Optimize for marshaling without preceding whitespace.
		if optimizeCommon && !enc.options.multiline && !enc.tokens.last.needObjectName() {
			enc.buf = enc.tokens.mayAppendDelim(enc.buf, 't')
			if va.Bool() {
				enc.buf = append(enc.buf, "true"...)
			} else {
				enc.buf = append(enc.buf, "false"...)
			}
			enc.tokens.last.increment()
			if enc.needFlush() {
				return enc.flush()
			}
			return nil
		}

		return enc.WriteToken(Bool(va.Bool()))
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			return newInvalidFormatError("unmarshal", t, uo.format)
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			va.SetBool(false)
			return nil
		case 't', 'f':
			va.SetBool(tok.Bool())
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

func makeStringArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			return newInvalidFormatError("marshal", t, mo.format)
		}
		return enc.WriteToken(String(va.String()))
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			return newInvalidFormatError("unmarshal", t, uo.format)
		}
		var flags valueFlags
		val, err := dec.readValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			va.SetString("")
			return nil
		case '"':
			val = unescapeStringMayCopy(val, flags.isVerbatim())
			if dec.stringCache == nil {
				dec.stringCache = new(stringCache)
			}
			str := dec.stringCache.make(val)
			va.SetString(str)
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

var (
	encodeBase16        = func(dst, src []byte) { hex.Encode(dst, src) }
	encodeBase32        = base32.StdEncoding.Encode
	encodeBase32Hex     = base32.HexEncoding.Encode
	encodeBase64        = base64.StdEncoding.Encode
	encodeBase64URL     = base64.URLEncoding.Encode
	encodedLenBase16    = hex.EncodedLen
	encodedLenBase32    = base32.StdEncoding.EncodedLen
	encodedLenBase32Hex = base32.HexEncoding.EncodedLen
	encodedLenBase64    = base64.StdEncoding.EncodedLen
	encodedLenBase64URL = base64.URLEncoding.EncodedLen
	decodeBase16        = hex.Decode
	decodeBase32        = base32.StdEncoding.Decode
	decodeBase32Hex     = base32.HexEncoding.Decode
	decodeBase64        = base64.StdEncoding.Decode
	decodeBase64URL     = base64.URLEncoding.Decode
	decodedLenBase16    = hex.DecodedLen
	decodedLenBase32    = base32.StdEncoding.WithPadding(base32.NoPadding).DecodedLen
	decodedLenBase32Hex = base32.HexEncoding.WithPadding(base32.NoPadding).DecodedLen
	decodedLenBase64    = base64.StdEncoding.WithPadding(base64.NoPadding).DecodedLen
	decodedLenBase64URL = base64.URLEncoding.WithPadding(base64.NoPadding).DecodedLen
)

func makeBytesArshaler(t reflect.Type, fncs *arshaler) *arshaler {
	// NOTE: This handles both []byte and [N]byte.
	marshalDefault := fncs.marshal
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		encode, encodedLen := encodeBase64, encodedLenBase64
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			switch mo.format {
			case "base64":
				encode, encodedLen = encodeBase64, encodedLenBase64
			case "base64url":
				encode, encodedLen = encodeBase64URL, encodedLenBase64URL
			case "base32":
				encode, encodedLen = encodeBase32, encodedLenBase32
			case "base32hex":
				encode, encodedLen = encodeBase32Hex, encodedLenBase32Hex
			case "base16", "hex":
				encode, encodedLen = encodeBase16, encodedLenBase16
			case "array":
				mo.format = ""
				return marshalDefault(mo, enc, va)
			default:
				return newInvalidFormatError("marshal", t, mo.format)
			}
		}
		val := enc.UnusedBuffer()
		var b []byte
		if va.Kind() == reflect.Array {
			// TODO(https://go.dev/issue/47066): Avoid reflect.Value.Slice.
			b = va.Slice(0, va.Len()).Bytes()
		} else {
			b = va.Bytes()
		}
		n := len(`"`) + encodedLen(len(b)) + len(`"`)
		if cap(val) < n {
			val = make([]byte, n)
		} else {
			val = val[:n]
		}
		val[0] = '"'
		encode(val[len(`"`):len(val)-len(`"`)], b)
		val[len(val)-1] = '"'
		return enc.WriteValue(val)
	}
	unmarshalDefault := fncs.unmarshal
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		decode, decodedLen := decodeBase64, decodedLenBase64
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			switch uo.format {
			case "base64":
				decode, decodedLen = decodeBase64, decodedLenBase64
			case "base64url":
				decode, decodedLen = decodeBase64URL, decodedLenBase64URL
			case "base32":
				decode, decodedLen = decodeBase32, decodedLenBase32
			case "base32hex":
				decode, decodedLen = decodeBase32Hex, decodedLenBase32Hex
			case "base16", "hex":
				decode, decodedLen = decodeBase16, decodedLenBase16
			case "array":
				uo.format = ""
				return unmarshalDefault(uo, dec, va)
			default:
				return newInvalidFormatError("unmarshal", t, uo.format)
			}
		}
		var flags valueFlags
		val, err := dec.readValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			va.Set(reflect.Zero(t))
			return nil
		case '"':
			val = unescapeStringMayCopy(val, flags.isVerbatim())

			// For base64 and base32, decodedLen computes the maximum output size
			// when given the original input size. To compute the exact size,
			// adjust the input size by excluding trailing padding characters.
			// This is unnecessary for base16, but also harmless.
			n := len(val)
			for n > 0 && val[n-1] == '=' {
				n--
			}
			n = decodedLen(n)
			var b []byte
			if va.Kind() == reflect.Array {
				// TODO(https://go.dev/issue/47066): Avoid reflect.Value.Slice.
				b = va.Slice(0, va.Len()).Bytes()
				if n != len(b) {
					err := fmt.Errorf("decoded base64 length of %d mismatches array length of %d", n, len(b))
					return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
				}
			} else {
				b = va.Bytes()
				if b == nil || cap(b) < n {
					b = make([]byte, n)
				} else {
					b = b[:n]
				}
			}
			if _, err := decode(b, val); err != nil {
				return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
			}
			if va.Kind() == reflect.Slice {
				va.SetBytes(b)
			}
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return fncs
}

func makeIntArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	bits := t.Bits()
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			return newInvalidFormatError("marshal", t, mo.format)
		}

		// Optimize for marshaling without preceding whitespace or string escaping.
		if optimizeCommon && !enc.options.multiline && !mo.StringifyNumbers && !enc.tokens.last.needObjectName() {
			enc.buf = enc.tokens.mayAppendDelim(enc.buf, '0')
			enc.buf = strconv.AppendInt(enc.buf, va.Int(), 10)
			enc.tokens.last.increment()
			if enc.needFlush() {
				return enc.flush()
			}
			return nil
		}

		x := math.Float64frombits(uint64(va.Int()))
		return enc.writeNumber(x, rawIntNumber, mo.StringifyNumbers)
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			return newInvalidFormatError("unmarshal", t, uo.format)
		}
		var flags valueFlags
		val, err := dec.readValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			va.SetInt(0)
			return nil
		case '"':
			if !uo.StringifyNumbers {
				break
			}
			val = unescapeStringMayCopy(val, flags.isVerbatim())
			fallthrough
		case '0':
			var negOffset int
			neg := val[0] == '-'
			if neg {
				negOffset = 1
			}
			n, ok := parseDecUint(val[negOffset:])
			maxInt := uint64(1) << (bits - 1)
			overflow := (neg && n > maxInt) || (!neg && n > maxInt-1)
			if !ok {
				if n != math.MaxUint64 {
					err := fmt.Errorf("cannot parse %q as signed integer: %w", val, strconv.ErrSyntax)
					return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
				}
				overflow = true
			}
			if overflow {
				err := fmt.Errorf("cannot parse %q as signed integer: %w", val, strconv.ErrRange)
				return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
			}
			if neg {
				va.SetInt(int64(-n))
			} else {
				va.SetInt(int64(+n))
			}
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

func makeUintArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	bits := t.Bits()
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			return newInvalidFormatError("marshal", t, mo.format)
		}

		// Optimize for marshaling without preceding whitespace or string escaping.
		if optimizeCommon && !enc.options.multiline && !mo.StringifyNumbers && !enc.tokens.last.needObjectName() {
			enc.buf = enc.tokens.mayAppendDelim(enc.buf, '0')
			enc.buf = strconv.AppendUint(enc.buf, va.Uint(), 10)
			enc.tokens.last.increment()
			if enc.needFlush() {
				return enc.flush()
			}
			return nil
		}

		x := math.Float64frombits(uint64(va.Uint()))
		return enc.writeNumber(x, rawUintNumber, mo.StringifyNumbers)
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			return newInvalidFormatError("unmarshal", t, uo.format)
		}
		var flags valueFlags
		val, err := dec.readValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			va.SetUint(0)
			return nil
		case '"':
			if !uo.StringifyNumbers {
				break
			}
			val = unescapeStringMayCopy(val, flags.isVerbatim())
			fallthrough
		case '0':
			n, ok := parseDecUint(val)
			maxUint := uint64(1) << bits
			overflow := n > maxUint-1
			if !ok {
				if n != math.MaxUint64 {
					err := fmt.Errorf("cannot parse %q as unsigned integer: %w", val, strconv.ErrSyntax)
					return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
				}
				overflow = true
			}
			if overflow {
				err := fmt.Errorf("cannot parse %q as unsigned integer: %w", val, strconv.ErrRange)
				return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
			}
			va.SetUint(uint64(n))
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

func makeFloatArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	bits := t.Bits()
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		var allowNonFinite bool
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			if mo.format == "nonfinite" {
				allowNonFinite = true
			} else {
				return newInvalidFormatError("marshal", t, mo.format)
			}
		}

		fv := va.Float()
		if math.IsNaN(fv) || math.IsInf(fv, 0) {
			if !allowNonFinite {
				err := fmt.Errorf("invalid value: %v", fv)
				return &SemanticError{action: "marshal", GoType: t, Err: err}
			}
			return enc.WriteToken(Float(fv))
		}

		// Optimize for marshaling without preceding whitespace or string escaping.
		if optimizeCommon && !enc.options.multiline && !mo.StringifyNumbers && !enc.tokens.last.needObjectName() {
			enc.buf = enc.tokens.mayAppendDelim(enc.buf, '0')
			enc.buf = appendNumber(enc.buf, fv, bits)
			enc.tokens.last.increment()
			if enc.needFlush() {
				return enc.flush()
			}
			return nil
		}

		return enc.writeNumber(fv, bits, mo.StringifyNumbers)
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		var allowNonFinite bool
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			if uo.format == "nonfinite" {
				allowNonFinite = true
			} else {
				return newInvalidFormatError("unmarshal", t, uo.format)
			}
		}
		var flags valueFlags
		val, err := dec.readValue(&flags)
		if err != nil {
			return err
		}
		k := val.Kind()
		switch k {
		case 'n':
			va.SetFloat(0)
			return nil
		case '"':
			val = unescapeStringMayCopy(val, flags.isVerbatim())
			if allowNonFinite {
				switch string(val) {
				case "NaN":
					va.SetFloat(math.NaN())
					return nil
				case "Infinity":
					va.SetFloat(math.Inf(+1))
					return nil
				case "-Infinity":
					va.SetFloat(math.Inf(-1))
					return nil
				}
			}
			if !uo.StringifyNumbers {
				break
			}
			if n, err := consumeNumber(val); n != len(val) || err != nil {
				err := fmt.Errorf("cannot parse %q as JSON number: %w", val, strconv.ErrSyntax)
				return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
			}
			fallthrough
		case '0':
			// NOTE: Floating-point parsing is by nature a lossy operation.
			// We never report an overflow condition since we can always
			// round the input to the closest representable finite value.
			// For extremely large numbers, the closest value is ±MaxFloat.
			fv, _ := parseFloat(val, bits)
			va.SetFloat(fv)
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

var mapIterPool = sync.Pool{
	New: func() any { return new(reflect.MapIter) },
}

func getMapIter(mv reflect.Value) *reflect.MapIter {
	iter := mapIterPool.Get().(*reflect.MapIter)
	iter.Reset(mv)
	return iter
}
func putMapIter(iter *reflect.MapIter) {
	iter.Reset(reflect.Value{}) // allow underlying map to be garbage collected
	mapIterPool.Put(iter)
}

func makeMapArshaler(t reflect.Type) *arshaler {
	// NOTE: The logic below disables namespaces for tracking duplicate names
	// when handling map keys with a unique represention.

	// NOTE: Values retrieved from a map are not addressable,
	// so we shallow copy the values to make them addressable and
	// store them back into the map afterwards.

	var fncs arshaler
	var (
		once    sync.Once
		keyFncs *arshaler
		valFncs *arshaler
	)
	init := func() {
		keyFncs = lookupArshaler(t.Key())
		valFncs = lookupArshaler(t.Elem())
	}
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		// Check for cycles.
		if enc.tokens.depth() > startDetectingCyclesAfter {
			if err := enc.seenPointers.visit(va.Value); err != nil {
				return err
			}
			defer enc.seenPointers.leave(va.Value)
		}

		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			if mo.format == "emitnull" {
				if va.IsNil() {
					return enc.WriteToken(Null)
				}
				mo.format = ""
			} else {
				return newInvalidFormatError("marshal", t, mo.format)
			}
		}

		// Optimize for marshaling an empty map without any preceding whitespace.
		n := va.Len()
		if optimizeCommon && n == 0 && !enc.options.multiline && !enc.tokens.last.needObjectName() {
			enc.buf = enc.tokens.mayAppendDelim(enc.buf, '{')
			enc.buf = append(enc.buf, "{}"...)
			enc.tokens.last.increment()
			if enc.needFlush() {
				return enc.flush()
			}
			return nil
		}

		once.Do(init)
		if err := enc.WriteToken(ObjectStart); err != nil {
			return err
		}
		if n > 0 {
			// Handle maps with numeric key types by stringifying them.
			mko := mo
			mko.StringifyNumbers = true

			nonDefaultKey := keyFncs.nonDefault
			marshalKey := keyFncs.marshal
			marshalVal := valFncs.marshal
			if mo.Marshalers != nil {
				var ok bool
				marshalKey, ok = mo.Marshalers.lookup(marshalKey, t.Key())
				marshalVal, _ = mo.Marshalers.lookup(marshalVal, t.Elem())
				nonDefaultKey = nonDefaultKey || ok
			}
			k := newAddressableValue(t.Key())
			v := newAddressableValue(t.Elem())

			// A Go map guarantees that each entry has a unique key.
			// As such, disable the expensive duplicate name check if we know
			// that every Go key will serialize as a unique JSON string.
			if !nonDefaultKey && mapKeyWithUniqueRepresentation(k.Kind(), enc.options.AllowInvalidUTF8) {
				enc.tokens.last.disableNamespace()
			}

			// NOTE: Map entries are serialized in a non-deterministic order.
			// Users that need stable output should call RawValue.Canonicalize.
			// TODO(go1.19): Remove use of a sync.Pool with reflect.MapIter.
			// Calling reflect.Value.MapRange no longer allocates.
			// See https://go.dev/cl/400675.
			iter := getMapIter(va.Value)
			defer putMapIter(iter)
			for iter.Next() {
				k.SetIterKey(iter)
				if err := marshalKey(mko, enc, k); err != nil {
					// TODO: If err is errMissingName, then wrap it as a
					// SemanticError since this key type cannot be serialized
					// as a JSON string.
					return err
				}
				v.SetIterValue(iter)
				if err := marshalVal(mo, enc, v); err != nil {
					return err
				}
			}
		}
		if err := enc.WriteToken(ObjectEnd); err != nil {
			return err
		}
		return nil
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			if uo.format == "emitnull" {
				uo.format = "" // only relevant for marshaling
			} else {
				return newInvalidFormatError("unmarshal", t, uo.format)
			}
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			va.Set(reflect.Zero(t))
			return nil
		case '{':
			once.Do(init)
			if va.IsNil() {
				va.Set(reflect.MakeMap(t))
			}

			// Handle maps with numeric key types by stringifying them.
			uko := uo
			uko.StringifyNumbers = true

			nonDefaultKey := keyFncs.nonDefault
			unmarshalKey := keyFncs.unmarshal
			unmarshalVal := valFncs.unmarshal
			if uo.Unmarshalers != nil {
				var ok bool
				unmarshalKey, ok = uo.Unmarshalers.lookup(unmarshalKey, t.Key())
				unmarshalVal, _ = uo.Unmarshalers.lookup(unmarshalVal, t.Elem())
				nonDefaultKey = nonDefaultKey || ok
			}
			k := newAddressableValue(t.Key())
			v := newAddressableValue(t.Elem())

			// Manually check for duplicate entries by virtue of whether the
			// unmarshaled key already exists in the destination Go map.
			// Consequently, syntactically different names (e.g., "0" and "-0")
			// will be rejected as duplicates since they semantically refer
			// to the same Go value. This is an unusual interaction
			// between syntax and semantics, but is more correct.
			if !nonDefaultKey && mapKeyWithUniqueRepresentation(k.Kind(), dec.options.AllowInvalidUTF8) {
				dec.tokens.last.disableNamespace()
			}

			// In the rare case where the map is not already empty,
			// then we need to manually track which keys we already saw
			// since existing presence alone is insufficient to indicate
			// whether the input had a duplicate name.
			var seen reflect.Value
			if !dec.options.AllowDuplicateNames && va.Len() > 0 {
				seen = reflect.MakeMap(reflect.MapOf(k.Type(), emptyStructType))
			}

			for dec.PeekKind() != '}' {
				k.Set(reflect.Zero(t.Key()))
				if err := unmarshalKey(uko, dec, k); err != nil {
					return err
				}
				if k.Kind() == reflect.Interface && !k.IsNil() && !k.Elem().Type().Comparable() {
					err := fmt.Errorf("invalid incomparable key type %v", k.Elem().Type())
					return &SemanticError{action: "unmarshal", GoType: t, Err: err}
				}

				if v2 := va.MapIndex(k.Value); v2.IsValid() {
					if !dec.options.AllowDuplicateNames && (!seen.IsValid() || seen.MapIndex(k.Value).IsValid()) {
						// TODO: Unread the object name.
						name := dec.previousBuffer()
						err := &SyntacticError{str: "duplicate name " + string(name) + " in object"}
						return err.withOffset(dec.InputOffset() - int64(len(name)))
					}
					v.Set(v2)
				} else {
					v.Set(reflect.Zero(v.Type()))
				}
				err := unmarshalVal(uo, dec, v)
				va.SetMapIndex(k.Value, v.Value)
				if seen.IsValid() {
					seen.SetMapIndex(k.Value, reflect.Zero(emptyStructType))
				}
				if err != nil {
					return err
				}
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

// mapKeyWithUniqueRepresentation reports whether all possible values of k
// marshal to a different JSON value, and whether all possible JSON values
// that can unmarshal into k unmarshal to different Go values.
// In other words, the representation must be a bijective.
func mapKeyWithUniqueRepresentation(k reflect.Kind, allowInvalidUTF8 bool) bool {
	switch k {
	case reflect.Bool,
		reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return true
	case reflect.String:
		// For strings, we have to be careful since names with invalid UTF-8
		// maybe unescape to the same Go string value.
		return !allowInvalidUTF8
	default:
		// Floating-point kinds are not listed above since NaNs
		// can appear multiple times and all serialize as "NaN".
		return false
	}
}

func makeStructArshaler(t reflect.Type) *arshaler {
	// NOTE: The logic below disables namespaces for tracking duplicate names
	// and does the tracking locally with an efficient bit-set based on which
	// Go struct fields were seen.

	var fncs arshaler
	var (
		once    sync.Once
		fields  structFields
		errInit *SemanticError
	)
	init := func() {
		fields, errInit = makeStructFields(t)
	}
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			return newInvalidFormatError("marshal", t, mo.format)
		}
		once.Do(init)
		if errInit != nil {
			err := *errInit // shallow copy SemanticError
			err.action = "marshal"
			return &err
		}
		if err := enc.WriteToken(ObjectStart); err != nil {
			return err
		}
		var seenIdxs uintSet
		prevIdx := -1
		enc.tokens.last.disableNamespace() // we manually ensure unique names below
		for i := range fields.flattened {
			f := &fields.flattened[i]
			v := addressableValue{va.Field(f.index[0])} // addressable if struct value is addressable
			if len(f.index) > 1 {
				v = v.fieldByIndex(f.index[1:], false)
				if !v.IsValid() {
					continue // implies a nil inlined field
				}
			}

			// OmitZero skips the field if the Go value is zero,
			// which we can determine up front without calling the marshaler.
			if f.omitzero && ((f.isZero == nil && v.IsZero()) || (f.isZero != nil && f.isZero(v))) {
				continue
			}

			marshal := f.fncs.marshal
			nonDefault := f.fncs.nonDefault
			if mo.Marshalers != nil {
				var ok bool
				marshal, ok = mo.Marshalers.lookup(marshal, f.typ)
				nonDefault = nonDefault || ok
			}

			// OmitEmpty skips the field if the marshaled JSON value is empty,
			// which we can know up front if there are no custom marshalers,
			// otherwise we must marshal the value and unwrite it if empty.
			if f.omitempty && !nonDefault && f.isEmpty != nil && f.isEmpty(v) {
				continue // fast path for omitempty
			}

			// Write the object member name.
			//
			// The logic below is semantically equivalent to:
			//	enc.WriteToken(String(f.name))
			// but specialized and simplified because:
			//	1. The Encoder must be expecting an object name.
			//	2. The object namespace is guaranteed to be disabled.
			//	3. The object name is guaranteed to be valid and pre-escaped.
			//	4. There is no need to flush the buffer (for unwrite purposes).
			//	5. There is no possibility of an error occuring.
			if optimizeCommon {
				// Append any delimiters or optional whitespace.
				if enc.tokens.last.length() > 0 {
					enc.buf = append(enc.buf, ',')
				}
				if enc.options.multiline {
					enc.buf = enc.appendIndent(enc.buf, enc.tokens.needIndent('"'))
				}

				// Append the token to the output and to the state machine.
				n0 := len(enc.buf) // offset before calling appendString
				if enc.options.EscapeRune == nil {
					enc.buf = append(enc.buf, f.quotedName...)
				} else {
					enc.buf, _ = appendString(enc.buf, f.name, false, enc.options.EscapeRune)
				}
				if !enc.options.AllowDuplicateNames {
					enc.names.replaceLastQuotedOffset(n0)
				}
				enc.tokens.last.increment()
			} else {
				if err := enc.WriteToken(String(f.name)); err != nil {
					return err
				}
			}

			// Write the object member value.
			mo2 := mo
			if f.string {
				mo2.StringifyNumbers = true
			}
			if f.format != "" {
				mo2.formatDepth = enc.tokens.depth()
				mo2.format = f.format
			}
			if err := marshal(mo2, enc, v); err != nil {
				return err
			}

			// Try unwriting the member if empty (slow path for omitempty).
			if f.omitempty {
				var prevName *string
				if prevIdx >= 0 {
					prevName = &fields.flattened[prevIdx].name
				}
				if enc.unwriteEmptyObjectMember(prevName) {
					continue
				}
			}

			// Remember the previous written object member.
			// The set of seen fields only needs to be updated to detect
			// duplicate names with those from the inlined fallback.
			if !enc.options.AllowDuplicateNames && fields.inlinedFallback != nil {
				seenIdxs.insert(uint(f.id))
			}
			prevIdx = f.id
		}
		if fields.inlinedFallback != nil && !(mo.DiscardUnknownMembers && fields.inlinedFallback.unknown) {
			var insertUnquotedName func([]byte) bool
			if !enc.options.AllowDuplicateNames {
				insertUnquotedName = func(name []byte) bool {
					// Check that the name from inlined fallback does not match
					// one of the previously marshaled names from known fields.
					if foldedFields := fields.byFoldedName[string(foldName(name))]; len(foldedFields) > 0 {
						if f := fields.byActualName[string(name)]; f != nil {
							return seenIdxs.insert(uint(f.id))
						}
						for _, f := range foldedFields {
							if f.nocase {
								return seenIdxs.insert(uint(f.id))
							}
						}
					}

					// Check that the name does not match any other name
					// previously marshaled from the inlined fallback.
					return enc.namespaces.last().insertUnquoted(name)
				}
			}
			if err := marshalInlinedFallbackAll(mo, enc, va, fields.inlinedFallback, insertUnquotedName); err != nil {
				return err
			}
		}
		if err := enc.WriteToken(ObjectEnd); err != nil {
			return err
		}
		return nil
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			return newInvalidFormatError("unmarshal", t, uo.format)
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			va.Set(reflect.Zero(t))
			return nil
		case '{':
			once.Do(init)
			if errInit != nil {
				err := *errInit // shallow copy SemanticError
				err.action = "unmarshal"
				return &err
			}
			var seenIdxs uintSet
			dec.tokens.last.disableNamespace()
			for dec.PeekKind() != '}' {
				// Process the object member name.
				var flags valueFlags
				val, err := dec.readValue(&flags)
				if err != nil {
					return err
				}
				name := unescapeStringMayCopy(val, flags.isVerbatim())
				f := fields.byActualName[string(name)]
				if f == nil {
					for _, f2 := range fields.byFoldedName[string(foldName(name))] {
						if f2.nocase {
							f = f2
							break
						}
					}
					if f == nil {
						if uo.RejectUnknownMembers && (fields.inlinedFallback == nil || fields.inlinedFallback.unknown) {
							return &SemanticError{action: "unmarshal", GoType: t, Err: fmt.Errorf("unknown name %s", val)}
						}
						if !dec.options.AllowDuplicateNames && !dec.namespaces.last().insertUnquoted(name) {
							// TODO: Unread the object name.
							err := &SyntacticError{str: "duplicate name " + string(val) + " in object"}
							return err.withOffset(dec.InputOffset() - int64(len(val)))
						}

						if fields.inlinedFallback == nil {
							// Skip unknown value since we have no place to store it.
							if err := dec.skipValue(); err != nil {
								return err
							}
						} else {
							// Marshal into value capable of storing arbitrary object members.
							if err := unmarshalInlinedFallbackNext(uo, dec, va, fields.inlinedFallback, val, name); err != nil {
								return err
							}
						}
						continue
					}
				}
				if !dec.options.AllowDuplicateNames && !seenIdxs.insert(uint(f.id)) {
					// TODO: Unread the object name.
					err := &SyntacticError{str: "duplicate name " + string(val) + " in object"}
					return err.withOffset(dec.InputOffset() - int64(len(val)))
				}

				// Process the object member value.
				unmarshal := f.fncs.unmarshal
				if uo.Unmarshalers != nil {
					unmarshal, _ = uo.Unmarshalers.lookup(unmarshal, f.typ)
				}
				uo2 := uo
				if f.string {
					uo2.StringifyNumbers = true
				}
				if f.format != "" {
					uo2.formatDepth = dec.tokens.depth()
					uo2.format = f.format
				}
				v := addressableValue{va.Field(f.index[0])} // addressable if struct value is addressable
				if len(f.index) > 1 {
					v = v.fieldByIndex(f.index[1:], true)
				}
				if err := unmarshal(uo2, dec, v); err != nil {
					return err
				}
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

func (va addressableValue) fieldByIndex(index []int, mayAlloc bool) addressableValue {
	for _, i := range index {
		va = va.indirect(mayAlloc)
		if !va.IsValid() {
			return va
		}
		va = addressableValue{va.Field(i)} // addressable if struct value is addressable
	}
	return va
}

func (va addressableValue) indirect(mayAlloc bool) addressableValue {
	if va.Kind() == reflect.Pointer {
		if va.IsNil() {
			if !mayAlloc {
				return addressableValue{}
			}
			va.Set(reflect.New(va.Type().Elem()))
		}
		va = addressableValue{va.Elem()} // dereferenced pointer is always addressable
	}
	return va
}

func makeSliceArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	var (
		once    sync.Once
		valFncs *arshaler
	)
	init := func() {
		valFncs = lookupArshaler(t.Elem())
	}
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		// Check for cycles.
		if enc.tokens.depth() > startDetectingCyclesAfter {
			if err := enc.seenPointers.visit(va.Value); err != nil {
				return err
			}
			defer enc.seenPointers.leave(va.Value)
		}

		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			if mo.format == "emitnull" {
				if va.IsNil() {
					return enc.WriteToken(Null)
				}
				mo.format = ""
			} else {
				return newInvalidFormatError("marshal", t, mo.format)
			}
		}

		// Optimize for marshaling an empty slice without any preceding whitespace.
		n := va.Len()
		if optimizeCommon && n == 0 && !enc.options.multiline && !enc.tokens.last.needObjectName() {
			enc.buf = enc.tokens.mayAppendDelim(enc.buf, '[')
			enc.buf = append(enc.buf, "[]"...)
			enc.tokens.last.increment()
			if enc.needFlush() {
				return enc.flush()
			}
			return nil
		}

		once.Do(init)
		if err := enc.WriteToken(ArrayStart); err != nil {
			return err
		}
		marshal := valFncs.marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.lookup(marshal, t.Elem())
		}
		for i := 0; i < n; i++ {
			v := addressableValue{va.Index(i)} // indexed slice element is always addressable
			if err := marshal(mo, enc, v); err != nil {
				return err
			}
		}
		if err := enc.WriteToken(ArrayEnd); err != nil {
			return err
		}
		return nil
	}
	emptySlice := reflect.MakeSlice(t, 0, 0)
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			if uo.format == "emitnull" {
				uo.format = "" // only relevant for marshaling
			} else {
				return newInvalidFormatError("unmarshal", t, uo.format)
			}
		}

		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			va.Set(reflect.Zero(t))
			return nil
		case '[':
			once.Do(init)
			unmarshal := valFncs.unmarshal
			if uo.Unmarshalers != nil {
				unmarshal, _ = uo.Unmarshalers.lookup(unmarshal, t.Elem())
			}
			mustZero := true // we do not know the cleanliness of unused capacity
			cap := va.Cap()
			if cap > 0 {
				va.SetLen(cap)
			}
			var i int
			for dec.PeekKind() != ']' {
				if i == cap {
					// TODO(https://go.dev/issue/48000): Use reflect.Value.Append.
					va.Set(reflect.Append(va.Value, reflect.Zero(t.Elem())))
					cap = va.Cap()
					va.SetLen(cap)
					mustZero = false // append guarantees that unused capacity is zero-initialized
				}
				v := addressableValue{va.Index(i)} // indexed slice element is always addressable
				i++
				if mustZero {
					v.Set(reflect.Zero(t.Elem()))
				}
				if err := unmarshal(uo, dec, v); err != nil {
					va.SetLen(i)
					return err
				}
			}
			if i == 0 {
				va.Set(emptySlice)
			} else {
				va.SetLen(i)
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

func makeArrayArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	var (
		once    sync.Once
		valFncs *arshaler
	)
	init := func() {
		valFncs = lookupArshaler(t.Elem())
	}
	n := t.Len()
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			return newInvalidFormatError("marshal", t, mo.format)
		}
		once.Do(init)
		if err := enc.WriteToken(ArrayStart); err != nil {
			return err
		}
		marshal := valFncs.marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.lookup(marshal, t.Elem())
		}
		for i := 0; i < n; i++ {
			v := addressableValue{va.Index(i)} // indexed array element is addressable if array is addressable
			if err := marshal(mo, enc, v); err != nil {
				return err
			}
		}
		if err := enc.WriteToken(ArrayEnd); err != nil {
			return err
		}
		return nil
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			return newInvalidFormatError("unmarshal", t, uo.format)
		}
		tok, err := dec.ReadToken()
		if err != nil {
			return err
		}
		k := tok.Kind()
		switch k {
		case 'n':
			va.Set(reflect.Zero(t))
			return nil
		case '[':
			once.Do(init)
			unmarshal := valFncs.unmarshal
			if uo.Unmarshalers != nil {
				unmarshal, _ = uo.Unmarshalers.lookup(unmarshal, t.Elem())
			}
			var i int
			for dec.PeekKind() != ']' {
				if i >= n {
					err := errors.New("too many array elements")
					return &SemanticError{action: "unmarshal", GoType: t, Err: err}
				}
				v := addressableValue{va.Index(i)} // indexed array element is addressable if array is addressable
				v.Set(reflect.Zero(v.Type()))
				if err := unmarshal(uo, dec, v); err != nil {
					return err
				}
				i++
			}
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			if i < n {
				err := errors.New("too few array elements")
				return &SemanticError{action: "unmarshal", GoType: t, Err: err}
			}
			return nil
		}
		return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
	}
	return &fncs
}

func makePointerArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	var (
		once    sync.Once
		valFncs *arshaler
	)
	init := func() {
		valFncs = lookupArshaler(t.Elem())
	}
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		// Check for cycles.
		if enc.tokens.depth() > startDetectingCyclesAfter {
			if err := enc.seenPointers.visit(va.Value); err != nil {
				return err
			}
			defer enc.seenPointers.leave(va.Value)
		}

		// NOTE: MarshalOptions.format is forwarded to underlying marshal.
		if va.IsNil() {
			return enc.WriteToken(Null)
		}
		once.Do(init)
		marshal := valFncs.marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.lookup(marshal, t.Elem())
		}
		v := addressableValue{va.Elem()} // dereferenced pointer is always addressable
		return marshal(mo, enc, v)
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		// NOTE: UnmarshalOptions.format is forwarded to underlying unmarshal.
		if dec.PeekKind() == 'n' {
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			va.Set(reflect.Zero(t))
			return nil
		}
		once.Do(init)
		unmarshal := valFncs.unmarshal
		if uo.Unmarshalers != nil {
			unmarshal, _ = uo.Unmarshalers.lookup(unmarshal, t.Elem())
		}
		if va.IsNil() {
			va.Set(reflect.New(t.Elem()))
		}
		v := addressableValue{va.Elem()} // dereferenced pointer is always addressable
		return unmarshal(uo, dec, v)
	}
	return &fncs
}

func makeInterfaceArshaler(t reflect.Type) *arshaler {
	// NOTE: Values retrieved from an interface are not addressable,
	// so we shallow copy the values to make them addressable and
	// store them back into the interface afterwards.

	var fncs arshaler
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
			return newInvalidFormatError("marshal", t, mo.format)
		}
		if va.IsNil() {
			return enc.WriteToken(Null)
		}
		v := newAddressableValue(va.Elem().Type())
		v.Set(va.Elem())
		marshal := lookupArshaler(v.Type()).marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.lookup(marshal, v.Type())
		}
		// Optimize for the any type if there are no special options.
		if optimizeCommon && t == anyType && !mo.StringifyNumbers && mo.format == "" && (mo.Marshalers == nil || !mo.Marshalers.fromAny) {
			return marshalValueAny(mo, enc, va.Elem().Interface())
		}
		return marshal(mo, enc, v)
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
			return newInvalidFormatError("unmarshal", t, uo.format)
		}
		if dec.PeekKind() == 'n' {
			if _, err := dec.ReadToken(); err != nil {
				return err
			}
			va.Set(reflect.Zero(t))
			return nil
		}
		var v addressableValue
		if va.IsNil() {
			// Optimize for the any type if there are no special options.
			// We do not care about stringified numbers since JSON strings
			// are always unmarshaled into an any value as Go strings.
			// Duplicate name check must be enforced since unmarshalValueAny
			// does not implement merge semantics.
			if optimizeCommon && t == anyType && uo.format == "" && (uo.Unmarshalers == nil || !uo.Unmarshalers.fromAny) && !dec.options.AllowDuplicateNames {
				v, err := unmarshalValueAny(uo, dec)
				// We must check for nil interface values up front.
				// See https://go.dev/issue/52310.
				if v != nil {
					va.Set(reflect.ValueOf(v))
				}
				return err
			}

			k := dec.PeekKind()
			if !isAnyType(t) {
				err := errors.New("cannot derive concrete type for non-empty interface")
				return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
			}
			switch k {
			case 'f', 't':
				v = newAddressableValue(boolType)
			case '"':
				v = newAddressableValue(stringType)
			case '0':
				v = newAddressableValue(float64Type)
			case '{':
				v = newAddressableValue(mapStringAnyType)
			case '[':
				v = newAddressableValue(sliceAnyType)
			default:
				// If k is invalid (e.g., due to an I/O or syntax error), then
				// that will be cached by PeekKind and returned by ReadValue.
				// If k is '}' or ']', then ReadValue must error since
				// those are invalid kinds at the start of a JSON value.
				_, err := dec.ReadValue()
				return err
			}
		} else {
			// Shallow copy the existing value to keep it addressable.
			// Any mutations at the top-level of the value will be observable
			// since we always store this value back into the interface value.
			v = newAddressableValue(va.Elem().Type())
			v.Set(va.Elem())
		}
		unmarshal := lookupArshaler(v.Type()).unmarshal
		if uo.Unmarshalers != nil {
			unmarshal, _ = uo.Unmarshalers.lookup(unmarshal, v.Type())
		}
		err := unmarshal(uo, dec, v)
		va.Set(v.Value)
		return err
	}
	return &fncs
}

// isAnyType reports wether t is equivalent to the any interface type.
func isAnyType(t reflect.Type) bool {
	// This is forward compatible if the Go language permits type sets within
	// ordinary interfaces where an interface with zero methods does not
	// necessarily mean it can hold every possible Go type.
	// See https://go.dev/issue/45346.
	return t == anyType || anyType.Implements(t)
}

func makeInvalidArshaler(t reflect.Type) *arshaler {
	var fncs arshaler
	fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
		return &SemanticError{action: "marshal", GoType: t}
	}
	fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
		return &SemanticError{action: "unmarshal", GoType: t}
	}
	return &fncs
}

func newInvalidFormatError(action string, t reflect.Type, format string) error {
	err := fmt.Errorf("invalid format flag: %q", format)
	return &SemanticError{action: action, GoType: t, Err: err}
}
