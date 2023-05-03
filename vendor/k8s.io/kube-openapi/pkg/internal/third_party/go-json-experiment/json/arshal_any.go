// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import "reflect"

// This files contains an optimized marshal and unmarshal implementation
// for the any type. This type is often used when the Go program has
// no knowledge of the JSON schema. This is a common enough occurrence
// to justify the complexity of adding logic for this.

func marshalValueAny(mo MarshalOptions, enc *Encoder, val any) error {
	switch val := val.(type) {
	case nil:
		return enc.WriteToken(Null)
	case bool:
		return enc.WriteToken(Bool(val))
	case string:
		return enc.WriteToken(String(val))
	case float64:
		return enc.WriteToken(Float(val))
	case map[string]any:
		return marshalObjectAny(mo, enc, val)
	case []any:
		return marshalArrayAny(mo, enc, val)
	default:
		v := newAddressableValue(reflect.TypeOf(val))
		v.Set(reflect.ValueOf(val))
		marshal := lookupArshaler(v.Type()).marshal
		if mo.Marshalers != nil {
			marshal, _ = mo.Marshalers.lookup(marshal, v.Type())
		}
		return marshal(mo, enc, v)
	}
}

func unmarshalValueAny(uo UnmarshalOptions, dec *Decoder) (any, error) {
	switch k := dec.PeekKind(); k {
	case '{':
		return unmarshalObjectAny(uo, dec)
	case '[':
		return unmarshalArrayAny(uo, dec)
	default:
		var flags valueFlags
		val, err := dec.readValue(&flags)
		if err != nil {
			return nil, err
		}
		switch val.Kind() {
		case 'n':
			return nil, nil
		case 'f':
			return false, nil
		case 't':
			return true, nil
		case '"':
			val = unescapeStringMayCopy(val, flags.isVerbatim())
			if dec.stringCache == nil {
				dec.stringCache = new(stringCache)
			}
			return dec.stringCache.make(val), nil
		case '0':
			fv, _ := parseFloat(val, 64) // ignore error since readValue guarantees val is valid
			return fv, nil
		default:
			panic("BUG: invalid kind: " + k.String())
		}
	}
}

func marshalObjectAny(mo MarshalOptions, enc *Encoder, obj map[string]any) error {
	// Check for cycles.
	if enc.tokens.depth() > startDetectingCyclesAfter {
		v := reflect.ValueOf(obj)
		if err := enc.seenPointers.visit(v); err != nil {
			return err
		}
		defer enc.seenPointers.leave(v)
	}

	// Optimize for marshaling an empty map without any preceding whitespace.
	if len(obj) == 0 && !enc.options.multiline && !enc.tokens.last.needObjectName() {
		enc.buf = enc.tokens.mayAppendDelim(enc.buf, '{')
		enc.buf = append(enc.buf, "{}"...)
		enc.tokens.last.increment()
		if enc.needFlush() {
			return enc.flush()
		}
		return nil
	}

	if err := enc.WriteToken(ObjectStart); err != nil {
		return err
	}
	// A Go map guarantees that each entry has a unique key
	// The only possibility of duplicates is due to invalid UTF-8.
	if !enc.options.AllowInvalidUTF8 {
		enc.tokens.last.disableNamespace()
	}
	if !mo.Deterministic || len(obj) <= 1 {
		for name, val := range obj {
			if err := enc.WriteToken(String(name)); err != nil {
				return err
			}
			if err := marshalValueAny(mo, enc, val); err != nil {
				return err
			}
		}
	} else {
		names := getStrings(len(obj))
		var i int
		for name := range obj {
			(*names)[i] = name
			i++
		}
		names.Sort()
		for _, name := range *names {
			if err := enc.WriteToken(String(name)); err != nil {
				return err
			}
			if err := marshalValueAny(mo, enc, obj[name]); err != nil {
				return err
			}
		}
		putStrings(names)
	}
	if err := enc.WriteToken(ObjectEnd); err != nil {
		return err
	}
	return nil
}

func unmarshalObjectAny(uo UnmarshalOptions, dec *Decoder) (map[string]any, error) {
	tok, err := dec.ReadToken()
	if err != nil {
		return nil, err
	}
	k := tok.Kind()
	switch k {
	case 'n':
		return nil, nil
	case '{':
		obj := make(map[string]any)
		// A Go map guarantees that each entry has a unique key
		// The only possibility of duplicates is due to invalid UTF-8.
		if !dec.options.AllowInvalidUTF8 {
			dec.tokens.last.disableNamespace()
		}
		for dec.PeekKind() != '}' {
			tok, err := dec.ReadToken()
			if err != nil {
				return obj, err
			}
			name := tok.String()

			// Manually check for duplicate names.
			if _, ok := obj[name]; ok {
				name := dec.previousBuffer()
				err := &SyntacticError{str: "duplicate name " + string(name) + " in object"}
				return obj, err.withOffset(dec.InputOffset() - int64(len(name)))
			}

			val, err := unmarshalValueAny(uo, dec)
			obj[name] = val
			if err != nil {
				return obj, err
			}
		}
		if _, err := dec.ReadToken(); err != nil {
			return obj, err
		}
		return obj, nil
	}
	return nil, &SemanticError{action: "unmarshal", JSONKind: k, GoType: mapStringAnyType}
}

func marshalArrayAny(mo MarshalOptions, enc *Encoder, arr []any) error {
	// Check for cycles.
	if enc.tokens.depth() > startDetectingCyclesAfter {
		v := reflect.ValueOf(arr)
		if err := enc.seenPointers.visit(v); err != nil {
			return err
		}
		defer enc.seenPointers.leave(v)
	}

	// Optimize for marshaling an empty slice without any preceding whitespace.
	if len(arr) == 0 && !enc.options.multiline && !enc.tokens.last.needObjectName() {
		enc.buf = enc.tokens.mayAppendDelim(enc.buf, '[')
		enc.buf = append(enc.buf, "[]"...)
		enc.tokens.last.increment()
		if enc.needFlush() {
			return enc.flush()
		}
		return nil
	}

	if err := enc.WriteToken(ArrayStart); err != nil {
		return err
	}
	for _, val := range arr {
		if err := marshalValueAny(mo, enc, val); err != nil {
			return err
		}
	}
	if err := enc.WriteToken(ArrayEnd); err != nil {
		return err
	}
	return nil
}

func unmarshalArrayAny(uo UnmarshalOptions, dec *Decoder) ([]any, error) {
	tok, err := dec.ReadToken()
	if err != nil {
		return nil, err
	}
	k := tok.Kind()
	switch k {
	case 'n':
		return nil, nil
	case '[':
		arr := []any{}
		for dec.PeekKind() != ']' {
			val, err := unmarshalValueAny(uo, dec)
			arr = append(arr, val)
			if err != nil {
				return arr, err
			}
		}
		if _, err := dec.ReadToken(); err != nil {
			return arr, err
		}
		return arr, nil
	}
	return nil, &SemanticError{action: "unmarshal", JSONKind: k, GoType: sliceAnyType}
}
