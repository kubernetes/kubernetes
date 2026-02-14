// SPDX-FileCopyrightText: Copyright (c) 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

// Package jsonpointer provides a golang implementation for json pointers.
package jsonpointer

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/go-openapi/swag/jsonname"
)

const (
	emptyPointer     = ``
	pointerSeparator = `/`
)

// JSONPointable is an interface for structs to implement,
// when they need to customize the json pointer process or want to avoid the use of reflection.
type JSONPointable interface {
	// JSONLookup returns a value pointed at this (unescaped) key.
	JSONLookup(key string) (any, error)
}

// JSONSetable is an interface for structs to implement,
// when they need to customize the json pointer process or want to avoid the use of reflection.
type JSONSetable interface {
	// JSONSet sets the value pointed at the (unescaped) key.
	JSONSet(key string, value any) error
}

// Pointer is a representation of a json pointer.
//
// Use [Pointer.Get] to retrieve a value or [Pointer.Set] to set a value.
//
// It works with any go type interpreted as a JSON document, which means:
//
//   - if a type implements [JSONPointable], its [JSONPointable.JSONLookup] method is used to resolve [Pointer.Get]
//   - if a type implements [JSONSetable], its [JSONPointable.JSONSet] method is used to resolve [Pointer.Set]
//   - a go map[K]V is interpreted as an object, with type K assignable to a string
//   - a go slice []T is interpreted as an array
//   - a go struct is interpreted as an object, with exported fields interpreted as keys
//   - promoted fields from an embedded struct are traversed
//   - scalars (e.g. int, float64 ...), channels, functions and go arrays cannot be traversed
//
// For struct s resolved by reflection, key mappings honor the conventional struct tag `json`.
//
// Fields that do not specify a `json` tag, or specify an empty one, or are tagged as `json:"-"` are ignored.
//
// # Limitations
//
//   - Unlike go standard marshaling, untagged fields do not default to the go field name and are ignored.
//   - anonymous fields are not traversed if untagged
type Pointer struct {
	referenceTokens []string
}

// New creates a new json pointer from its string representation.
func New(jsonPointerString string) (Pointer, error) {
	var p Pointer
	err := p.parse(jsonPointerString)

	return p, err
}

// Get uses the pointer to retrieve a value from a JSON document.
//
// It returns the value with its type as a [reflect.Kind] or an error.
func (p *Pointer) Get(document any) (any, reflect.Kind, error) {
	return p.get(document, jsonname.DefaultJSONNameProvider)
}

// Set uses the pointer to set a value from a data type
// that represent a JSON document.
//
// It returns the updated document.
func (p *Pointer) Set(document any, value any) (any, error) {
	return document, p.set(document, value, jsonname.DefaultJSONNameProvider)
}

// DecodedTokens returns the decoded (unescaped) tokens of this JSON pointer.
func (p *Pointer) DecodedTokens() []string {
	result := make([]string, 0, len(p.referenceTokens))
	for _, token := range p.referenceTokens {
		result = append(result, Unescape(token))
	}

	return result
}

// IsEmpty returns true if this is an empty json pointer.
//
// This indicates that it points to the root document.
func (p *Pointer) IsEmpty() bool {
	return len(p.referenceTokens) == 0
}

// String representation of a pointer.
func (p *Pointer) String() string {
	if len(p.referenceTokens) == 0 {
		return emptyPointer
	}

	return pointerSeparator + strings.Join(p.referenceTokens, pointerSeparator)
}

func (p *Pointer) Offset(document string) (int64, error) {
	dec := json.NewDecoder(strings.NewReader(document))
	var offset int64
	for _, ttk := range p.DecodedTokens() {
		tk, err := dec.Token()
		if err != nil {
			return 0, err
		}
		switch tk := tk.(type) {
		case json.Delim:
			switch tk {
			case '{':
				offset, err = offsetSingleObject(dec, ttk)
				if err != nil {
					return 0, err
				}
			case '[':
				offset, err = offsetSingleArray(dec, ttk)
				if err != nil {
					return 0, err
				}
			default:
				return 0, fmt.Errorf("invalid token %#v: %w", tk, ErrPointer)
			}
		default:
			return 0, fmt.Errorf("invalid token %#v: %w", tk, ErrPointer)
		}
	}
	return offset, nil
}

// "Constructor", parses the given string JSON pointer.
func (p *Pointer) parse(jsonPointerString string) error {
	if jsonPointerString == emptyPointer {
		return nil
	}

	if !strings.HasPrefix(jsonPointerString, pointerSeparator) {
		// non empty pointer must start with "/"
		return errors.Join(ErrInvalidStart, ErrPointer)
	}

	referenceTokens := strings.Split(jsonPointerString, pointerSeparator)
	p.referenceTokens = append(p.referenceTokens, referenceTokens[1:]...)

	return nil
}

func (p *Pointer) get(node any, nameProvider *jsonname.NameProvider) (any, reflect.Kind, error) {
	if nameProvider == nil {
		nameProvider = jsonname.DefaultJSONNameProvider
	}

	kind := reflect.Invalid

	// full document when empty
	if len(p.referenceTokens) == 0 {
		return node, kind, nil
	}

	for _, token := range p.referenceTokens {
		decodedToken := Unescape(token)

		r, knd, err := getSingleImpl(node, decodedToken, nameProvider)
		if err != nil {
			return nil, knd, err
		}
		node = r
	}

	rValue := reflect.ValueOf(node)
	kind = rValue.Kind()

	return node, kind, nil
}

func (p *Pointer) set(node, data any, nameProvider *jsonname.NameProvider) error {
	knd := reflect.ValueOf(node).Kind()

	if knd != reflect.Pointer && knd != reflect.Struct && knd != reflect.Map && knd != reflect.Slice && knd != reflect.Array {
		return errors.Join(
			fmt.Errorf("unexpected type: %T", node), //nolint:err113 // err wrapping is carried out by errors.Join, not fmt.Errorf.
			ErrUnsupportedValueType,
			ErrPointer,
		)
	}

	l := len(p.referenceTokens)

	// full document when empty
	if l == 0 {
		return nil
	}

	if nameProvider == nil {
		nameProvider = jsonname.DefaultJSONNameProvider
	}

	var decodedToken string
	lastIndex := l - 1

	if lastIndex > 0 { // skip if we only have one token in pointer
		for _, token := range p.referenceTokens[:lastIndex] {
			decodedToken = Unescape(token)
			next, err := p.resolveNodeForToken(node, decodedToken, nameProvider)
			if err != nil {
				return err
			}

			node = next
		}
	}

	// last token
	decodedToken = Unescape(p.referenceTokens[lastIndex])

	return setSingleImpl(node, data, decodedToken, nameProvider)
}

func (p *Pointer) resolveNodeForToken(node any, decodedToken string, nameProvider *jsonname.NameProvider) (next any, err error) {
	// check for nil during traversal
	if isNil(node) {
		return nil, fmt.Errorf("cannot traverse through nil value at %q: %w", decodedToken, ErrPointer)
	}

	pointable, ok := node.(JSONPointable)
	if ok {
		r, err := pointable.JSONLookup(decodedToken)
		if err != nil {
			return nil, err
		}

		fld := reflect.ValueOf(r)
		if fld.CanAddr() && fld.Kind() != reflect.Interface && fld.Kind() != reflect.Map && fld.Kind() != reflect.Slice && fld.Kind() != reflect.Pointer {
			return fld.Addr().Interface(), nil
		}

		return r, nil
	}

	rValue := reflect.Indirect(reflect.ValueOf(node))
	kind := rValue.Kind()

	switch kind {
	case reflect.Struct:
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return nil, fmt.Errorf("object has no field %q: %w", decodedToken, ErrPointer)
		}

		return typeFromValue(rValue.FieldByName(nm)), nil

	case reflect.Map:
		kv := reflect.ValueOf(decodedToken)
		mv := rValue.MapIndex(kv)

		if !mv.IsValid() {
			return nil, errNoKey(decodedToken)
		}

		return typeFromValue(mv), nil

	case reflect.Slice:
		tokenIndex, err := strconv.Atoi(decodedToken)
		if err != nil {
			return nil, errors.Join(err, ErrPointer)
		}

		sLength := rValue.Len()
		if tokenIndex < 0 || tokenIndex >= sLength {
			return nil, errOutOfBounds(sLength, tokenIndex)
		}

		return typeFromValue(rValue.Index(tokenIndex)), nil

	default:
		return nil, errInvalidReference(decodedToken)
	}
}

func isNil(input any) bool {
	if input == nil {
		return true
	}

	kind := reflect.TypeOf(input).Kind()
	switch kind {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Chan:
		return reflect.ValueOf(input).IsNil()
	default:
		return false
	}
}

func typeFromValue(v reflect.Value) any {
	if v.CanAddr() && v.Kind() != reflect.Interface && v.Kind() != reflect.Map && v.Kind() != reflect.Slice && v.Kind() != reflect.Pointer {
		return v.Addr().Interface()
	}

	return v.Interface()
}

// GetForToken gets a value for a json pointer token 1 level deep.
func GetForToken(document any, decodedToken string) (any, reflect.Kind, error) {
	return getSingleImpl(document, decodedToken, jsonname.DefaultJSONNameProvider)
}

// SetForToken sets a value for a json pointer token 1 level deep.
func SetForToken(document any, decodedToken string, value any) (any, error) {
	return document, setSingleImpl(document, value, decodedToken, jsonname.DefaultJSONNameProvider)
}

func getSingleImpl(node any, decodedToken string, nameProvider *jsonname.NameProvider) (any, reflect.Kind, error) {
	rValue := reflect.Indirect(reflect.ValueOf(node))
	kind := rValue.Kind()
	if isNil(node) {
		return nil, kind, fmt.Errorf("nil value has no field %q: %w", decodedToken, ErrPointer)
	}

	switch typed := node.(type) {
	case JSONPointable:
		r, err := typed.JSONLookup(decodedToken)
		if err != nil {
			return nil, kind, err
		}
		return r, kind, nil
	case *any: // case of a pointer to interface, that is not resolved by reflect.Indirect
		return getSingleImpl(*typed, decodedToken, nameProvider)
	}

	switch kind {
	case reflect.Struct:
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return nil, kind, fmt.Errorf("object has no field %q: %w", decodedToken, ErrPointer)
		}

		fld := rValue.FieldByName(nm)

		return fld.Interface(), kind, nil

	case reflect.Map:
		kv := reflect.ValueOf(decodedToken)
		mv := rValue.MapIndex(kv)

		if mv.IsValid() {
			return mv.Interface(), kind, nil
		}

		return nil, kind, errNoKey(decodedToken)

	case reflect.Slice:
		tokenIndex, err := strconv.Atoi(decodedToken)
		if err != nil {
			return nil, kind, errors.Join(err, ErrPointer)
		}
		sLength := rValue.Len()
		if tokenIndex < 0 || tokenIndex >= sLength {
			return nil, kind, errOutOfBounds(sLength, tokenIndex)
		}

		elem := rValue.Index(tokenIndex)
		return elem.Interface(), kind, nil

	default:
		return nil, kind, errInvalidReference(decodedToken)
	}
}

func setSingleImpl(node, data any, decodedToken string, nameProvider *jsonname.NameProvider) error {
	// check for nil to prevent panic when calling rValue.Type()
	if isNil(node) {
		return fmt.Errorf("cannot set field %q on nil value: %w", decodedToken, ErrPointer)
	}

	if ns, ok := node.(JSONSetable); ok {
		return ns.JSONSet(decodedToken, data)
	}

	rValue := reflect.Indirect(reflect.ValueOf(node))

	switch rValue.Kind() {
	case reflect.Struct:
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return fmt.Errorf("object has no field %q: %w", decodedToken, ErrPointer)
		}

		fld := rValue.FieldByName(nm)
		if !fld.CanSet() {
			return fmt.Errorf("can't set struct field %s to %v: %w", nm, data, ErrPointer)
		}

		value := reflect.ValueOf(data)
		valueType := value.Type()
		assignedType := fld.Type()

		if !valueType.AssignableTo(assignedType) {
			return fmt.Errorf("can't set value with type %T to field %s with type %v: %w", data, nm, assignedType, ErrPointer)
		}

		fld.Set(value)

		return nil

	case reflect.Map:
		kv := reflect.ValueOf(decodedToken)
		rValue.SetMapIndex(kv, reflect.ValueOf(data))

		return nil

	case reflect.Slice:
		tokenIndex, err := strconv.Atoi(decodedToken)
		if err != nil {
			return errors.Join(err, ErrPointer)
		}

		sLength := rValue.Len()
		if tokenIndex < 0 || tokenIndex >= sLength {
			return errOutOfBounds(sLength, tokenIndex)
		}

		elem := rValue.Index(tokenIndex)
		if !elem.CanSet() {
			return fmt.Errorf("can't set slice index %s to %v: %w", decodedToken, data, ErrPointer)
		}

		value := reflect.ValueOf(data)
		valueType := value.Type()
		assignedType := elem.Type()

		if !valueType.AssignableTo(assignedType) {
			return fmt.Errorf("can't set value with type %T to slice element %d with type %v: %w", data, tokenIndex, assignedType, ErrPointer)
		}

		elem.Set(value)

		return nil

	default:
		return errInvalidReference(decodedToken)
	}
}

func offsetSingleObject(dec *json.Decoder, decodedToken string) (int64, error) {
	for dec.More() {
		offset := dec.InputOffset()
		tk, err := dec.Token()
		if err != nil {
			return 0, err
		}
		switch tk := tk.(type) {
		case json.Delim:
			switch tk {
			case '{':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			case '[':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			}
		case string:
			if tk == decodedToken {
				return offset, nil
			}
		default:
			return 0, fmt.Errorf("invalid token %#v: %w", tk, ErrPointer)
		}
	}

	return 0, fmt.Errorf("token reference %q not found: %w", decodedToken, ErrPointer)
}

func offsetSingleArray(dec *json.Decoder, decodedToken string) (int64, error) {
	idx, err := strconv.Atoi(decodedToken)
	if err != nil {
		return 0, fmt.Errorf("token reference %q is not a number: %w: %w", decodedToken, err, ErrPointer)
	}
	var i int
	for i = 0; i < idx && dec.More(); i++ {
		tk, err := dec.Token()
		if err != nil {
			return 0, err
		}

		if delim, isDelim := tk.(json.Delim); isDelim {
			switch delim {
			case '{':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			case '[':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			}
		}
	}

	if !dec.More() {
		return 0, fmt.Errorf("token reference %q not found: %w", decodedToken, ErrPointer)
	}

	return dec.InputOffset(), nil
}

// drainSingle drains a single level of object or array.
//
// The decoder has to guarantee the beginning delim (i.e. '{' or '[') has been consumed.
func drainSingle(dec *json.Decoder) error {
	for dec.More() {
		tk, err := dec.Token()
		if err != nil {
			return err
		}
		if delim, isDelim := tk.(json.Delim); isDelim {
			switch delim {
			case '{':
				if err = drainSingle(dec); err != nil {
					return err
				}
			case '[':
				if err = drainSingle(dec); err != nil {
					return err
				}
			}
		}
	}

	// consumes the ending delim
	if _, err := dec.Token(); err != nil {
		return err
	}

	return nil
}

// JSON pointer encoding:
// ~0 => ~
// ~1 => /
// ... and vice versa

const (
	encRefTok0 = `~0`
	encRefTok1 = `~1`
	decRefTok0 = `~`
	decRefTok1 = `/`
)

var (
	encRefTokReplacer = strings.NewReplacer(encRefTok1, decRefTok1, encRefTok0, decRefTok0) //nolint:gochecknoglobals // it's okay to declare a replacer as a private global
	decRefTokReplacer = strings.NewReplacer(decRefTok1, encRefTok1, decRefTok0, encRefTok0) //nolint:gochecknoglobals // it's okay to declare a replacer as a private global
)

// Unescape unescapes a json pointer reference token string to the original representation.
func Unescape(token string) string {
	return encRefTokReplacer.Replace(token)
}

// Escape escapes a pointer reference token string.
//
// The JSONPointer specification defines "/" as a separator and "~" as an escape prefix.
//
// Keys containing such characters are escaped with the following rules:
//
//   - "~" is escaped as "~0"
//   - "/" is escaped as "~1"
func Escape(token string) string {
	return decRefTokReplacer.Replace(token)
}
