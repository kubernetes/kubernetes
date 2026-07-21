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
)

const (
	emptyPointer     = ``
	pointerSeparator = `/`
)

// Pointer is a representation of a json pointer.
//
// Use [Pointer.Get] to retrieve a value or [Pointer.Set] to set a value.
//
// It works with any go type interpreted as a JSON document, which means:
//
//   - if a type implements [JSONPointable], its [JSONPointable.JSONLookup] method is used to resolve [Pointer.Get]
//   - if a type implements [JSONSetable], its [JSONSetable.JSONSet] method is used to resolve [Pointer.Set]
//   - a go map[K]V is interpreted as an object, with type K assignable to a string
//   - a go slice []T is interpreted as an array
//   - a go struct is interpreted as an object, with exported fields interpreted as keys
//   - promoted fields from an embedded struct are traversed
//   - scalars (e.g. int, float64 ...), channels, functions and go arrays cannot be traversed
//
// For struct s resolved by reflection, key mappings honor the conventional struct tag `json`.
//
// Fields that do not specify a `json` tag, or specify an empty one, or are tagged as `json:"-"` are
// ignored.
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
func (p *Pointer) Get(document any, opts ...Option) (any, reflect.Kind, error) {
	o := optionsWithDefaults(opts)

	return p.get(document, o.provider)
}

// Set uses the pointer to set a value from a data type that represent a JSON document.
//
// # Mutation contract
//
// Set mutates the provided document in place whenever Go's type system allows it: when document is
// a map, a pointer, or when the targeted value is reached through an addressable ancestor (e.g. a
// struct field traversed via a pointer, a slice element).
//
// Callers that rely on this in-place behavior may continue to ignore the returned document.
//
// The returned document is only load-bearing when Set cannot mutate in place.
//
// This happens in one specific case: appending to a top-level slice passed by value (e.g. document
// of type []T rather than *[]T) via the RFC 6901 "-" terminal token. reflect.Append produces a new
// slice header that the library cannot rebind into the caller's variable; the updated document is
// returned instead.
//
// Pass *[]T if you want in-place rebind for that case as well.
//
// See [ErrDashToken] for the semantics of the "-" token.
func (p *Pointer) Set(document any, value any, opts ...Option) (any, error) {
	o := optionsWithDefaults(opts)

	return p.set(document, value, o.provider)
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

// Offset returns the byte offset, in the raw JSON text of document, of the location referenced by
// this pointer's terminal token.
//
// Unlike [Pointer.Get] and [Pointer.Set], which operate on a decoded Go value, Offset operates
// directly on the textual JSON source.
//
// It drives an [encoding/json.Decoder] over the string and stops at the terminal token, returning
// the position at which the decoder was about to read that token.
//
// It is primarily intended for tooling that needs to map a pointer back to a region of the original
// source: reporting line/column for validation or parse diagnostics, extracting a sub-document by
// slicing the raw bytes, or highlighting the referenced span in an editor.
//
// # Offset semantics
//
// The meaning of the returned offset depends on whether the terminal token addresses an object
// property or an array element:
//
//   - Object property: the offset points to the first byte of the key (its
//     opening quote character), not to the associated value. For example,
//     pointer "/foo/bar" against {"foo": {"bar": 21}} returns 9, the index of
//     the opening quote of "bar".
//   - Array element: the offset points to the first byte of the value at that
//     index. For example, pointer "/0/1" against [[1,2], [3,4]] returns 4,
//     the index of the digit 2.
//
// # Errors
//
// Offset returns an error in any of these cases:
//
//   - document is not syntactically valid JSON;
//   - the structure of document does not match the pointer (e.g. traversing
//     into a scalar, or a token that is neither a valid key nor a valid
//     numeric index);
//   - a referenced key or index does not exist in document;
//   - the pointer's terminal token is the RFC 6901 "-" array token, which
//     designates a nonexistent element and therefore has no offset in the
//     source. The returned error wraps [ErrDashToken].
//
// All errors wrap [ErrPointer].
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
	return skipJSONSeparator(document, offset), nil
}

// skipJSONSeparator advances offset past trailing JSON whitespace and at most one value separator
// (comma) in document, so the result points at the first byte of the next JSON token.
//
// The streaming decoder's InputOffset sits right after the most recently consumed token, which
// between values is the comma (or whitespace) — not the following token.
//
// Normalizing here keeps Offset's contract uniform: for both object keys and array elements, and
// regardless of position within the parent container, the returned offset always points at the
// first byte of the addressed token.
func skipJSONSeparator(document string, offset int64) int64 {
	n := int64(len(document))
	for offset < n && isJSONWhitespace(document[offset]) {
		offset++
	}
	if offset < n && document[offset] == ',' {
		offset++
	}
	for offset < n && isJSONWhitespace(document[offset]) {
		offset++
	}
	return offset
}

func isJSONWhitespace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
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

func (p *Pointer) get(node any, nameProvider NameProvider) (any, reflect.Kind, error) {
	if nameProvider == nil {
		nameProvider = defaultOptions.provider
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

func (p *Pointer) set(node, data any, nameProvider NameProvider) (any, error) {
	knd := reflect.ValueOf(node).Kind()

	if knd != reflect.Pointer && knd != reflect.Struct && knd != reflect.Map && knd != reflect.Slice && knd != reflect.Array {
		return node, errors.Join(
			fmt.Errorf("unexpected type: %T", node), //nolint:err113 // err wrapping is carried out by errors.Join, not fmt.Errorf.
			ErrUnsupportedValueType,
			ErrPointer,
		)
	}

	// full document when empty
	if len(p.referenceTokens) == 0 {
		return node, nil
	}

	if nameProvider == nil {
		nameProvider = defaultOptions.provider
	}

	return p.setAt(node, p.referenceTokens, data, nameProvider)
}

// setAt recursively walks the token list, setting the data at the terminal token and rebinding any
// new child reference (e.g. a slice header returned by an "-" append) into its parent on the way
// back up.
//
// Returning the (possibly new) node at each level is what makes append work at any depth without
// requiring the caller to pass a pointer to the containing slice: the new slice header propagates
// up and each parent rebinds it via the appropriate kind-specific setter.
func (p *Pointer) setAt(node any, tokens []string, data any, nameProvider NameProvider) (any, error) {
	decodedToken := Unescape(tokens[0])

	if len(tokens) == 1 {
		return setSingleImpl(node, data, decodedToken, nameProvider)
	}

	child, err := p.resolveNodeForToken(node, decodedToken, nameProvider)
	if err != nil {
		return node, err
	}

	newChild, err := p.setAt(child, tokens[1:], data, nameProvider)
	if err != nil {
		return node, err
	}

	return rebindChild(node, decodedToken, newChild, nameProvider)
}

// rebindChild writes newChild back into node at decodedToken.
//
// For cases where the child was already mutated in place (pointer aliasing, addressable slice
// elements) the rebind is a safe no-op.
//
// For cases where the child was returned by value (map entries holding a slice, slices reached
// through a non-addressable ancestor), the rebind propagates the new value into the parent.
//
// Parents implementing [JSONPointable] are left alone: they took ownership of the child via
// JSONLookup and did not opt into a JSONSet-based rebind on intermediate tokens.
func rebindChild(node any, decodedToken string, newChild any, nameProvider NameProvider) (any, error) {
	if _, ok := node.(JSONPointable); ok {
		return node, nil
	}

	rValue := reflect.Indirect(reflect.ValueOf(node))

	switch rValue.Kind() {
	case reflect.Struct:
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return node, fmt.Errorf("object has no field %q: %w", decodedToken, ErrPointer)
		}
		fld := rValue.FieldByName(nm)
		if !fld.CanSet() {
			return node, nil
		}
		assignReflectValue(fld, newChild)
		return node, nil

	case reflect.Map:
		rValue.SetMapIndex(reflect.ValueOf(decodedToken), reflect.ValueOf(newChild))
		return node, nil

	case reflect.Slice:
		if decodedToken == dashToken {
			return node, errDashIntermediate()
		}
		idx, err := strconv.Atoi(decodedToken)
		if err != nil {
			return node, errors.Join(err, ErrPointer)
		}
		elem := rValue.Index(idx)
		if !elem.CanSet() {
			return node, nil
		}
		assignReflectValue(elem, newChild)
		return node, nil

	default:
		return node, errInvalidReference(decodedToken)
	}
}

// assignReflectValue assigns src into dst, unwrapping a pointer when dst expects the pointee type.
//
// This tolerates the pointer-wrapping performed by [typeFromValue] for addressable fields.
func assignReflectValue(dst reflect.Value, src any) {
	nv := reflect.ValueOf(src)
	if !nv.IsValid() {
		return
	}
	if nv.Type().AssignableTo(dst.Type()) {
		dst.Set(nv)
		return
	}
	if nv.Kind() == reflect.Pointer && nv.Elem().Type().AssignableTo(dst.Type()) {
		dst.Set(nv.Elem())
	}
}

func (p *Pointer) resolveNodeForToken(node any, decodedToken string, nameProvider NameProvider) (next any, err error) {
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
		if decodedToken == dashToken {
			return nil, errDashIntermediate()
		}
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
func GetForToken(document any, decodedToken string, opts ...Option) (any, reflect.Kind, error) {
	o := optionsWithDefaults(opts)

	return getSingleImpl(document, decodedToken, o.provider)
}

// SetForToken sets a value for a json pointer token 1 level deep.
//
// See [Pointer.Set] for the mutation contract, in particular the handling of the RFC 6901 "-" token
// on slices.
func SetForToken(document any, decodedToken string, value any, opts ...Option) (any, error) {
	o := optionsWithDefaults(opts)

	return setSingleImpl(document, value, decodedToken, o.provider)
}

func getSingleImpl(node any, decodedToken string, nameProvider NameProvider) (any, reflect.Kind, error) {
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
		if decodedToken == dashToken {
			return nil, kind, errDashOnGet()
		}
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

func setSingleImpl(node, data any, decodedToken string, nameProvider NameProvider) (any, error) {
	// check for nil to prevent panic when calling rValue.Type()
	if isNil(node) {
		return node, fmt.Errorf("cannot set field %q on nil value: %w", decodedToken, ErrPointer)
	}

	if ns, ok := node.(JSONSetable); ok {
		return node, ns.JSONSet(decodedToken, data)
	}

	rValue := reflect.Indirect(reflect.ValueOf(node))

	switch rValue.Kind() {
	case reflect.Struct:
		nm, ok := nameProvider.GetGoNameForType(rValue.Type(), decodedToken)
		if !ok {
			return node, fmt.Errorf("object has no field %q: %w", decodedToken, ErrPointer)
		}

		fld := rValue.FieldByName(nm)
		if !fld.CanSet() {
			return node, fmt.Errorf("can't set struct field %s to %v: %w", nm, data, ErrPointer)
		}

		value := reflect.ValueOf(data)
		valueType := value.Type()
		assignedType := fld.Type()

		if !valueType.AssignableTo(assignedType) {
			return node, fmt.Errorf("can't set value with type %T to field %s with type %v: %w", data, nm, assignedType, ErrPointer)
		}

		fld.Set(value)

		return node, nil

	case reflect.Map:
		kv := reflect.ValueOf(decodedToken)
		rValue.SetMapIndex(kv, reflect.ValueOf(data))

		return node, nil

	case reflect.Slice:
		if decodedToken == dashToken {
			// RFC 6901 §4 / RFC 6902 append semantics: terminal "-" appends the value to the slice.
			//
			// We rebind in place when the slice is reachable via an addressable ancestor; otherwise we
			// return the new slice header for the parent (or the public Set) to rebind.
			value := reflect.ValueOf(data)
			elemType := rValue.Type().Elem()
			if !value.Type().AssignableTo(elemType) {
				return node, fmt.Errorf("can't append value of type %T to slice of %v: %w", data, elemType, ErrPointer)
			}
			newSlice := reflect.Append(rValue, value)
			if rValue.CanSet() {
				rValue.Set(newSlice)
				return node, nil
			}
			return newSlice.Interface(), nil
		}

		tokenIndex, err := strconv.Atoi(decodedToken)
		if err != nil {
			return node, errors.Join(err, ErrPointer)
		}

		sLength := rValue.Len()
		if tokenIndex < 0 || tokenIndex >= sLength {
			return node, errOutOfBounds(sLength, tokenIndex)
		}

		elem := rValue.Index(tokenIndex)
		if !elem.CanSet() {
			return node, fmt.Errorf("can't set slice index %s to %v: %w", decodedToken, data, ErrPointer)
		}

		value := reflect.ValueOf(data)
		valueType := value.Type()
		assignedType := elem.Type()

		if !valueType.AssignableTo(assignedType) {
			return node, fmt.Errorf("can't set value with type %T to slice element %d with type %v: %w", data, tokenIndex, assignedType, ErrPointer)
		}

		elem.Set(value)

		return node, nil

	default:
		return node, errInvalidReference(decodedToken)
	}
}

func offsetSingleObject(dec *json.Decoder, decodedToken string) (int64, error) {
	for dec.More() {
		offset := dec.InputOffset()
		tk, err := dec.Token()
		if err != nil {
			return 0, err
		}
		key, ok := tk.(string)
		if !ok {
			return 0, fmt.Errorf("invalid key token %#v: %w", tk, ErrPointer)
		}
		if key == decodedToken {
			return offset, nil
		}

		// Consume the associated value.
		// Scalars are fully read by a single Token() call; composite values must be drained.
		tk, err = dec.Token()
		if err != nil {
			return 0, err
		}
		if delim, isDelim := tk.(json.Delim); isDelim {
			switch delim {
			case '{', '[':
				if err = drainSingle(dec); err != nil {
					return 0, err
				}
			}
		}
	}

	return 0, fmt.Errorf("token reference %q not found: %w", decodedToken, ErrPointer)
}

func offsetSingleArray(dec *json.Decoder, decodedToken string) (int64, error) {
	if decodedToken == dashToken {
		return 0, errDashOnOffset()
	}
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

// JSON pointer encoding: ~0 => ~ ~1 => / ... and vice versa.

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
