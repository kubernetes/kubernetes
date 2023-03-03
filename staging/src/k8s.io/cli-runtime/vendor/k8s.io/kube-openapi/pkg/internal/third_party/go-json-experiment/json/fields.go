// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

var errIgnoredField = errors.New("ignored field")

type isZeroer interface {
	IsZero() bool
}

var isZeroerType = reflect.TypeOf((*isZeroer)(nil)).Elem()

type structFields struct {
	flattened       []structField // listed in depth-first ordering
	byActualName    map[string]*structField
	byFoldedName    map[string][]*structField
	inlinedFallback *structField
}

type structField struct {
	id      int   // unique numeric ID in breadth-first ordering
	index   []int // index into a struct according to reflect.Type.FieldByIndex
	typ     reflect.Type
	fncs    *arshaler
	isZero  func(addressableValue) bool
	isEmpty func(addressableValue) bool
	fieldOptions
}

func makeStructFields(root reflect.Type) (structFields, *SemanticError) {
	var fs structFields
	fs.byActualName = make(map[string]*structField, root.NumField())
	fs.byFoldedName = make(map[string][]*structField, root.NumField())

	// ambiguous is a sentinel value to indicate that at least two fields
	// at the same depth have the same name, and thus cancel each other out.
	// This follows the same rules as selecting a field on embedded structs
	// where the shallowest field takes precedence. If more than one field
	// exists at the shallowest depth, then the selection is illegal.
	// See https://go.dev/ref/spec#Selectors.
	ambiguous := new(structField)

	// Setup a queue for a breath-first search.
	var queueIndex int
	type queueEntry struct {
		typ           reflect.Type
		index         []int
		visitChildren bool // whether to recursively visit inlined field in this struct
	}
	queue := []queueEntry{{root, nil, true}}
	seen := map[reflect.Type]bool{root: true}

	// Perform a breadth-first search over all reachable fields.
	// This ensures that len(f.index) will be monotonically increasing.
	for queueIndex < len(queue) {
		qe := queue[queueIndex]
		queueIndex++

		t := qe.typ
		inlinedFallbackIndex := -1         // index of last inlined fallback field in current struct
		namesIndex := make(map[string]int) // index of each field with a given JSON object name in current struct
		var hasAnyJSONTag bool             // whether any Go struct field has a `json` tag
		var hasAnyJSONField bool           // whether any JSON serializable fields exist in current struct
		for i := 0; i < t.NumField(); i++ {
			sf := t.Field(i)
			_, hasTag := sf.Tag.Lookup("json")
			hasAnyJSONTag = hasAnyJSONTag || hasTag
			options, err := parseFieldOptions(sf)
			if err != nil {
				if err == errIgnoredField {
					continue
				}
				return structFields{}, &SemanticError{GoType: t, Err: err}
			}
			hasAnyJSONField = true
			f := structField{
				// Allocate a new slice (len=N+1) to hold both
				// the parent index (len=N) and the current index (len=1).
				// Do this to avoid clobbering the memory of the parent index.
				index:        append(append(make([]int, 0, len(qe.index)+1), qe.index...), i),
				typ:          sf.Type,
				fieldOptions: options,
			}
			if sf.Anonymous && !f.hasName {
				f.inline = true // implied by use of Go embedding without an explicit name
			}
			if f.inline || f.unknown {
				// Handle an inlined field that serializes to/from
				// zero or more JSON object members.

				if f.inline && f.unknown {
					err := fmt.Errorf("Go struct field %s cannot have both `inline` and `unknown` specified", sf.Name)
					return structFields{}, &SemanticError{GoType: t, Err: err}
				}
				switch f.fieldOptions {
				case fieldOptions{name: f.name, quotedName: f.quotedName, inline: true}:
				case fieldOptions{name: f.name, quotedName: f.quotedName, unknown: true}:
				default:
					err := fmt.Errorf("Go struct field %s cannot have any options other than `inline` or `unknown` specified", sf.Name)
					return structFields{}, &SemanticError{GoType: t, Err: err}
				}

				// Unwrap one level of pointer indirection similar to how Go
				// only allows embedding either T or *T, but not **T.
				tf := f.typ
				if tf.Kind() == reflect.Pointer && tf.Name() == "" {
					tf = tf.Elem()
				}
				// Reject any types with custom serialization otherwise
				// it becomes impossible to know what sub-fields to inline.
				if which, _ := implementsWhich(tf,
					jsonMarshalerV2Type, jsonMarshalerV1Type, textMarshalerType,
					jsonUnmarshalerV2Type, jsonUnmarshalerV1Type, textUnmarshalerType,
				); which != nil && tf != rawValueType {
					err := fmt.Errorf("inlined Go struct field %s of type %s must not implement JSON marshal or unmarshal methods", sf.Name, tf)
					return structFields{}, &SemanticError{GoType: t, Err: err}
				}

				// Handle an inlined field that serializes to/from
				// a finite number of JSON object members backed by a Go struct.
				if tf.Kind() == reflect.Struct {
					if f.unknown {
						err := fmt.Errorf("inlined Go struct field %s of type %s with `unknown` tag must be a Go map of string key or a json.RawValue", sf.Name, tf)
						return structFields{}, &SemanticError{GoType: t, Err: err}
					}
					if qe.visitChildren {
						queue = append(queue, queueEntry{tf, f.index, !seen[tf]})
					}
					seen[tf] = true
					continue
				}

				// Handle an inlined field that serializes to/from any number of
				// JSON object members back by a Go map or RawValue.
				switch {
				case tf == rawValueType:
					f.fncs = nil // specially handled in arshal_inlined.go
				case tf.Kind() == reflect.Map && tf.Key() == stringType:
					f.fncs = lookupArshaler(tf.Elem())
				default:
					err := fmt.Errorf("inlined Go struct field %s of type %s must be a Go struct, Go map of string key, or json.RawValue", sf.Name, tf)
					return structFields{}, &SemanticError{GoType: t, Err: err}
				}

				// Reject multiple inlined fallback fields within the same struct.
				if inlinedFallbackIndex >= 0 {
					err := fmt.Errorf("inlined Go struct fields %s and %s cannot both be a Go map or json.RawValue", t.Field(inlinedFallbackIndex).Name, sf.Name)
					return structFields{}, &SemanticError{GoType: t, Err: err}
				}
				inlinedFallbackIndex = i

				// Multiple inlined fallback fields across different structs
				// follow the same precedence rules as Go struct embedding.
				if fs.inlinedFallback == nil {
					fs.inlinedFallback = &f // store first occurrence at lowest depth
				} else if len(fs.inlinedFallback.index) == len(f.index) {
					fs.inlinedFallback = ambiguous // at least two occurrences at same depth
				}
			} else {
				// Handle normal Go struct field that serializes to/from
				// a single JSON object member.

				// Provide a function that uses a type's IsZero method.
				switch {
				case sf.Type.Kind() == reflect.Interface && sf.Type.Implements(isZeroerType):
					f.isZero = func(va addressableValue) bool {
						// Avoid panics calling IsZero on a nil interface or
						// non-nil interface with nil pointer.
						return va.IsNil() || (va.Elem().Kind() == reflect.Pointer && va.Elem().IsNil()) || va.Interface().(isZeroer).IsZero()
					}
				case sf.Type.Kind() == reflect.Pointer && sf.Type.Implements(isZeroerType):
					f.isZero = func(va addressableValue) bool {
						// Avoid panics calling IsZero on nil pointer.
						return va.IsNil() || va.Interface().(isZeroer).IsZero()
					}
				case sf.Type.Implements(isZeroerType):
					f.isZero = func(va addressableValue) bool { return va.Interface().(isZeroer).IsZero() }
				case reflect.PointerTo(sf.Type).Implements(isZeroerType):
					f.isZero = func(va addressableValue) bool { return va.Addr().Interface().(isZeroer).IsZero() }
				}

				// Provide a function that can determine whether the value would
				// serialize as an empty JSON value.
				switch sf.Type.Kind() {
				case reflect.String, reflect.Map, reflect.Array, reflect.Slice:
					f.isEmpty = func(va addressableValue) bool { return va.Len() == 0 }
				case reflect.Pointer, reflect.Interface:
					f.isEmpty = func(va addressableValue) bool { return va.IsNil() }
				}

				f.id = len(fs.flattened)
				f.fncs = lookupArshaler(sf.Type)
				fs.flattened = append(fs.flattened, f)

				// Reject user-specified names with invalid UTF-8.
				if !utf8.ValidString(f.name) {
					err := fmt.Errorf("Go struct field %s has JSON object name %q with invalid UTF-8", sf.Name, f.name)
					return structFields{}, &SemanticError{GoType: t, Err: err}
				}
				// Reject multiple fields with same name within the same struct.
				if j, ok := namesIndex[f.name]; ok {
					err := fmt.Errorf("Go struct fields %s and %s conflict over JSON object name %q", t.Field(j).Name, sf.Name, f.name)
					return structFields{}, &SemanticError{GoType: t, Err: err}
				}
				namesIndex[f.name] = i

				// Multiple fields of the same name across different structs
				// follow the same precedence rules as Go struct embedding.
				if f2 := fs.byActualName[f.name]; f2 == nil {
					fs.byActualName[f.name] = &fs.flattened[len(fs.flattened)-1] // store first occurrence at lowest depth
				} else if len(f2.index) == len(f.index) {
					fs.byActualName[f.name] = ambiguous // at least two occurrences at same depth
				}
			}
		}

		// NOTE: New users to the json package are occasionally surprised that
		// unexported fields are ignored. This occurs by necessity due to our
		// inability to directly introspect such fields with Go reflection
		// without the use of unsafe.
		//
		// To reduce friction here, refuse to serialize any Go struct that
		// has no JSON serializable fields, has at least one Go struct field,
		// and does not have any `json` tags present. For example,
		// errors returned by errors.New would fail to serialize.
		isEmptyStruct := t.NumField() == 0
		if !isEmptyStruct && !hasAnyJSONTag && !hasAnyJSONField {
			err := errors.New("Go struct has no exported fields")
			return structFields{}, &SemanticError{GoType: t, Err: err}
		}
	}

	// Remove all fields that are duplicates.
	// This may move elements forward to fill the holes from removed fields.
	var n int
	for _, f := range fs.flattened {
		switch f2 := fs.byActualName[f.name]; {
		case f2 == ambiguous:
			delete(fs.byActualName, f.name)
		case f2 == nil:
			continue // may be nil due to previous delete
		// TODO(https://go.dev/issue/45955): Use slices.Equal.
		case reflect.DeepEqual(f.index, f2.index):
			f.id = n
			fs.flattened[n] = f
			fs.byActualName[f.name] = &fs.flattened[n] // fix pointer to new location
			n++
		}
	}
	fs.flattened = fs.flattened[:n]
	if fs.inlinedFallback == ambiguous {
		fs.inlinedFallback = nil
	}
	if len(fs.flattened) != len(fs.byActualName) {
		panic(fmt.Sprintf("BUG: flattened list of fields mismatches fields mapped by name: %d != %d", len(fs.flattened), len(fs.byActualName)))
	}

	// Sort the fields according to a depth-first ordering.
	// This operation will cause pointers in byActualName to become incorrect,
	// which we will correct in another loop shortly thereafter.
	sort.Slice(fs.flattened, func(i, j int) bool {
		si := fs.flattened[i].index
		sj := fs.flattened[j].index
		for len(si) > 0 && len(sj) > 0 {
			switch {
			case si[0] < sj[0]:
				return true
			case si[0] > sj[0]:
				return false
			default:
				si = si[1:]
				sj = sj[1:]
			}
		}
		return len(si) < len(sj)
	})

	// Recompute the mapping of fields in the byActualName map.
	// Pre-fold all names so that we can lookup folded names quickly.
	for i, f := range fs.flattened {
		foldedName := string(foldName([]byte(f.name)))
		fs.byActualName[f.name] = &fs.flattened[i]
		fs.byFoldedName[foldedName] = append(fs.byFoldedName[foldedName], &fs.flattened[i])
	}
	for foldedName, fields := range fs.byFoldedName {
		if len(fields) > 1 {
			// The precedence order for conflicting nocase names
			// is by breadth-first order, rather than depth-first order.
			sort.Slice(fields, func(i, j int) bool {
				return fields[i].id < fields[j].id
			})
			fs.byFoldedName[foldedName] = fields
		}
	}

	return fs, nil
}

type fieldOptions struct {
	name       string
	quotedName string // quoted name per RFC 8785, section 3.2.2.2.
	hasName    bool
	nocase     bool
	inline     bool
	unknown    bool
	omitzero   bool
	omitempty  bool
	string     bool
	format     string
}

// parseFieldOptions parses the `json` tag in a Go struct field as
// a structured set of options configuring parameters such as
// the JSON member name and other features.
// As a special case, it returns errIgnoredField if the field is ignored.
func parseFieldOptions(sf reflect.StructField) (out fieldOptions, err error) {
	tag, hasTag := sf.Tag.Lookup("json")

	// Check whether this field is explicitly ignored.
	if tag == "-" {
		return fieldOptions{}, errIgnoredField
	}

	// Check whether this field is unexported.
	if !sf.IsExported() {
		// In contrast to v1, v2 no longer forwards exported fields from
		// embedded fields of unexported types since Go reflection does not
		// allow the same set of operations that are available in normal cases
		// of purely exported fields.
		// See https://go.dev/issue/21357 and https://go.dev/issue/24153.
		if sf.Anonymous {
			return fieldOptions{}, fmt.Errorf("embedded Go struct field %s of an unexported type must be explicitly ignored with a `json:\"-\"` tag", sf.Type.Name())
		}
		// Tag options specified on an unexported field suggests user error.
		if hasTag {
			return fieldOptions{}, fmt.Errorf("unexported Go struct field %s cannot have non-ignored `json:%q` tag", sf.Name, tag)
		}
		return fieldOptions{}, errIgnoredField
	}

	// Determine the JSON member name for this Go field. A user-specified name
	// may be provided as either an identifier or a single-quoted string.
	// The single-quoted string allows arbitrary characters in the name.
	// See https://go.dev/issue/2718 and https://go.dev/issue/3546.
	out.name = sf.Name // always starts with an uppercase character
	if len(tag) > 0 && !strings.HasPrefix(tag, ",") {
		// For better compatibility with v1, accept almost any unescaped name.
		n := len(tag) - len(strings.TrimLeftFunc(tag, func(r rune) bool {
			return !strings.ContainsRune(",\\'\"`", r) // reserve comma, backslash, and quotes
		}))
		opt := tag[:n]
		if n == 0 {
			// Allow a single quoted string for arbitrary names.
			opt, n, err = consumeTagOption(tag)
			if err != nil {
				return fieldOptions{}, fmt.Errorf("Go struct field %s has malformed `json` tag: %v", sf.Name, err)
			}
		}
		out.hasName = true
		out.name = opt
		tag = tag[n:]
	}
	b, _ := appendString(nil, out.name, false, nil)
	out.quotedName = string(b)

	// Handle any additional tag options (if any).
	var wasFormat bool
	seenOpts := make(map[string]bool)
	for len(tag) > 0 {
		// Consume comma delimiter.
		if tag[0] != ',' {
			return fieldOptions{}, fmt.Errorf("Go struct field %s has malformed `json` tag: invalid character %q before next option (expecting ',')", sf.Name, tag[0])
		}
		tag = tag[len(","):]
		if len(tag) == 0 {
			return fieldOptions{}, fmt.Errorf("Go struct field %s has malformed `json` tag: invalid trailing ',' character", sf.Name)
		}

		// Consume and process the tag option.
		opt, n, err := consumeTagOption(tag)
		if err != nil {
			return fieldOptions{}, fmt.Errorf("Go struct field %s has malformed `json` tag: %v", sf.Name, err)
		}
		rawOpt := tag[:n]
		tag = tag[n:]
		switch {
		case wasFormat:
			return fieldOptions{}, fmt.Errorf("Go struct field %s has `format` tag option that was not specified last", sf.Name)
		case strings.HasPrefix(rawOpt, "'") && strings.TrimFunc(opt, isLetterOrDigit) == "":
			return fieldOptions{}, fmt.Errorf("Go struct field %s has unnecessarily quoted appearance of `%s` tag option; specify `%s` instead", sf.Name, rawOpt, opt)
		}
		switch opt {
		case "nocase":
			out.nocase = true
		case "inline":
			out.inline = true
		case "unknown":
			out.unknown = true
		case "omitzero":
			out.omitzero = true
		case "omitempty":
			out.omitempty = true
		case "string":
			out.string = true
		case "format":
			if !strings.HasPrefix(tag, ":") {
				return fieldOptions{}, fmt.Errorf("Go struct field %s is missing value for `format` tag option", sf.Name)
			}
			tag = tag[len(":"):]
			opt, n, err := consumeTagOption(tag)
			if err != nil {
				return fieldOptions{}, fmt.Errorf("Go struct field %s has malformed value for `format` tag option: %v", sf.Name, err)
			}
			tag = tag[n:]
			out.format = opt
			wasFormat = true
		default:
			// Reject keys that resemble one of the supported options.
			// This catches invalid mutants such as "omitEmpty" or "omit_empty".
			normOpt := strings.ReplaceAll(strings.ToLower(opt), "_", "")
			switch normOpt {
			case "nocase", "inline", "unknown", "omitzero", "omitempty", "string", "format":
				return fieldOptions{}, fmt.Errorf("Go struct field %s has invalid appearance of `%s` tag option; specify `%s` instead", sf.Name, opt, normOpt)
			}

			// NOTE: Everything else is ignored. This does not mean it is
			// forward compatible to insert arbitrary tag options since
			// a future version of this package may understand that tag.
		}

		// Reject duplicates.
		if seenOpts[opt] {
			return fieldOptions{}, fmt.Errorf("Go struct field %s has duplicate appearance of `%s` tag option", sf.Name, rawOpt)
		}
		seenOpts[opt] = true
	}
	return out, nil
}

func consumeTagOption(in string) (string, int, error) {
	switch r, _ := utf8.DecodeRuneInString(in); {
	// Option as a Go identifier.
	case r == '_' || unicode.IsLetter(r):
		n := len(in) - len(strings.TrimLeftFunc(in, isLetterOrDigit))
		return in[:n], n, nil
	// Option as a single-quoted string.
	case r == '\'':
		// The grammar is nearly identical to a double-quoted Go string literal,
		// but uses single quotes as the terminators. The reason for a custom
		// grammar is because both backtick and double quotes cannot be used
		// verbatim in a struct tag.
		//
		// Convert a single-quoted string to a double-quote string and rely on
		// strconv.Unquote to handle the rest.
		var inEscape bool
		b := []byte{'"'}
		n := len(`'`)
		for len(in) > n {
			r, rn := utf8.DecodeRuneInString(in[n:])
			switch {
			case inEscape:
				if r == '\'' {
					b = b[:len(b)-1] // remove escape character: `\'` => `'`
				}
				inEscape = false
			case r == '\\':
				inEscape = true
			case r == '"':
				b = append(b, '\\') // insert escape character: `"` => `\"`
			case r == '\'':
				b = append(b, '"')
				n += len(`'`)
				out, err := strconv.Unquote(string(b))
				if err != nil {
					return "", 0, fmt.Errorf("invalid single-quoted string: %s", in[:n])
				}
				return out, n, nil
			}
			b = append(b, in[n:][:rn]...)
			n += rn
		}
		if n > 10 {
			n = 10 // limit the amount of context printed in the error
		}
		return "", 0, fmt.Errorf("single-quoted string not terminated: %s...", in[:n])
	case len(in) == 0:
		return "", 0, io.ErrUnexpectedEOF
	default:
		return "", 0, fmt.Errorf("invalid character %q at start of option (expecting Unicode letter or single quote)", r)
	}
}

func isLetterOrDigit(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}
