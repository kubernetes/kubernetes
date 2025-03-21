/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package printers

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
	"unicode"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utiljson "k8s.io/apimachinery/pkg/util/json"
)

// KYAMLPrinter is an implementation of ResourcePrinter which formats data into
// a specific dialect of YAML, known as KYAML. KYAML is halfway between YAML
// and JSON, but is a strict subset of YAML, and so it should should be
// readable by any YAML parser. It is designed to be explicit and unambiguous,
// and eschews significant whitespace.
//
// Any type that implements json.Marshaler will be marshaled to JSON and then
// to YAML.
type KYAMLPrinter struct {
	printCount int64
}

// PrintObj prints the data as KYAML to the specified writer.
func (p *KYAMLPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	// We use reflect.Indirect here in order to obtain the actual value from a pointer.
	// We need an actual value in order to retrieve the package path for an object.
	// Using reflect.Indirect indiscriminately is valid here, as all runtime.Objects are supposed to be pointers.
	if InternalObjectPreventer.IsForbidden(reflect.Indirect(reflect.ValueOf(obj)).Type().PkgPath()) {
		return errors.New(InternalObjectPrinterErr)
	}

	// Print a comment shebang to indicate that this is a YAML document.
	count := atomic.AddInt64(&p.printCount, 1)
	if count == 1 {
		if _, err := fmt.Fprintln(w, "#!yaml"); err != nil {
			return fmt.Errorf("error writing to output: %w", err)
		}
	}

	// Always emit a document separator, which helps disambiguate between YAML
	// and JSON.
	if _, err := fmt.Fprintln(w, "---"); err != nil {
		return fmt.Errorf("error writing to output: %w", err)
	}

	switch obj := obj.(type) {
	case *metav1.WatchEvent:
		if InternalObjectPreventer.IsForbidden(reflect.Indirect(reflect.ValueOf(obj.Object.Object)).Type().PkgPath()) {
			return errors.New(InternalObjectPrinterErr)
		}
	case *runtime.Unknown:
		return p.fromJSON(obj.Raw, w)
	}

	if obj.GetObjectKind().GroupVersionKind().Empty() {
		return fmt.Errorf("missing apiVersion or kind; try GetObjectKind().SetGroupVersionKind() if you know the type")
	}

	return p.fromAny(obj, w)
}

func (p *KYAMLPrinter) fromJSON(jsonBytes []byte, w io.Writer) error {
	jsonObj, err := p.unmarshalJSON(jsonBytes)
	if err != nil {
		return err
	}
	return p.fromAny(jsonObj, w)
}

func (p *KYAMLPrinter) unmarshalJSON(jsonBytes []byte) (interface{}, error) {
	// We are using our own JSON here(instead of json.Unmarshal) because the
	// Go JSON library doesn't try to pick the right number type (int, float,
	// etc.) when unmarshalling to interface{}, it just picks float64
	// universally.
	var jsonObj interface{}
	if err := utiljson.Unmarshal(jsonBytes, &jsonObj); err != nil {
		return nil, fmt.Errorf("error unmarshaling from JSON: %w", err)
	}
	return jsonObj, nil
}

func (p *KYAMLPrinter) fromAny(obj any, w io.Writer) error {
	buf := &strings.Builder{}
	if err := p.formatValue(reflect.ValueOf(obj), 0, flagDefault, buf); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "%s\n", buf.String()); err != nil {
		return fmt.Errorf("error writing to output: %w", err)
	}
	return nil
}

type flagMask uint64

const (
	flagDefault   flagMask = 0
	flagLazyQuote flagMask = 1 << iota
)

func (p *KYAMLPrinter) formatValue(v reflect.Value, indent int, flags flagMask, buf *strings.Builder) error {
	if !v.IsValid() {
		buf.WriteString("null")
		return nil
	}

	// Handle things that implement json.Marshaler through that interface.
	if v.CanInterface() {
		if m, ok := v.Interface().(json.Marshaler); ok {
			jsonBytes, err := m.MarshalJSON()
			if err != nil {
				return fmt.Errorf("error marshaling to JSON: %w", err)
			}
			jsonObj, err := p.unmarshalJSON(jsonBytes)
			if err != nil {
				return err
			}
			// If something chose to marshal as "null", we should respect that.
			// Any nil-pointer will work.
			if jsonObj == nil {
				jsonObj = (*int)(nil)
			}
			v = reflect.ValueOf(jsonObj)
		}
	}

	switch v.Kind() {
	case reflect.String:
		val := v.String()
		if (flags&flagLazyQuote != 0) && !needsQuotes(val) {
			fmt.Fprintf(buf, "%s", val)
		} else {
			fmt.Fprintf(buf, "%q", val)
		}
		return nil

	case reflect.Bool:
		val := v.Bool()
		fmt.Fprintf(buf, "%v", val)
		return nil

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		val := v.Int()
		fmt.Fprintf(buf, "%d", val)
		return nil

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		val := v.Uint()
		fmt.Fprintf(buf, "%d", val)
		return nil

	case reflect.Float32, reflect.Float64:
		val := v.Float()
		fmt.Fprintf(buf, "%g", val)
		return nil

	case reflect.Interface, reflect.Ptr:
		if v.IsNil() {
			buf.WriteString("null")
			return nil
		}
		// Note: retain flags here.
		return p.formatValue(v.Elem(), indent, flags, buf)

	case reflect.Struct:
		buf.WriteString("{\n")
		if err := p.writeStructFields(v, indent, buf); err != nil {
			return err
		}
		p.writeIndent(indent, buf)
		buf.WriteString("}")
		return nil

	case reflect.Map:
		if v.Len() == 0 {
			buf.WriteString("{}")
			return nil
		}
		buf.WriteString("{\n")

		mapKeys := v.MapKeys()
		sort.Slice(mapKeys, func(i, j int) bool {
			return mapKeys[i].String() < mapKeys[j].String()
		})
		for _, key := range mapKeys {
			p.writeIndent(indent+1, buf)
			if err := p.formatValue(key, indent+1, flagLazyQuote, buf); err != nil {
				return err
			}
			buf.WriteString(": ")
			if err := p.formatValue(v.MapIndex(key), indent+1, flagDefault, buf); err != nil {
				return err
			}
			buf.WriteString(",\n")
		}
		p.writeIndent(indent, buf)
		buf.WriteString("}")
		return nil

	case reflect.Array, reflect.Slice:
		if v.Len() == 0 {
			buf.WriteString("[]")
			return nil
		}

		// Check if elements are structs or maps
		elemKind := v.Index(0).Kind()
		if elemKind == reflect.Interface || elemKind == reflect.Ptr {
			elemKind = v.Index(0).Elem().Kind()
		}
		isCuddledType := elemKind == reflect.Struct || elemKind == reflect.Map || elemKind == reflect.Array || elemKind == reflect.Slice

		buf.WriteString("[")
		if !isCuddledType {
			buf.WriteByte('\n')
		}

		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				if !isCuddledType {
					buf.WriteByte('\n')
				} else {
					buf.WriteByte(' ')
				}
			}
			if !isCuddledType {
				// Indent beyond the opening bracket.
				p.writeIndent(indent+1, buf)
			}
			// Indent is not +1 here because it's only used for cuddled types,
			// which will increment it internally, so the final closing brace
			// lines up with the current key.
			if err := p.formatValue(v.Index(i), indent, flagDefault, buf); err != nil {
				return err
			}
			if !isCuddledType || i != v.Len()-1 {
				buf.WriteByte(',')
			}
		}

		if !isCuddledType {
			buf.WriteString("\n")
			p.writeIndent(indent, buf)
		}
		buf.WriteByte(']')
		return nil

	default:
		return fmt.Errorf("KYAML: unsupported type: %v (kind %v)", v.Type(), v.Kind())
	}
}

func (p *KYAMLPrinter) writeStructFields(v reflect.Value, indent int, buf *strings.Builder) error {
	fbuf := &strings.Builder{}

	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		if field.PkgPath != "" { // Skip unexported fields
			continue
		}

		fieldName := ""
		fieldValue := v.Field(i)
		omitempty := false

		if jsonTag, found := field.Tag.Lookup("json"); found {
			parts := strings.Split(jsonTag, ",")
			if parts[0] == "-" {
				continue
			}
			if parts[0] != "" {
				fieldName = parts[0]
			}
			for _, opt := range parts[1:] {
				if opt == "omitempty" {
					omitempty = true
					break
				}
			}
		}

		// Check for empty value when omitempty is set
		if omitempty && isEmptyValue(fieldValue) {
			continue
		}

		// Handle embedded structs
		if field.Anonymous && fieldName == "" {
			if err := p.writeStructFields(fieldValue, indent, fbuf); err != nil {
				return err
			}
			continue
		}
		if fieldName == "" {
			// Do this AFTER handling embedded.
			fieldName = field.Name
		}

		p.writeIndent(indent+1, fbuf)
		fbuf.WriteString(fieldName)
		fbuf.WriteString(": ")
		if err := p.formatValue(fieldValue, indent+1, flagDefault, fbuf); err != nil {
			return err
		}
		fbuf.WriteString(",\n")
	}

	if str := fbuf.String(); len(str) > 0 {
		buf.WriteString(str)
	}
	return nil
}

func (p *KYAMLPrinter) writeIndent(level int, buf *strings.Builder) {
	for i := 0; i < level*2; i++ {
		buf.WriteByte(' ')
	}
}

func isEmptyValue(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return v.IsNil()
	}
	return false
}

func needsQuotes(s string) bool {
	if s == "" {
		return true
	}
	if yamlGuessesWrong(s) {
		return true
	}
	for _, r := range s {
		if !unicode.IsLetter(r) && !unicode.IsNumber(r) && r != '_' && r != '-' {
			return true
		}
	}
	return false
}

func yamlGuessesWrong(s string) bool {
	// Null-like strings
	switch s {
	case "null", "Null", "NULL", "~":
		return true
	}

	// Boolean-like strings
	if _, err := strconv.ParseBool(s); err == nil {
		return true
	}
	switch strings.ToLower(s) {
	case "true", "y", "yes", "on", "false", "n", "no", "off":
		return true
	}

	// Number-like strings (the stripping of underscores is gross)
	if _, err := strconv.ParseInt(strings.ReplaceAll(s, "_", ""), 0, 64); err == nil && !isSyntaxError(err) {
		return true
	}
	if _, err := strconv.ParseFloat(s, 64); err == nil && !isSyntaxError(err) {
		return true
	}

	// Infinity and NaN
	switch strings.ToLower(s) {
	case ".inf", "-.inf", "+.inf", ".nan":
		return true
	}

	// Time-like strings
	if _, matches := parseTimestamp(s); matches {
		return true
	}

	return false
}

func isSyntaxError(err error) bool {
	var numerr *strconv.NumError
	if ok := errors.As(err, &numerr); ok {
		return errors.Is(numerr.Err, strconv.ErrSyntax)
	}
	return false
}

// This is a subset of the formats allowed by the regular expression
// defined at http://yaml.org/type/timestamp.html.
//
// NOTE: This was copied from sigs.k8s.io/yaml/goyaml.v2
var allowedTimestampFormats = []string{
	"2006-1-2T15:4:5.999999999Z07:00", // RCF3339Nano with short date fields.
	"2006-1-2t15:4:5.999999999Z07:00", // RFC3339Nano with short date fields and lower-case "t".
	"2006-1-2 15:4:5.999999999",       // space separated with no time zone
	"2006-1-2",                        // date only
	// Notable exception: time.Parse cannot handle: "2001-12-14 21:59:43.10 -5"
	// from the set of examples.
}

// parseTimestamp parses s as a timestamp string and
// returns the timestamp and reports whether it succeeded.
// Timestamp formats are defined at http://yaml.org/type/timestamp.html
//
// NOTE: This was copied from sigs.k8s.io/yaml/goyaml.v2
func parseTimestamp(s string) (time.Time, bool) {
	// TODO write code to check all the formats supported by
	// http://yaml.org/type/timestamp.html instead of using time.Parse.

	// Quick check: all date formats start with YYYY-.
	i := 0
	for ; i < len(s); i++ {
		if c := s[i]; c < '0' || c > '9' {
			break
		}
	}
	if i != 4 || i == len(s) || s[i] != '-' {
		return time.Time{}, false
	}
	for _, format := range allowedTimestampFormats {
		if t, err := time.Parse(format, s); err == nil {
			return t, true
		}
	}
	return time.Time{}, false
}
