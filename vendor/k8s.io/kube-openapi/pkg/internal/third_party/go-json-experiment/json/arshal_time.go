// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
)

var (
	timeDurationType = reflect.TypeOf((*time.Duration)(nil)).Elem()
	timeTimeType     = reflect.TypeOf((*time.Time)(nil)).Elem()
)

func makeTimeArshaler(fncs *arshaler, t reflect.Type) *arshaler {
	// Ideally, time types would implement MarshalerV2 and UnmarshalerV2,
	// but that would incur a dependency on package json from package time.
	// Given how widely used time is, it is more acceptable that we incur a
	// dependency on time from json.
	//
	// Injecting the arshaling functionality like this will not be identical
	// to actually declaring methods on the time types since embedding of the
	// time types will not be able to forward this functionality.
	switch t {
	case timeDurationType:
		fncs.nonDefault = true
		marshalNanos := fncs.marshal
		fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
			if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
				if mo.format == "nanos" {
					mo.format = ""
					return marshalNanos(mo, enc, va)
				} else {
					return newInvalidFormatError("marshal", t, mo.format)
				}
			}

			td := va.Interface().(time.Duration)
			b := enc.UnusedBuffer()
			b = append(b, '"')
			b = append(b, td.String()...) // never contains special characters
			b = append(b, '"')
			return enc.WriteValue(b)
		}
		unmarshalNanos := fncs.unmarshal
		fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
			// TODO: Should there be a flag that specifies that we can unmarshal
			// from either form since there would be no ambiguity?
			if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
				if uo.format == "nanos" {
					uo.format = ""
					return unmarshalNanos(uo, dec, va)
				} else {
					return newInvalidFormatError("unmarshal", t, uo.format)
				}
			}

			var flags valueFlags
			td := va.Addr().Interface().(*time.Duration)
			val, err := dec.readValue(&flags)
			if err != nil {
				return err
			}
			switch k := val.Kind(); k {
			case 'n':
				*td = time.Duration(0)
				return nil
			case '"':
				val = unescapeStringMayCopy(val, flags.isVerbatim())
				td2, err := time.ParseDuration(string(val))
				if err != nil {
					return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
				}
				*td = td2
				return nil
			default:
				return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
			}
		}
	case timeTimeType:
		fncs.nonDefault = true
		fncs.marshal = func(mo MarshalOptions, enc *Encoder, va addressableValue) error {
			format := time.RFC3339Nano
			isRFC3339 := true
			if mo.format != "" && mo.formatDepth == enc.tokens.depth() {
				var err error
				format, isRFC3339, err = checkTimeFormat(mo.format)
				if err != nil {
					return &SemanticError{action: "marshal", GoType: t, Err: err}
				}
			}

			tt := va.Interface().(time.Time)
			b := enc.UnusedBuffer()
			b = append(b, '"')
			b = tt.AppendFormat(b, format)
			b = append(b, '"')
			if isRFC3339 {
				// Not all Go timestamps can be represented as valid RFC 3339.
				// Explicitly check for these edge cases.
				// See https://go.dev/issue/4556 and https://go.dev/issue/54580.
				var err error
				switch b := b[len(`"`) : len(b)-len(`"`)]; {
				case b[len("9999")] != '-': // year must be exactly 4 digits wide
					err = errors.New("year outside of range [0,9999]")
				case b[len(b)-1] != 'Z':
					c := b[len(b)-len("Z07:00")]
					if ('0' <= c && c <= '9') || parseDec2(b[len(b)-len("07:00"):]) >= 24 {
						err = errors.New("timezone hour outside of range [0,23]")
					}
				}
				if err != nil {
					return &SemanticError{action: "marshal", GoType: t, Err: err}
				}
				return enc.WriteValue(b) // RFC 3339 never needs JSON escaping
			}
			// The format may contain special characters that need escaping.
			// Verify that the result is a valid JSON string (common case),
			// otherwise escape the string correctly (slower case).
			if consumeSimpleString(b) != len(b) {
				b, _ = appendString(nil, string(b[len(`"`):len(b)-len(`"`)]), true, nil)
			}
			return enc.WriteValue(b)
		}
		fncs.unmarshal = func(uo UnmarshalOptions, dec *Decoder, va addressableValue) error {
			format := time.RFC3339
			isRFC3339 := true
			if uo.format != "" && uo.formatDepth == dec.tokens.depth() {
				var err error
				format, isRFC3339, err = checkTimeFormat(uo.format)
				if err != nil {
					return &SemanticError{action: "unmarshal", GoType: t, Err: err}
				}
			}

			var flags valueFlags
			tt := va.Addr().Interface().(*time.Time)
			val, err := dec.readValue(&flags)
			if err != nil {
				return err
			}
			k := val.Kind()
			switch k {
			case 'n':
				*tt = time.Time{}
				return nil
			case '"':
				val = unescapeStringMayCopy(val, flags.isVerbatim())
				tt2, err := time.Parse(format, string(val))
				if isRFC3339 && err == nil {
					// TODO(https://go.dev/issue/54580): RFC 3339 specifies
					// the exact grammar of a valid timestamp. However,
					// the parsing functionality in "time" is too loose and
					// incorrectly accepts invalid timestamps as valid.
					// Remove these manual checks when "time" checks it for us.
					newParseError := func(layout, value, layoutElem, valueElem, message string) error {
						return &time.ParseError{Layout: layout, Value: value, LayoutElem: layoutElem, ValueElem: valueElem, Message: message}
					}
					switch {
					case val[len("2006-01-02T")+1] == ':': // hour must be two digits
						err = newParseError(format, string(val), "15", string(val[len("2006-01-02T"):][:1]), "")
					case val[len("2006-01-02T15:04:05")] == ',': // sub-second separator must be a period
						err = newParseError(format, string(val), ".", ",", "")
					case val[len(val)-1] != 'Z':
						switch {
						case parseDec2(val[len(val)-len("07:00"):]) >= 24: // timezone hour must be in range
							err = newParseError(format, string(val), "Z07:00", string(val[len(val)-len("Z07:00"):]), ": timezone hour out of range")
						case parseDec2(val[len(val)-len("00"):]) >= 60: // timezone minute must be in range
							err = newParseError(format, string(val), "Z07:00", string(val[len(val)-len("Z07:00"):]), ": timezone minute out of range")
						}
					}
				}
				if err != nil {
					return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t, Err: err}
				}
				*tt = tt2
				return nil
			default:
				return &SemanticError{action: "unmarshal", JSONKind: k, GoType: t}
			}
		}
	}
	return fncs
}

func checkTimeFormat(format string) (string, bool, error) {
	// We assume that an exported constant in the time package will
	// always start with an uppercase ASCII letter.
	if len(format) > 0 && 'A' <= format[0] && format[0] <= 'Z' {
		switch format {
		case "ANSIC":
			return time.ANSIC, false, nil
		case "UnixDate":
			return time.UnixDate, false, nil
		case "RubyDate":
			return time.RubyDate, false, nil
		case "RFC822":
			return time.RFC822, false, nil
		case "RFC822Z":
			return time.RFC822Z, false, nil
		case "RFC850":
			return time.RFC850, false, nil
		case "RFC1123":
			return time.RFC1123, false, nil
		case "RFC1123Z":
			return time.RFC1123Z, false, nil
		case "RFC3339":
			return time.RFC3339, true, nil
		case "RFC3339Nano":
			return time.RFC3339Nano, true, nil
		case "Kitchen":
			return time.Kitchen, false, nil
		case "Stamp":
			return time.Stamp, false, nil
		case "StampMilli":
			return time.StampMilli, false, nil
		case "StampMicro":
			return time.StampMicro, false, nil
		case "StampNano":
			return time.StampNano, false, nil
		default:
			// Reject any format that is an exported Go identifier in case
			// new format constants are added to the time package.
			if strings.TrimFunc(format, isLetterOrDigit) == "" {
				return "", false, fmt.Errorf("undefined format layout: %v", format)
			}
		}
	}
	return format, false, nil
}

// parseDec2 parses b as an unsigned, base-10, 2-digit number.
// It panics if len(b) < 2. The result is undefined if digits are not base-10.
func parseDec2(b []byte) byte {
	return 10*(b[0]-'0') + (b[1] - '0')
}
