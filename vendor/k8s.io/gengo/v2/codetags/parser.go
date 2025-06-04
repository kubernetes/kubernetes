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

package codetags

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Parse parses a comment tag into a TypedTag, or returns an error if the tag
// string fails to parse.
//
// A tag consists of a name, optional arguments, and an optional value. For example,
//
//	"name"
//	"name=50"
//	"name("featureX")=50"
//	"name(limit: 10, path: "/xyz")=text value"
//
// Arguments are optional and may be either:
//   - A single positional argument.
//   - One or more named arguments (in the format `name: value`).
//   - (Positional and named arguments cannot be mixed.)
//
// For example,
//
//	"name()"
//	"name(arg)"
//	"name(namedArg1: argValue1)"
//	"name(namedArg1: argValue1, namedArg2: argValue2)"
//
// Argument values may be strings, ints, booleans, or identifiers.
//
// For example,
//
//	"name("double-quoted")"
//	"name(`backtick-quoted`)"
//	"name(100)"
//	"name(true)"
//	"name(arg1: identifier)"
//	"name(arg1:`string value`)"
//	"name(arg1: 100)"
//	"name(arg1: true)"
//
// Note: When processing Go source code comments, the Extract function is
// typically used first to find and isolate tag strings matching a specific
// prefix. Those extracted strings can then be parsed using this function.
//
// The value part of the tag is optional and follows an equals sign "=".
// If no equals sign and value are present, the `Value` field of the
// resulting TypedTag will be an empty string.
//
// For example,
//
//	"name" // no value
//	"name=value"
//	"name=values include all the content after the = sign including any special characters (@*#^&...)"
//
// Comments are treated as part of the value, if a value is present.
//
// For example,
//
//	"key // This tag has no value, so this comment is ignored"
//	"key=value // Comments after values are treated as part of the value"
//
// Formal Grammar:
//
// <tag>             ::= <tagName> [ "(" [ <args> ] ")" ] [ "=" <tagValue> ]
// <args>            ::= <argValue> | <namedArgs>
// <namedArgs>       ::= <argNameAndValue> [ "," <namedArgs> ]*
// <argNameAndValue> ::= <identifier> ":" <argValue>
// <argValue>        ::= <identifier> | <string> | <int> | <bool>
//
// <tagName>       ::= [a-zA-Z_][a-zA-Z0-9_-.:]*
// <identifier>    ::= [a-zA-Z_][a-zA-Z0-9_-.]*
// <string>        ::= /* Go-style double-quoted or backtick-quoted strings,
// ...                    with standard Go escape sequences for double-quoted strings. */
// <int>           ::= /* Standard Go integer literals (decimal, 0x hex, 0o octal, 0b binary),
// ...                    with an optional +/- prefix. */
// <bool>          ::= "true" | "false"
// <tagValue>      ::= /* All text following the "=" sign to the end of the string. */
func Parse(tagText string) (TypedTag, error) {
	tagText = strings.TrimSpace(tagText)
	return parseTagKey(tagText)
}

// ParseAll calls Parse on each tag in the input slice.
func ParseAll(tags []string) ([]TypedTag, error) {
	var out []TypedTag
	for _, tag := range tags {
		parsed, err := Parse(tag)
		if err != nil {
			return nil, err
		}
		out = append(out, parsed)
	}
	return out, nil
}

const (
	stBegin           = "stBegin"
	stTag             = "stTag"
	stArg             = "stArg"
	stNumber          = "stNumber"
	stPrefixedNumber  = "stPrefixedNumber"
	stQuotedString    = "stQuotedString"
	stNakedString     = "stNakedString"
	stEscape          = "stEscape"
	stEndOfToken      = "stEndOfToken"
	stMaybeValue      = "stMaybeValue"
	stValue           = "stValue"
	stMaybeComment    = "stMaybeComment"
	stTrailingSlash   = "stTrailingSlash"
	stTrailingComment = "stTrailingComment"
)

func parseTagKey(input string) (TypedTag, error) {
	tag := bytes.Buffer{}   // current tag name
	args := []Arg{}         // all tag arguments
	value := bytes.Buffer{} // current tag value

	cur := Arg{}          // current argument accumulator
	buf := bytes.Buffer{} // string accumulator

	// These are defined outside the loop to make errors easier.
	var i int
	var r rune
	var incomplete bool
	var quote rune

	saveInt := func() error {
		s := buf.String()
		if _, err := strconv.ParseInt(s, 0, 64); err != nil {
			return fmt.Errorf("invalid number %q", s)
		} else {
			cur.Value = s
			cur.Type = ArgTypeInt
		}
		args = append(args, cur)
		cur = Arg{}
		buf.Reset()
		return nil
	}
	saveString := func() {
		s := buf.String()
		cur.Value = s
		cur.Type = ArgTypeString
		args = append(args, cur)
		cur = Arg{}
		buf.Reset()
	}
	saveBoolOrString := func() {
		s := buf.String()
		if s == "true" || s == "false" {
			cur.Value = s
			cur.Type = ArgTypeBool
		} else {
			cur.Value = s
			cur.Type = ArgTypeString
		}
		args = append(args, cur)
		cur = Arg{}
		buf.Reset()
	}
	saveName := func() {
		cur.Name = buf.String()
		buf.Reset()
	}

	runes := []rune(input)
	st := stBegin
parseLoop:
	for i, r = range runes {
		switch st {
		case stBegin:
			switch {
			case unicode.IsSpace(r):
				continue
			case isIdentBegin(r):
				tag.WriteRune(r)
				st = stTag
			default:
				break parseLoop
			}
		case stTag:
			switch {
			case isIdentInterior(r) || r == ':':
				tag.WriteRune(r)
			case r == '(':
				incomplete = true
				st = stArg
			case r == '=':
				st = stValue
			case unicode.IsSpace(r):
				st = stMaybeComment
			default:
				break parseLoop
			}
		case stArg:
			switch {
			case unicode.IsSpace(r):
				continue
			case r == ')':
				incomplete = false
				st = stMaybeValue
			case r == '0':
				buf.WriteRune(r)
				st = stPrefixedNumber
			case r == '-' || r == '+' || unicode.IsDigit(r):
				buf.WriteRune(r)
				st = stNumber
			case r == '"' || r == '`':
				quote = r
				st = stQuotedString
			case isIdentBegin(r):
				buf.WriteRune(r)
				st = stNakedString
			default:
				break parseLoop
			}
		case stNumber:
			hexits := "abcdefABCDEF"
			switch {
			case unicode.IsDigit(r) || strings.Contains(hexits, string(r)):
				buf.WriteRune(r)
				continue
			case r == ',':
				if err := saveInt(); err != nil {
					return TypedTag{}, err
				}
				st = stArg
			case r == ')':
				if err := saveInt(); err != nil {
					return TypedTag{}, err
				}
				incomplete = false
				st = stMaybeValue
			case unicode.IsSpace(r):
				if err := saveInt(); err != nil {
					return TypedTag{}, err
				}
				st = stEndOfToken
			default:
				break parseLoop
			}
		case stPrefixedNumber:
			switch {
			case unicode.IsDigit(r):
				buf.WriteRune(r)
				st = stNumber
			case r == 'x' || r == 'o' || r == 'b':
				buf.WriteRune(r)
				st = stNumber
			default:
				break parseLoop
			}
		case stQuotedString:
			switch {
			case r == '\\':
				st = stEscape
			case r == quote:
				saveString()
				st = stEndOfToken
			default:
				buf.WriteRune(r)
			}
		case stEscape:
			switch {
			case r == quote || r == '\\':
				buf.WriteRune(r)
				st = stQuotedString
			default:
				return TypedTag{}, fmt.Errorf("unhandled escaped character %q", r)
			}
		case stNakedString:
			switch {
			case isIdentInterior(r):
				buf.WriteRune(r)
			case r == ',':
				saveBoolOrString()
				st = stArg
			case r == ')':
				saveBoolOrString()
				incomplete = false
				st = stMaybeValue
			case unicode.IsSpace(r):
				saveBoolOrString()
				st = stEndOfToken
			case r == ':':
				saveName()
				st = stArg
			default:
				break parseLoop
			}
		case stEndOfToken:
			switch {
			case unicode.IsSpace(r):
				continue
			case r == ',':
				st = stArg
			case r == ')':
				incomplete = false
				st = stMaybeValue
			default:
				break parseLoop
			}
		case stMaybeValue:
			switch {
			case r == '=':
				st = stValue
			case unicode.IsSpace(r):
				st = stMaybeComment
			default:
				break parseLoop
			}
		case stValue: // This is a terminal state, it consumes the rest of the input as an opaque value.
			value.WriteRune(r)
		case stMaybeComment:
			switch {
			case unicode.IsSpace(r):
				continue
			case r == '/':
				incomplete = true
				st = stTrailingSlash
			default:
				break parseLoop
			}
		case stTrailingSlash:
			switch {
			case r == '/':
				incomplete = false
				st = stTrailingComment
			default:
				break parseLoop
			}
		case stTrailingComment:
			i = len(runes) - 1
			break parseLoop
		default:
			return TypedTag{}, fmt.Errorf("unknown state reached in parser: %s at position %d", st, i)
		}
	}
	if i != len(runes)-1 {
		return TypedTag{}, fmt.Errorf("unexpected character %q at position %d", r, i)
	}
	if incomplete {
		return TypedTag{}, fmt.Errorf("unexpected end of input")
	}
	usingNamedArgs := false
	for i, arg := range args {
		if (usingNamedArgs && arg.Name == "") || (!usingNamedArgs && arg.Name != "" && i > 0) {
			return TypedTag{}, fmt.Errorf("can't mix named and positional arguments")
		}
		if arg.Name != "" {
			usingNamedArgs = true
		}
	}
	if !usingNamedArgs && len(args) > 1 {
		return TypedTag{}, fmt.Errorf("multiple arguments must use 'name: value' syntax")
	}
	return TypedTag{Name: tag.String(), Args: args, Value: value.String()}, nil
}

func isIdentBegin(r rune) bool {
	return unicode.IsLetter(r) || r == '_'
}

func isIdentInterior(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '.' || r == '-'
}
