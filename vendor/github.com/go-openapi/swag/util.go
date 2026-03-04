// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"reflect"
	"strings"
	"unicode"
	"unicode/utf8"
)

// GoNamePrefixFunc sets an optional rule to prefix go names
// which do not start with a letter.
//
// The prefix function is assumed to return a string that starts with an upper case letter.
//
// e.g. to help convert "123" into "{prefix}123"
//
// The default is to prefix with "X"
var GoNamePrefixFunc func(string) string

func prefixFunc(name, in string) string {
	if GoNamePrefixFunc == nil {
		return "X" + in
	}

	return GoNamePrefixFunc(name) + in
}

const (
	// collectionFormatComma = "csv"
	collectionFormatSpace = "ssv"
	collectionFormatTab   = "tsv"
	collectionFormatPipe  = "pipes"
	collectionFormatMulti = "multi"
)

// JoinByFormat joins a string array by a known format (e.g. swagger's collectionFormat attribute):
//
//	ssv: space separated value
//	tsv: tab separated value
//	pipes: pipe (|) separated value
//	csv: comma separated value (default)
func JoinByFormat(data []string, format string) []string {
	if len(data) == 0 {
		return data
	}
	var sep string
	switch format {
	case collectionFormatSpace:
		sep = " "
	case collectionFormatTab:
		sep = "\t"
	case collectionFormatPipe:
		sep = "|"
	case collectionFormatMulti:
		return data
	default:
		sep = ","
	}
	return []string{strings.Join(data, sep)}
}

// SplitByFormat splits a string by a known format:
//
//	ssv: space separated value
//	tsv: tab separated value
//	pipes: pipe (|) separated value
//	csv: comma separated value (default)
func SplitByFormat(data, format string) []string {
	if data == "" {
		return nil
	}
	var sep string
	switch format {
	case collectionFormatSpace:
		sep = " "
	case collectionFormatTab:
		sep = "\t"
	case collectionFormatPipe:
		sep = "|"
	case collectionFormatMulti:
		return nil
	default:
		sep = ","
	}
	var result []string
	for _, s := range strings.Split(data, sep) {
		if ts := strings.TrimSpace(s); ts != "" {
			result = append(result, ts)
		}
	}
	return result
}

// Removes leading whitespaces
func trim(str string) string {
	return strings.TrimSpace(str)
}

// Shortcut to strings.ToUpper()
func upper(str string) string {
	return strings.ToUpper(trim(str))
}

// Shortcut to strings.ToLower()
func lower(str string) string {
	return strings.ToLower(trim(str))
}

// Camelize an uppercased word
func Camelize(word string) string {
	camelized := poolOfBuffers.BorrowBuffer(len(word))
	defer func() {
		poolOfBuffers.RedeemBuffer(camelized)
	}()

	for pos, ru := range []rune(word) {
		if pos > 0 {
			camelized.WriteRune(unicode.ToLower(ru))
		} else {
			camelized.WriteRune(unicode.ToUpper(ru))
		}
	}
	return camelized.String()
}

// ToFileName lowercases and underscores a go type name
func ToFileName(name string) string {
	in := split(name)
	out := make([]string, 0, len(in))

	for _, w := range in {
		out = append(out, lower(w))
	}

	return strings.Join(out, "_")
}

// ToCommandName lowercases and underscores a go type name
func ToCommandName(name string) string {
	in := split(name)
	out := make([]string, 0, len(in))

	for _, w := range in {
		out = append(out, lower(w))
	}
	return strings.Join(out, "-")
}

// ToHumanNameLower represents a code name as a human series of words
func ToHumanNameLower(name string) string {
	s := poolOfSplitters.BorrowSplitter(withPostSplitInitialismCheck)
	in := s.split(name)
	poolOfSplitters.RedeemSplitter(s)
	out := make([]string, 0, len(*in))

	for _, w := range *in {
		if !w.IsInitialism() {
			out = append(out, lower(w.GetOriginal()))
		} else {
			out = append(out, trim(w.GetOriginal()))
		}
	}
	poolOfLexems.RedeemLexems(in)

	return strings.Join(out, " ")
}

// ToHumanNameTitle represents a code name as a human series of words with the first letters titleized
func ToHumanNameTitle(name string) string {
	s := poolOfSplitters.BorrowSplitter(withPostSplitInitialismCheck)
	in := s.split(name)
	poolOfSplitters.RedeemSplitter(s)

	out := make([]string, 0, len(*in))
	for _, w := range *in {
		original := trim(w.GetOriginal())
		if !w.IsInitialism() {
			out = append(out, Camelize(original))
		} else {
			out = append(out, original)
		}
	}
	poolOfLexems.RedeemLexems(in)

	return strings.Join(out, " ")
}

// ToJSONName camelcases a name which can be underscored or pascal cased
func ToJSONName(name string) string {
	in := split(name)
	out := make([]string, 0, len(in))

	for i, w := range in {
		if i == 0 {
			out = append(out, lower(w))
			continue
		}
		out = append(out, Camelize(trim(w)))
	}
	return strings.Join(out, "")
}

// ToVarName camelcases a name which can be underscored or pascal cased
func ToVarName(name string) string {
	res := ToGoName(name)
	if isInitialism(res) {
		return lower(res)
	}
	if len(res) <= 1 {
		return lower(res)
	}
	return lower(res[:1]) + res[1:]
}

// ToGoName translates a swagger name which can be underscored or camel cased to a name that golint likes
func ToGoName(name string) string {
	s := poolOfSplitters.BorrowSplitter(withPostSplitInitialismCheck)
	lexems := s.split(name)
	poolOfSplitters.RedeemSplitter(s)
	defer func() {
		poolOfLexems.RedeemLexems(lexems)
	}()
	lexemes := *lexems

	if len(lexemes) == 0 {
		return ""
	}

	result := poolOfBuffers.BorrowBuffer(len(name))
	defer func() {
		poolOfBuffers.RedeemBuffer(result)
	}()

	// check if not starting with a letter, upper case
	firstPart := lexemes[0].GetUnsafeGoName()
	if lexemes[0].IsInitialism() {
		firstPart = upper(firstPart)
	}

	if c := firstPart[0]; c < utf8.RuneSelf {
		// ASCII
		switch {
		case 'A' <= c && c <= 'Z':
			result.WriteString(firstPart)
		case 'a' <= c && c <= 'z':
			result.WriteByte(c - 'a' + 'A')
			result.WriteString(firstPart[1:])
		default:
			result.WriteString(prefixFunc(name, firstPart))
			// NOTE: no longer check if prefixFunc returns a string that starts with uppercase:
			// assume this is always the case
		}
	} else {
		// unicode
		firstRune, _ := utf8.DecodeRuneInString(firstPart)
		switch {
		case !unicode.IsLetter(firstRune):
			result.WriteString(prefixFunc(name, firstPart))
		case !unicode.IsUpper(firstRune):
			result.WriteString(prefixFunc(name, firstPart))
			/*
				result.WriteRune(unicode.ToUpper(firstRune))
				result.WriteString(firstPart[offset:])
			*/
		default:
			result.WriteString(firstPart)
		}
	}

	for _, lexem := range lexemes[1:] {
		goName := lexem.GetUnsafeGoName()

		// to support old behavior
		if lexem.IsInitialism() {
			goName = upper(goName)
		}
		result.WriteString(goName)
	}

	return result.String()
}

// ContainsStrings searches a slice of strings for a case-sensitive match
func ContainsStrings(coll []string, item string) bool {
	for _, a := range coll {
		if a == item {
			return true
		}
	}
	return false
}

// ContainsStringsCI searches a slice of strings for a case-insensitive match
func ContainsStringsCI(coll []string, item string) bool {
	for _, a := range coll {
		if strings.EqualFold(a, item) {
			return true
		}
	}
	return false
}

type zeroable interface {
	IsZero() bool
}

// IsZero returns true when the value passed into the function is a zero value.
// This allows for safer checking of interface values.
func IsZero(data interface{}) bool {
	v := reflect.ValueOf(data)
	// check for nil data
	switch v.Kind() { //nolint:exhaustive
	case reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		if v.IsNil() {
			return true
		}
	}

	// check for things that have an IsZero method instead
	if vv, ok := data.(zeroable); ok {
		return vv.IsZero()
	}

	// continue with slightly more complex reflection
	switch v.Kind() { //nolint:exhaustive
	case reflect.String:
		return v.Len() == 0
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return v.Float() == 0
	case reflect.Struct, reflect.Array:
		return reflect.DeepEqual(data, reflect.Zero(v.Type()).Interface())
	case reflect.Invalid:
		return true
	default:
		return false
	}
}

// CommandLineOptionsGroup represents a group of user-defined command line options
type CommandLineOptionsGroup struct {
	ShortDescription string
	LongDescription  string
	Options          interface{}
}
