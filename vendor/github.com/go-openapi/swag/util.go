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
)

// commonInitialisms are common acronyms that are kept as whole uppercased words.
var commonInitialisms *indexOfInitialisms

// initialisms is a slice of sorted initialisms
var initialisms []string

var isInitialism func(string) bool

// GoNamePrefixFunc sets an optional rule to prefix go names
// which do not start with a letter.
//
// e.g. to help convert "123" into "{prefix}123"
//
// The default is to prefix with "X"
var GoNamePrefixFunc func(string) string

func init() {
	// Taken from https://github.com/golang/lint/blob/3390df4df2787994aea98de825b964ac7944b817/lint.go#L732-L769
	var configuredInitialisms = map[string]bool{
		"ACL":   true,
		"API":   true,
		"ASCII": true,
		"CPU":   true,
		"CSS":   true,
		"DNS":   true,
		"EOF":   true,
		"GUID":  true,
		"HTML":  true,
		"HTTPS": true,
		"HTTP":  true,
		"ID":    true,
		"IP":    true,
		"IPv4":  true,
		"IPv6":  true,
		"JSON":  true,
		"LHS":   true,
		"OAI":   true,
		"QPS":   true,
		"RAM":   true,
		"RHS":   true,
		"RPC":   true,
		"SLA":   true,
		"SMTP":  true,
		"SQL":   true,
		"SSH":   true,
		"TCP":   true,
		"TLS":   true,
		"TTL":   true,
		"UDP":   true,
		"UI":    true,
		"UID":   true,
		"UUID":  true,
		"URI":   true,
		"URL":   true,
		"UTF8":  true,
		"VM":    true,
		"XML":   true,
		"XMPP":  true,
		"XSRF":  true,
		"XSS":   true,
	}

	// a thread-safe index of initialisms
	commonInitialisms = newIndexOfInitialisms().load(configuredInitialisms)
	initialisms = commonInitialisms.sorted()

	// a test function
	isInitialism = commonInitialisms.isInitialism
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

type byInitialism []string

func (s byInitialism) Len() int {
	return len(s)
}
func (s byInitialism) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s byInitialism) Less(i, j int) bool {
	if len(s[i]) != len(s[j]) {
		return len(s[i]) < len(s[j])
	}

	return strings.Compare(s[i], s[j]) > 0
}

// Removes leading whitespaces
func trim(str string) string {
	return strings.Trim(str, " ")
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
func Camelize(word string) (camelized string) {
	for pos, ru := range []rune(word) {
		if pos > 0 {
			camelized += string(unicode.ToLower(ru))
		} else {
			camelized += string(unicode.ToUpper(ru))
		}
	}
	return
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
	in := newSplitter(withPostSplitInitialismCheck).split(name)
	out := make([]string, 0, len(in))

	for _, w := range in {
		if !w.IsInitialism() {
			out = append(out, lower(w.GetOriginal()))
		} else {
			out = append(out, w.GetOriginal())
		}
	}

	return strings.Join(out, " ")
}

// ToHumanNameTitle represents a code name as a human series of words with the first letters titleized
func ToHumanNameTitle(name string) string {
	in := newSplitter(withPostSplitInitialismCheck).split(name)

	out := make([]string, 0, len(in))
	for _, w := range in {
		original := w.GetOriginal()
		if !w.IsInitialism() {
			out = append(out, Camelize(original))
		} else {
			out = append(out, original)
		}
	}
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
		out = append(out, Camelize(w))
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
	lexems := newSplitter(withPostSplitInitialismCheck).split(name)

	result := ""
	for _, lexem := range lexems {
		goName := lexem.GetUnsafeGoName()

		// to support old behavior
		if lexem.IsInitialism() {
			goName = upper(goName)
		}
		result += goName
	}

	if len(result) > 0 {
		// Only prefix with X when the first character isn't an ascii letter
		first := []rune(result)[0]
		if !unicode.IsLetter(first) || (first > unicode.MaxASCII && !unicode.IsUpper(first)) {
			if GoNamePrefixFunc == nil {
				return "X" + result
			}
			result = GoNamePrefixFunc(name) + result
		}
		first = []rune(result)[0]
		if unicode.IsLetter(first) && !unicode.IsUpper(first) {
			result = string(append([]rune{unicode.ToUpper(first)}, []rune(result)[1:]...))
		}
	}

	return result
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
	// check for things that have an IsZero method instead
	if vv, ok := data.(zeroable); ok {
		return vv.IsZero()
	}
	// continue with slightly more complex reflection
	v := reflect.ValueOf(data)
	switch v.Kind() {
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
	case reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		return v.IsNil()
	case reflect.Struct, reflect.Array:
		return reflect.DeepEqual(data, reflect.Zero(v.Type()).Interface())
	case reflect.Invalid:
		return true
	}
	return false
}

// AddInitialisms add additional initialisms
func AddInitialisms(words ...string) {
	for _, word := range words {
		// commonInitialisms[upper(word)] = true
		commonInitialisms.add(upper(word))
	}
	// sort again
	initialisms = commonInitialisms.sorted()
}

// CommandLineOptionsGroup represents a group of user-defined command line options
type CommandLineOptionsGroup struct {
	ShortDescription string
	LongDescription  string
	Options          interface{}
}
