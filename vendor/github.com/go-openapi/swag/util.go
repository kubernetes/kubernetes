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
	"math"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Taken from https://github.com/golang/lint/blob/3390df4df2787994aea98de825b964ac7944b817/lint.go#L732-L769
var commonInitialisms = map[string]bool{
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
var initialisms []string

var once sync.Once

func sortInitialisms() {
	for k := range commonInitialisms {
		initialisms = append(initialisms, k)
	}
	sort.Sort(sort.Reverse(byLength(initialisms)))
}

// JoinByFormat joins a string array by a known format:
//		ssv: space separated value
//		tsv: tab separated value
//		pipes: pipe (|) separated value
//		csv: comma separated value (default)
func JoinByFormat(data []string, format string) []string {
	if len(data) == 0 {
		return data
	}
	var sep string
	switch format {
	case "ssv":
		sep = " "
	case "tsv":
		sep = "\t"
	case "pipes":
		sep = "|"
	case "multi":
		return data
	default:
		sep = ","
	}
	return []string{strings.Join(data, sep)}
}

// SplitByFormat splits a string by a known format:
//		ssv: space separated value
//		tsv: tab separated value
//		pipes: pipe (|) separated value
//		csv: comma separated value (default)
func SplitByFormat(data, format string) []string {
	if data == "" {
		return nil
	}
	var sep string
	switch format {
	case "ssv":
		sep = " "
	case "tsv":
		sep = "\t"
	case "pipes":
		sep = "|"
	case "multi":
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

type byLength []string

func (s byLength) Len() int {
	return len(s)
}
func (s byLength) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
func (s byLength) Less(i, j int) bool {
	return len(s[i]) < len(s[j])
}

// Prepares strings by splitting by caps, spaces, dashes, and underscore
func split(str string) (words []string) {
	repl := strings.NewReplacer(
		"@", "At ",
		"&", "And ",
		"|", "Pipe ",
		"$", "Dollar ",
		"!", "Bang ",
		"-", " ",
		"_", " ",
	)

	rex1 := regexp.MustCompile(`(\p{Lu})`)
	rex2 := regexp.MustCompile(`(\pL|\pM|\pN|\p{Pc})+`)

	str = trim(str)

	// Convert dash and underscore to spaces
	str = repl.Replace(str)

	// Split when uppercase is found (needed for Snake)
	str = rex1.ReplaceAllString(str, " $1")

	// check if consecutive single char things make up an initialism
	once.Do(sortInitialisms)
	for _, k := range initialisms {
		str = strings.Replace(str, rex1.ReplaceAllString(k, " $1"), " "+k, -1)
	}
	// Get the final list of words
	words = rex2.FindAllString(str, -1)

	return
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
	for pos, ru := range word {
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
	var out []string

	for _, w := range split(name) {
		out = append(out, lower(w))
	}

	return strings.Join(out, "_")
}

// ToCommandName lowercases and underscores a go type name
func ToCommandName(name string) string {
	var out []string
	for _, w := range split(name) {
		out = append(out, lower(w))
	}
	return strings.Join(out, "-")
}

// ToHumanNameLower represents a code name as a human series of words
func ToHumanNameLower(name string) string {
	var out []string
	for _, w := range split(name) {
		if !commonInitialisms[upper(w)] {
			out = append(out, lower(w))
		} else {
			out = append(out, w)
		}
	}
	return strings.Join(out, " ")
}

// ToHumanNameTitle represents a code name as a human series of words with the first letters titleized
func ToHumanNameTitle(name string) string {
	var out []string
	for _, w := range split(name) {
		uw := upper(w)
		if !commonInitialisms[uw] {
			out = append(out, upper(w[:1])+lower(w[1:]))
		} else {
			out = append(out, w)
		}
	}
	return strings.Join(out, " ")
}

// ToJSONName camelcases a name which can be underscored or pascal cased
func ToJSONName(name string) string {
	var out []string
	for i, w := range split(name) {
		if i == 0 {
			out = append(out, lower(w))
			continue
		}
		out = append(out, upper(w[:1])+lower(w[1:]))
	}
	return strings.Join(out, "")
}

// ToVarName camelcases a name which can be underscored or pascal cased
func ToVarName(name string) string {
	res := ToGoName(name)
	if _, ok := commonInitialisms[res]; ok {
		return lower(res)
	}
	if len(res) <= 1 {
		return lower(res)
	}
	return lower(res[:1]) + res[1:]
}

// ToGoName translates a swagger name which can be underscored or camel cased to a name that golint likes
func ToGoName(name string) string {
	var out []string
	for _, w := range split(name) {
		uw := upper(w)
		mod := int(math.Min(float64(len(uw)), 2))
		if !commonInitialisms[uw] && !commonInitialisms[uw[:len(uw)-mod]] {
			uw = upper(w[:1]) + lower(w[1:])
		}
		out = append(out, uw)
	}

	result := strings.Join(out, "")
	if len(result) > 0 {
		ud := upper(result[:1])
		ru := []rune(ud)
		if unicode.IsUpper(ru[0]) {
			result = ud + result[1:]
		} else {
			result = "X" + ud + result[1:]
		}
	}
	return result
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
		commonInitialisms[upper(word)] = true
	}
}

// CommandLineOptionsGroup represents a group of user-defined command line options
type CommandLineOptionsGroup struct {
	ShortDescription string
	LongDescription  string
	Options          interface{}
}
