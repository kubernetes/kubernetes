// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package ext contains CEL extension libraries where each library defines a related set of
// constants, functions, macros, or other configuration settings which may not be covered by
// the core CEL spec.
package ext

import (
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/language"
	"golang.org/x/text/message"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/google/cel-go/interpreter"
)

const (
	defaultLocale    = "en-US"
	defaultPrecision = 6
)

// Strings returns a cel.EnvOption to configure extended functions for string manipulation.
// As a general note, all indices are zero-based.
//
// # CharAt
//
// Returns the character at the given position. If the position is negative, or greater than
// the length of the string, the function will produce an error:
//
//	<string>.charAt(<int>) -> <string>
//
// Examples:
//
//	'hello'.charAt(4)  // return 'o'
//	'hello'.charAt(5)  // return ''
//	'hello'.charAt(-1) // error
//
// # Format
//
// Introduced at version: 1
//
// Returns a new string with substitutions being performed, printf-style.
// The valid formatting clauses are:
//
// `%s` - substitutes a string. This can also be used on bools, lists, maps, bytes,
// Duration and Timestamp, in addition to all numerical types (int, uint, and double).
// Note that the dot/period decimal separator will always be used when printing a list
// or map that contains a double, and that null can be passed (which results in the
// string "null") in addition to types.
// `%d` - substitutes an integer.
// `%f` - substitutes a double with fixed-point precision. The default precision is 6, but
// this can be adjusted. The strings `Infinity`, `-Infinity`, and `NaN` are also valid input
// for this clause.
// `%e` - substitutes a double in scientific notation. The default precision is 6, but this
// can be adjusted.
// `%b` - substitutes an integer with its equivalent binary string. Can also be used on bools.
// `%x` - substitutes an integer with its equivalent in hexadecimal, or if given a string or
// bytes, will output each character's equivalent in hexadecimal.
// `%X` - same as above, but with A-F capitalized.
// `%o` - substitutes an integer with its equivalent in octal.
//
//	<string>.format(<list>) -> <string>
//
// Examples:
//
//	"this is a string: %s\nand an integer: %d".format(["str", 42]) // returns "this is a string: str\nand an integer: 42"
//	"a double substituted with %%s: %s".format([64.2]) // returns "a double substituted with %s: 64.2"
//	"string type: %s".format([type(string)]) // returns "string type: string"
//	"timestamp: %s".format([timestamp("2023-02-03T23:31:20+00:00")]) // returns "timestamp: 2023-02-03T23:31:20Z"
//	"duration: %s".format([duration("1h45m47s")]) // returns "duration: 6347s"
//	"%f".format([3.14]) // returns "3.140000"
//	"scientific notation: %e".format([2.71828]) // returns "scientific notation: 2.718280\u202f\u00d7\u202f10\u2070\u2070"
//	"5 in binary: %b".format([5]), // returns "5 in binary; 101"
//	"26 in hex: %x".format([26]), // returns "26 in hex: 1a"
//	"26 in hex (uppercase): %X".format([26]) // returns "26 in hex (uppercase): 1A"
//	"30 in octal: %o".format([30]) // returns "30 in octal: 36"
//	"a map inside a list: %s".format([[1, 2, 3, {"a": "x", "b": "y", "c": "z"}]]) // returns "a map inside a list: [1, 2, 3, {"a":"x", "b":"y", "c":"d"}]"
//	"true bool: %s - false bool: %s\nbinary bool: %b".format([true, false, true]) // returns "true bool: true - false bool: false\nbinary bool: 1"
//
// Passing an incorrect type (an integer to `%s`) is considered an error, as well as attempting
// to use more formatting clauses than there are arguments (`%d %d %d` while passing two ints, for instance).
// If compile-time checking is enabled, and the formatting string is a constant, and the argument list is a literal,
// then letting any arguments go unused/unformatted is also considered an error.
//
// # IndexOf
//
// Returns the integer index of the first occurrence of the search string. If the search string is
// not found the function returns -1.
//
// The function also accepts an optional position from which to begin the substring search. If the
// substring is the empty string, the index where the search starts is returned (zero or custom).
//
//	<string>.indexOf(<string>) -> <int>
//	<string>.indexOf(<string>, <int>) -> <int>
//
// Examples:
//
//	'hello mellow'.indexOf('')         // returns 0
//	'hello mellow'.indexOf('ello')     // returns 1
//	'hello mellow'.indexOf('jello')    // returns -1
//	'hello mellow'.indexOf('', 2)      // returns 2
//	'hello mellow'.indexOf('ello', 2)  // returns 7
//	'hello mellow'.indexOf('ello', 20) // error
//
// # Join
//
// Returns a new string where the elements of string list are concatenated.
//
// The function also accepts an optional separator which is placed between elements in the resulting string.
//
// <list<string>>.join() -> <string>
// <list<string>>.join(<string>) -> <string>
//
// Examples:
//
//	['hello', 'mellow'].join() // returns 'hellomellow'
//	['hello', 'mellow'].join(' ') // returns 'hello mellow'
//	[].join() // returns ''
//	[].join('/') // returns ''
//
// # LastIndexOf
//
// Returns the integer index at the start of the last occurrence of the search string. If the
// search string is not found the function returns -1.
//
// The function also accepts an optional position which represents the last index to be
// considered as the beginning of the substring match. If the substring is the empty string,
// the index where the search starts is returned (string length or custom).
//
//	<string>.lastIndexOf(<string>) -> <int>
//	<string>.lastIndexOf(<string>, <int>) -> <int>
//
// Examples:
//
//	'hello mellow'.lastIndexOf('')         // returns 12
//	'hello mellow'.lastIndexOf('ello')     // returns 7
//	'hello mellow'.lastIndexOf('jello')    // returns -1
//	'hello mellow'.lastIndexOf('ello', 6)  // returns 1
//	'hello mellow'.lastIndexOf('ello', -1) // error
//
// # LowerAscii
//
// Returns a new string where all ASCII characters are lower-cased.
//
// This function does not perform Unicode case-mapping for characters outside the ASCII range.
//
//	<string>.lowerAscii() -> <string>
//
// Examples:
//
//	'TacoCat'.lowerAscii()      // returns 'tacocat'
//	'TacoCÆt Xii'.lowerAscii()  // returns 'tacocÆt xii'
//
// # Quote
//
// Introduced in version: 1
//
// Takes the given string and makes it safe to print (without any formatting due to escape sequences).
// If any invalid UTF-8 characters are encountered, they are replaced with \uFFFD.
//
// strings.quote(<string>)
//
// Examples:
//
// strings.quote('single-quote with "double quote"') // returns '"single-quote with \"double quote\""'
// strings.quote("two escape sequences \a\n") // returns '"two escape sequences \\a\\n"'
//
// # Replace
//
// Returns a new string based on the target, which replaces the occurrences of a search string
// with a replacement string if present. The function accepts an optional limit on the number of
// substring replacements to be made.
//
// When the replacement limit is 0, the result is the original string. When the limit is a negative
// number, the function behaves the same as replace all.
//
//	<string>.replace(<string>, <string>) -> <string>
//	<string>.replace(<string>, <string>, <int>) -> <string>
//
// Examples:
//
//	'hello hello'.replace('he', 'we')     // returns 'wello wello'
//	'hello hello'.replace('he', 'we', -1) // returns 'wello wello'
//	'hello hello'.replace('he', 'we', 1)  // returns 'wello hello'
//	'hello hello'.replace('he', 'we', 0)  // returns 'hello hello'
//
// # Split
//
// Returns a list of strings split from the input by the given separator. The function accepts
// an optional argument specifying a limit on the number of substrings produced by the split.
//
// When the split limit is 0, the result is an empty list. When the limit is 1, the result is the
// target string to split. When the limit is a negative number, the function behaves the same as
// split all.
//
//	<string>.split(<string>) -> <list<string>>
//	<string>.split(<string>, <int>) -> <list<string>>
//
// Examples:
//
//	'hello hello hello'.split(' ')     // returns ['hello', 'hello', 'hello']
//	'hello hello hello'.split(' ', 0)  // returns []
//	'hello hello hello'.split(' ', 1)  // returns ['hello hello hello']
//	'hello hello hello'.split(' ', 2)  // returns ['hello', 'hello hello']
//	'hello hello hello'.split(' ', -1) // returns ['hello', 'hello', 'hello']
//
// # Substring
//
// Returns the substring given a numeric range corresponding to character positions. Optionally
// may omit the trailing range for a substring from a given character position until the end of
// a string.
//
// Character offsets are 0-based with an inclusive start range and exclusive end range. It is an
// error to specify an end range that is lower than the start range, or for either the start or end
// index to be negative or exceed the string length.
//
//	<string>.substring(<int>) -> <string>
//	<string>.substring(<int>, <int>) -> <string>
//
// Examples:
//
//	'tacocat'.substring(4)    // returns 'cat'
//	'tacocat'.substring(0, 4) // returns 'taco'
//	'tacocat'.substring(-1)   // error
//	'tacocat'.substring(2, 1) // error
//
// # Trim
//
// Returns a new string which removes the leading and trailing whitespace in the target string.
// The trim function uses the Unicode definition of whitespace which does not include the
// zero-width spaces. See: https://en.wikipedia.org/wiki/Whitespace_character#Unicode
//
//	<string>.trim() -> <string>
//
// Examples:
//
//	'  \ttrim\n    '.trim() // returns 'trim'
//
// # UpperAscii
//
// Returns a new string where all ASCII characters are upper-cased.
//
// This function does not perform Unicode case-mapping for characters outside the ASCII range.
//
//	<string>.upperAscii() -> <string>
//
// Examples:
//
//	'TacoCat'.upperAscii()      // returns 'TACOCAT'
//	'TacoCÆt Xii'.upperAscii()  // returns 'TACOCÆT XII'
func Strings(options ...StringsOption) cel.EnvOption {
	s := &stringLib{version: math.MaxUint32}
	for _, o := range options {
		s = o(s)
	}
	return cel.Lib(s)
}

type stringLib struct {
	locale  string
	version uint32
}

// LibraryName implements the SingletonLibrary interface method.
func (*stringLib) LibraryName() string {
	return "cel.lib.ext.strings"
}

// StringsOption is a functional interface for configuring the strings library.
type StringsOption func(*stringLib) *stringLib

// StringsLocale configures the library with the given locale. The locale tag will
// be checked for validity at the time that EnvOptions are configured. If this option
// is not passed, string.format will behave as if en_US was passed as the locale.
func StringsLocale(locale string) StringsOption {
	return func(sl *stringLib) *stringLib {
		sl.locale = locale
		return sl
	}
}

// StringsVersion configures the version of the string library. The version limits which
// functions are available. Only functions introduced below or equal to the given
// version included in the library. See the library documentation to determine
// which version a function was introduced at. If the documentation does not
// state which version a function was introduced at, it can be assumed to be
// introduced at version 0, when the library was first created.
// If this option is not set, all functions are available.
func StringsVersion(version uint32) func(lib *stringLib) *stringLib {
	return func(sl *stringLib) *stringLib {
		sl.version = version
		return sl
	}
}

// CompileOptions implements the Library interface method.
func (sl *stringLib) CompileOptions() []cel.EnvOption {
	formatLocale := "en_US"
	if sl.locale != "" {
		// ensure locale is properly-formed if set
		_, err := language.Parse(sl.locale)
		if err != nil {
			return []cel.EnvOption{
				func(e *cel.Env) (*cel.Env, error) {
					return nil, fmt.Errorf("failed to parse locale: %w", err)
				},
			}
		}
		formatLocale = sl.locale
	}

	opts := []cel.EnvOption{
		cel.Function("charAt",
			cel.MemberOverload("string_char_at_int", []*cel.Type{cel.StringType, cel.IntType}, cel.StringType,
				cel.BinaryBinding(func(str, ind ref.Val) ref.Val {
					s := str.(types.String)
					i := ind.(types.Int)
					return stringOrError(charAt(string(s), int64(i)))
				}))),
		cel.Function("indexOf",
			cel.MemberOverload("string_index_of_string", []*cel.Type{cel.StringType, cel.StringType}, cel.IntType,
				cel.BinaryBinding(func(str, substr ref.Val) ref.Val {
					s := str.(types.String)
					sub := substr.(types.String)
					return intOrError(indexOf(string(s), string(sub)))
				})),
			cel.MemberOverload("string_index_of_string_int", []*cel.Type{cel.StringType, cel.StringType, cel.IntType}, cel.IntType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					s := args[0].(types.String)
					sub := args[1].(types.String)
					offset := args[2].(types.Int)
					return intOrError(indexOfOffset(string(s), string(sub), int64(offset)))
				}))),
		cel.Function("lastIndexOf",
			cel.MemberOverload("string_last_index_of_string", []*cel.Type{cel.StringType, cel.StringType}, cel.IntType,
				cel.BinaryBinding(func(str, substr ref.Val) ref.Val {
					s := str.(types.String)
					sub := substr.(types.String)
					return intOrError(lastIndexOf(string(s), string(sub)))
				})),
			cel.MemberOverload("string_last_index_of_string_int", []*cel.Type{cel.StringType, cel.StringType, cel.IntType}, cel.IntType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					s := args[0].(types.String)
					sub := args[1].(types.String)
					offset := args[2].(types.Int)
					return intOrError(lastIndexOfOffset(string(s), string(sub), int64(offset)))
				}))),
		cel.Function("lowerAscii",
			cel.MemberOverload("string_lower_ascii", []*cel.Type{cel.StringType}, cel.StringType,
				cel.UnaryBinding(func(str ref.Val) ref.Val {
					s := str.(types.String)
					return stringOrError(lowerASCII(string(s)))
				}))),
		cel.Function("replace",
			cel.MemberOverload(
				"string_replace_string_string", []*cel.Type{cel.StringType, cel.StringType, cel.StringType}, cel.StringType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					str := args[0].(types.String)
					old := args[1].(types.String)
					new := args[2].(types.String)
					return stringOrError(replace(string(str), string(old), string(new)))
				})),
			cel.MemberOverload(
				"string_replace_string_string_int", []*cel.Type{cel.StringType, cel.StringType, cel.StringType, cel.IntType}, cel.StringType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					str := args[0].(types.String)
					old := args[1].(types.String)
					new := args[2].(types.String)
					n := args[3].(types.Int)
					return stringOrError(replaceN(string(str), string(old), string(new), int64(n)))
				}))),
		cel.Function("split",
			cel.MemberOverload("string_split_string", []*cel.Type{cel.StringType, cel.StringType}, cel.ListType(cel.StringType),
				cel.BinaryBinding(func(str, separator ref.Val) ref.Val {
					s := str.(types.String)
					sep := separator.(types.String)
					return listStringOrError(split(string(s), string(sep)))
				})),
			cel.MemberOverload("string_split_string_int", []*cel.Type{cel.StringType, cel.StringType, cel.IntType}, cel.ListType(cel.StringType),
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					s := args[0].(types.String)
					sep := args[1].(types.String)
					n := args[2].(types.Int)
					return listStringOrError(splitN(string(s), string(sep), int64(n)))
				}))),
		cel.Function("substring",
			cel.MemberOverload("string_substring_int", []*cel.Type{cel.StringType, cel.IntType}, cel.StringType,
				cel.BinaryBinding(func(str, offset ref.Val) ref.Val {
					s := str.(types.String)
					off := offset.(types.Int)
					return stringOrError(substr(string(s), int64(off)))
				})),
			cel.MemberOverload("string_substring_int_int", []*cel.Type{cel.StringType, cel.IntType, cel.IntType}, cel.StringType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					s := args[0].(types.String)
					start := args[1].(types.Int)
					end := args[2].(types.Int)
					return stringOrError(substrRange(string(s), int64(start), int64(end)))
				}))),
		cel.Function("trim",
			cel.MemberOverload("string_trim", []*cel.Type{cel.StringType}, cel.StringType,
				cel.UnaryBinding(func(str ref.Val) ref.Val {
					s := str.(types.String)
					return stringOrError(trimSpace(string(s)))
				}))),
		cel.Function("upperAscii",
			cel.MemberOverload("string_upper_ascii", []*cel.Type{cel.StringType}, cel.StringType,
				cel.UnaryBinding(func(str ref.Val) ref.Val {
					s := str.(types.String)
					return stringOrError(upperASCII(string(s)))
				}))),
	}
	if sl.version >= 1 {
		opts = append(opts, cel.Function("format",
			cel.MemberOverload("string_format", []*cel.Type{cel.StringType, cel.ListType(cel.DynType)}, cel.StringType,
				cel.FunctionBinding(func(args ...ref.Val) ref.Val {
					s := string(args[0].(types.String))
					formatArgs := args[1].(traits.Lister)
					return stringOrError(interpreter.ParseFormatString(s, &stringFormatter{}, &stringArgList{formatArgs}, formatLocale))
				}))),
			cel.Function("strings.quote", cel.Overload("strings_quote", []*cel.Type{cel.StringType}, cel.StringType,
				cel.UnaryBinding(func(str ref.Val) ref.Val {
					s := str.(types.String)
					return stringOrError(quote(string(s)))
				}))))

	}
	if sl.version >= 2 {
		opts = append(opts,
			cel.Function("join",
				cel.MemberOverload("list_join", []*cel.Type{cel.ListType(cel.StringType)}, cel.StringType,
					cel.UnaryBinding(func(list ref.Val) ref.Val {
						l := list.(traits.Lister)
						return stringOrError(joinValSeparator(l, ""))
					})),
				cel.MemberOverload("list_join_string", []*cel.Type{cel.ListType(cel.StringType), cel.StringType}, cel.StringType,
					cel.BinaryBinding(func(list, delim ref.Val) ref.Val {
						l := list.(traits.Lister)
						d := delim.(types.String)
						return stringOrError(joinValSeparator(l, string(d)))
					}))),
		)
	} else {
		opts = append(opts,
			cel.Function("join",
				cel.MemberOverload("list_join", []*cel.Type{cel.ListType(cel.StringType)}, cel.StringType,
					cel.UnaryBinding(func(list ref.Val) ref.Val {
						l, err := list.ConvertToNative(stringListType)
						if err != nil {
							return types.NewErr(err.Error())
						}
						return stringOrError(join(l.([]string)))
					})),
				cel.MemberOverload("list_join_string", []*cel.Type{cel.ListType(cel.StringType), cel.StringType}, cel.StringType,
					cel.BinaryBinding(func(list, delim ref.Val) ref.Val {
						l, err := list.ConvertToNative(stringListType)
						if err != nil {
							return types.NewErr(err.Error())
						}
						d := delim.(types.String)
						return stringOrError(joinSeparator(l.([]string), string(d)))
					}))),
		)
	}
	return opts
}

// ProgramOptions implements the Library interface method.
func (*stringLib) ProgramOptions() []cel.ProgramOption {
	return []cel.ProgramOption{}
}

func charAt(str string, ind int64) (string, error) {
	i := int(ind)
	runes := []rune(str)
	if i < 0 || i > len(runes) {
		return "", fmt.Errorf("index out of range: %d", ind)
	}
	if i == len(runes) {
		return "", nil
	}
	return string(runes[i]), nil
}

func indexOf(str, substr string) (int64, error) {
	return indexOfOffset(str, substr, int64(0))
}

func indexOfOffset(str, substr string, offset int64) (int64, error) {
	if substr == "" {
		return offset, nil
	}
	off := int(offset)
	runes := []rune(str)
	subrunes := []rune(substr)
	if off < 0 || off >= len(runes) {
		return -1, fmt.Errorf("index out of range: %d", off)
	}
	for i := off; i < len(runes)-(len(subrunes)-1); i++ {
		found := true
		for j := 0; j < len(subrunes); j++ {
			if runes[i+j] != subrunes[j] {
				found = false
				break
			}
		}
		if found {
			return int64(i), nil
		}
	}
	return -1, nil
}

func lastIndexOf(str, substr string) (int64, error) {
	runes := []rune(str)
	if substr == "" {
		return int64(len(runes)), nil
	}
	return lastIndexOfOffset(str, substr, int64(len(runes)-1))
}

func lastIndexOfOffset(str, substr string, offset int64) (int64, error) {
	if substr == "" {
		return offset, nil
	}
	off := int(offset)
	runes := []rune(str)
	subrunes := []rune(substr)
	if off < 0 || off >= len(runes) {
		return -1, fmt.Errorf("index out of range: %d", off)
	}
	if off > len(runes)-len(subrunes) {
		off = len(runes) - len(subrunes)
	}
	for i := off; i >= 0; i-- {
		found := true
		for j := 0; j < len(subrunes); j++ {
			if runes[i+j] != subrunes[j] {
				found = false
				break
			}
		}
		if found {
			return int64(i), nil
		}
	}
	return -1, nil
}

func lowerASCII(str string) (string, error) {
	runes := []rune(str)
	for i, r := range runes {
		if r <= unicode.MaxASCII {
			r = unicode.ToLower(r)
			runes[i] = r
		}
	}
	return string(runes), nil
}

func replace(str, old, new string) (string, error) {
	return strings.ReplaceAll(str, old, new), nil
}

func replaceN(str, old, new string, n int64) (string, error) {
	return strings.Replace(str, old, new, int(n)), nil
}

func split(str, sep string) ([]string, error) {
	return strings.Split(str, sep), nil
}

func splitN(str, sep string, n int64) ([]string, error) {
	return strings.SplitN(str, sep, int(n)), nil
}

func substr(str string, start int64) (string, error) {
	runes := []rune(str)
	if int(start) < 0 || int(start) > len(runes) {
		return "", fmt.Errorf("index out of range: %d", start)
	}
	return string(runes[start:]), nil
}

func substrRange(str string, start, end int64) (string, error) {
	runes := []rune(str)
	l := len(runes)
	if start > end {
		return "", fmt.Errorf("invalid substring range. start: %d, end: %d", start, end)
	}
	if int(start) < 0 || int(start) > l {
		return "", fmt.Errorf("index out of range: %d", start)
	}
	if int(end) < 0 || int(end) > l {
		return "", fmt.Errorf("index out of range: %d", end)
	}
	return string(runes[int(start):int(end)]), nil
}

func trimSpace(str string) (string, error) {
	return strings.TrimSpace(str), nil
}

func upperASCII(str string) (string, error) {
	runes := []rune(str)
	for i, r := range runes {
		if r <= unicode.MaxASCII {
			r = unicode.ToUpper(r)
			runes[i] = r
		}
	}
	return string(runes), nil
}

func joinSeparator(strs []string, separator string) (string, error) {
	return strings.Join(strs, separator), nil
}

func join(strs []string) (string, error) {
	return strings.Join(strs, ""), nil
}

func joinValSeparator(strs traits.Lister, separator string) (string, error) {
	sz := strs.Size().(types.Int)
	var sb strings.Builder
	for i := types.Int(0); i < sz; i++ {
		if i != 0 {
			sb.WriteString(separator)
		}
		elem := strs.Get(i)
		str, ok := elem.(types.String)
		if !ok {
			return "", fmt.Errorf("join: invalid input: %v", elem)
		}
		sb.WriteString(string(str))
	}
	return sb.String(), nil
}

type clauseImpl func(ref.Val, string) (string, error)

func clauseForType(argType ref.Type) (clauseImpl, error) {
	switch argType {
	case types.IntType, types.UintType:
		return formatDecimal, nil
	case types.StringType, types.BytesType, types.BoolType, types.NullType, types.TypeType:
		return FormatString, nil
	case types.TimestampType, types.DurationType:
		// special case to ensure timestamps/durations get printed as CEL literals
		return func(arg ref.Val, locale string) (string, error) {
			argStrVal := arg.ConvertToType(types.StringType)
			argStr := argStrVal.Value().(string)
			if arg.Type() == types.TimestampType {
				return fmt.Sprintf("timestamp(%q)", argStr), nil
			}
			if arg.Type() == types.DurationType {
				return fmt.Sprintf("duration(%q)", argStr), nil
			}
			return "", fmt.Errorf("cannot convert argument of type %s to timestamp/duration", arg.Type().TypeName())
		}, nil
	case types.ListType:
		return formatList, nil
	case types.MapType:
		return formatMap, nil
	case types.DoubleType:
		// avoid formatFixed so we can output a period as the decimal separator in order
		// to always be a valid CEL literal
		return func(arg ref.Val, locale string) (string, error) {
			argDouble, ok := arg.Value().(float64)
			if !ok {
				return "", fmt.Errorf("couldn't convert %s to float64", arg.Type().TypeName())
			}
			fmtStr := fmt.Sprintf("%%.%df", defaultPrecision)
			return fmt.Sprintf(fmtStr, argDouble), nil
		}, nil
	case types.TypeType:
		return func(arg ref.Val, locale string) (string, error) {
			return fmt.Sprintf("type(%s)", arg.Value().(string)), nil
		}, nil
	default:
		return nil, fmt.Errorf("no formatting function for %s", argType.TypeName())
	}
}

func formatList(arg ref.Val, locale string) (string, error) {
	argList := arg.(traits.Lister)
	argIterator := argList.Iterator()
	var listStrBuilder strings.Builder
	_, err := listStrBuilder.WriteRune('[')
	if err != nil {
		return "", fmt.Errorf("error writing to list string: %w", err)
	}
	for argIterator.HasNext() == types.True {
		member := argIterator.Next()
		memberFormat, err := clauseForType(member.Type())
		if err != nil {
			return "", err
		}
		unquotedStr, err := memberFormat(member, locale)
		if err != nil {
			return "", err
		}
		str := quoteForCEL(member, unquotedStr)
		_, err = listStrBuilder.WriteString(str)
		if err != nil {
			return "", fmt.Errorf("error writing to list string: %w", err)
		}
		if argIterator.HasNext() == types.True {
			_, err = listStrBuilder.WriteString(", ")
			if err != nil {
				return "", fmt.Errorf("error writing to list string: %w", err)
			}
		}
	}
	_, err = listStrBuilder.WriteRune(']')
	if err != nil {
		return "", fmt.Errorf("error writing to list string: %w", err)
	}
	return listStrBuilder.String(), nil
}

func formatMap(arg ref.Val, locale string) (string, error) {
	argMap := arg.(traits.Mapper)
	argIterator := argMap.Iterator()
	type mapPair struct {
		key   string
		value string
	}
	argPairs := make([]mapPair, argMap.Size().Value().(int64))
	i := 0
	for argIterator.HasNext() == types.True {
		key := argIterator.Next()
		var keyFormat clauseImpl
		switch key.Type() {
		case types.StringType, types.BoolType:
			keyFormat = FormatString
		case types.IntType, types.UintType:
			keyFormat = formatDecimal
		default:
			return "", fmt.Errorf("no formatting function for map key of type %s", key.Type().TypeName())
		}
		unquotedKeyStr, err := keyFormat(key, locale)
		if err != nil {
			return "", err
		}
		keyStr := quoteForCEL(key, unquotedKeyStr)
		value, found := argMap.Find(key)
		if !found {
			return "", fmt.Errorf("could not find key: %q", key)
		}
		valueFormat, err := clauseForType(value.Type())
		if err != nil {
			return "", err
		}
		unquotedValueStr, err := valueFormat(value, locale)
		if err != nil {
			return "", err
		}
		valueStr := quoteForCEL(value, unquotedValueStr)
		argPairs[i] = mapPair{keyStr, valueStr}
		i++
	}
	sort.SliceStable(argPairs, func(x, y int) bool {
		return argPairs[x].key < argPairs[y].key
	})
	var mapStrBuilder strings.Builder
	_, err := mapStrBuilder.WriteRune('{')
	if err != nil {
		return "", fmt.Errorf("error writing to map string: %w", err)
	}
	for i, entry := range argPairs {
		_, err = mapStrBuilder.WriteString(fmt.Sprintf("%s:%s", entry.key, entry.value))
		if err != nil {
			return "", fmt.Errorf("error writing to map string: %w", err)
		}
		if i < len(argPairs)-1 {
			_, err = mapStrBuilder.WriteString(", ")
			if err != nil {
				return "", fmt.Errorf("error writing to map string: %w", err)
			}
		}
	}
	_, err = mapStrBuilder.WriteRune('}')
	if err != nil {
		return "", fmt.Errorf("error writing to map string: %w", err)
	}
	return mapStrBuilder.String(), nil
}

// quoteForCEL takes a formatted, unquoted value and quotes it in a manner
// suitable for embedding directly in CEL.
func quoteForCEL(refVal ref.Val, unquotedValue string) string {
	switch refVal.Type() {
	case types.StringType:
		return fmt.Sprintf("%q", unquotedValue)
	case types.BytesType:
		return fmt.Sprintf("b%q", unquotedValue)
	case types.DoubleType:
		// special case to handle infinity/NaN
		num := refVal.Value().(float64)
		if math.IsInf(num, 1) || math.IsInf(num, -1) || math.IsNaN(num) {
			return fmt.Sprintf("%q", unquotedValue)
		}
		return unquotedValue
	default:
		return unquotedValue
	}
}

// FormatString returns the string representation of a CEL value.
// It is used to implement the %s specifier in the (string).format() extension
// function.
func FormatString(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.ListType:
		return formatList(arg, locale)
	case types.MapType:
		return formatMap(arg, locale)
	case types.IntType, types.UintType, types.DoubleType,
		types.BoolType, types.StringType, types.TimestampType, types.BytesType, types.DurationType, types.TypeType:
		argStrVal := arg.ConvertToType(types.StringType)
		argStr, ok := argStrVal.Value().(string)
		if !ok {
			return "", fmt.Errorf("could not convert argument %q to string", argStrVal)
		}
		return argStr, nil
	case types.NullType:
		return "null", nil
	default:
		return "", fmt.Errorf("string clause can only be used on strings, bools, bytes, ints, doubles, maps, lists, types, durations, and timestamps, was given %s", arg.Type().TypeName())
	}
}

func formatDecimal(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt, ok := arg.ConvertToType(types.IntType).Value().(int64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%s\" to int64", arg.Value())
		}
		return fmt.Sprintf("%d", argInt), nil
	case types.UintType:
		argInt, ok := arg.ConvertToType(types.UintType).Value().(uint64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%s\" to uint64", arg.Value())
		}
		return fmt.Sprintf("%d", argInt), nil
	default:
		return "", fmt.Errorf("decimal clause can only be used on integers, was given %s", arg.Type().TypeName())
	}
}

func matchLanguage(locale string) (language.Tag, error) {
	matcher, err := makeMatcher(locale)
	if err != nil {
		return language.Und, err
	}
	tag, _ := language.MatchStrings(matcher, locale)
	return tag, nil
}

func makeMatcher(locale string) (language.Matcher, error) {
	tags := make([]language.Tag, 0)
	tag, err := language.Parse(locale)
	if err != nil {
		return nil, err
	}
	tags = append(tags, tag)
	return language.NewMatcher(tags), nil
}

// quote implements a string quoting function. The string will be wrapped in
// double quotes, and all valid CEL escape sequences will be escaped to show up
// literally if printed. If the input contains any invalid UTF-8, the invalid runes
// will be replaced with utf8.RuneError.
func quote(s string) (string, error) {
	var quotedStrBuilder strings.Builder
	for _, c := range sanitize(s) {
		switch c {
		case '\a':
			quotedStrBuilder.WriteString("\\a")
		case '\b':
			quotedStrBuilder.WriteString("\\b")
		case '\f':
			quotedStrBuilder.WriteString("\\f")
		case '\n':
			quotedStrBuilder.WriteString("\\n")
		case '\r':
			quotedStrBuilder.WriteString("\\r")
		case '\t':
			quotedStrBuilder.WriteString("\\t")
		case '\v':
			quotedStrBuilder.WriteString("\\v")
		case '\\':
			quotedStrBuilder.WriteString("\\\\")
		case '"':
			quotedStrBuilder.WriteString("\\\"")
		default:
			quotedStrBuilder.WriteRune(c)
		}
	}
	escapedStr := quotedStrBuilder.String()
	return "\"" + escapedStr + "\"", nil
}

// sanitize replaces all invalid runes in the given string with utf8.RuneError.
func sanitize(s string) string {
	var sanitizedStringBuilder strings.Builder
	for _, r := range s {
		if !utf8.ValidRune(r) {
			sanitizedStringBuilder.WriteRune(utf8.RuneError)
		} else {
			sanitizedStringBuilder.WriteRune(r)
		}
	}
	return sanitizedStringBuilder.String()
}

type stringFormatter struct{}

func (c *stringFormatter) String(arg ref.Val, locale string) (string, error) {
	return FormatString(arg, locale)
}

func (c *stringFormatter) Decimal(arg ref.Val, locale string) (string, error) {
	return formatDecimal(arg, locale)
}

func (c *stringFormatter) Fixed(precision *int) func(ref.Val, string) (string, error) {
	if precision == nil {
		precision = new(int)
		*precision = defaultPrecision
	}
	return func(arg ref.Val, locale string) (string, error) {
		strException := false
		if arg.Type() == types.StringType {
			argStr := arg.Value().(string)
			if argStr == "NaN" || argStr == "Infinity" || argStr == "-Infinity" {
				strException = true
			}
		}
		if arg.Type() != types.DoubleType && !strException {
			return "", fmt.Errorf("fixed-point clause can only be used on doubles, was given %s", arg.Type().TypeName())
		}
		argFloatVal := arg.ConvertToType(types.DoubleType)
		argFloat, ok := argFloatVal.Value().(float64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%s\" to float64", argFloatVal.Value())
		}
		fmtStr := fmt.Sprintf("%%.%df", *precision)

		matchedLocale, err := matchLanguage(locale)
		if err != nil {
			return "", fmt.Errorf("error matching locale: %w", err)
		}
		return message.NewPrinter(matchedLocale).Sprintf(fmtStr, argFloat), nil
	}
}

func (c *stringFormatter) Scientific(precision *int) func(ref.Val, string) (string, error) {
	if precision == nil {
		precision = new(int)
		*precision = defaultPrecision
	}
	return func(arg ref.Val, locale string) (string, error) {
		strException := false
		if arg.Type() == types.StringType {
			argStr := arg.Value().(string)
			if argStr == "NaN" || argStr == "Infinity" || argStr == "-Infinity" {
				strException = true
			}
		}
		if arg.Type() != types.DoubleType && !strException {
			return "", fmt.Errorf("scientific clause can only be used on doubles, was given %s", arg.Type().TypeName())
		}
		argFloatVal := arg.ConvertToType(types.DoubleType)
		argFloat, ok := argFloatVal.Value().(float64)
		if !ok {
			return "", fmt.Errorf("could not convert \"%s\" to float64", argFloatVal.Value())
		}
		matchedLocale, err := matchLanguage(locale)
		if err != nil {
			return "", fmt.Errorf("error matching locale: %w", err)
		}
		fmtStr := fmt.Sprintf("%%%de", *precision)
		return message.NewPrinter(matchedLocale).Sprintf(fmtStr, argFloat), nil
	}
}

func (c *stringFormatter) Binary(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt := arg.Value().(int64)
		// locale is intentionally unused as integers formatted as binary
		// strings are locale-independent
		return fmt.Sprintf("%b", argInt), nil
	case types.UintType:
		argInt := arg.Value().(uint64)
		return fmt.Sprintf("%b", argInt), nil
	case types.BoolType:
		argBool := arg.Value().(bool)
		if argBool {
			return "1", nil
		}
		return "0", nil
	default:
		return "", fmt.Errorf("only integers and bools can be formatted as binary, was given %s", arg.Type().TypeName())
	}
}

func (c *stringFormatter) Hex(useUpper bool) func(ref.Val, string) (string, error) {
	return func(arg ref.Val, locale string) (string, error) {
		fmtStr := "%x"
		if useUpper {
			fmtStr = "%X"
		}
		switch arg.Type() {
		case types.StringType, types.BytesType:
			if arg.Type() == types.BytesType {
				return fmt.Sprintf(fmtStr, arg.Value().([]byte)), nil
			}
			return fmt.Sprintf(fmtStr, arg.Value().(string)), nil
		case types.IntType:
			argInt, ok := arg.Value().(int64)
			if !ok {
				return "", fmt.Errorf("could not convert \"%s\" to int64", arg.Value())
			}
			return fmt.Sprintf(fmtStr, argInt), nil
		case types.UintType:
			argInt, ok := arg.Value().(uint64)
			if !ok {
				return "", fmt.Errorf("could not convert \"%s\" to uint64", arg.Value())
			}
			return fmt.Sprintf(fmtStr, argInt), nil
		default:
			return "", fmt.Errorf("only integers, byte buffers, and strings can be formatted as hex, was given %s", arg.Type().TypeName())
		}
	}
}

func (c *stringFormatter) Octal(arg ref.Val, locale string) (string, error) {
	switch arg.Type() {
	case types.IntType:
		argInt := arg.Value().(int64)
		return fmt.Sprintf("%o", argInt), nil
	case types.UintType:
		argInt := arg.Value().(uint64)
		return fmt.Sprintf("%o", argInt), nil
	default:
		return "", fmt.Errorf("octal clause can only be used on integers, was given %s", arg.Type().TypeName())
	}
}

type stringArgList struct {
	args traits.Lister
}

func (c *stringArgList) Arg(index int64) (ref.Val, error) {
	if index >= c.args.Size().Value().(int64) {
		return nil, fmt.Errorf("index %d out of range", index)
	}
	return c.args.Get(types.Int(index)), nil
}

func (c *stringArgList) ArgSize() int64 {
	return c.args.Size().Value().(int64)
}

var (
	stringListType = reflect.TypeOf([]string{})
)
