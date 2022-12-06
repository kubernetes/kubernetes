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
	"reflect"
	"strings"
	"unicode"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

// Strings returns a cel.EnvOption to configure extended functions for string manipulation.
// As a general note, all indices are zero-based.
//
// CharAt
//
// Returns the character at the given position. If the position is negative, or greater than
// the length of the string, the function will produce an error:
//
//     <string>.charAt(<int>) -> <string>
//
// Examples:
//
//     'hello'.charAt(4)  // return 'o'
//     'hello'.charAt(5)  // return ''
//     'hello'.charAt(-1) // error
//
// IndexOf
//
// Returns the integer index of the first occurrence of the search string. If the search string is
// not found the function returns -1.
//
// The function also accepts an optional position from which to begin the substring search. If the
// substring is the empty string, the index where the search starts is returned (zero or custom).
//
//     <string>.indexOf(<string>) -> <int>
//     <string>.indexOf(<string>, <int>) -> <int>
//
// Examples:
//
//     'hello mellow'.indexOf('')         // returns 0
//     'hello mellow'.indexOf('ello')     // returns 1
//     'hello mellow'.indexOf('jello')    // returns -1
//     'hello mellow'.indexOf('', 2)      // returns 2
//     'hello mellow'.indexOf('ello', 2)  // returns 7
//     'hello mellow'.indexOf('ello', 20) // error
//
// Join
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
//     ['hello', 'mellow'].join() // returns 'hellomellow'
//     ['hello', 'mellow'].join(' ') // returns 'hello mellow'
//     [].join() // returns ''
//     [].join('/') // returns ''
//
// LastIndexOf
//
// Returns the integer index at the start of the last occurrence of the search string. If the
// search string is not found the function returns -1.
//
// The function also accepts an optional position which represents the last index to be
// considered as the beginning of the substring match. If the substring is the empty string,
// the index where the search starts is returned (string length or custom).
//
//     <string>.lastIndexOf(<string>) -> <int>
//     <string>.lastIndexOf(<string>, <int>) -> <int>
//
// Examples:
//
//     'hello mellow'.lastIndexOf('')         // returns 12
//     'hello mellow'.lastIndexOf('ello')     // returns 7
//     'hello mellow'.lastIndexOf('jello')    // returns -1
//     'hello mellow'.lastIndexOf('ello', 6)  // returns 1
//     'hello mellow'.lastIndexOf('ello', -1) // error
//
// LowerAscii
//
// Returns a new string where all ASCII characters are lower-cased.
//
// This function does not perform Unicode case-mapping for characters outside the ASCII range.
//
//     <string>.lowerAscii() -> <string>
//
// Examples:
//
//     'TacoCat'.lowerAscii()      // returns 'tacocat'
//     'TacoCÆt Xii'.lowerAscii()  // returns 'tacocÆt xii'
//
// Replace
//
// Returns a new string based on the target, which replaces the occurrences of a search string
// with a replacement string if present. The function accepts an optional limit on the number of
// substring replacements to be made.
//
// When the replacement limit is 0, the result is the original string. When the limit is a negative
// number, the function behaves the same as replace all.
//
//     <string>.replace(<string>, <string>) -> <string>
//     <string>.replace(<string>, <string>, <int>) -> <string>
//
// Examples:
//
//     'hello hello'.replace('he', 'we')     // returns 'wello wello'
//     'hello hello'.replace('he', 'we', -1) // returns 'wello wello'
//     'hello hello'.replace('he', 'we', 1)  // returns 'wello hello'
//     'hello hello'.replace('he', 'we', 0)  // returns 'hello hello'
//
// Split
//
// Returns a list of strings split from the input by the given separator. The function accepts
// an optional argument specifying a limit on the number of substrings produced by the split.
//
// When the split limit is 0, the result is an empty list. When the limit is 1, the result is the
// target string to split. When the limit is a negative number, the function behaves the same as
// split all.
//
//     <string>.split(<string>) -> <list<string>>
//     <string>.split(<string>, <int>) -> <list<string>>
//
// Examples:
//
//     'hello hello hello'.split(' ')     // returns ['hello', 'hello', 'hello']
//     'hello hello hello'.split(' ', 0)  // returns []
//     'hello hello hello'.split(' ', 1)  // returns ['hello hello hello']
//     'hello hello hello'.split(' ', 2)  // returns ['hello', 'hello hello']
//     'hello hello hello'.split(' ', -1) // returns ['hello', 'hello', 'hello']
//
// Substring
//
// Returns the substring given a numeric range corresponding to character positions. Optionally
// may omit the trailing range for a substring from a given character position until the end of
// a string.
//
// Character offsets are 0-based with an inclusive start range and exclusive end range. It is an
// error to specify an end range that is lower than the start range, or for either the start or end
// index to be negative or exceed the string length.
//
//     <string>.substring(<int>) -> <string>
//     <string>.substring(<int>, <int>) -> <string>
//
// Examples:
//
//     'tacocat'.substring(4)    // returns 'cat'
//     'tacocat'.substring(0, 4) // returns 'taco'
//     'tacocat'.substring(-1)   // error
//     'tacocat'.substring(2, 1) // error
//
// Trim
//
// Returns a new string which removes the leading and trailing whitespace in the target string.
// The trim function uses the Unicode definition of whitespace which does not include the
// zero-width spaces. See: https://en.wikipedia.org/wiki/Whitespace_character#Unicode
//
//      <string>.trim() -> <string>
//
// Examples:
//
//     '  \ttrim\n    '.trim() // returns 'trim'
//
// UpperAscii
//
// Returns a new string where all ASCII characters are upper-cased.
//
// This function does not perform Unicode case-mapping for characters outside the ASCII range.
//
//    <string>.upperAscii() -> <string>
//
// Examples:
//
//     'TacoCat'.upperAscii()      // returns 'TACOCAT'
//     'TacoCÆt Xii'.upperAscii()  // returns 'TACOCÆT XII'
func Strings() cel.EnvOption {
	return cel.Lib(stringLib{})
}

type stringLib struct{}

func (stringLib) CompileOptions() []cel.EnvOption {
	return []cel.EnvOption{
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
	}
}

func (stringLib) ProgramOptions() []cel.ProgramOption {
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

var (
	stringListType = reflect.TypeOf([]string{})
)
