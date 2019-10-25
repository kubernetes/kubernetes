/*
Copyright 2014 Alexander Okoli

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

/*
Package goutils provides utility functions to manipulate strings in various ways.
The code snippets below show examples of how to use goutils. Some functions return
errors while others do not, so usage would vary as a result.

Example:

    package main

    import (
        "fmt"
        "github.com/aokoli/goutils"
    )

    func main() {

        // EXAMPLE 1: A goutils function which returns no errors
        fmt.Println (goutils.Initials("John Doe Foo")) // Prints out "JDF"



        // EXAMPLE 2: A goutils function which returns an error
        rand1, err1 := goutils.Random (-1, 0, 0, true, true)

        if err1 != nil {
            fmt.Println(err1) // Prints out error message because -1 was entered as the first parameter in goutils.Random(...)
        } else {
            fmt.Println(rand1)
        }
    }
*/
package goutils

import (
	"bytes"
	"strings"
	"unicode"
)

// VERSION indicates the current version of goutils
const VERSION = "1.0.0"

/*
Wrap wraps a single line of text, identifying words by ' '.
New lines will be separated by '\n'. Very long words, such as URLs will not be wrapped.
Leading spaces on a new line are stripped. Trailing spaces are not stripped.

Parameters:
    str - the string to be word wrapped
    wrapLength - the column (a column can fit only one character) to wrap the words at, less than 1 is treated as 1

Returns:
    a line with newlines inserted
*/
func Wrap(str string, wrapLength int) string {
	return WrapCustom(str, wrapLength, "", false)
}

/*
WrapCustom wraps a single line of text, identifying words by ' '.
Leading spaces on a new line are stripped. Trailing spaces are not stripped.

Parameters:
    str - the string to be word wrapped
    wrapLength - the column number (a column can fit only one character) to wrap the words at, less than 1 is treated as 1
    newLineStr - the string to insert for a new line, "" uses '\n'
    wrapLongWords - true if long words (such as URLs) should be wrapped

Returns:
    a line with newlines inserted
*/
func WrapCustom(str string, wrapLength int, newLineStr string, wrapLongWords bool) string {

	if str == "" {
		return ""
	}
	if newLineStr == "" {
		newLineStr = "\n" // TODO Assumes "\n" is seperator. Explore SystemUtils.LINE_SEPARATOR from Apache Commons
	}
	if wrapLength < 1 {
		wrapLength = 1
	}

	inputLineLength := len(str)
	offset := 0

	var wrappedLine bytes.Buffer

	for inputLineLength-offset > wrapLength {

		if rune(str[offset]) == ' ' {
			offset++
			continue
		}

		end := wrapLength + offset + 1
		spaceToWrapAt := strings.LastIndex(str[offset:end], " ") + offset

		if spaceToWrapAt >= offset {
			// normal word (not longer than wrapLength)
			wrappedLine.WriteString(str[offset:spaceToWrapAt])
			wrappedLine.WriteString(newLineStr)
			offset = spaceToWrapAt + 1

		} else {
			// long word or URL
			if wrapLongWords {
				end := wrapLength + offset
				// long words are wrapped one line at a time
				wrappedLine.WriteString(str[offset:end])
				wrappedLine.WriteString(newLineStr)
				offset += wrapLength
			} else {
				// long words aren't wrapped, just extended beyond limit
				end := wrapLength + offset
				index := strings.IndexRune(str[end:len(str)], ' ')
				if index == -1 {
					wrappedLine.WriteString(str[offset:len(str)])
					offset = inputLineLength
				} else {
					spaceToWrapAt = index + end
					wrappedLine.WriteString(str[offset:spaceToWrapAt])
					wrappedLine.WriteString(newLineStr)
					offset = spaceToWrapAt + 1
				}
			}
		}
	}

	wrappedLine.WriteString(str[offset:len(str)])

	return wrappedLine.String()

}

/*
Capitalize capitalizes all the delimiter separated words in a string. Only the first letter of each word is changed.
To convert the rest of each word to lowercase at the same time, use CapitalizeFully(str string, delimiters ...rune).
The delimiters represent a set of characters understood to separate words. The first string character
and the first non-delimiter character after a delimiter will be capitalized. A "" input string returns "".
Capitalization uses the Unicode title case, normally equivalent to upper case.

Parameters:
    str - the string to capitalize
    delimiters - set of characters to determine capitalization, exclusion of this parameter means whitespace would be delimeter

Returns:
    capitalized string
*/
func Capitalize(str string, delimiters ...rune) string {

	var delimLen int

	if delimiters == nil {
		delimLen = -1
	} else {
		delimLen = len(delimiters)
	}

	if str == "" || delimLen == 0 {
		return str
	}

	buffer := []rune(str)
	capitalizeNext := true
	for i := 0; i < len(buffer); i++ {
		ch := buffer[i]
		if isDelimiter(ch, delimiters...) {
			capitalizeNext = true
		} else if capitalizeNext {
			buffer[i] = unicode.ToTitle(ch)
			capitalizeNext = false
		}
	}
	return string(buffer)

}

/*
CapitalizeFully converts all the delimiter separated words in a string into capitalized words, that is each word is made up of a
titlecase character and then a series of lowercase characters. The delimiters represent a set of characters understood
to separate words. The first string character and the first non-delimiter character after a delimiter will be capitalized.
Capitalization uses the Unicode title case, normally equivalent to upper case.

Parameters:
    str - the string to capitalize fully
    delimiters - set of characters to determine capitalization, exclusion of this parameter means whitespace would be delimeter

Returns:
    capitalized string
*/
func CapitalizeFully(str string, delimiters ...rune) string {

	var delimLen int

	if delimiters == nil {
		delimLen = -1
	} else {
		delimLen = len(delimiters)
	}

	if str == "" || delimLen == 0 {
		return str
	}
	str = strings.ToLower(str)
	return Capitalize(str, delimiters...)
}

/*
Uncapitalize uncapitalizes all the whitespace separated words in a string. Only the first letter of each word is changed.
The delimiters represent a set of characters understood to separate words. The first string character and the first non-delimiter
character after a delimiter will be uncapitalized. Whitespace is defined by unicode.IsSpace(char).

Parameters:
    str - the string to uncapitalize fully
    delimiters - set of characters to determine capitalization, exclusion of this parameter means whitespace would be delimeter

Returns:
    uncapitalized string
*/
func Uncapitalize(str string, delimiters ...rune) string {

	var delimLen int

	if delimiters == nil {
		delimLen = -1
	} else {
		delimLen = len(delimiters)
	}

	if str == "" || delimLen == 0 {
		return str
	}

	buffer := []rune(str)
	uncapitalizeNext := true // TODO Always makes capitalize/un apply to first char.
	for i := 0; i < len(buffer); i++ {
		ch := buffer[i]
		if isDelimiter(ch, delimiters...) {
			uncapitalizeNext = true
		} else if uncapitalizeNext {
			buffer[i] = unicode.ToLower(ch)
			uncapitalizeNext = false
		}
	}
	return string(buffer)
}

/*
SwapCase swaps the case of a string using a word based algorithm.

Conversion algorithm:

    Upper case character converts to Lower case
    Title case character converts to Lower case
    Lower case character after Whitespace or at start converts to Title case
    Other Lower case character converts to Upper case
    Whitespace is defined by unicode.IsSpace(char).

Parameters:
    str - the string to swap case

Returns:
    the changed string
*/
func SwapCase(str string) string {
	if str == "" {
		return str
	}
	buffer := []rune(str)

	whitespace := true

	for i := 0; i < len(buffer); i++ {
		ch := buffer[i]
		if unicode.IsUpper(ch) {
			buffer[i] = unicode.ToLower(ch)
			whitespace = false
		} else if unicode.IsTitle(ch) {
			buffer[i] = unicode.ToLower(ch)
			whitespace = false
		} else if unicode.IsLower(ch) {
			if whitespace {
				buffer[i] = unicode.ToTitle(ch)
				whitespace = false
			} else {
				buffer[i] = unicode.ToUpper(ch)
			}
		} else {
			whitespace = unicode.IsSpace(ch)
		}
	}
	return string(buffer)
}

/*
Initials extracts the initial letters from each word in the string. The first letter of the string and all first
letters after the defined delimiters are returned as a new string. Their case is not changed. If the delimiters
parameter is excluded, then Whitespace is used. Whitespace is defined by unicode.IsSpacea(char). An empty delimiter array returns an empty string.

Parameters:
    str - the string to get initials from
    delimiters - set of characters to determine words, exclusion of this parameter means whitespace would be delimeter
Returns:
    string of initial letters
*/
func Initials(str string, delimiters ...rune) string {
	if str == "" {
		return str
	}
	if delimiters != nil && len(delimiters) == 0 {
		return ""
	}
	strLen := len(str)
	var buf bytes.Buffer
	lastWasGap := true
	for i := 0; i < strLen; i++ {
		ch := rune(str[i])

		if isDelimiter(ch, delimiters...) {
			lastWasGap = true
		} else if lastWasGap {
			buf.WriteRune(ch)
			lastWasGap = false
		}
	}
	return buf.String()
}

// private function (lower case func name)
func isDelimiter(ch rune, delimiters ...rune) bool {
	if delimiters == nil {
		return unicode.IsSpace(ch)
	}
	for _, delimiter := range delimiters {
		if ch == delimiter {
			return true
		}
	}
	return false
}
