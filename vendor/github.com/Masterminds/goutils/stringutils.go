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

package goutils

import (
	"bytes"
	"fmt"
	"strings"
	"unicode"
)

// Typically returned by functions where a searched item cannot be found
const INDEX_NOT_FOUND = -1

/*
Abbreviate abbreviates a string using ellipses. This will turn  the string "Now is the time for all good men" into "Now is the time for..."

Specifically, the algorithm is as follows:

    - If str is less than maxWidth characters long, return it.
    - Else abbreviate it to (str[0:maxWidth - 3] + "...").
    - If maxWidth is less than 4, return an illegal argument error.
    - In no case will it return a string of length greater than maxWidth.

Parameters:
    str -  the string to check
    maxWidth - maximum length of result string, must be at least 4

Returns:
    string - abbreviated string
    error - if the width is too small
*/
func Abbreviate(str string, maxWidth int) (string, error) {
	return AbbreviateFull(str, 0, maxWidth)
}

/*
AbbreviateFull abbreviates a string using ellipses. This will turn the string "Now is the time for all good men" into "...is the time for..."
This function works like Abbreviate(string, int), but allows you to specify a "left edge" offset. Note that this left edge is not
necessarily going to be the leftmost character in the result, or the first character following the ellipses, but it will appear
somewhere in the result.
In no case will it return a string of length greater than maxWidth.

Parameters:
    str - the string to check
    offset - left edge of source string
    maxWidth - maximum length of result string, must be at least 4

Returns:
    string - abbreviated string
    error - if the width is too small
*/
func AbbreviateFull(str string, offset int, maxWidth int) (string, error) {
	if str == "" {
		return "", nil
	}
	if maxWidth < 4 {
		err := fmt.Errorf("stringutils illegal argument: Minimum abbreviation width is 4")
		return "", err
	}
	if len(str) <= maxWidth {
		return str, nil
	}
	if offset > len(str) {
		offset = len(str)
	}
	if len(str)-offset < (maxWidth - 3) { // 15 - 5 < 10 - 3 =  10 < 7
		offset = len(str) - (maxWidth - 3)
	}
	abrevMarker := "..."
	if offset <= 4 {
		return str[0:maxWidth-3] + abrevMarker, nil // str.substring(0, maxWidth - 3) + abrevMarker;
	}
	if maxWidth < 7 {
		err := fmt.Errorf("stringutils illegal argument: Minimum abbreviation width with offset is 7")
		return "", err
	}
	if (offset + maxWidth - 3) < len(str) { // 5 + (10-3) < 15 = 12 < 15
		abrevStr, _ := Abbreviate(str[offset:len(str)], (maxWidth - 3))
		return abrevMarker + abrevStr, nil // abrevMarker + abbreviate(str.substring(offset), maxWidth - 3);
	}
	return abrevMarker + str[(len(str)-(maxWidth-3)):len(str)], nil // abrevMarker + str.substring(str.length() - (maxWidth - 3));
}

/*
DeleteWhiteSpace deletes all whitespaces from a string as defined by unicode.IsSpace(rune).
It returns the string without whitespaces.

Parameter:
    str - the string to delete whitespace from, may be nil

Returns:
    the string without whitespaces
*/
func DeleteWhiteSpace(str string) string {
	if str == "" {
		return str
	}
	sz := len(str)
	var chs bytes.Buffer
	count := 0
	for i := 0; i < sz; i++ {
		ch := rune(str[i])
		if !unicode.IsSpace(ch) {
			chs.WriteRune(ch)
			count++
		}
	}
	if count == sz {
		return str
	}
	return chs.String()
}

/*
IndexOfDifference compares two strings, and returns the index at which the strings begin to differ.

Parameters:
    str1 - the first string
    str2 - the second string

Returns:
    the index where str1 and str2 begin to differ; -1 if they are equal
*/
func IndexOfDifference(str1 string, str2 string) int {
	if str1 == str2 {
		return INDEX_NOT_FOUND
	}
	if IsEmpty(str1) || IsEmpty(str2) {
		return 0
	}
	var i int
	for i = 0; i < len(str1) && i < len(str2); i++ {
		if rune(str1[i]) != rune(str2[i]) {
			break
		}
	}
	if i < len(str2) || i < len(str1) {
		return i
	}
	return INDEX_NOT_FOUND
}

/*
IsBlank checks if a string is whitespace or empty (""). Observe the following behavior:

    goutils.IsBlank("")        = true
    goutils.IsBlank(" ")       = true
    goutils.IsBlank("bob")     = false
    goutils.IsBlank("  bob  ") = false

Parameter:
    str - the string to check

Returns:
    true - if the string is whitespace or empty ("")
*/
func IsBlank(str string) bool {
	strLen := len(str)
	if str == "" || strLen == 0 {
		return true
	}
	for i := 0; i < strLen; i++ {
		if unicode.IsSpace(rune(str[i])) == false {
			return false
		}
	}
	return true
}

/*
IndexOf returns the index of the first instance of sub in str, with the search beginning from the
index start point specified. -1 is returned if sub is not present in str.

An empty string ("") will return -1 (INDEX_NOT_FOUND). A negative start position is treated as zero.
A start position greater than the string length returns -1.

Parameters:
    str - the string to check
    sub - the substring to find
    start - the start position; negative treated as zero

Returns:
    the first index where the sub string was found  (always >= start)
*/
func IndexOf(str string, sub string, start int) int {

	if start < 0 {
		start = 0
	}

	if len(str) < start {
		return INDEX_NOT_FOUND
	}

	if IsEmpty(str) || IsEmpty(sub) {
		return INDEX_NOT_FOUND
	}

	partialIndex := strings.Index(str[start:len(str)], sub)
	if partialIndex == -1 {
		return INDEX_NOT_FOUND
	}
	return partialIndex + start
}

// IsEmpty checks if a string is empty (""). Returns true if empty, and false otherwise.
func IsEmpty(str string) bool {
	return len(str) == 0
}
