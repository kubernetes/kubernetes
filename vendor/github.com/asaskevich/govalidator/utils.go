package govalidator

import (
	"errors"
	"fmt"
	"html"
	"path"
	"regexp"
	"strings"
	"unicode"
)

// Contains check if the string contains the substring.
func Contains(str, substring string) bool {
	return strings.Contains(str, substring)
}

// Matches check if string matches the pattern (pattern is regular expression)
// In case of error return false
func Matches(str, pattern string) bool {
	match, _ := regexp.MatchString(pattern, str)
	return match
}

// LeftTrim trim characters from the left-side of the input.
// If second argument is empty, it's will be remove leading spaces.
func LeftTrim(str, chars string) string {
	pattern := ""
	if chars == "" {
		pattern = "^\\s+"
	} else {
		pattern = "^[" + chars + "]+"
	}
	r, _ := regexp.Compile(pattern)
	return string(r.ReplaceAll([]byte(str), []byte("")))
}

// RightTrim trim characters from the right-side of the input.
// If second argument is empty, it's will be remove spaces.
func RightTrim(str, chars string) string {
	pattern := ""
	if chars == "" {
		pattern = "\\s+$"
	} else {
		pattern = "[" + chars + "]+$"
	}
	r, _ := regexp.Compile(pattern)
	return string(r.ReplaceAll([]byte(str), []byte("")))
}

// Trim trim characters from both sides of the input.
// If second argument is empty, it's will be remove spaces.
func Trim(str, chars string) string {
	return LeftTrim(RightTrim(str, chars), chars)
}

// WhiteList remove characters that do not appear in the whitelist.
func WhiteList(str, chars string) string {
	pattern := "[^" + chars + "]+"
	r, _ := regexp.Compile(pattern)
	return string(r.ReplaceAll([]byte(str), []byte("")))
}

// BlackList remove characters that appear in the blacklist.
func BlackList(str, chars string) string {
	pattern := "[" + chars + "]+"
	r, _ := regexp.Compile(pattern)
	return string(r.ReplaceAll([]byte(str), []byte("")))
}

// StripLow remove characters with a numerical value < 32 and 127, mostly control characters.
// If keep_new_lines is true, newline characters are preserved (\n and \r, hex 0xA and 0xD).
func StripLow(str string, keepNewLines bool) string {
	chars := ""
	if keepNewLines {
		chars = "\x00-\x09\x0B\x0C\x0E-\x1F\x7F"
	} else {
		chars = "\x00-\x1F\x7F"
	}
	return BlackList(str, chars)
}

// ReplacePattern replace regular expression pattern in string
func ReplacePattern(str, pattern, replace string) string {
	r, _ := regexp.Compile(pattern)
	return string(r.ReplaceAll([]byte(str), []byte(replace)))
}

// Escape replace <, >, & and " with HTML entities.
var Escape = html.EscapeString

func addSegment(inrune, segment []rune) []rune {
	if len(segment) == 0 {
		return inrune
	}
	if len(inrune) != 0 {
		inrune = append(inrune, '_')
	}
	inrune = append(inrune, segment...)
	return inrune
}

// UnderscoreToCamelCase converts from underscore separated form to camel case form.
// Ex.: my_func => MyFunc
func UnderscoreToCamelCase(s string) string {
	return strings.Replace(strings.Title(strings.Replace(strings.ToLower(s), "_", " ", -1)), " ", "", -1)
}

// CamelCaseToUnderscore converts from camel case form to underscore separated form.
// Ex.: MyFunc => my_func
func CamelCaseToUnderscore(str string) string {
	var output []rune
	var segment []rune
	for _, r := range str {
		if !unicode.IsLower(r) {
			output = addSegment(output, segment)
			segment = nil
		}
		segment = append(segment, unicode.ToLower(r))
	}
	output = addSegment(output, segment)
	return string(output)
}

// Reverse return reversed string
func Reverse(s string) string {
	r := []rune(s)
	for i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {
		r[i], r[j] = r[j], r[i]
	}
	return string(r)
}

// GetLines split string by "\n" and return array of lines
func GetLines(s string) []string {
	return strings.Split(s, "\n")
}

// GetLine return specified line of multiline string
func GetLine(s string, index int) (string, error) {
	lines := GetLines(s)
	if index < 0 || index >= len(lines) {
		return "", errors.New("line index out of bounds")
	}
	return lines[index], nil
}

// RemoveTags remove all tags from HTML string
func RemoveTags(s string) string {
	return ReplacePattern(s, "<[^>]*>", "")
}

// SafeFileName return safe string that can be used in file names
func SafeFileName(str string) string {
	name := strings.ToLower(str)
	name = path.Clean(path.Base(name))
	name = strings.Trim(name, " ")
	separators, err := regexp.Compile(`[ &_=+:]`)
	if err == nil {
		name = separators.ReplaceAllString(name, "-")
	}
	legal, err := regexp.Compile(`[^[:alnum:]-.]`)
	if err == nil {
		name = legal.ReplaceAllString(name, "")
	}
	for strings.Contains(name, "--") {
		name = strings.Replace(name, "--", "-", -1)
	}
	return name
}

// NormalizeEmail canonicalize an email address.
// The local part of the email address is lowercased for all domains; the hostname is always lowercased and
// the local part of the email address is always lowercased for hosts that are known to be case-insensitive (currently only GMail).
// Normalization follows special rules for known providers: currently, GMail addresses have dots removed in the local part and
// are stripped of tags (e.g. some.one+tag@gmail.com becomes someone@gmail.com) and all @googlemail.com addresses are
// normalized to @gmail.com.
func NormalizeEmail(str string) (string, error) {
	if !IsEmail(str) {
		return "", fmt.Errorf("%s is not an email", str)
	}
	parts := strings.Split(str, "@")
	parts[0] = strings.ToLower(parts[0])
	parts[1] = strings.ToLower(parts[1])
	if parts[1] == "gmail.com" || parts[1] == "googlemail.com" {
		parts[1] = "gmail.com"
		parts[0] = strings.Split(ReplacePattern(parts[0], `\.`, ""), "+")[0]
	}
	return strings.Join(parts, "@"), nil
}

// Truncate a string to the closest length without breaking words.
func Truncate(str string, length int, ending string) string {
	var aftstr, befstr string
	if len(str) > length {
		words := strings.Fields(str)
		before, present := 0, 0
		for i := range words {
			befstr = aftstr
			before = present
			aftstr = aftstr + words[i] + " "
			present = len(aftstr)
			if present > length && i != 0 {
				if (length - before) < (present - length) {
					return Trim(befstr, " /\\.,\"'#!?&@+-") + ending
				}
				return Trim(aftstr, " /\\.,\"'#!?&@+-") + ending
			}
		}
	}

	return str
}
