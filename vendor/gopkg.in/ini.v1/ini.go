// +build go1.6

// Copyright 2014 Unknwon
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

// Package ini provides INI file read and write functionality in Go.
package ini

import (
	"regexp"
	"runtime"
)

const (
	// DefaultSection is the name of default section. You can use this constant or the string literal.
	// In most of cases, an empty string is all you need to access the section.
	DefaultSection = "DEFAULT"

	// Maximum allowed depth when recursively substituing variable names.
	depthValues = 99
	version     = "1.51.0"
)

// Version returns current package version literal.
func Version() string {
	return version
}

var (
	// LineBreak is the delimiter to determine or compose a new line.
	// This variable will be changed to "\r\n" automatically on Windows at package init time.
	LineBreak = "\n"

	// Variable regexp pattern: %(variable)s
	varPattern = regexp.MustCompile(`%\(([^)]+)\)s`)

	// DefaultHeader explicitly writes default section header.
	DefaultHeader = false

	// PrettySection indicates whether to put a line between sections.
	PrettySection = true
	// PrettyFormat indicates whether to align "=" sign with spaces to produce pretty output
	// or reduce all possible spaces for compact format.
	PrettyFormat = true
	// PrettyEqual places spaces around "=" sign even when PrettyFormat is false.
	PrettyEqual = false
	// DefaultFormatLeft places custom spaces on the left when PrettyFormat and PrettyEqual are both disabled.
	DefaultFormatLeft = ""
	// DefaultFormatRight places custom spaces on the right when PrettyFormat and PrettyEqual are both disabled.
	DefaultFormatRight = ""
)

func init() {
	if runtime.GOOS == "windows" {
		LineBreak = "\r\n"
	}
}

// LoadOptions contains all customized options used for load data source(s).
type LoadOptions struct {
	// Loose indicates whether the parser should ignore nonexistent files or return error.
	Loose bool
	// Insensitive indicates whether the parser forces all section and key names to lowercase.
	Insensitive bool
	// IgnoreContinuation indicates whether to ignore continuation lines while parsing.
	IgnoreContinuation bool
	// IgnoreInlineComment indicates whether to ignore comments at the end of value and treat it as part of value.
	IgnoreInlineComment bool
	// SkipUnrecognizableLines indicates whether to skip unrecognizable lines that do not conform to key/value pairs.
	SkipUnrecognizableLines bool
	// AllowBooleanKeys indicates whether to allow boolean type keys or treat as value is missing.
	// This type of keys are mostly used in my.cnf.
	AllowBooleanKeys bool
	// AllowShadows indicates whether to keep track of keys with same name under same section.
	AllowShadows bool
	// AllowNestedValues indicates whether to allow AWS-like nested values.
	// Docs: http://docs.aws.amazon.com/cli/latest/topic/config-vars.html#nested-values
	AllowNestedValues bool
	// AllowPythonMultilineValues indicates whether to allow Python-like multi-line values.
	// Docs: https://docs.python.org/3/library/configparser.html#supported-ini-file-structure
	// Relevant quote:  Values can also span multiple lines, as long as they are indented deeper
	// than the first line of the value.
	AllowPythonMultilineValues bool
	// SpaceBeforeInlineComment indicates whether to allow comment symbols (\# and \;) inside value.
	// Docs: https://docs.python.org/2/library/configparser.html
	// Quote: Comments may appear on their own in an otherwise empty line, or may be entered in lines holding values or section names.
	// In the latter case, they need to be preceded by a whitespace character to be recognized as a comment.
	SpaceBeforeInlineComment bool
	// UnescapeValueDoubleQuotes indicates whether to unescape double quotes inside value to regular format
	// when value is surrounded by double quotes, e.g. key="a \"value\"" => key=a "value"
	UnescapeValueDoubleQuotes bool
	// UnescapeValueCommentSymbols indicates to unescape comment symbols (\# and \;) inside value to regular format
	// when value is NOT surrounded by any quotes.
	// Note: UNSTABLE, behavior might change to only unescape inside double quotes but may noy necessary at all.
	UnescapeValueCommentSymbols bool
	// UnparseableSections stores a list of blocks that are allowed with raw content which do not otherwise
	// conform to key/value pairs. Specify the names of those blocks here.
	UnparseableSections []string
	// KeyValueDelimiters is the sequence of delimiters that are used to separate key and value. By default, it is "=:".
	KeyValueDelimiters string
	// PreserveSurroundedQuote indicates whether to preserve surrounded quote (single and double quotes).
	PreserveSurroundedQuote bool
	// DebugFunc is called to collect debug information (currently only useful to debug parsing Python-style multiline values).
	DebugFunc DebugFunc
	// ReaderBufferSize is the buffer size of the reader in bytes.
	ReaderBufferSize int
}

// DebugFunc is the type of function called to log parse events.
type DebugFunc func(message string)

// LoadSources allows caller to apply customized options for loading from data source(s).
func LoadSources(opts LoadOptions, source interface{}, others ...interface{}) (_ *File, err error) {
	sources := make([]dataSource, len(others)+1)
	sources[0], err = parseDataSource(source)
	if err != nil {
		return nil, err
	}
	for i := range others {
		sources[i+1], err = parseDataSource(others[i])
		if err != nil {
			return nil, err
		}
	}
	f := newFile(sources, opts)
	if err = f.Reload(); err != nil {
		return nil, err
	}
	return f, nil
}

// Load loads and parses from INI data sources.
// Arguments can be mixed of file name with string type, or raw data in []byte.
// It will return error if list contains nonexistent files.
func Load(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{}, source, others...)
}

// LooseLoad has exactly same functionality as Load function
// except it ignores nonexistent files instead of returning error.
func LooseLoad(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{Loose: true}, source, others...)
}

// InsensitiveLoad has exactly same functionality as Load function
// except it forces all section and key names to be lowercased.
func InsensitiveLoad(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{Insensitive: true}, source, others...)
}

// ShadowLoad has exactly same functionality as Load function
// except it allows have shadow keys.
func ShadowLoad(source interface{}, others ...interface{}) (*File, error) {
	return LoadSources(LoadOptions{AllowShadows: true}, source, others...)
}
