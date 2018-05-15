/*
Copyright 2017 The Kubernetes Authors.

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

package explain

import (
	"fmt"
	"io"
	"strings"
)

// Formatter helps you write with indentation, and can wrap text as needed.
type Formatter struct {
	IndentLevel int
	Wrap        int
	Writer      io.Writer
}

// Indent creates a new Formatter that will indent the code by that much more.
func (f Formatter) Indent(indent int) *Formatter {
	f.IndentLevel = f.IndentLevel + indent
	return &f
}

// Write writes a string with the indentation set for the
// Formatter. This is not wrapping text.
func (f *Formatter) Write(str string, a ...interface{}) error {
	// Don't indent empty lines
	if str == "" {
		_, err := io.WriteString(f.Writer, "\n")
		return err
	}

	indent := ""
	for i := 0; i < f.IndentLevel; i++ {
		indent = indent + " "
	}

	if len(a) > 0 {
		str = fmt.Sprintf(str, a...)
	}
	_, err := io.WriteString(f.Writer, indent+str+"\n")
	return err
}

// WriteWrapped writes a string with the indentation set for the
// Formatter, and wraps as needed.
func (f *Formatter) WriteWrapped(str string, a ...interface{}) error {
	if f.Wrap == 0 {
		return f.Write(str, a...)
	}
	text := fmt.Sprintf(str, a...)
	strs := wrapString(text, f.Wrap-f.IndentLevel)
	for _, substr := range strs {
		if err := f.Write(substr); err != nil {
			return err
		}
	}
	return nil
}

type line struct {
	wrap  int
	words []string
}

func (l *line) String() string {
	return strings.Join(l.words, " ")
}

func (l *line) Empty() bool {
	return len(l.words) == 0
}

func (l *line) Len() int {
	return len(l.String())
}

// Add adds the word to the line, returns true if we could, false if we
// didn't have enough room. It's always possible to add to an empty line.
func (l *line) Add(word string) bool {
	newLine := line{
		wrap:  l.wrap,
		words: append(l.words, word),
	}
	if newLine.Len() <= l.wrap || len(l.words) == 0 {
		l.words = newLine.words
		return true
	}
	return false
}

func wrapString(str string, wrap int) []string {
	words := strings.Fields(str)
	wrapped := []string{}
	l := line{wrap: wrap}

	for _, word := range words {
		if !l.Add(word) {
			wrapped = append(wrapped, l.String())
			l = line{wrap: wrap}
			if !l.Add(word) {
				panic("Couldn't add to empty line.")
			}
		}
	}
	if !l.Empty() {
		wrapped = append(wrapped, l.String())
	}
	return wrapped
}
