/*
Copyright 2023 The Kubernetes Authors.

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

package prettyjson

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/fatih/color"
)

// Formatter is a struct to format JSON data. `color` is github.com/fatih/color: https://github.com/fatih/color
type Formatter struct {
	// JSON key color. Default is `color.New(color.FgBlue, color.Bold)`.
	KeyColor *color.Color

	// JSON string value color. Default is `color.New(color.FgGreen, color.Bold)`.
	StringColor *color.Color

	// JSON boolean value color. Default is `color.New(color.FgYellow, color.Bold)`.
	BoolColor *color.Color

	// JSON number value color. Default is `color.New(color.FgCyan, color.Bold)`.
	NumberColor *color.Color

	// JSON null value color. Default is `color.New(color.FgBlack, color.Bold)`.
	NullColor *color.Color

	// Max length of JSON string value. When the value is 1 and over, string is truncated to length of the value.
	// Default is 0 (not truncated).
	StringMaxLength int

	// Boolean to disable color. Default is false.
	DisabledColor bool

	// Indent space number. Default is 2.
	Indent int

	// Newline string. To print without new lines set it to empty string. Default is \n.
	Newline string
}

// NewFormatter returns a new formatter with following default values.
func NewFormatter() *Formatter {
	return &Formatter{
		KeyColor:        color.New(color.FgBlue, color.Bold),
		StringColor:     color.New(color.FgGreen, color.Bold),
		BoolColor:       color.New(color.FgYellow, color.Bold),
		NumberColor:     color.New(color.FgCyan, color.Bold),
		NullColor:       color.New(color.FgBlack, color.Bold),
		StringMaxLength: 0,
		DisabledColor:   false,
		Indent:          2,
		Newline:         "\n",
	}
}

// Marshal marshals and formats JSON data.
func (f *Formatter) Marshal(v interface{}) ([]byte, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}

	return f.Format(data)
}

// Format formats JSON string.
func (f *Formatter) Format(data []byte) ([]byte, error) {
	var v interface{}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&v); err != nil {
		return nil, err
	}

	return []byte(f.pretty(v, 1)), nil
}

func (f *Formatter) sprintfColor(c *color.Color, format string, args ...interface{}) string {
	if f.DisabledColor || c == nil {
		return fmt.Sprintf(format, args...)
	}
	return c.SprintfFunc()(format, args...)
}

func (f *Formatter) sprintColor(c *color.Color, s string) string {
	if f.DisabledColor || c == nil {
		return fmt.Sprint(s)
	}
	return c.SprintFunc()(s)
}

func (f *Formatter) pretty(v interface{}, depth int) string {
	switch val := v.(type) {
	case string:
		return f.processString(val)
	case float64:
		return f.sprintColor(f.NumberColor, strconv.FormatFloat(val, 'f', -1, 64))
	case json.Number:
		return f.sprintColor(f.NumberColor, string(val))
	case bool:
		return f.sprintColor(f.BoolColor, strconv.FormatBool(val))
	case nil:
		return f.sprintColor(f.NullColor, "null")
	case map[string]interface{}:
		return f.processMap(val, depth)
	case []interface{}:
		return f.processArray(val, depth)
	}

	return ""
}

func (f *Formatter) processString(s string) string {
	r := []rune(s)
	if f.StringMaxLength != 0 && len(r) >= f.StringMaxLength {
		s = string(r[0:f.StringMaxLength]) + "..."
	}

	buf := &bytes.Buffer{}
	encoder := json.NewEncoder(buf)
	encoder.SetEscapeHTML(false)
	encoder.Encode(s)
	s = string(buf.Bytes())
	s = strings.TrimSuffix(s, "\n")

	return f.sprintColor(f.StringColor, s)
}

func (f *Formatter) processMap(m map[string]interface{}, depth int) string {
	if len(m) == 0 {
		return "{}"
	}

	currentIndent := f.generateIndent(depth - 1)
	nextIndent := f.generateIndent(depth)
	rows := []string{}
	keys := []string{}

	for key := range m {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	for _, key := range keys {
		val := m[key]
		buf := &bytes.Buffer{}
		encoder := json.NewEncoder(buf)
		encoder.SetEscapeHTML(false)
		encoder.Encode(key)
		s := strings.TrimSuffix(string(buf.Bytes()), "\n")
		k := f.sprintColor(f.KeyColor, s)
		v := f.pretty(val, depth+1)

		valueIndent := " "
		if f.Newline == "" {
			valueIndent = ""
		}
		row := fmt.Sprintf("%s%s:%s%s", nextIndent, k, valueIndent, v)
		rows = append(rows, row)
	}

	return fmt.Sprintf("{%s%s%s%s}", f.Newline, strings.Join(rows, ","+f.Newline), f.Newline, currentIndent)
}

func (f *Formatter) processArray(a []interface{}, depth int) string {
	if len(a) == 0 {
		return "[]"
	}

	currentIndent := f.generateIndent(depth - 1)
	nextIndent := f.generateIndent(depth)
	rows := []string{}

	for _, val := range a {
		c := f.pretty(val, depth+1)
		row := nextIndent + c
		rows = append(rows, row)
	}
	return fmt.Sprintf("[%s%s%s%s]", f.Newline, strings.Join(rows, ","+f.Newline), f.Newline, currentIndent)
}

func (f *Formatter) generateIndent(depth int) string {
	return strings.Repeat(" ", f.Indent*depth)
}

// Marshal JSON data with default options.
func Marshal(v interface{}) ([]byte, error) {
	return NewFormatter().Marshal(v)
}

// Format JSON string with default options.
func Format(data []byte) ([]byte, error) {
	return NewFormatter().Format(data)
}
