// Copyright 2013 Google Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pretty

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// a formatter stores stateful formatting information as well as being
// an io.Writer for simplicity.
type formatter struct {
	*bufio.Writer
	*Config

	// Self-referential structure tracking
	tagNumbers map[int]int // tagNumbers[id] = <#n>
}

// newFormatter creates a new buffered formatter.  For the output to be written
// to the given writer, this must be accompanied by a call to write (or Flush).
func newFormatter(cfg *Config, w io.Writer) *formatter {
	return &formatter{
		Writer:     bufio.NewWriter(w),
		Config:     cfg,
		tagNumbers: make(map[int]int),
	}
}

func (f *formatter) write(n node) {
	defer f.Flush()
	n.format(f, "")
}

func (f *formatter) tagFor(id int) int {
	if tag, ok := f.tagNumbers[id]; ok {
		return tag
	}
	if f.tagNumbers == nil {
		return 0
	}
	tag := len(f.tagNumbers) + 1
	f.tagNumbers[id] = tag
	return tag
}

type node interface {
	format(f *formatter, indent string)
}

func (f *formatter) compactString(n node) string {
	switch k := n.(type) {
	case stringVal:
		return string(k)
	case rawVal:
		return string(k)
	}

	buf := new(bytes.Buffer)
	f2 := newFormatter(&Config{Compact: true}, buf)
	f2.tagNumbers = f.tagNumbers // reuse tagNumbers just in case
	f2.write(n)
	return buf.String()
}

type stringVal string

func (str stringVal) format(f *formatter, indent string) {
	f.WriteString(strconv.Quote(string(str)))
}

type rawVal string

func (r rawVal) format(f *formatter, indent string) {
	f.WriteString(string(r))
}

type keyval struct {
	key string
	val node
}

type keyvals []keyval

func (l keyvals) format(f *formatter, indent string) {
	f.WriteByte('{')

	switch {
	case f.Compact:
		// All on one line:
		for i, kv := range l {
			if i > 0 {
				f.WriteByte(',')
			}
			f.WriteString(kv.key)
			f.WriteByte(':')
			kv.val.format(f, indent)
		}
	case f.Diffable:
		f.WriteByte('\n')
		inner := indent + " "
		// Each value gets its own line:
		for _, kv := range l {
			f.WriteString(inner)
			f.WriteString(kv.key)
			f.WriteString(": ")
			kv.val.format(f, inner)
			f.WriteString(",\n")
		}
		f.WriteString(indent)
	default:
		keyWidth := 0
		for _, kv := range l {
			if kw := len(kv.key); kw > keyWidth {
				keyWidth = kw
			}
		}
		alignKey := indent + " "
		alignValue := strings.Repeat(" ", keyWidth)
		inner := alignKey + alignValue + "  "
		// First and last line shared with bracket:
		for i, kv := range l {
			if i > 0 {
				f.WriteString(",\n")
				f.WriteString(alignKey)
			}
			f.WriteString(kv.key)
			f.WriteString(": ")
			f.WriteString(alignValue[len(kv.key):])
			kv.val.format(f, inner)
		}
	}

	f.WriteByte('}')
}

type list []node

func (l list) format(f *formatter, indent string) {
	if max := f.ShortList; max > 0 {
		short := f.compactString(l)
		if len(short) <= max {
			f.WriteString(short)
			return
		}
	}

	f.WriteByte('[')

	switch {
	case f.Compact:
		// All on one line:
		for i, v := range l {
			if i > 0 {
				f.WriteByte(',')
			}
			v.format(f, indent)
		}
	case f.Diffable:
		f.WriteByte('\n')
		inner := indent + " "
		// Each value gets its own line:
		for _, v := range l {
			f.WriteString(inner)
			v.format(f, inner)
			f.WriteString(",\n")
		}
		f.WriteString(indent)
	default:
		inner := indent + " "
		// First and last line shared with bracket:
		for i, v := range l {
			if i > 0 {
				f.WriteString(",\n")
				f.WriteString(inner)
			}
			v.format(f, inner)
		}
	}

	f.WriteByte(']')
}

type ref struct {
	id int
}

func (r ref) format(f *formatter, indent string) {
	fmt.Fprintf(f, "<see #%d>", f.tagFor(r.id))
}

type target struct {
	id    int
	value node
}

func (t target) format(f *formatter, indent string) {
	tag := fmt.Sprintf("<#%d> ", f.tagFor(t.id))
	switch {
	case f.Diffable, f.Compact:
		// no indent changes
	default:
		indent += strings.Repeat(" ", len(tag))
	}
	f.WriteString(tag)
	t.value.format(f, indent)
}
