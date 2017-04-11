// Copyright 2015 CoreOS, Inc.
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

package unit

import (
	"bytes"
	"io"
)

// Serialize encodes all of the given UnitOption objects into a
// unit file. When serialized the options are sorted in their
// supplied order but grouped by section.
func Serialize(opts []*UnitOption) io.Reader {
	var buf bytes.Buffer

	if len(opts) == 0 {
		return &buf
	}

	// Index of sections -> ordered options
	idx := map[string][]*UnitOption{}
	// Separately preserve order in which sections were seen
	sections := []string{}
	for _, opt := range opts {
		sec := opt.Section
		if _, ok := idx[sec]; !ok {
			sections = append(sections, sec)
		}
		idx[sec] = append(idx[sec], opt)
	}

	for i, sect := range sections {
		writeSectionHeader(&buf, sect)
		writeNewline(&buf)

		opts := idx[sect]
		for _, opt := range opts {
			writeOption(&buf, opt)
			writeNewline(&buf)
		}
		if i < len(sections)-1 {
			writeNewline(&buf)
		}
	}

	return &buf
}

func writeNewline(buf *bytes.Buffer) {
	buf.WriteRune('\n')
}

func writeSectionHeader(buf *bytes.Buffer, section string) {
	buf.WriteRune('[')
	buf.WriteString(section)
	buf.WriteRune(']')
}

func writeOption(buf *bytes.Buffer, opt *UnitOption) {
	buf.WriteString(opt.Name)
	buf.WriteRune('=')
	buf.WriteString(opt.Value)
}
