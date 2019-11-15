/*
Copyright 2019 The Kubernetes Authors.

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

package strings

import (
	"bytes"
	"fmt"
	"io"
)

type readerWithStringTable struct {
	src io.Reader
	dst *bytes.Buffer

	stringTable [][]byte

	inputBuffer  []byte
	outputBuffer []byte

	readingIndex bool
	index        int

	inEscapeSequence bool
	inQuotes         bool
}

func NewReaderWithStringTable(r io.Reader) (io.Reader, error) {
	version, err := parseStringTableVersion(r)
	if err != nil {
		return nil, err
	}
	stringTable, err := getTable(version)
	if err != nil {
		return nil, err
	}
	return &readerWithStringTable{
		src:         r,
		dst:         bytes.NewBuffer(make([]byte, 0)),
		inputBuffer: make([]byte, 1024),
		stringTable: stringTable,
	}, nil
}

func parseStringTableVersion(r io.Reader) (int, error) {
	version := 0
	for {
		b := make([]byte, 1)
		if n, err := r.Read(b); n != 1 || err != nil {
			return 0, err
		} else if b[0] == byte(',') || b[0] == byte(':') || b[0] == byte('}') || b[0] == byte(']') {
			return version, nil
		} else if b[0] >= byte('0') && b[0] <= byte('9') {
			version = 10*version + int(b[0]) - int('0')
		} else {
			return 0, fmt.Errorf("expecting a digit between 0 and 9 but got: %v", b[0])
		}
	}
}

func (r *readerWithStringTable) Read(b []byte) (int, error) {
	for r.dst.Len() < len(b) {
		n, readErr := r.src.Read(r.inputBuffer)

		r.outputBuffer = r.outputBuffer[:0]
		for _, b := range r.inputBuffer[:n] {
			if err := r.parseByte(b); err != nil {
				return 0, err
			}
		}
		_, writeErr := r.dst.Write(r.outputBuffer)
		if writeErr != nil {
			return 0, writeErr
		}
		// EOF
		if readErr != nil {
			b = b[:r.dst.Len()]
		}
	}
	return r.dst.Read(b)
}

func (r *readerWithStringTable) parseByte(b byte) (err error) {
	if r.readingIndex {
		if b == byte(',') || b == byte(':') || b == byte('}') || b == byte(']') {
			if r.index >= len(r.stringTable) {
				return fmt.Errorf("unable to look up %v in the string table", r.index)
			}
			r.outputBuffer = append(r.outputBuffer, r.stringTable[r.index]...)
			r.outputBuffer = append(r.outputBuffer, b)
			r.readingIndex = false
		} else if b >= byte('A') && b <= byte('Z') {
			r.index = 64*r.index + int(b) - int('A')
		} else if b >= byte('a') && b <= byte('z') {
			r.index = 64*r.index + int(b) - int('a') + 26
		} else if b >= byte('0') && b <= byte('9') {
			r.index = 64*r.index + int(b) - int('0') + 52
		} else if b == byte('+') {
			r.index = 64*r.index + 62
		} else if b == byte('/') {
			r.index = 64*r.index + 63
		} else {
			return fmt.Errorf("expecting a digit in base64 but got: %v", b)
		}
	} else if !r.inQuotes && b == byte('!') {
		// Identify and parse an index of an item in the string table
		// This will start with a '!'.
		r.index = 0
		r.readingIndex = true
	} else {
		// Update the state of the parser so it knows what part of json it's reading
		r.inQuotes = !r.inQuotes && (b == byte('"')) || r.inQuotes && (r.inEscapeSequence || !(b == byte('"')))
		r.inEscapeSequence = !r.inEscapeSequence && b == byte('\\')
		r.outputBuffer = append(r.outputBuffer, b)
	}
	return nil
}
