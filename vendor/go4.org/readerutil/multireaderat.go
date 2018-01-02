/*
Copyright 2016 The go4 Authors

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

package readerutil

import (
	"io"
	"sort"
)

// NewMultiReaderAt is like io.MultiReader but produces a ReaderAt
// (and Size), instead of just a reader.
func NewMultiReaderAt(parts ...SizeReaderAt) SizeReaderAt {
	m := &multiRA{
		parts: make([]offsetAndSource, 0, len(parts)),
	}
	var off int64
	for _, p := range parts {
		m.parts = append(m.parts, offsetAndSource{off, p})
		off += p.Size()
	}
	m.size = off
	return m
}

type offsetAndSource struct {
	off int64
	SizeReaderAt
}

type multiRA struct {
	parts []offsetAndSource
	size  int64
}

func (m *multiRA) Size() int64 { return m.size }

func (m *multiRA) ReadAt(p []byte, off int64) (n int, err error) {
	wantN := len(p)

	// Skip past the requested offset.
	skipParts := sort.Search(len(m.parts), func(i int) bool {
		// This function returns whether parts[i] will
		// contribute any bytes to our output.
		part := m.parts[i]
		return part.off+part.Size() > off
	})
	parts := m.parts[skipParts:]

	// How far to skip in the first part.
	needSkip := off
	if len(parts) > 0 {
		needSkip -= parts[0].off
	}

	for len(parts) > 0 && len(p) > 0 {
		readP := p
		partSize := parts[0].Size()
		if int64(len(readP)) > partSize-needSkip {
			readP = readP[:partSize-needSkip]
		}
		pn, err0 := parts[0].ReadAt(readP, needSkip)
		if err0 != nil {
			return n, err0
		}
		n += pn
		p = p[pn:]
		if int64(pn)+needSkip == partSize {
			parts = parts[1:]
		}
		needSkip = 0
	}

	if n != wantN {
		err = io.ErrUnexpectedEOF
	}
	return
}
