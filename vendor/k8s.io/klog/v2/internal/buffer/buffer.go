// Copyright 2013 Google Inc. All Rights Reserved.
// Copyright 2022 The Kubernetes Authors.
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

// Package buffer provides a cache for byte.Buffer instances that can be reused
// to avoid frequent allocation and deallocation. It also has utility code
// for log header formatting that use these buffers.
package buffer

import (
	"bytes"
	"os"
	"sync"
	"time"

	"k8s.io/klog/v2/internal/severity"
)

var (
	// Pid is inserted into log headers. Can be overridden for tests.
	Pid = os.Getpid()
)

// Buffer holds a single byte.Buffer for reuse. The zero value is ready for
// use. It also provides some helper methods for output formatting.
type Buffer struct {
	bytes.Buffer
	Tmp  [64]byte // temporary byte array for creating headers.
	next *Buffer
}

// Buffers manages the reuse of individual buffer instances. It is thread-safe.
type Buffers struct {
	// mu protects the free list. It is separate from the main mutex
	// so buffers can be grabbed and printed to without holding the main lock,
	// for better parallelization.
	mu sync.Mutex

	// freeList is a list of byte buffers, maintained under mu.
	freeList *Buffer
}

// GetBuffer returns a new, ready-to-use buffer.
func (bl *Buffers) GetBuffer() *Buffer {
	bl.mu.Lock()
	b := bl.freeList
	if b != nil {
		bl.freeList = b.next
	}
	bl.mu.Unlock()
	if b == nil {
		b = new(Buffer)
	} else {
		b.next = nil
		b.Reset()
	}
	return b
}

// PutBuffer returns a buffer to the free list.
func (bl *Buffers) PutBuffer(b *Buffer) {
	if b.Len() >= 256 {
		// Let big buffers die a natural death.
		return
	}
	bl.mu.Lock()
	b.next = bl.freeList
	bl.freeList = b
	bl.mu.Unlock()
}

// Some custom tiny helper functions to print the log header efficiently.

const digits = "0123456789"

// twoDigits formats a zero-prefixed two-digit integer at buf.Tmp[i].
func (buf *Buffer) twoDigits(i, d int) {
	buf.Tmp[i+1] = digits[d%10]
	d /= 10
	buf.Tmp[i] = digits[d%10]
}

// nDigits formats an n-digit integer at buf.Tmp[i],
// padding with pad on the left.
// It assumes d >= 0.
func (buf *Buffer) nDigits(n, i, d int, pad byte) {
	j := n - 1
	for ; j >= 0 && d > 0; j-- {
		buf.Tmp[i+j] = digits[d%10]
		d /= 10
	}
	for ; j >= 0; j-- {
		buf.Tmp[i+j] = pad
	}
}

// someDigits formats a zero-prefixed variable-width integer at buf.Tmp[i].
func (buf *Buffer) someDigits(i, d int) int {
	// Print into the top, then copy down. We know there's space for at least
	// a 10-digit number.
	j := len(buf.Tmp)
	for {
		j--
		buf.Tmp[j] = digits[d%10]
		d /= 10
		if d == 0 {
			break
		}
	}
	return copy(buf.Tmp[i:], buf.Tmp[j:])
}

// FormatHeader formats a log header using the provided file name and line number.
func (buf *Buffer) FormatHeader(s severity.Severity, file string, line int, now time.Time) {
	if line < 0 {
		line = 0 // not a real line number, but acceptable to someDigits
	}
	if s > severity.FatalLog {
		s = severity.InfoLog // for safety.
	}

	// Avoid Fprintf, for speed. The format is so simple that we can do it quickly by hand.
	// It's worth about 3X. Fprintf is hard.
	_, month, day := now.Date()
	hour, minute, second := now.Clock()
	// Lmmdd hh:mm:ss.uuuuuu threadid file:line]
	buf.Tmp[0] = severity.Char[s]
	buf.twoDigits(1, int(month))
	buf.twoDigits(3, day)
	buf.Tmp[5] = ' '
	buf.twoDigits(6, hour)
	buf.Tmp[8] = ':'
	buf.twoDigits(9, minute)
	buf.Tmp[11] = ':'
	buf.twoDigits(12, second)
	buf.Tmp[14] = '.'
	buf.nDigits(6, 15, now.Nanosecond()/1000, '0')
	buf.Tmp[21] = ' '
	buf.nDigits(7, 22, Pid, ' ') // TODO: should be TID
	buf.Tmp[29] = ' '
	buf.Write(buf.Tmp[:30])
	buf.WriteString(file)
	buf.Tmp[0] = ':'
	n := buf.someDigits(1, line)
	buf.Tmp[n+1] = ']'
	buf.Tmp[n+2] = ' '
	buf.Write(buf.Tmp[:n+3])
}
