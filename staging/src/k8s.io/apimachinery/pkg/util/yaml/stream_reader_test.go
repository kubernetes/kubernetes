/*
Copyright 2025 The Kubernetes Authors.

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

package yaml

import (
	"io"
	"strings"
	"testing"
)

// srt = StreamReaderTest
type srtStep struct {
	op       string
	expected string
	err      error
	size     int
}

func srtRead(size int, expected string, err error) srtStep {
	return srtStep{op: "Read", size: size, expected: expected, err: err}
}
func srtReadN(size int, expected string, err error) srtStep {
	return srtStep{op: "ReadN", size: size, expected: expected, err: err}
}
func srtPeek(size int, expected string, err error) srtStep {
	return srtStep{op: "Peek", size: size, expected: expected, err: err}
}
func srtRewind() srtStep {
	return srtStep{op: "Rewind"}
}
func srtRewindN(size int) srtStep {
	return srtStep{op: "RewindN", size: size}
}
func srtConsume(size int) srtStep {
	return srtStep{op: "Consume", size: size}
}
func srtConsumed(exp int) srtStep {
	return srtStep{op: "Consumed", size: exp}
}

func srtRun(t *testing.T, reader *StreamReader, steps []srtStep) {
	t.Helper()

	checkRead := func(i int, step srtStep, buf []byte, err error) {
		t.Helper()
		if err != nil && step.err == nil {
			t.Errorf("step %d: unexpected error: %v", i, err)
		} else if err == nil && step.err != nil {
			t.Errorf("step %d: expected error %v", i, step.err)
		} else if err != nil && err != step.err { //nolint:errorlint
			t.Errorf("step %d: expected error %v, got %v", i, step.err, err)
		}
		if got := string(buf); got != step.expected {
			t.Errorf("step %d: expected %q, got %q", i, step.expected, got)
		}
	}

	for i, step := range steps {
		switch step.op {
		case "Read":
			buf := make([]byte, step.size)
			n, err := reader.Read(buf)
			buf = buf[:n]
			checkRead(i, step, buf, err)
		case "ReadN":
			buf, err := reader.ReadN(step.size)
			checkRead(i, step, buf, err)
		case "Peek":
			buf, err := reader.Peek(step.size)
			checkRead(i, step, buf, err)
		case "Rewind":
			reader.Rewind()
		case "RewindN":
			reader.RewindN(step.size)
		case "Consume":
			reader.Consume(step.size)
		case "Consumed":
			if n := reader.Consumed(); n != step.size {
				t.Errorf("step %d: expected %d consumed, got %d", i, step.size, n)
			}
		default:
			t.Fatalf("step %d: unknown operation %q", i, step.op)
		}
	}
}

func TestStreamReader_Read(t *testing.T) {
	tests := []struct {
		name  string
		input string
		steps []srtStep
	}{{
		name:  "empty input",
		input: "",
		steps: []srtStep{
			srtRead(1, "", io.EOF),
			srtRead(1, "", io.EOF), // still EOF
		},
	}, {
		name:  "simple reads",
		input: "0123456789",
		steps: []srtStep{
			srtRead(5, "01234", nil),
			srtRead(5, "56789", nil),
			srtRead(1, "", io.EOF),
		},
	}, {
		name:  "short read at EOF",
		input: "0123456789",
		steps: []srtStep{
			srtRead(8, "01234567", nil),
			srtRead(8, "89", nil), // short read, no error
			srtRead(1, "", io.EOF),
		},
	}, {
		name:  "short reads from buffer",
		input: "0123456789",
		steps: []srtStep{
			srtRead(3, "012", nil), // fill buffer
			srtRewind(),
			srtRead(4, "012", nil), // short read from buffer
			srtRewind(),
			srtRead(4, "012", nil),  // still short
			srtRead(4, "3456", nil), // from reader
			srtRewind(),
			srtRead(10, "0123456", nil), // short read from buffer
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := NewStreamReader(strings.NewReader(tt.input), 4) // small initial buffer
			srtRun(t, reader, tt.steps)
		})
	}
}

func TestStreamReader_Rewind(t *testing.T) {
	tests := []struct {
		name  string
		input string
		steps []srtStep
	}{{
		name:  "simple read and rewind",
		input: "0123456789",
		steps: []srtStep{
			srtRead(4, "0123", nil),
			srtRead(4, "4567", nil),
			srtRead(4, "89", nil),
			srtRead(1, "", io.EOF),
			srtRewind(),
			srtRead(4, "0123", nil),
			srtRead(4, "4567", nil),
			srtRead(4, "89", nil),
			srtRead(1, "", io.EOF),
		},
	}, {
		name:  "multiple rewinds",
		input: "01234",
		steps: []srtStep{
			srtRead(2, "01", nil),
			srtRewind(),
			srtRead(2, "01", nil),
			srtRead(2, "23", nil),
			srtRewind(),
			srtRead(2, "01", nil),
			srtRead(2, "23", nil),
			srtRead(2, "4", nil),
			srtRead(1, "", io.EOF),
			srtRewind(),
			srtRead(100, "01234", nil),
			srtRead(1, "", io.EOF),
		},
	}, {
		name:  "empty input",
		input: "",
		steps: []srtStep{
			srtRead(1, "", io.EOF),
			srtRewind(),
			srtRead(1, "", io.EOF),
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := NewStreamReader(strings.NewReader(tt.input), 4) // small initial buffer
			srtRun(t, reader, tt.steps)
		})
	}
}

func TestStreamReader_RewindN(t *testing.T) {
	tests := []struct {
		name  string
		input string
		steps []srtStep
	}{{
		name:  "simple rewindn",
		input: "0123456789",
		steps: []srtStep{
			srtRead(4, "0123", nil),
			srtRead(4, "4567", nil),
			srtRead(4, "89", nil),
			srtRead(1, "", io.EOF),
			srtRewindN(4),
			srtRead(2, "67", nil),
			srtRewindN(4),
			srtRead(10, "456789", nil),
			srtRead(1, "", io.EOF),
		},
	}, {
		name:  "empty input",
		input: "",
		steps: []srtStep{
			srtRead(1, "", io.EOF),
			srtRewindN(100),
			srtRead(1, "", io.EOF),
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := NewStreamReader(strings.NewReader(tt.input), 4) // small initial buffer
			srtRun(t, reader, tt.steps)
		})
	}
}

func TestStreamReader_Consume(t *testing.T) {
	tests := []struct {
		name  string
		input string
		steps []srtStep
	}{{
		name:  "simple consume",
		input: "0123456789",
		steps: []srtStep{
			srtConsumed(0),
			srtRead(4, "0123", nil),
			srtRead(4, "4567", nil),
			srtConsume(2), // drops 01
			srtConsumed(2),
			srtRead(4, "89", nil),
			srtRead(1, "", io.EOF),
			srtRewind(),
			srtRead(5, "23456", nil),
			srtRead(5, "789", nil),
			srtRead(1, "", io.EOF),
			srtConsumed(2),
		},
	}, {
		name:  "consume too much",
		input: "01234",
		steps: []srtStep{
			srtConsumed(0),
			srtRead(5, "01234", nil),
			srtConsume(5),
			srtConsumed(5),
			srtConsume(5),
			srtConsumed(5),
			srtRead(1, "", io.EOF),
			srtConsume(5),
			srtConsumed(5),
			srtRead(1, "", io.EOF),
			srtConsumed(5),
		},
	}, {
		name:  "empty input",
		input: "",
		steps: []srtStep{
			srtConsumed(0),
			srtConsume(5),
			srtRead(1, "", io.EOF),
			srtConsumed(0),
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := NewStreamReader(strings.NewReader(tt.input), 4) // small initial buffer
			srtRun(t, reader, tt.steps)
		})
	}
}

func TestStreamReader_ReadN(t *testing.T) {
	tests := []struct {
		name  string
		input string
		steps []srtStep
	}{{
		name:  "short read full readN",
		input: "0123456789",
		steps: []srtStep{
			srtRead(3, "012", nil), // fill buffer
			srtRewind(),
			srtRead(5, "012", nil), // short read from buffer
			srtRewind(),
			srtReadN(5, "01234", nil), // full readN
			srtRewind(),
			srtRead(10, "01234", nil), // short read from buffer
			srtRewind(),
			srtReadN(10, "0123456789", nil), // full readN
			srtRewind(),
			srtRead(10, "0123456789", nil), // full read from buffer
			srtRead(1, "", io.EOF),
		},
	}, {
		name:  "short read consume readN",
		input: "0123456789",
		steps: []srtStep{
			srtRead(3, "012", nil), // fill buffer
			srtRewind(),
			srtRead(4, "012", nil), // short read from buffer
			srtConsume(1),
			srtRewind(),
			srtRead(4, "12", nil), // short read from buffer
			srtRewind(),
			srtReadN(4, "1234", nil), // full read
			srtConsume(1),
			srtRewind(),
			srtRead(4, "234", nil), // short read from buffer
			srtRewind(),
			srtReadN(10, "23456789", io.EOF), // short readN, EOF
			srtRewind(),
			srtRead(10, "23456789", nil), // full read from buffer
			srtRead(1, "", io.EOF),
		},
	}, {
		name:  "empty input",
		input: "",
		steps: []srtStep{
			srtReadN(1, "", io.EOF),
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := NewStreamReader(strings.NewReader(tt.input), 4) // small initial buffer
			srtRun(t, reader, tt.steps)
		})
	}
}

func TestStreamReader_Peek(t *testing.T) {
	tests := []struct {
		name  string
		input string
		steps []srtStep
	}{{
		name:  "simple peek",
		input: "0123456789",
		steps: []srtStep{
			srtPeek(3, "012", nil), // fill buffer
			srtRead(5, "012", nil), // short read from buffer
			srtRewind(),
			srtPeek(6, "012345", nil),  // fill buffer
			srtRead(10, "012345", nil), // short read from buffer
		},
	}, {
		name:  "empty input",
		input: "",
		steps: []srtStep{
			srtPeek(1, "", io.EOF),
		},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := NewStreamReader(strings.NewReader(tt.input), 0)
			srtRun(t, reader, tt.steps)
		})
	}
}
