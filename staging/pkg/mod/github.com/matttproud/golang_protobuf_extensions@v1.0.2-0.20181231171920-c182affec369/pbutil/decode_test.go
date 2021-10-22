// Copyright 2016 Matt T. Proud
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

package pbutil

import (
	"bytes"
	"io"
	"testing"
	"testing/iotest"
)

func TestReadDelimitedIllegalVarint(t *testing.T) {
	t.Parallel()
	var tests = []struct {
		in  []byte
		n   int
		err error
	}{
		{
			in:  []byte{255, 255, 255, 255, 255},
			n:   5,
			err: errInvalidVarint,
		},
		{
			in:  []byte{255, 255, 255, 255, 255, 255},
			n:   5,
			err: errInvalidVarint,
		},
	}
	for _, test := range tests {
		n, err := ReadDelimited(bytes.NewReader(test.in), nil)
		if got, want := n, test.n; got != want {
			t.Errorf("ReadDelimited(%#v, nil) = %#v, ?; want = %v#, ?", test.in, got, want)
		}
		if got, want := err, test.err; got != want {
			t.Errorf("ReadDelimited(%#v, nil) = ?, %#v; want = ?, %#v", test.in, got, want)
		}
	}
}

func TestReadDelimitedPrematureHeader(t *testing.T) {
	t.Parallel()
	var data = []byte{128, 5} // 256 + 256 + 128
	n, err := ReadDelimited(bytes.NewReader(data[0:1]), nil)
	if got, want := n, 1; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = %#v, ?; want = %v#, ?", data[0:1], got, want)
	}
	if got, want := err, io.EOF; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = ?, %#v; want = ?, %#v", data[0:1], got, want)
	}
}

func TestReadDelimitedPrematureBody(t *testing.T) {
	t.Parallel()
	var data = []byte{128, 5, 0, 0, 0} // 256 + 256 + 128
	n, err := ReadDelimited(bytes.NewReader(data[:]), nil)
	if got, want := n, 5; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = %#v, ?; want = %v#, ?", data, got, want)
	}
	if got, want := err, io.ErrUnexpectedEOF; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = ?, %#v; want = ?, %#v", data, got, want)
	}
}

func TestReadDelimitedPrematureHeaderIncremental(t *testing.T) {
	t.Parallel()
	var data = []byte{128, 5} // 256 + 256 + 128
	n, err := ReadDelimited(iotest.OneByteReader(bytes.NewReader(data[0:1])), nil)
	if got, want := n, 1; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = %#v, ?; want = %v#, ?", data[0:1], got, want)
	}
	if got, want := err, io.EOF; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = ?, %#v; want = ?, %#v", data[0:1], got, want)
	}
}

func TestReadDelimitedPrematureBodyIncremental(t *testing.T) {
	t.Parallel()
	var data = []byte{128, 5, 0, 0, 0} // 256 + 256 + 128
	n, err := ReadDelimited(iotest.OneByteReader(bytes.NewReader(data[:])), nil)
	if got, want := n, 5; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = %#v, ?; want = %v#, ?", data, got, want)
	}
	if got, want := err, io.ErrUnexpectedEOF; got != want {
		t.Errorf("ReadDelimited(%#v, nil) = ?, %#v; want = ?, %#v", data, got, want)
	}
}
