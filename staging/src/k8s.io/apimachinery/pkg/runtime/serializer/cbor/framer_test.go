/*
Copyright 2024 The Kubernetes Authors.

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

package cbor_test

import (
	"bytes"
	"errors"
	"io"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"

	"github.com/google/go-cmp/cmp"
)

// TestFrameReaderReadError tests that the frame reader does not resume after encountering a
// well-formedness error in the input stream. According to RFC 8742 Section 2.8: "[...] if any data
// item in the sequence is not well formed, it is not possible to reliably decode the rest of the
// sequence."
func TestFrameReaderReadError(t *testing.T) {
	input := []byte{
		0xff, // ill-formed initial break
		0xa0, // followed by well-formed empty map
	}
	fr := cbor.NewFramer().NewFrameReader(io.NopCloser(bytes.NewReader(input)))
	for i := 0; i < 3; i++ {
		n, err := fr.Read(nil)
		if err == nil || errors.Is(err, io.ErrShortBuffer) {
			t.Fatalf("expected a non-nil error other than io.ErrShortBuffer, got: %v", err)
		}
		if n != 0 {
			t.Fatalf("expected 0 bytes read on error, got %d", n)
		}
	}
}

func TestFrameReaderRead(t *testing.T) {
	type ChunkedFrame [][]byte

	for _, tc := range []struct {
		Name   string
		Frames []ChunkedFrame
	}{
		{
			Name: "consecutive frames",
			Frames: []ChunkedFrame{
				[][]byte{{0xa0}},
				[][]byte{{0xa0}},
			},
		},
		{
			Name: "zero-length destination buffer",
			Frames: []ChunkedFrame{
				[][]byte{{}, {0xa0}},
			},
		},
		{
			Name: "overflow",
			Frames: []ChunkedFrame{
				[][]byte{
					{0x43},
					{'x'},
					{'y', 'z'},
				},
				[][]byte{
					{0xa1, 0x43, 'f', 'o', 'o'},
					{'b'},
					{'a', 'r'},
				},
			},
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			var concatenation []byte
			for _, f := range tc.Frames {
				for _, c := range f {
					concatenation = append(concatenation, c...)
				}
			}

			fr := cbor.NewFramer().NewFrameReader(io.NopCloser(bytes.NewReader(concatenation)))

			for _, frame := range tc.Frames {
				var want, got []byte
				for i, chunk := range frame {
					dst := make([]byte, len(chunk), 2*len(chunk))
					for i := len(dst); i < cap(dst); i++ {
						dst[:cap(dst)][i] = 0xff
					}
					n, err := fr.Read(dst)
					if n != len(chunk) {
						t.Errorf("expected %d bytes read, got %d", len(chunk), n)
					}
					if i == len(frame)-1 && err != nil {
						t.Errorf("unexpected non-nil error on last read of frame: %v", err)
					} else if i < len(frame)-1 && !errors.Is(err, io.ErrShortBuffer) {
						t.Errorf("expected io.ErrShortBuffer on all but the last read of a frame, got: %v", err)
					}
					for i := len(dst); i < cap(dst); i++ {
						if dst[:cap(dst)][i] != 0xff {
							t.Errorf("read mutated underlying array beyond slice length: %#v", dst[len(dst):cap(dst)])
							break
						}
					}
					want = append(want, chunk...)
					got = append(got, dst...)
				}
				if diff := cmp.Diff(want, got); diff != "" {
					t.Errorf("reassembled frame differs:\n%s", diff)
				}
			}
		})
	}
}

type fakeReadCloser struct {
	err error
}

func (rc fakeReadCloser) Read(_ []byte) (int, error) {
	return 0, nil
}

func (rc fakeReadCloser) Close() error {
	return rc.err
}

func TestFrameReaderClose(t *testing.T) {
	want := errors.New("test")
	if got := cbor.NewFramer().NewFrameReader(fakeReadCloser{err: want}).Close(); !errors.Is(got, want) {
		t.Errorf("got error %v, want %v", got, want)
	}
}
