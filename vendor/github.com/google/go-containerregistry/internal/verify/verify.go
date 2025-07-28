// Copyright 2020 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package verify provides a ReadCloser that verifies content matches the
// expected hash values.
package verify

import (
	"bytes"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"

	"github.com/google/go-containerregistry/internal/and"
	v1 "github.com/google/go-containerregistry/pkg/v1"
)

// SizeUnknown is a sentinel value to indicate that the expected size is not known.
const SizeUnknown = -1

type verifyReader struct {
	inner             io.Reader
	hasher            hash.Hash
	expected          v1.Hash
	gotSize, wantSize int64
}

// Error provides information about the failed hash verification.
type Error struct {
	got     string
	want    v1.Hash
	gotSize int64
}

func (v Error) Error() string {
	return fmt.Sprintf("error verifying %s checksum after reading %d bytes; got %q, want %q",
		v.want.Algorithm, v.gotSize, v.got, v.want)
}

// Read implements io.Reader
func (vc *verifyReader) Read(b []byte) (int, error) {
	n, err := vc.inner.Read(b)
	vc.gotSize += int64(n)
	if err == io.EOF {
		if vc.wantSize != SizeUnknown && vc.gotSize != vc.wantSize {
			return n, fmt.Errorf("error verifying size; got %d, want %d", vc.gotSize, vc.wantSize)
		}
		got := hex.EncodeToString(vc.hasher.Sum(nil))
		if want := vc.expected.Hex; got != want {
			return n, Error{
				got:     vc.expected.Algorithm + ":" + got,
				want:    vc.expected,
				gotSize: vc.gotSize,
			}
		}
	}
	return n, err
}

// ReadCloser wraps the given io.ReadCloser to verify that its contents match
// the provided v1.Hash before io.EOF is returned.
//
// The reader will only be read up to size bytes, to prevent resource
// exhaustion. If EOF is returned before size bytes are read, an error is
// returned.
//
// A size of SizeUnknown (-1) indicates disables size verification when the size
// is unknown ahead of time.
func ReadCloser(r io.ReadCloser, size int64, h v1.Hash) (io.ReadCloser, error) {
	w, err := v1.Hasher(h.Algorithm)
	if err != nil {
		return nil, err
	}
	r2 := io.TeeReader(r, w) // pass all writes to the hasher.
	if size != SizeUnknown {
		r2 = io.LimitReader(r2, size) // if we know the size, limit to that size.
	}
	return &and.ReadCloser{
		Reader: &verifyReader{
			inner:    r2,
			hasher:   w,
			expected: h,
			wantSize: size,
		},
		CloseFunc: r.Close,
	}, nil
}

// Descriptor verifies that the embedded Data field matches the Size and Digest
// fields of the given v1.Descriptor, returning an error if the Data field is
// missing or if it contains incorrect data.
func Descriptor(d v1.Descriptor) error {
	if d.Data == nil {
		return errors.New("error verifying descriptor; Data == nil")
	}

	h, sz, err := v1.SHA256(bytes.NewReader(d.Data))
	if err != nil {
		return err
	}
	if h != d.Digest {
		return fmt.Errorf("error verifying Digest; got %q, want %q", h, d.Digest)
	}
	if sz != d.Size {
		return fmt.Errorf("error verifying Size; got %d, want %d", sz, d.Size)
	}

	return nil
}
