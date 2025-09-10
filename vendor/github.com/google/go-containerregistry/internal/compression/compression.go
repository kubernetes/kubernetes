// Copyright 2022 Google LLC All Rights Reserved.
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

// Package compression abstracts over gzip and zstd.
package compression

import (
	"bufio"
	"bytes"
	"io"

	"github.com/google/go-containerregistry/internal/gzip"
	"github.com/google/go-containerregistry/internal/zstd"
	"github.com/google/go-containerregistry/pkg/compression"
)

// Opener represents e.g. opening a file.
type Opener = func() (io.ReadCloser, error)

// GetCompression detects whether an Opener is compressed and which algorithm is used.
func GetCompression(opener Opener) (compression.Compression, error) {
	rc, err := opener()
	if err != nil {
		return compression.None, err
	}
	defer rc.Close()

	cp, _, err := PeekCompression(rc)
	if err != nil {
		return compression.None, err
	}

	return cp, nil
}

// PeekCompression detects whether the input stream is compressed and which algorithm is used.
//
// If r implements Peek, we will use that directly, otherwise a small number
// of bytes are buffered to Peek at the gzip/zstd header, and the returned
// PeekReader can be used as a replacement for the consumed input io.Reader.
func PeekCompression(r io.Reader) (compression.Compression, PeekReader, error) {
	pr := intoPeekReader(r)

	if isGZip, _, err := checkHeader(pr, gzip.MagicHeader); err != nil {
		return compression.None, pr, err
	} else if isGZip {
		return compression.GZip, pr, nil
	}

	if isZStd, _, err := checkHeader(pr, zstd.MagicHeader); err != nil {
		return compression.None, pr, err
	} else if isZStd {
		return compression.ZStd, pr, nil
	}

	return compression.None, pr, nil
}

// PeekReader is an io.Reader that also implements Peek a la bufio.Reader.
type PeekReader interface {
	io.Reader
	Peek(n int) ([]byte, error)
}

// IntoPeekReader creates a PeekReader from an io.Reader.
// If the reader already has a Peek method, it will just return the passed reader.
func intoPeekReader(r io.Reader) PeekReader {
	if p, ok := r.(PeekReader); ok {
		return p
	}

	return bufio.NewReader(r)
}

// CheckHeader checks whether the first bytes from a PeekReader match an expected header
func checkHeader(pr PeekReader, expectedHeader []byte) (bool, PeekReader, error) {
	header, err := pr.Peek(len(expectedHeader))
	if err != nil {
		// https://github.com/google/go-containerregistry/issues/367
		if err == io.EOF {
			return false, pr, nil
		}
		return false, pr, err
	}
	return bytes.Equal(header, expectedHeader), pr, nil
}
