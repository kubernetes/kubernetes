// Copyright 2015 The appc Authors
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

package aci

import (
	"archive/tar"
	"bytes"
	"compress/bzip2"
	"compress/gzip"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os/exec"
	"path/filepath"

	"github.com/appc/spec/schema"
)

type FileType string

const (
	TypeGzip    = FileType("gz")
	TypeBzip2   = FileType("bz2")
	TypeXz      = FileType("xz")
	TypeTar     = FileType("tar")
	TypeText    = FileType("text")
	TypeUnknown = FileType("unknown")

	readLen = 512 // max bytes to sniff

	hexHdrGzip  = "1f8b"
	hexHdrBzip2 = "425a68"
	hexHdrXz    = "fd377a585a00"
	hexSigTar   = "7573746172"

	tarOffset = 257

	textMime = "text/plain; charset=utf-8"
)

var (
	hdrGzip  []byte
	hdrBzip2 []byte
	hdrXz    []byte
	sigTar   []byte
	tarEnd   int
)

func mustDecodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func init() {
	hdrGzip = mustDecodeHex(hexHdrGzip)
	hdrBzip2 = mustDecodeHex(hexHdrBzip2)
	hdrXz = mustDecodeHex(hexHdrXz)
	sigTar = mustDecodeHex(hexSigTar)
	tarEnd = tarOffset + len(sigTar)
}

// DetectFileType attempts to detect the type of file that the given reader
// represents by comparing it against known file signatures (magic numbers)
func DetectFileType(r io.Reader) (FileType, error) {
	var b bytes.Buffer
	n, err := io.CopyN(&b, r, readLen)
	if err != nil && err != io.EOF {
		return TypeUnknown, err
	}
	bs := b.Bytes()
	switch {
	case bytes.HasPrefix(bs, hdrGzip):
		return TypeGzip, nil
	case bytes.HasPrefix(bs, hdrBzip2):
		return TypeBzip2, nil
	case bytes.HasPrefix(bs, hdrXz):
		return TypeXz, nil
	case n > int64(tarEnd) && bytes.Equal(bs[tarOffset:tarEnd], sigTar):
		return TypeTar, nil
	case http.DetectContentType(bs) == textMime:
		return TypeText, nil
	default:
		return TypeUnknown, nil
	}
}

// XzReader is an io.ReadCloser which decompresses xz compressed data.
type XzReader struct {
	io.ReadCloser
	cmd     *exec.Cmd
	closech chan error
}

// NewXzReader shells out to a command line xz executable (if
// available) to decompress the given io.Reader using the xz
// compression format and returns an *XzReader.
// It is the caller's responsibility to call Close on the XzReader when done.
func NewXzReader(r io.Reader) (*XzReader, error) {
	rpipe, wpipe := io.Pipe()
	ex, err := exec.LookPath("xz")
	if err != nil {
		log.Fatalf("couldn't find xz executable: %v", err)
	}
	cmd := exec.Command(ex, "--decompress", "--stdout")

	closech := make(chan error)

	cmd.Stdin = r
	cmd.Stdout = wpipe

	go func() {
		err := cmd.Run()
		wpipe.CloseWithError(err)
		closech <- err
	}()

	return &XzReader{rpipe, cmd, closech}, nil
}

func (r *XzReader) Close() error {
	r.ReadCloser.Close()
	r.cmd.Process.Kill()
	return <-r.closech
}

// ManifestFromImage extracts a new schema.ImageManifest from the given ACI image.
func ManifestFromImage(rs io.ReadSeeker) (*schema.ImageManifest, error) {
	var im schema.ImageManifest

	tr, err := NewCompressedTarReader(rs)
	if err != nil {
		return nil, err
	}
	defer tr.Close()

	for {
		hdr, err := tr.Next()
		switch err {
		case io.EOF:
			return nil, errors.New("missing manifest")
		case nil:
			if filepath.Clean(hdr.Name) == ManifestFile {
				data, err := ioutil.ReadAll(tr)
				if err != nil {
					return nil, err
				}
				if err := im.UnmarshalJSON(data); err != nil {
					return nil, err
				}
				return &im, nil
			}
		default:
			return nil, fmt.Errorf("error extracting tarball: %v", err)
		}
	}
}

// TarReadCloser embeds a *tar.Reader and the related io.Closer
// It is the caller's responsibility to call Close on TarReadCloser when
// done.
type TarReadCloser struct {
	*tar.Reader
	io.Closer
}

func (r *TarReadCloser) Close() error {
	return r.Closer.Close()
}

// NewCompressedTarReader creates a new TarReadCloser reading from the
// given ACI image.
// It is the caller's responsibility to call Close on the TarReadCloser
// when done.
func NewCompressedTarReader(rs io.ReadSeeker) (*TarReadCloser, error) {
	cr, err := NewCompressedReader(rs)
	if err != nil {
		return nil, err
	}
	return &TarReadCloser{tar.NewReader(cr), cr}, nil
}

// NewCompressedReader creates a new io.ReaderCloser from the given ACI image.
// It is the caller's responsibility to call Close on the Reader when done.
func NewCompressedReader(rs io.ReadSeeker) (io.ReadCloser, error) {

	var (
		dr  io.ReadCloser
		err error
	)

	_, err = rs.Seek(0, 0)
	if err != nil {
		return nil, err
	}

	ftype, err := DetectFileType(rs)
	if err != nil {
		return nil, err
	}

	_, err = rs.Seek(0, 0)
	if err != nil {
		return nil, err
	}

	switch ftype {
	case TypeGzip:
		dr, err = gzip.NewReader(rs)
		if err != nil {
			return nil, err
		}
	case TypeBzip2:
		dr = ioutil.NopCloser(bzip2.NewReader(rs))
	case TypeXz:
		dr, err = NewXzReader(rs)
		if err != nil {
			return nil, err
		}
	case TypeTar:
		dr = ioutil.NopCloser(rs)
	case TypeUnknown:
		return nil, errors.New("error: unknown image filetype")
	default:
		return nil, errors.New("no type returned from DetectFileType?")
	}
	return dr, nil
}
