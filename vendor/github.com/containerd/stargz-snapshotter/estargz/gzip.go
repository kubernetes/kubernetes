/*
   Copyright The containerd Authors.

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

/*
   Copyright 2019 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.
*/

package estargz

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"strconv"

	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

type gzipCompression struct {
	*GzipCompressor
	*GzipDecompressor
}

func newGzipCompressionWithLevel(level int) Compression {
	return &gzipCompression{
		&GzipCompressor{level},
		&GzipDecompressor{},
	}
}

func NewGzipCompressor() *GzipCompressor {
	return &GzipCompressor{gzip.BestCompression}
}

func NewGzipCompressorWithLevel(level int) *GzipCompressor {
	return &GzipCompressor{level}
}

type GzipCompressor struct {
	compressionLevel int
}

func (gc *GzipCompressor) Writer(w io.Writer) (io.WriteCloser, error) {
	return gzip.NewWriterLevel(w, gc.compressionLevel)
}

func (gc *GzipCompressor) WriteTOCAndFooter(w io.Writer, off int64, toc *JTOC, diffHash hash.Hash) (digest.Digest, error) {
	tocJSON, err := json.MarshalIndent(toc, "", "\t")
	if err != nil {
		return "", err
	}
	gz, _ := gzip.NewWriterLevel(w, gc.compressionLevel)
	gw := io.Writer(gz)
	if diffHash != nil {
		gw = io.MultiWriter(gz, diffHash)
	}
	tw := tar.NewWriter(gw)
	if err := tw.WriteHeader(&tar.Header{
		Typeflag: tar.TypeReg,
		Name:     TOCTarName,
		Size:     int64(len(tocJSON)),
	}); err != nil {
		return "", err
	}
	if _, err := tw.Write(tocJSON); err != nil {
		return "", err
	}

	if err := tw.Close(); err != nil {
		return "", err
	}
	if err := gz.Close(); err != nil {
		return "", err
	}
	if _, err := w.Write(gzipFooterBytes(off)); err != nil {
		return "", err
	}
	return digest.FromBytes(tocJSON), nil
}

// gzipFooterBytes returns the 51 bytes footer.
func gzipFooterBytes(tocOff int64) []byte {
	buf := bytes.NewBuffer(make([]byte, 0, FooterSize))
	gz, _ := gzip.NewWriterLevel(buf, gzip.NoCompression) // MUST be NoCompression to keep 51 bytes

	// Extra header indicating the offset of TOCJSON
	// https://tools.ietf.org/html/rfc1952#section-2.3.1.1
	header := make([]byte, 4)
	header[0], header[1] = 'S', 'G'
	subfield := fmt.Sprintf("%016xSTARGZ", tocOff)
	binary.LittleEndian.PutUint16(header[2:4], uint16(len(subfield))) // little-endian per RFC1952
	gz.Header.Extra = append(header, []byte(subfield)...)
	gz.Close()
	if buf.Len() != FooterSize {
		panic(fmt.Sprintf("footer buffer = %d, not %d", buf.Len(), FooterSize))
	}
	return buf.Bytes()
}

type GzipDecompressor struct{}

func (gz *GzipDecompressor) Reader(r io.Reader) (io.ReadCloser, error) {
	return gzip.NewReader(r)
}

func (gz *GzipDecompressor) ParseTOC(r io.Reader) (toc *JTOC, tocDgst digest.Digest, err error) {
	return parseTOCEStargz(r)
}

func (gz *GzipDecompressor) ParseFooter(p []byte) (blobPayloadSize, tocOffset, tocSize int64, err error) {
	if len(p) != FooterSize {
		return 0, 0, 0, fmt.Errorf("invalid length %d cannot be parsed", len(p))
	}
	zr, err := gzip.NewReader(bytes.NewReader(p))
	if err != nil {
		return 0, 0, 0, err
	}
	defer zr.Close()
	extra := zr.Header.Extra
	si1, si2, subfieldlen, subfield := extra[0], extra[1], extra[2:4], extra[4:]
	if si1 != 'S' || si2 != 'G' {
		return 0, 0, 0, fmt.Errorf("invalid subfield IDs: %q, %q; want E, S", si1, si2)
	}
	if slen := binary.LittleEndian.Uint16(subfieldlen); slen != uint16(16+len("STARGZ")) {
		return 0, 0, 0, fmt.Errorf("invalid length of subfield %d; want %d", slen, 16+len("STARGZ"))
	}
	if string(subfield[16:]) != "STARGZ" {
		return 0, 0, 0, fmt.Errorf("STARGZ magic string must be included in the footer subfield")
	}
	tocOffset, err = strconv.ParseInt(string(subfield[:16]), 16, 64)
	if err != nil {
		return 0, 0, 0, errors.Wrapf(err, "legacy: failed to parse toc offset")
	}
	return tocOffset, tocOffset, 0, nil
}

func (gz *GzipDecompressor) FooterSize() int64 {
	return FooterSize
}

func (gz *GzipDecompressor) DecompressTOC(r io.Reader) (tocJSON io.ReadCloser, err error) {
	return decompressTOCEStargz(r)
}

type LegacyGzipDecompressor struct{}

func (gz *LegacyGzipDecompressor) Reader(r io.Reader) (io.ReadCloser, error) {
	return gzip.NewReader(r)
}

func (gz *LegacyGzipDecompressor) ParseTOC(r io.Reader) (toc *JTOC, tocDgst digest.Digest, err error) {
	return parseTOCEStargz(r)
}

func (gz *LegacyGzipDecompressor) ParseFooter(p []byte) (blobPayloadSize, tocOffset, tocSize int64, err error) {
	if len(p) != legacyFooterSize {
		return 0, 0, 0, fmt.Errorf("legacy: invalid length %d cannot be parsed", len(p))
	}
	zr, err := gzip.NewReader(bytes.NewReader(p))
	if err != nil {
		return 0, 0, 0, errors.Wrapf(err, "legacy: failed to get footer gzip reader")
	}
	defer zr.Close()
	extra := zr.Header.Extra
	if len(extra) != 16+len("STARGZ") {
		return 0, 0, 0, fmt.Errorf("legacy: invalid stargz's extra field size")
	}
	if string(extra[16:]) != "STARGZ" {
		return 0, 0, 0, fmt.Errorf("legacy: magic string STARGZ not found")
	}
	tocOffset, err = strconv.ParseInt(string(extra[:16]), 16, 64)
	if err != nil {
		return 0, 0, 0, errors.Wrapf(err, "legacy: failed to parse toc offset")
	}
	return tocOffset, tocOffset, 0, nil
}

func (gz *LegacyGzipDecompressor) FooterSize() int64 {
	return legacyFooterSize
}

func (gz *LegacyGzipDecompressor) DecompressTOC(r io.Reader) (tocJSON io.ReadCloser, err error) {
	return decompressTOCEStargz(r)
}

func parseTOCEStargz(r io.Reader) (toc *JTOC, tocDgst digest.Digest, err error) {
	tr, err := decompressTOCEStargz(r)
	if err != nil {
		return nil, "", err
	}
	dgstr := digest.Canonical.Digester()
	toc = new(JTOC)
	if err := json.NewDecoder(io.TeeReader(tr, dgstr.Hash())).Decode(&toc); err != nil {
		return nil, "", fmt.Errorf("error decoding TOC JSON: %v", err)
	}
	if err := tr.Close(); err != nil {
		return nil, "", err
	}
	return toc, dgstr.Digest(), nil
}

func decompressTOCEStargz(r io.Reader) (tocJSON io.ReadCloser, err error) {
	zr, err := gzip.NewReader(r)
	if err != nil {
		return nil, fmt.Errorf("malformed TOC gzip header: %v", err)
	}
	zr.Multistream(false)
	tr := tar.NewReader(zr)
	h, err := tr.Next()
	if err != nil {
		return nil, fmt.Errorf("failed to find tar header in TOC gzip stream: %v", err)
	}
	if h.Name != TOCTarName {
		return nil, fmt.Errorf("TOC tar entry had name %q; expected %q", h.Name, TOCTarName)
	}
	return readCloser{tr, zr.Close}, nil
}
