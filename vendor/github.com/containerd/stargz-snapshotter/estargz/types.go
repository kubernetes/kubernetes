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
	"hash"
	"io"
	"os"
	"path"
	"time"

	digest "github.com/opencontainers/go-digest"
)

const (
	// TOCTarName is the name of the JSON file in the tar archive in the
	// table of contents gzip stream.
	TOCTarName = "stargz.index.json"

	// FooterSize is the number of bytes in the footer
	//
	// The footer is an empty gzip stream with no compression and an Extra
	// header of the form "%016xSTARGZ", where the 64 bit hex-encoded
	// number is the offset to the gzip stream of JSON TOC.
	//
	// 51 comes from:
	//
	// 10 bytes  gzip header
	// 2  bytes  XLEN (length of Extra field) = 26 (4 bytes header + 16 hex digits + len("STARGZ"))
	// 2  bytes  Extra: SI1 = 'S', SI2 = 'G'
	// 2  bytes  Extra: LEN = 22 (16 hex digits + len("STARGZ"))
	// 22 bytes  Extra: subfield = fmt.Sprintf("%016xSTARGZ", offsetOfTOC)
	// 5  bytes  flate header
	// 8  bytes  gzip footer
	// (End of the eStargz blob)
	//
	// NOTE: For Extra fields, subfield IDs SI1='S' SI2='G' is used for eStargz.
	FooterSize = 51

	// legacyFooterSize is the number of bytes in the legacy stargz footer.
	//
	// 47 comes from:
	//
	//   10 byte gzip header +
	//   2 byte (LE16) length of extra, encoding 22 (16 hex digits + len("STARGZ")) == "\x16\x00" +
	//   22 bytes of extra (fmt.Sprintf("%016xSTARGZ", tocGzipOffset))
	//   5 byte flate header
	//   8 byte gzip footer (two little endian uint32s: digest, size)
	legacyFooterSize = 47

	// TOCJSONDigestAnnotation is an annotation for an image layer. This stores the
	// digest of the TOC JSON.
	// This annotation is valid only when it is specified in `.[]layers.annotations`
	// of an image manifest.
	TOCJSONDigestAnnotation = "containerd.io/snapshot/stargz/toc.digest"

	// StoreUncompressedSizeAnnotation is an additional annotation key for eStargz to enable lazy
	// pulling on containers/storage. Stargz Store is required to expose the layer's uncompressed size
	// to the runtime but current OCI image doesn't ship this information by default. So we store this
	// to the special annotation.
	StoreUncompressedSizeAnnotation = "io.containers.estargz.uncompressed-size"

	// PrefetchLandmark is a file entry which indicates the end position of
	// prefetch in the stargz file.
	PrefetchLandmark = ".prefetch.landmark"

	// NoPrefetchLandmark is a file entry which indicates that no prefetch should
	// occur in the stargz file.
	NoPrefetchLandmark = ".no.prefetch.landmark"

	landmarkContents = 0xf
)

// JTOC is the JSON-serialized table of contents index of the files in the stargz file.
type JTOC struct {
	Version int         `json:"version"`
	Entries []*TOCEntry `json:"entries"`
}

// TOCEntry is an entry in the stargz file's TOC (Table of Contents).
type TOCEntry struct {
	// Name is the tar entry's name. It is the complete path
	// stored in the tar file, not just the base name.
	Name string `json:"name"`

	// Type is one of "dir", "reg", "symlink", "hardlink", "char",
	// "block", "fifo", or "chunk".
	// The "chunk" type is used for regular file data chunks past the first
	// TOCEntry; the 2nd chunk and on have only Type ("chunk"), Offset,
	// ChunkOffset, and ChunkSize populated.
	Type string `json:"type"`

	// Size, for regular files, is the logical size of the file.
	Size int64 `json:"size,omitempty"`

	// ModTime3339 is the modification time of the tar entry. Empty
	// means zero or unknown. Otherwise it's in UTC RFC3339
	// format. Use the ModTime method to access the time.Time value.
	ModTime3339 string `json:"modtime,omitempty"`
	modTime     time.Time

	// LinkName, for symlinks and hardlinks, is the link target.
	LinkName string `json:"linkName,omitempty"`

	// Mode is the permission and mode bits.
	Mode int64 `json:"mode,omitempty"`

	// UID is the user ID of the owner.
	UID int `json:"uid,omitempty"`

	// GID is the group ID of the owner.
	GID int `json:"gid,omitempty"`

	// Uname is the username of the owner.
	//
	// In the serialized JSON, this field may only be present for
	// the first entry with the same UID.
	Uname string `json:"userName,omitempty"`

	// Gname is the group name of the owner.
	//
	// In the serialized JSON, this field may only be present for
	// the first entry with the same GID.
	Gname string `json:"groupName,omitempty"`

	// Offset, for regular files, provides the offset in the
	// stargz file to the file's data bytes. See ChunkOffset and
	// ChunkSize.
	Offset int64 `json:"offset,omitempty"`

	// InnerOffset is an optional field indicates uncompressed offset
	// of this "reg" or "chunk" payload in a stream starts from Offset.
	// This field enables to put multiple "reg" or "chunk" payloads
	// in one chunk with having the same Offset but different InnerOffset.
	InnerOffset int64 `json:"innerOffset,omitempty"`

	nextOffset int64 // the Offset of the next entry with a non-zero Offset

	// DevMajor is the major device number for "char" and "block" types.
	DevMajor int `json:"devMajor,omitempty"`

	// DevMinor is the major device number for "char" and "block" types.
	DevMinor int `json:"devMinor,omitempty"`

	// NumLink is the number of entry names pointing to this entry.
	// Zero means one name references this entry.
	// This field is calculated during runtime and not recorded in TOC JSON.
	NumLink int `json:"-"`

	// Xattrs are the extended attribute for the entry.
	Xattrs map[string][]byte `json:"xattrs,omitempty"`

	// Digest stores the OCI checksum for regular files payload.
	// It has the form "sha256:abcdef01234....".
	Digest string `json:"digest,omitempty"`

	// ChunkOffset is non-zero if this is a chunk of a large,
	// regular file. If so, the Offset is where the gzip header of
	// ChunkSize bytes at ChunkOffset in Name begin.
	//
	// In serialized form, a "chunkSize" JSON field of zero means
	// that the chunk goes to the end of the file. After reading
	// from the stargz TOC, though, the ChunkSize is initialized
	// to a non-zero file for when Type is either "reg" or
	// "chunk".
	ChunkOffset int64 `json:"chunkOffset,omitempty"`
	ChunkSize   int64 `json:"chunkSize,omitempty"`

	// ChunkDigest stores an OCI digest of the chunk. This must be formed
	// as "sha256:0123abcd...".
	ChunkDigest string `json:"chunkDigest,omitempty"`

	children map[string]*TOCEntry

	// chunkTopIndex is index of the entry where Offset starts in the blob.
	chunkTopIndex int
}

// ModTime returns the entry's modification time.
func (e *TOCEntry) ModTime() time.Time { return e.modTime }

// NextOffset returns the position (relative to the start of the
// stargz file) of the next gzip boundary after e.Offset.
func (e *TOCEntry) NextOffset() int64 { return e.nextOffset }

func (e *TOCEntry) addChild(baseName string, child *TOCEntry) {
	if e.children == nil {
		e.children = make(map[string]*TOCEntry)
	}
	if child.Type == "dir" {
		e.NumLink++ // Entry ".." in the subdirectory links to this directory
	}
	e.children[baseName] = child
}

// isDataType reports whether TOCEntry is a regular file or chunk (something that
// contains regular file data).
func (e *TOCEntry) isDataType() bool { return e.Type == "reg" || e.Type == "chunk" }

// Stat returns a FileInfo value representing e.
func (e *TOCEntry) Stat() os.FileInfo { return fileInfo{e} }

// ForeachChild calls f for each child item. If f returns false, iteration ends.
// If e is not a directory, f is not called.
func (e *TOCEntry) ForeachChild(f func(baseName string, ent *TOCEntry) bool) {
	for name, ent := range e.children {
		if !f(name, ent) {
			return
		}
	}
}

// LookupChild returns the directory e's child by its base name.
func (e *TOCEntry) LookupChild(baseName string) (child *TOCEntry, ok bool) {
	child, ok = e.children[baseName]
	return
}

// fileInfo implements os.FileInfo using the wrapped *TOCEntry.
type fileInfo struct{ e *TOCEntry }

var _ os.FileInfo = fileInfo{}

func (fi fileInfo) Name() string       { return path.Base(fi.e.Name) }
func (fi fileInfo) IsDir() bool        { return fi.e.Type == "dir" }
func (fi fileInfo) Size() int64        { return fi.e.Size }
func (fi fileInfo) ModTime() time.Time { return fi.e.ModTime() }
func (fi fileInfo) Sys() interface{}   { return fi.e }
func (fi fileInfo) Mode() (m os.FileMode) {
	// TOCEntry.Mode is tar.Header.Mode so we can understand the these bits using `tar` pkg.
	m = (&tar.Header{Mode: fi.e.Mode}).FileInfo().Mode() &
		(os.ModePerm | os.ModeSetuid | os.ModeSetgid | os.ModeSticky)
	switch fi.e.Type {
	case "dir":
		m |= os.ModeDir
	case "symlink":
		m |= os.ModeSymlink
	case "char":
		m |= os.ModeDevice | os.ModeCharDevice
	case "block":
		m |= os.ModeDevice
	case "fifo":
		m |= os.ModeNamedPipe
	}
	return m
}

// TOCEntryVerifier holds verifiers that are usable for verifying chunks contained
// in a eStargz blob.
type TOCEntryVerifier interface {

	// Verifier provides a content verifier that can be used for verifying the
	// contents of the specified TOCEntry.
	Verifier(ce *TOCEntry) (digest.Verifier, error)
}

// Compression provides the compression helper to be used creating and parsing eStargz.
// This package provides gzip-based Compression by default, but any compression
// algorithm (e.g. zstd) can be used as long as it implements Compression.
type Compression interface {
	Compressor
	Decompressor
}

// Compressor represents the helper mothods to be used for creating eStargz.
type Compressor interface {
	// Writer returns WriteCloser to be used for writing a chunk to eStargz.
	// Everytime a chunk is written, the WriteCloser is closed and Writer is
	// called again for writing the next chunk.
	//
	// The returned writer should implement "Flush() error" function that flushes
	// any pending compressed data to the underlying writer.
	Writer(w io.Writer) (WriteFlushCloser, error)

	// WriteTOCAndFooter is called to write JTOC to the passed Writer.
	// diffHash calculates the DiffID (uncompressed sha256 hash) of the blob
	// WriteTOCAndFooter can optionally write anything that affects DiffID calculation
	// (e.g. uncompressed TOC JSON).
	//
	// This function returns tocDgst that represents the digest of TOC that will be used
	// to verify this blob when it's parsed.
	WriteTOCAndFooter(w io.Writer, off int64, toc *JTOC, diffHash hash.Hash) (tocDgst digest.Digest, err error)
}

// Decompressor represents the helper mothods to be used for parsing eStargz.
type Decompressor interface {
	// Reader returns ReadCloser to be used for decompressing file payload.
	Reader(r io.Reader) (io.ReadCloser, error)

	// FooterSize returns the size of the footer of this blob.
	FooterSize() int64

	// ParseFooter parses the footer and returns the offset and (compressed) size of TOC.
	// payloadBlobSize is the (compressed) size of the blob payload (i.e. the size between
	// the top until the TOC JSON).
	//
	// If tocOffset < 0, we assume that TOC isn't contained in the blob and pass nil reader
	// to ParseTOC. We expect that ParseTOC acquire TOC from the external location and return it.
	//
	// tocSize is optional. If tocSize <= 0, it's by default the size of the range from tocOffset until the beginning of the
	// footer (blob size - tocOff - FooterSize).
	// If blobPayloadSize < 0, blobPayloadSize become the blob size.
	ParseFooter(p []byte) (blobPayloadSize, tocOffset, tocSize int64, err error)

	// ParseTOC parses TOC from the passed reader. The reader provides the partial contents
	// of the underlying blob that has the range specified by ParseFooter method.
	//
	// This function returns tocDgst that represents the digest of TOC that will be used
	// to verify this blob. This must match to the value returned from
	// Compressor.WriteTOCAndFooter that is used when creating this blob.
	//
	// If tocOffset returned by ParseFooter is < 0, we assume that TOC isn't contained in the blob.
	// Pass nil reader to ParseTOC then we expect that ParseTOC acquire TOC from the external location
	// and return it.
	ParseTOC(r io.Reader) (toc *JTOC, tocDgst digest.Digest, err error)
}

type WriteFlushCloser interface {
	io.WriteCloser
	Flush() error
}
