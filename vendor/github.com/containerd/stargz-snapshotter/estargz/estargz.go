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
	"bufio"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"fmt"
	"hash"
	"io"
	"io/ioutil"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/containerd/stargz-snapshotter/estargz/errorutil"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
	"github.com/vbatts/tar-split/archive/tar"
)

// A Reader permits random access reads from a stargz file.
type Reader struct {
	sr        *io.SectionReader
	toc       *JTOC
	tocDigest digest.Digest

	// m stores all non-chunk entries, keyed by name.
	m map[string]*TOCEntry

	// chunks stores all TOCEntry values for regular files that
	// are split up. For a file with a single chunk, it's only
	// stored in m.
	chunks map[string][]*TOCEntry

	decompressor Decompressor
}

type openOpts struct {
	tocOffset     int64
	decompressors []Decompressor
	telemetry     *Telemetry
}

// OpenOption is an option used during opening the layer
type OpenOption func(o *openOpts) error

// WithTOCOffset option specifies the offset of TOC
func WithTOCOffset(tocOffset int64) OpenOption {
	return func(o *openOpts) error {
		o.tocOffset = tocOffset
		return nil
	}
}

// WithDecompressors option specifies decompressors to use.
// Default is gzip-based decompressor.
func WithDecompressors(decompressors ...Decompressor) OpenOption {
	return func(o *openOpts) error {
		o.decompressors = decompressors
		return nil
	}
}

// WithTelemetry option specifies the telemetry hooks
func WithTelemetry(telemetry *Telemetry) OpenOption {
	return func(o *openOpts) error {
		o.telemetry = telemetry
		return nil
	}
}

// MeasureLatencyHook is a func which takes start time and records the diff
type MeasureLatencyHook func(time.Time)

// Telemetry is a struct which defines telemetry hooks. By implementing these hooks you should be able to record
// the latency metrics of the respective steps of estargz open operation. To be used with estargz.OpenWithTelemetry(...)
type Telemetry struct {
	GetFooterLatency      MeasureLatencyHook // measure time to get stargz footer (in milliseconds)
	GetTocLatency         MeasureLatencyHook // measure time to GET TOC JSON (in milliseconds)
	DeserializeTocLatency MeasureLatencyHook // measure time to deserialize TOC JSON (in milliseconds)
}

// Open opens a stargz file for reading.
// The behaviour is configurable using options.
//
// Note that each entry name is normalized as the path that is relative to root.
func Open(sr *io.SectionReader, opt ...OpenOption) (*Reader, error) {
	var opts openOpts
	for _, o := range opt {
		if err := o(&opts); err != nil {
			return nil, err
		}
	}

	gzipCompressors := []Decompressor{new(GzipDecompressor), new(LegacyGzipDecompressor)}
	decompressors := append(gzipCompressors, opts.decompressors...)

	// Determine the size to fetch. Try to fetch as many bytes as possible.
	fetchSize := maxFooterSize(sr.Size(), decompressors...)
	if maybeTocOffset := opts.tocOffset; maybeTocOffset > fetchSize {
		if maybeTocOffset > sr.Size() {
			return nil, fmt.Errorf("blob size %d is smaller than the toc offset", sr.Size())
		}
		fetchSize = sr.Size() - maybeTocOffset
	}

	start := time.Now() // before getting layer footer
	footer := make([]byte, fetchSize)
	if _, err := sr.ReadAt(footer, sr.Size()-fetchSize); err != nil {
		return nil, fmt.Errorf("error reading footer: %v", err)
	}
	if opts.telemetry != nil && opts.telemetry.GetFooterLatency != nil {
		opts.telemetry.GetFooterLatency(start)
	}

	var allErr []error
	var found bool
	var r *Reader
	for _, d := range decompressors {
		fSize := d.FooterSize()
		fOffset := positive(int64(len(footer)) - fSize)
		maybeTocBytes := footer[:fOffset]
		_, tocOffset, tocSize, err := d.ParseFooter(footer[fOffset:])
		if err != nil {
			allErr = append(allErr, err)
			continue
		}
		if tocSize <= 0 {
			tocSize = sr.Size() - tocOffset - fSize
		}
		if tocSize < int64(len(maybeTocBytes)) {
			maybeTocBytes = maybeTocBytes[:tocSize]
		}
		r, err = parseTOC(d, sr, tocOffset, tocSize, maybeTocBytes, opts)
		if err == nil {
			found = true
			break
		}
		allErr = append(allErr, err)
	}
	if !found {
		return nil, errorutil.Aggregate(allErr)
	}
	if err := r.initFields(); err != nil {
		return nil, fmt.Errorf("failed to initialize fields of entries: %v", err)
	}
	return r, nil
}

// OpenFooter extracts and parses footer from the given blob.
// only supports gzip-based eStargz.
func OpenFooter(sr *io.SectionReader) (tocOffset int64, footerSize int64, rErr error) {
	if sr.Size() < FooterSize && sr.Size() < legacyFooterSize {
		return 0, 0, fmt.Errorf("blob size %d is smaller than the footer size", sr.Size())
	}
	var footer [FooterSize]byte
	if _, err := sr.ReadAt(footer[:], sr.Size()-FooterSize); err != nil {
		return 0, 0, fmt.Errorf("error reading footer: %v", err)
	}
	var allErr []error
	for _, d := range []Decompressor{new(GzipDecompressor), new(LegacyGzipDecompressor)} {
		fSize := d.FooterSize()
		fOffset := positive(int64(len(footer)) - fSize)
		_, tocOffset, _, err := d.ParseFooter(footer[fOffset:])
		if err == nil {
			return tocOffset, fSize, err
		}
		allErr = append(allErr, err)
	}
	return 0, 0, errorutil.Aggregate(allErr)
}

// initFields populates the Reader from r.toc after decoding it from
// JSON.
//
// Unexported fields are populated and TOCEntry fields that were
// implicit in the JSON are populated.
func (r *Reader) initFields() error {
	r.m = make(map[string]*TOCEntry, len(r.toc.Entries))
	r.chunks = make(map[string][]*TOCEntry)
	var lastPath string
	uname := map[int]string{}
	gname := map[int]string{}
	var lastRegEnt *TOCEntry
	for _, ent := range r.toc.Entries {
		ent.Name = cleanEntryName(ent.Name)
		if ent.Type == "reg" {
			lastRegEnt = ent
		}
		if ent.Type == "chunk" {
			ent.Name = lastPath
			r.chunks[ent.Name] = append(r.chunks[ent.Name], ent)
			if ent.ChunkSize == 0 && lastRegEnt != nil {
				ent.ChunkSize = lastRegEnt.Size - ent.ChunkOffset
			}
		} else {
			lastPath = ent.Name

			if ent.Uname != "" {
				uname[ent.UID] = ent.Uname
			} else {
				ent.Uname = uname[ent.UID]
			}
			if ent.Gname != "" {
				gname[ent.GID] = ent.Gname
			} else {
				ent.Gname = uname[ent.GID]
			}

			ent.modTime, _ = time.Parse(time.RFC3339, ent.ModTime3339)

			if ent.Type == "dir" {
				ent.NumLink++ // Parent dir links to this directory
			}
			r.m[ent.Name] = ent
		}
		if ent.Type == "reg" && ent.ChunkSize > 0 && ent.ChunkSize < ent.Size {
			r.chunks[ent.Name] = make([]*TOCEntry, 0, ent.Size/ent.ChunkSize+1)
			r.chunks[ent.Name] = append(r.chunks[ent.Name], ent)
		}
		if ent.ChunkSize == 0 && ent.Size != 0 {
			ent.ChunkSize = ent.Size
		}
	}

	// Populate children, add implicit directories:
	for _, ent := range r.toc.Entries {
		if ent.Type == "chunk" {
			continue
		}
		// add "foo/":
		//    add "foo" child to "" (creating "" if necessary)
		//
		// add "foo/bar/":
		//    add "bar" child to "foo" (creating "foo" if necessary)
		//
		// add "foo/bar.txt":
		//    add "bar.txt" child to "foo" (creating "foo" if necessary)
		//
		// add "a/b/c/d/e/f.txt":
		//    create "a/b/c/d/e" node
		//    add "f.txt" child to "e"

		name := ent.Name
		pdirName := parentDir(name)
		if name == pdirName {
			// This entry and its parent are the same.
			// Ignore this for avoiding infinite loop of the reference.
			// The example case where this can occur is when tar contains the root
			// directory itself (e.g. "./", "/").
			continue
		}
		pdir := r.getOrCreateDir(pdirName)
		ent.NumLink++ // at least one name(ent.Name) references this entry.
		if ent.Type == "hardlink" {
			org, err := r.getSource(ent)
			if err != nil {
				return err
			}
			org.NumLink++ // original entry is referenced by this ent.Name.
			ent = org
		}
		pdir.addChild(path.Base(name), ent)
	}

	lastOffset := r.sr.Size()
	for i := len(r.toc.Entries) - 1; i >= 0; i-- {
		e := r.toc.Entries[i]
		if e.isDataType() {
			e.nextOffset = lastOffset
		}
		if e.Offset != 0 {
			lastOffset = e.Offset
		}
	}

	return nil
}

func (r *Reader) getSource(ent *TOCEntry) (_ *TOCEntry, err error) {
	if ent.Type == "hardlink" {
		org, ok := r.m[cleanEntryName(ent.LinkName)]
		if !ok {
			return nil, fmt.Errorf("%q is a hardlink but the linkname %q isn't found", ent.Name, ent.LinkName)
		}
		ent, err = r.getSource(org)
		if err != nil {
			return nil, err
		}
	}
	return ent, nil
}

func parentDir(p string) string {
	dir, _ := path.Split(p)
	return strings.TrimSuffix(dir, "/")
}

func (r *Reader) getOrCreateDir(d string) *TOCEntry {
	e, ok := r.m[d]
	if !ok {
		e = &TOCEntry{
			Name:    d,
			Type:    "dir",
			Mode:    0755,
			NumLink: 2, // The directory itself(.) and the parent link to this directory.
		}
		r.m[d] = e
		if d != "" {
			pdir := r.getOrCreateDir(parentDir(d))
			pdir.addChild(path.Base(d), e)
		}
	}
	return e
}

func (r *Reader) TOCDigest() digest.Digest {
	return r.tocDigest
}

// VerifyTOC checks that the TOC JSON in the passed blob matches the
// passed digests and that the TOC JSON contains digests for all chunks
// contained in the blob. If the verification succceeds, this function
// returns TOCEntryVerifier which holds all chunk digests in the stargz blob.
func (r *Reader) VerifyTOC(tocDigest digest.Digest) (TOCEntryVerifier, error) {
	// Verify the digest of TOC JSON
	if r.tocDigest != tocDigest {
		return nil, fmt.Errorf("invalid TOC JSON %q; want %q", r.tocDigest, tocDigest)
	}
	return r.Verifiers()
}

// Verifiers returns TOCEntryVerifier of this chunk. Use VerifyTOC instead in most cases
// because this doesn't verify TOC.
func (r *Reader) Verifiers() (TOCEntryVerifier, error) {
	chunkDigestMap := make(map[int64]digest.Digest) // map from chunk offset to the chunk digest
	regDigestMap := make(map[int64]digest.Digest)   // map from chunk offset to the reg file digest
	var chunkDigestMapIncomplete bool
	var regDigestMapIncomplete bool
	var containsChunk bool
	for _, e := range r.toc.Entries {
		if e.Type != "reg" && e.Type != "chunk" {
			continue
		}

		// offset must be unique in stargz blob
		_, dOK := chunkDigestMap[e.Offset]
		_, rOK := regDigestMap[e.Offset]
		if dOK || rOK {
			return nil, fmt.Errorf("offset %d found twice", e.Offset)
		}

		if e.Type == "reg" {
			if e.Size == 0 {
				continue // ignores empty file
			}

			// record the digest of regular file payload
			if e.Digest != "" {
				d, err := digest.Parse(e.Digest)
				if err != nil {
					return nil, errors.Wrapf(err,
						"failed to parse regular file digest %q", e.Digest)
				}
				regDigestMap[e.Offset] = d
			} else {
				regDigestMapIncomplete = true
			}
		} else {
			containsChunk = true // this layer contains "chunk" entries.
		}

		// "reg" also can contain ChunkDigest (e.g. when "reg" is the first entry of
		// chunked file)
		if e.ChunkDigest != "" {
			d, err := digest.Parse(e.ChunkDigest)
			if err != nil {
				return nil, errors.Wrapf(err,
					"failed to parse chunk digest %q", e.ChunkDigest)
			}
			chunkDigestMap[e.Offset] = d
		} else {
			chunkDigestMapIncomplete = true
		}
	}

	if chunkDigestMapIncomplete {
		// Though some chunk digests are not found, if this layer doesn't contain
		// "chunk"s and all digest of "reg" files are recorded, we can use them instead.
		if !containsChunk && !regDigestMapIncomplete {
			return &verifier{digestMap: regDigestMap}, nil
		}
		return nil, fmt.Errorf("some ChunkDigest not found in TOC JSON")
	}

	return &verifier{digestMap: chunkDigestMap}, nil
}

// verifier is an implementation of TOCEntryVerifier which holds verifiers keyed by
// offset of the chunk.
type verifier struct {
	digestMap   map[int64]digest.Digest
	digestMapMu sync.Mutex
}

// Verifier returns a content verifier specified by TOCEntry.
func (v *verifier) Verifier(ce *TOCEntry) (digest.Verifier, error) {
	v.digestMapMu.Lock()
	defer v.digestMapMu.Unlock()
	d, ok := v.digestMap[ce.Offset]
	if !ok {
		return nil, fmt.Errorf("verifier for offset=%d,size=%d hasn't been registered",
			ce.Offset, ce.ChunkSize)
	}
	return d.Verifier(), nil
}

// ChunkEntryForOffset returns the TOCEntry containing the byte of the
// named file at the given offset within the file.
// Name must be absolute path or one that is relative to root.
func (r *Reader) ChunkEntryForOffset(name string, offset int64) (e *TOCEntry, ok bool) {
	name = cleanEntryName(name)
	e, ok = r.Lookup(name)
	if !ok || !e.isDataType() {
		return nil, false
	}
	ents := r.chunks[name]
	if len(ents) < 2 {
		if offset >= e.ChunkSize {
			return nil, false
		}
		return e, true
	}
	i := sort.Search(len(ents), func(i int) bool {
		e := ents[i]
		return e.ChunkOffset >= offset || (offset > e.ChunkOffset && offset < e.ChunkOffset+e.ChunkSize)
	})
	if i == len(ents) {
		return nil, false
	}
	return ents[i], true
}

// Lookup returns the Table of Contents entry for the given path.
//
// To get the root directory, use the empty string.
// Path must be absolute path or one that is relative to root.
func (r *Reader) Lookup(path string) (e *TOCEntry, ok bool) {
	path = cleanEntryName(path)
	if r == nil {
		return
	}
	e, ok = r.m[path]
	if ok && e.Type == "hardlink" {
		var err error
		e, err = r.getSource(e)
		if err != nil {
			return nil, false
		}
	}
	return
}

// OpenFile returns the reader of the specified file payload.
//
// Name must be absolute path or one that is relative to root.
func (r *Reader) OpenFile(name string) (*io.SectionReader, error) {
	name = cleanEntryName(name)
	ent, ok := r.Lookup(name)
	if !ok {
		// TODO: come up with some error plan. This is lazy:
		return nil, &os.PathError{
			Path: name,
			Op:   "OpenFile",
			Err:  os.ErrNotExist,
		}
	}
	if ent.Type != "reg" {
		return nil, &os.PathError{
			Path: name,
			Op:   "OpenFile",
			Err:  errors.New("not a regular file"),
		}
	}
	fr := &fileReader{
		r:    r,
		size: ent.Size,
		ents: r.getChunks(ent),
	}
	return io.NewSectionReader(fr, 0, fr.size), nil
}

func (r *Reader) getChunks(ent *TOCEntry) []*TOCEntry {
	if ents, ok := r.chunks[ent.Name]; ok {
		return ents
	}
	return []*TOCEntry{ent}
}

type fileReader struct {
	r    *Reader
	size int64
	ents []*TOCEntry // 1 or more reg/chunk entries
}

func (fr *fileReader) ReadAt(p []byte, off int64) (n int, err error) {
	if off >= fr.size {
		return 0, io.EOF
	}
	if off < 0 {
		return 0, errors.New("invalid offset")
	}
	var i int
	if len(fr.ents) > 1 {
		i = sort.Search(len(fr.ents), func(i int) bool {
			return fr.ents[i].ChunkOffset >= off
		})
		if i == len(fr.ents) {
			i = len(fr.ents) - 1
		}
	}
	ent := fr.ents[i]
	if ent.ChunkOffset > off {
		if i == 0 {
			return 0, errors.New("internal error; first chunk offset is non-zero")
		}
		ent = fr.ents[i-1]
	}

	//  If ent is a chunk of a large file, adjust the ReadAt
	//  offset by the chunk's offset.
	off -= ent.ChunkOffset

	finalEnt := fr.ents[len(fr.ents)-1]
	compressedOff := ent.Offset
	// compressedBytesRemain is the number of compressed bytes in this
	// file remaining, over 1+ chunks.
	compressedBytesRemain := finalEnt.NextOffset() - compressedOff

	sr := io.NewSectionReader(fr.r.sr, compressedOff, compressedBytesRemain)

	const maxRead = 2 << 20
	var bufSize = maxRead
	if compressedBytesRemain < maxRead {
		bufSize = int(compressedBytesRemain)
	}

	br := bufio.NewReaderSize(sr, bufSize)
	if _, err := br.Peek(bufSize); err != nil {
		return 0, fmt.Errorf("fileReader.ReadAt.peek: %v", err)
	}

	dr, err := fr.r.decompressor.Reader(br)
	if err != nil {
		return 0, fmt.Errorf("fileReader.ReadAt.decompressor.Reader: %v", err)
	}
	defer dr.Close()
	if n, err := io.CopyN(ioutil.Discard, dr, off); n != off || err != nil {
		return 0, fmt.Errorf("discard of %d bytes = %v, %v", off, n, err)
	}
	return io.ReadFull(dr, p)
}

// A Writer writes stargz files.
//
// Use NewWriter to create a new Writer.
type Writer struct {
	bw       *bufio.Writer
	cw       *countWriter
	toc      *JTOC
	diffHash hash.Hash // SHA-256 of uncompressed tar

	closed        bool
	gz            io.WriteCloser
	lastUsername  map[int]string
	lastGroupname map[int]string
	compressor    Compressor

	// ChunkSize optionally controls the maximum number of bytes
	// of data of a regular file that can be written in one gzip
	// stream before a new gzip stream is started.
	// Zero means to use a default, currently 4 MiB.
	ChunkSize int
}

// currentCompressionWriter writes to the current w.gz field, which can
// change throughout writing a tar entry.
//
// Additionally, it updates w's SHA-256 of the uncompressed bytes
// of the tar file.
type currentCompressionWriter struct{ w *Writer }

func (ccw currentCompressionWriter) Write(p []byte) (int, error) {
	ccw.w.diffHash.Write(p)
	if ccw.w.gz == nil {
		if err := ccw.w.condOpenGz(); err != nil {
			return 0, err
		}
	}
	return ccw.w.gz.Write(p)
}

func (w *Writer) chunkSize() int {
	if w.ChunkSize <= 0 {
		return 4 << 20
	}
	return w.ChunkSize
}

// Unpack decompresses the given estargz blob and returns a ReadCloser of the tar blob.
// TOC JSON and footer are removed.
func Unpack(sr *io.SectionReader, c Decompressor) (io.ReadCloser, error) {
	footerSize := c.FooterSize()
	if sr.Size() < footerSize {
		return nil, fmt.Errorf("blob is too small; %d < %d", sr.Size(), footerSize)
	}
	footerOffset := sr.Size() - footerSize
	footer := make([]byte, footerSize)
	if _, err := sr.ReadAt(footer, footerOffset); err != nil {
		return nil, err
	}
	blobPayloadSize, _, _, err := c.ParseFooter(footer)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse footer")
	}
	return c.Reader(io.LimitReader(sr, blobPayloadSize))
}

// NewWriter returns a new stargz writer (gzip-based) writing to w.
//
// The writer must be closed to write its trailing table of contents.
func NewWriter(w io.Writer) *Writer {
	return NewWriterLevel(w, gzip.BestCompression)
}

// NewWriterLevel returns a new stargz writer (gzip-based) writing to w.
// The compression level is configurable.
//
// The writer must be closed to write its trailing table of contents.
func NewWriterLevel(w io.Writer, compressionLevel int) *Writer {
	return NewWriterWithCompressor(w, NewGzipCompressorWithLevel(compressionLevel))
}

// NewWriterWithCompressor returns a new stargz writer writing to w.
// The compression method is configurable.
//
// The writer must be closed to write its trailing table of contents.
func NewWriterWithCompressor(w io.Writer, c Compressor) *Writer {
	bw := bufio.NewWriter(w)
	cw := &countWriter{w: bw}
	return &Writer{
		bw:         bw,
		cw:         cw,
		toc:        &JTOC{Version: 1},
		diffHash:   sha256.New(),
		compressor: c,
	}
}

// Close writes the stargz's table of contents and flushes all the
// buffers, returning any error.
func (w *Writer) Close() (digest.Digest, error) {
	if w.closed {
		return "", nil
	}
	defer func() { w.closed = true }()

	if err := w.closeGz(); err != nil {
		return "", err
	}

	// Write the TOC index and footer.
	tocDigest, err := w.compressor.WriteTOCAndFooter(w.cw, w.cw.n, w.toc, w.diffHash)
	if err != nil {
		return "", err
	}
	if err := w.bw.Flush(); err != nil {
		return "", err
	}

	return tocDigest, nil
}

func (w *Writer) closeGz() error {
	if w.closed {
		return errors.New("write on closed Writer")
	}
	if w.gz != nil {
		if err := w.gz.Close(); err != nil {
			return err
		}
		w.gz = nil
	}
	return nil
}

// nameIfChanged returns name, unless it was the already the value of (*mp)[id],
// in which case it returns the empty string.
func (w *Writer) nameIfChanged(mp *map[int]string, id int, name string) string {
	if name == "" {
		return ""
	}
	if *mp == nil {
		*mp = make(map[int]string)
	}
	if (*mp)[id] == name {
		return ""
	}
	(*mp)[id] = name
	return name
}

func (w *Writer) condOpenGz() (err error) {
	if w.gz == nil {
		w.gz, err = w.compressor.Writer(w.cw)
	}
	return
}

// AppendTar reads the tar or tar.gz file from r and appends
// each of its contents to w.
//
// The input r can optionally be gzip compressed but the output will
// always be compressed by the specified compressor.
func (w *Writer) AppendTar(r io.Reader) error {
	return w.appendTar(r, false)
}

// AppendTarLossLess reads the tar or tar.gz file from r and appends
// each of its contents to w.
//
// The input r can optionally be gzip compressed but the output will
// always be compressed by the specified compressor.
//
// The difference of this func with AppendTar is that this writes
// the input tar stream into w without any modification (e.g. to header bytes).
//
// Note that if the input tar stream already contains TOC JSON, this returns
// error because w cannot overwrite the TOC JSON to the one generated by w without
// lossy modification. To avoid this error, if the input stream is known to be stargz/estargz,
// you shoud decompress it and remove TOC JSON in advance.
func (w *Writer) AppendTarLossLess(r io.Reader) error {
	return w.appendTar(r, true)
}

func (w *Writer) appendTar(r io.Reader, lossless bool) error {
	var src io.Reader
	br := bufio.NewReader(r)
	if isGzip(br) {
		zr, _ := gzip.NewReader(br)
		src = zr
	} else {
		src = io.Reader(br)
	}
	dst := currentCompressionWriter{w}
	var tw *tar.Writer
	if !lossless {
		tw = tar.NewWriter(dst) // use tar writer only when this isn't lossless mode.
	}
	tr := tar.NewReader(src)
	if lossless {
		tr.RawAccounting = true
	}
	for {
		h, err := tr.Next()
		if err == io.EOF {
			if lossless {
				if remain := tr.RawBytes(); len(remain) > 0 {
					// Collect the remaining null bytes.
					// https://github.com/vbatts/tar-split/blob/80a436fd6164c557b131f7c59ed69bd81af69761/concept/main.go#L49-L53
					if _, err := dst.Write(remain); err != nil {
						return err
					}
				}
			}
			break
		}
		if err != nil {
			return fmt.Errorf("error reading from source tar: tar.Reader.Next: %v", err)
		}
		if cleanEntryName(h.Name) == TOCTarName {
			// It is possible for a layer to be "stargzified" twice during the
			// distribution lifecycle. So we reserve "TOCTarName" here to avoid
			// duplicated entries in the resulting layer.
			if lossless {
				// We cannot handle this in lossless way.
				return fmt.Errorf("existing TOC JSON is not allowed; decompress layer before append")
			}
			continue
		}

		xattrs := make(map[string][]byte)
		const xattrPAXRecordsPrefix = "SCHILY.xattr."
		if h.PAXRecords != nil {
			for k, v := range h.PAXRecords {
				if strings.HasPrefix(k, xattrPAXRecordsPrefix) {
					xattrs[k[len(xattrPAXRecordsPrefix):]] = []byte(v)
				}
			}
		}
		ent := &TOCEntry{
			Name:        h.Name,
			Mode:        h.Mode,
			UID:         h.Uid,
			GID:         h.Gid,
			Uname:       w.nameIfChanged(&w.lastUsername, h.Uid, h.Uname),
			Gname:       w.nameIfChanged(&w.lastGroupname, h.Gid, h.Gname),
			ModTime3339: formatModtime(h.ModTime),
			Xattrs:      xattrs,
		}
		if err := w.condOpenGz(); err != nil {
			return err
		}
		if tw != nil {
			if err := tw.WriteHeader(h); err != nil {
				return err
			}
		} else {
			if _, err := dst.Write(tr.RawBytes()); err != nil {
				return err
			}
		}
		switch h.Typeflag {
		case tar.TypeLink:
			ent.Type = "hardlink"
			ent.LinkName = h.Linkname
		case tar.TypeSymlink:
			ent.Type = "symlink"
			ent.LinkName = h.Linkname
		case tar.TypeDir:
			ent.Type = "dir"
		case tar.TypeReg:
			ent.Type = "reg"
			ent.Size = h.Size
		case tar.TypeChar:
			ent.Type = "char"
			ent.DevMajor = int(h.Devmajor)
			ent.DevMinor = int(h.Devminor)
		case tar.TypeBlock:
			ent.Type = "block"
			ent.DevMajor = int(h.Devmajor)
			ent.DevMinor = int(h.Devminor)
		case tar.TypeFifo:
			ent.Type = "fifo"
		default:
			return fmt.Errorf("unsupported input tar entry %q", h.Typeflag)
		}

		// We need to keep a reference to the TOC entry for regular files, so that we
		// can fill the digest later.
		var regFileEntry *TOCEntry
		var payloadDigest digest.Digester
		if h.Typeflag == tar.TypeReg {
			regFileEntry = ent
			payloadDigest = digest.Canonical.Digester()
		}

		if h.Typeflag == tar.TypeReg && ent.Size > 0 {
			var written int64
			totalSize := ent.Size // save it before we destroy ent
			tee := io.TeeReader(tr, payloadDigest.Hash())
			for written < totalSize {
				if err := w.closeGz(); err != nil {
					return err
				}

				chunkSize := int64(w.chunkSize())
				remain := totalSize - written
				if remain < chunkSize {
					chunkSize = remain
				} else {
					ent.ChunkSize = chunkSize
				}
				ent.Offset = w.cw.n
				ent.ChunkOffset = written
				chunkDigest := digest.Canonical.Digester()

				if err := w.condOpenGz(); err != nil {
					return err
				}

				teeChunk := io.TeeReader(tee, chunkDigest.Hash())
				var out io.Writer
				if tw != nil {
					out = tw
				} else {
					out = dst
				}
				if _, err := io.CopyN(out, teeChunk, chunkSize); err != nil {
					return fmt.Errorf("error copying %q: %v", h.Name, err)
				}
				ent.ChunkDigest = chunkDigest.Digest().String()
				w.toc.Entries = append(w.toc.Entries, ent)
				written += chunkSize
				ent = &TOCEntry{
					Name: h.Name,
					Type: "chunk",
				}
			}
		} else {
			w.toc.Entries = append(w.toc.Entries, ent)
		}
		if payloadDigest != nil {
			regFileEntry.Digest = payloadDigest.Digest().String()
		}
		if tw != nil {
			if err := tw.Flush(); err != nil {
				return err
			}
		}
	}
	remainDest := ioutil.Discard
	if lossless {
		remainDest = dst // Preserve the remaining bytes in lossless mode
	}
	_, err := io.Copy(remainDest, src)
	return err
}

// DiffID returns the SHA-256 of the uncompressed tar bytes.
// It is only valid to call DiffID after Close.
func (w *Writer) DiffID() string {
	return fmt.Sprintf("sha256:%x", w.diffHash.Sum(nil))
}

func maxFooterSize(blobSize int64, decompressors ...Decompressor) (res int64) {
	for _, d := range decompressors {
		if s := d.FooterSize(); res < s && s <= blobSize {
			res = s
		}
	}
	return
}

func parseTOC(d Decompressor, sr *io.SectionReader, tocOff, tocSize int64, tocBytes []byte, opts openOpts) (*Reader, error) {
	if len(tocBytes) > 0 {
		start := time.Now()
		toc, tocDgst, err := d.ParseTOC(bytes.NewReader(tocBytes))
		if err == nil {
			if opts.telemetry != nil && opts.telemetry.DeserializeTocLatency != nil {
				opts.telemetry.DeserializeTocLatency(start)
			}
			return &Reader{
				sr:           sr,
				toc:          toc,
				tocDigest:    tocDgst,
				decompressor: d,
			}, nil
		}
	}

	start := time.Now()
	tocBytes = make([]byte, tocSize)
	if _, err := sr.ReadAt(tocBytes, tocOff); err != nil {
		return nil, fmt.Errorf("error reading %d byte TOC targz: %v", len(tocBytes), err)
	}
	if opts.telemetry != nil && opts.telemetry.GetTocLatency != nil {
		opts.telemetry.GetTocLatency(start)
	}
	start = time.Now()
	toc, tocDgst, err := d.ParseTOC(bytes.NewReader(tocBytes))
	if err != nil {
		return nil, err
	}
	if opts.telemetry != nil && opts.telemetry.DeserializeTocLatency != nil {
		opts.telemetry.DeserializeTocLatency(start)
	}
	return &Reader{
		sr:           sr,
		toc:          toc,
		tocDigest:    tocDgst,
		decompressor: d,
	}, nil
}

func formatModtime(t time.Time) string {
	if t.IsZero() || t.Unix() == 0 {
		return ""
	}
	return t.UTC().Round(time.Second).Format(time.RFC3339)
}

func cleanEntryName(name string) string {
	// Use path.Clean to consistently deal with path separators across platforms.
	return strings.TrimPrefix(path.Clean("/"+name), "/")
}

// countWriter counts how many bytes have been written to its wrapped
// io.Writer.
type countWriter struct {
	w io.Writer
	n int64
}

func (cw *countWriter) Write(p []byte) (n int, err error) {
	n, err = cw.w.Write(p)
	cw.n += int64(n)
	return
}

// isGzip reports whether br is positioned right before an upcoming gzip stream.
// It does not consume any bytes from br.
func isGzip(br *bufio.Reader) bool {
	const (
		gzipID1     = 0x1f
		gzipID2     = 0x8b
		gzipDeflate = 8
	)
	peek, _ := br.Peek(3)
	return len(peek) >= 3 && peek[0] == gzipID1 && peek[1] == gzipID2 && peek[2] == gzipDeflate
}

func positive(n int64) int64 {
	if n < 0 {
		return 0
	}
	return n
}
