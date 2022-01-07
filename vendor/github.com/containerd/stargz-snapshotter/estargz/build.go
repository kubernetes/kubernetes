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
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"runtime"
	"strings"
	"sync"

	"github.com/containerd/stargz-snapshotter/estargz/errorutil"
	"github.com/klauspost/compress/zstd"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
	"golang.org/x/sync/errgroup"
)

type options struct {
	chunkSize              int
	compressionLevel       int
	prioritizedFiles       []string
	missedPrioritizedFiles *[]string
	compression            Compression
}

type Option func(o *options) error

// WithChunkSize option specifies the chunk size of eStargz blob to build.
func WithChunkSize(chunkSize int) Option {
	return func(o *options) error {
		o.chunkSize = chunkSize
		return nil
	}
}

// WithCompressionLevel option specifies the gzip compression level.
// The default is gzip.BestCompression.
// See also: https://godoc.org/compress/gzip#pkg-constants
func WithCompressionLevel(level int) Option {
	return func(o *options) error {
		o.compressionLevel = level
		return nil
	}
}

// WithPrioritizedFiles option specifies the list of prioritized files.
// These files must be complete paths that are absolute or relative to "/"
// For example, all of "foo/bar", "/foo/bar", "./foo/bar" and "../foo/bar"
// are treated as "/foo/bar".
func WithPrioritizedFiles(files []string) Option {
	return func(o *options) error {
		o.prioritizedFiles = files
		return nil
	}
}

// WithAllowPrioritizeNotFound makes Build continue the execution even if some
// of prioritized files specified by WithPrioritizedFiles option aren't found
// in the input tar. Instead, this records all missed file names to the passed
// slice.
func WithAllowPrioritizeNotFound(missedFiles *[]string) Option {
	return func(o *options) error {
		if missedFiles == nil {
			return fmt.Errorf("WithAllowPrioritizeNotFound: slice must be passed")
		}
		o.missedPrioritizedFiles = missedFiles
		return nil
	}
}

// WithCompression specifies compression algorithm to be used.
// Default is gzip.
func WithCompression(compression Compression) Option {
	return func(o *options) error {
		o.compression = compression
		return nil
	}
}

// Blob is an eStargz blob.
type Blob struct {
	io.ReadCloser
	diffID    digest.Digester
	tocDigest digest.Digest
}

// DiffID returns the digest of uncompressed blob.
// It is only valid to call DiffID after Close.
func (b *Blob) DiffID() digest.Digest {
	return b.diffID.Digest()
}

// TOCDigest returns the digest of uncompressed TOC JSON.
func (b *Blob) TOCDigest() digest.Digest {
	return b.tocDigest
}

// Build builds an eStargz blob which is an extended version of stargz, from a blob (gzip, zstd
// or plain tar) passed through the argument. If there are some prioritized files are listed in
// the option, these files are grouped as "prioritized" and can be used for runtime optimization
// (e.g. prefetch). This function builds a blob in parallel, with dividing that blob into several
// (at least the number of runtime.GOMAXPROCS(0)) sub-blobs.
func Build(tarBlob *io.SectionReader, opt ...Option) (_ *Blob, rErr error) {
	var opts options
	opts.compressionLevel = gzip.BestCompression // BestCompression by default
	for _, o := range opt {
		if err := o(&opts); err != nil {
			return nil, err
		}
	}
	if opts.compression == nil {
		opts.compression = newGzipCompressionWithLevel(opts.compressionLevel)
	}
	layerFiles := newTempFiles()
	defer func() {
		if rErr != nil {
			if err := layerFiles.CleanupAll(); err != nil {
				rErr = errors.Wrapf(rErr, "failed to cleanup tmp files: %v", err)
			}
		}
	}()
	tarBlob, err := decompressBlob(tarBlob, layerFiles)
	if err != nil {
		return nil, err
	}
	entries, err := sortEntries(tarBlob, opts.prioritizedFiles, opts.missedPrioritizedFiles)
	if err != nil {
		return nil, err
	}
	tarParts := divideEntries(entries, runtime.GOMAXPROCS(0))
	writers := make([]*Writer, len(tarParts))
	payloads := make([]*os.File, len(tarParts))
	var mu sync.Mutex
	var eg errgroup.Group
	for i, parts := range tarParts {
		i, parts := i, parts
		// builds verifiable stargz sub-blobs
		eg.Go(func() error {
			esgzFile, err := layerFiles.TempFile("", "esgzdata")
			if err != nil {
				return err
			}
			sw := NewWriterWithCompressor(esgzFile, opts.compression)
			sw.ChunkSize = opts.chunkSize
			if err := sw.AppendTar(readerFromEntries(parts...)); err != nil {
				return err
			}
			mu.Lock()
			writers[i] = sw
			payloads[i] = esgzFile
			mu.Unlock()
			return nil
		})
	}
	if err := eg.Wait(); err != nil {
		rErr = err
		return nil, err
	}
	tocAndFooter, tocDgst, err := closeWithCombine(opts.compressionLevel, writers...)
	if err != nil {
		rErr = err
		return nil, err
	}
	var rs []io.Reader
	for _, p := range payloads {
		fs, err := fileSectionReader(p)
		if err != nil {
			return nil, err
		}
		rs = append(rs, fs)
	}
	diffID := digest.Canonical.Digester()
	pr, pw := io.Pipe()
	go func() {
		r, err := opts.compression.Reader(io.TeeReader(io.MultiReader(append(rs, tocAndFooter)...), pw))
		if err != nil {
			pw.CloseWithError(err)
			return
		}
		defer r.Close()
		if _, err := io.Copy(diffID.Hash(), r); err != nil {
			pw.CloseWithError(err)
			return
		}
		pw.Close()
	}()
	return &Blob{
		ReadCloser: readCloser{
			Reader:    pr,
			closeFunc: layerFiles.CleanupAll,
		},
		tocDigest: tocDgst,
		diffID:    diffID,
	}, nil
}

// closeWithCombine takes unclosed Writers and close them. This also returns the
// toc that combined all Writers into.
// Writers doesn't write TOC and footer to the underlying writers so they can be
// combined into a single eStargz and tocAndFooter returned by this function can
// be appended at the tail of that combined blob.
func closeWithCombine(compressionLevel int, ws ...*Writer) (tocAndFooterR io.Reader, tocDgst digest.Digest, err error) {
	if len(ws) == 0 {
		return nil, "", fmt.Errorf("at least one writer must be passed")
	}
	for _, w := range ws {
		if w.closed {
			return nil, "", fmt.Errorf("writer must be unclosed")
		}
		defer func(w *Writer) { w.closed = true }(w)
		if err := w.closeGz(); err != nil {
			return nil, "", err
		}
		if err := w.bw.Flush(); err != nil {
			return nil, "", err
		}
	}
	var (
		mtoc          = new(JTOC)
		currentOffset int64
	)
	mtoc.Version = ws[0].toc.Version
	for _, w := range ws {
		for _, e := range w.toc.Entries {
			// Recalculate Offset of non-empty files/chunks
			if (e.Type == "reg" && e.Size > 0) || e.Type == "chunk" {
				e.Offset += currentOffset
			}
			mtoc.Entries = append(mtoc.Entries, e)
		}
		if w.toc.Version > mtoc.Version {
			mtoc.Version = w.toc.Version
		}
		currentOffset += w.cw.n
	}

	return tocAndFooter(ws[0].compressor, mtoc, currentOffset)
}

func tocAndFooter(compressor Compressor, toc *JTOC, offset int64) (io.Reader, digest.Digest, error) {
	buf := new(bytes.Buffer)
	tocDigest, err := compressor.WriteTOCAndFooter(buf, offset, toc, nil)
	if err != nil {
		return nil, "", err
	}
	return buf, tocDigest, nil
}

// divideEntries divides passed entries to the parts at least the number specified by the
// argument.
func divideEntries(entries []*entry, minPartsNum int) (set [][]*entry) {
	var estimatedSize int64
	for _, e := range entries {
		estimatedSize += e.header.Size
	}
	unitSize := estimatedSize / int64(minPartsNum)
	var (
		nextEnd = unitSize
		offset  int64
	)
	set = append(set, []*entry{})
	for _, e := range entries {
		set[len(set)-1] = append(set[len(set)-1], e)
		offset += e.header.Size
		if offset > nextEnd {
			set = append(set, []*entry{})
			nextEnd += unitSize
		}
	}
	return
}

var errNotFound = errors.New("not found")

// sortEntries reads the specified tar blob and returns a list of tar entries.
// If some of prioritized files are specified, the list starts from these
// files with keeping the order specified by the argument.
func sortEntries(in io.ReaderAt, prioritized []string, missedPrioritized *[]string) ([]*entry, error) {

	// Import tar file.
	intar, err := importTar(in)
	if err != nil {
		return nil, errors.Wrap(err, "failed to sort")
	}

	// Sort the tar file respecting to the prioritized files list.
	sorted := &tarFile{}
	for _, l := range prioritized {
		if err := moveRec(l, intar, sorted); err != nil {
			if errors.Is(err, errNotFound) && missedPrioritized != nil {
				*missedPrioritized = append(*missedPrioritized, l)
				continue // allow not found
			}
			return nil, errors.Wrap(err, "failed to sort tar entries")
		}
	}
	if len(prioritized) == 0 {
		sorted.add(&entry{
			header: &tar.Header{
				Name:     NoPrefetchLandmark,
				Typeflag: tar.TypeReg,
				Size:     int64(len([]byte{landmarkContents})),
			},
			payload: bytes.NewReader([]byte{landmarkContents}),
		})
	} else {
		sorted.add(&entry{
			header: &tar.Header{
				Name:     PrefetchLandmark,
				Typeflag: tar.TypeReg,
				Size:     int64(len([]byte{landmarkContents})),
			},
			payload: bytes.NewReader([]byte{landmarkContents}),
		})
	}

	// Dump all entry and concatinate them.
	return append(sorted.dump(), intar.dump()...), nil
}

// readerFromEntries returns a reader of tar archive that contains entries passed
// through the arguments.
func readerFromEntries(entries ...*entry) io.Reader {
	pr, pw := io.Pipe()
	go func() {
		tw := tar.NewWriter(pw)
		defer tw.Close()
		for _, entry := range entries {
			if err := tw.WriteHeader(entry.header); err != nil {
				pw.CloseWithError(fmt.Errorf("Failed to write tar header: %v", err))
				return
			}
			if _, err := io.Copy(tw, entry.payload); err != nil {
				pw.CloseWithError(fmt.Errorf("Failed to write tar payload: %v", err))
				return
			}
		}
		pw.Close()
	}()
	return pr
}

func importTar(in io.ReaderAt) (*tarFile, error) {
	tf := &tarFile{}
	pw, err := newCountReader(in)
	if err != nil {
		return nil, errors.Wrap(err, "failed to make position watcher")
	}
	tr := tar.NewReader(pw)

	// Walk through all nodes.
	for {
		// Fetch and parse next header.
		h, err := tr.Next()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return nil, errors.Wrap(err, "failed to parse tar file")
			}
		}
		switch cleanEntryName(h.Name) {
		case PrefetchLandmark, NoPrefetchLandmark:
			// Ignore existing landmark
			continue
		}

		// Add entry. If it already exists, replace it.
		if _, ok := tf.get(h.Name); ok {
			tf.remove(h.Name)
		}
		tf.add(&entry{
			header:  h,
			payload: io.NewSectionReader(in, pw.currentPos(), h.Size),
		})
	}

	return tf, nil
}

func moveRec(name string, in *tarFile, out *tarFile) error {
	name = cleanEntryName(name)
	if name == "" { // root directory. stop recursion.
		if e, ok := in.get(name); ok {
			// entry of the root directory exists. we should move it as well.
			// this case will occur if tar entries are prefixed with "./", "/", etc.
			out.add(e)
			in.remove(name)
		}
		return nil
	}

	_, okIn := in.get(name)
	_, okOut := out.get(name)
	if !okIn && !okOut {
		return errors.Wrapf(errNotFound, "file: %q", name)
	}

	parent, _ := path.Split(strings.TrimSuffix(name, "/"))
	if err := moveRec(parent, in, out); err != nil {
		return err
	}
	if e, ok := in.get(name); ok && e.header.Typeflag == tar.TypeLink {
		if err := moveRec(e.header.Linkname, in, out); err != nil {
			return err
		}
	}
	if e, ok := in.get(name); ok {
		out.add(e)
		in.remove(name)
	}
	return nil
}

type entry struct {
	header  *tar.Header
	payload io.ReadSeeker
}

type tarFile struct {
	index  map[string]*entry
	stream []*entry
}

func (f *tarFile) add(e *entry) {
	if f.index == nil {
		f.index = make(map[string]*entry)
	}
	f.index[cleanEntryName(e.header.Name)] = e
	f.stream = append(f.stream, e)
}

func (f *tarFile) remove(name string) {
	name = cleanEntryName(name)
	if f.index != nil {
		delete(f.index, name)
	}
	var filtered []*entry
	for _, e := range f.stream {
		if cleanEntryName(e.header.Name) == name {
			continue
		}
		filtered = append(filtered, e)
	}
	f.stream = filtered
}

func (f *tarFile) get(name string) (e *entry, ok bool) {
	if f.index == nil {
		return nil, false
	}
	e, ok = f.index[cleanEntryName(name)]
	return
}

func (f *tarFile) dump() []*entry {
	return f.stream
}

type readCloser struct {
	io.Reader
	closeFunc func() error
}

func (rc readCloser) Close() error {
	return rc.closeFunc()
}

func fileSectionReader(file *os.File) (*io.SectionReader, error) {
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	return io.NewSectionReader(file, 0, info.Size()), nil
}

func newTempFiles() *tempFiles {
	return &tempFiles{}
}

type tempFiles struct {
	files   []*os.File
	filesMu sync.Mutex
}

func (tf *tempFiles) TempFile(dir, pattern string) (*os.File, error) {
	f, err := ioutil.TempFile(dir, pattern)
	if err != nil {
		return nil, err
	}
	tf.filesMu.Lock()
	tf.files = append(tf.files, f)
	tf.filesMu.Unlock()
	return f, nil
}

func (tf *tempFiles) CleanupAll() error {
	tf.filesMu.Lock()
	defer tf.filesMu.Unlock()
	var allErr []error
	for _, f := range tf.files {
		if err := f.Close(); err != nil {
			allErr = append(allErr, err)
		}
		if err := os.Remove(f.Name()); err != nil {
			allErr = append(allErr, err)
		}
	}
	tf.files = nil
	return errorutil.Aggregate(allErr)
}

func newCountReader(r io.ReaderAt) (*countReader, error) {
	pos := int64(0)
	return &countReader{r: r, cPos: &pos}, nil
}

type countReader struct {
	r    io.ReaderAt
	cPos *int64

	mu sync.Mutex
}

func (cr *countReader) Read(p []byte) (int, error) {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	n, err := cr.r.ReadAt(p, *cr.cPos)
	if err == nil {
		*cr.cPos += int64(n)
	}
	return n, err
}

func (cr *countReader) Seek(offset int64, whence int) (int64, error) {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	switch whence {
	default:
		return 0, fmt.Errorf("Unknown whence: %v", whence)
	case io.SeekStart:
	case io.SeekCurrent:
		offset += *cr.cPos
	case io.SeekEnd:
		return 0, fmt.Errorf("Unsupported whence: %v", whence)
	}

	if offset < 0 {
		return 0, fmt.Errorf("invalid offset")
	}
	*cr.cPos = offset
	return offset, nil
}

func (cr *countReader) currentPos() int64 {
	cr.mu.Lock()
	defer cr.mu.Unlock()

	return *cr.cPos
}

func decompressBlob(org *io.SectionReader, tmp *tempFiles) (*io.SectionReader, error) {
	if org.Size() < 4 {
		return org, nil
	}
	src := make([]byte, 4)
	if _, err := org.Read(src); err != nil && err != io.EOF {
		return nil, err
	}
	var dR io.Reader
	if bytes.Equal([]byte{0x1F, 0x8B, 0x08}, src[:3]) {
		// gzip
		dgR, err := gzip.NewReader(io.NewSectionReader(org, 0, org.Size()))
		if err != nil {
			return nil, err
		}
		defer dgR.Close()
		dR = io.Reader(dgR)
	} else if bytes.Equal([]byte{0x28, 0xb5, 0x2f, 0xfd}, src[:4]) {
		// zstd
		dzR, err := zstd.NewReader(io.NewSectionReader(org, 0, org.Size()))
		if err != nil {
			return nil, err
		}
		defer dzR.Close()
		dR = io.Reader(dzR)
	} else {
		// uncompressed
		return io.NewSectionReader(org, 0, org.Size()), nil
	}
	b, err := tmp.TempFile("", "uncompresseddata")
	if err != nil {
		return nil, err
	}
	if _, err := io.Copy(b, dR); err != nil {
		return nil, err
	}
	return fileSectionReader(b)
}
