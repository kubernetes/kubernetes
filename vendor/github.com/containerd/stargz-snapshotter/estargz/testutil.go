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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/containerd/stargz-snapshotter/estargz/errorutil"
	"github.com/klauspost/compress/zstd"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

// TestingController is Compression with some helper methods necessary for testing.
type TestingController interface {
	Compression
	CountStreams(*testing.T, []byte) int
	DiffIDOf(*testing.T, []byte) string
	String() string
}

// CompressionTestSuite tests this pkg with controllers can build valid eStargz blobs and parse them.
func CompressionTestSuite(t *testing.T, controllers ...TestingController) {
	t.Run("testBuild", func(t *testing.T) { t.Parallel(); testBuild(t, controllers...) })
	t.Run("testDigestAndVerify", func(t *testing.T) { t.Parallel(); testDigestAndVerify(t, controllers...) })
	t.Run("testWriteAndOpen", func(t *testing.T) { t.Parallel(); testWriteAndOpen(t, controllers...) })
}

const (
	uncompressedType int = iota
	gzipType
	zstdType
)

var srcCompressions = []int{
	uncompressedType,
	gzipType,
	zstdType,
}

var allowedPrefix = [4]string{"", "./", "/", "../"}

// testBuild tests the resulting stargz blob built by this pkg has the same
// contents as the normal stargz blob.
func testBuild(t *testing.T, controllers ...TestingController) {
	tests := []struct {
		name      string
		chunkSize int
		in        []tarEntry
	}{
		{
			name:      "regfiles and directories",
			chunkSize: 4,
			in: tarOf(
				file("foo", "test1"),
				dir("foo2/"),
				file("foo2/bar", "test2", xAttr(map[string]string{"test": "sample"})),
			),
		},
		{
			name:      "empty files",
			chunkSize: 4,
			in: tarOf(
				file("foo", "tttttt"),
				file("foo_empty", ""),
				file("foo2", "tttttt"),
				file("foo_empty2", ""),
				file("foo3", "tttttt"),
				file("foo_empty3", ""),
				file("foo4", "tttttt"),
				file("foo_empty4", ""),
				file("foo5", "tttttt"),
				file("foo_empty5", ""),
				file("foo6", "tttttt"),
			),
		},
		{
			name:      "various files",
			chunkSize: 4,
			in: tarOf(
				file("baz.txt", "bazbazbazbazbazbazbaz"),
				file("foo.txt", "a"),
				symlink("barlink", "test/bar.txt"),
				dir("test/"),
				dir("dev/"),
				blockdev("dev/testblock", 3, 4),
				fifo("dev/testfifo"),
				chardev("dev/testchar1", 5, 6),
				file("test/bar.txt", "testbartestbar", xAttr(map[string]string{"test2": "sample2"})),
				dir("test2/"),
				link("test2/bazlink", "baz.txt"),
				chardev("dev/testchar2", 1, 2),
			),
		},
		{
			name:      "no contents",
			chunkSize: 4,
			in: tarOf(
				file("baz.txt", ""),
				symlink("barlink", "test/bar.txt"),
				dir("test/"),
				dir("dev/"),
				blockdev("dev/testblock", 3, 4),
				fifo("dev/testfifo"),
				chardev("dev/testchar1", 5, 6),
				file("test/bar.txt", "", xAttr(map[string]string{"test2": "sample2"})),
				dir("test2/"),
				link("test2/bazlink", "baz.txt"),
				chardev("dev/testchar2", 1, 2),
			),
		},
	}
	for _, tt := range tests {
		for _, srcCompression := range srcCompressions {
			srcCompression := srcCompression
			for _, cl := range controllers {
				cl := cl
				for _, srcTarFormat := range []tar.Format{tar.FormatUSTAR, tar.FormatPAX, tar.FormatGNU} {
					srcTarFormat := srcTarFormat
					for _, prefix := range allowedPrefix {
						prefix := prefix
						t.Run(tt.name+"-"+fmt.Sprintf("compression=%v,prefix=%q,src=%d,format=%s", cl, prefix, srcCompression, srcTarFormat), func(t *testing.T) {
							tarBlob := buildTar(t, tt.in, prefix, srcTarFormat)
							// Test divideEntries()
							entries, err := sortEntries(tarBlob, nil, nil) // identical order
							if err != nil {
								t.Fatalf("failed to parse tar: %v", err)
							}
							var merged []*entry
							for _, part := range divideEntries(entries, 4) {
								merged = append(merged, part...)
							}
							if !reflect.DeepEqual(entries, merged) {
								for _, e := range entries {
									t.Logf("Original: %v", e.header)
								}
								for _, e := range merged {
									t.Logf("Merged: %v", e.header)
								}
								t.Errorf("divided entries couldn't be merged")
								return
							}

							// Prepare sample data
							wantBuf := new(bytes.Buffer)
							sw := NewWriterWithCompressor(wantBuf, cl)
							sw.ChunkSize = tt.chunkSize
							if err := sw.AppendTar(tarBlob); err != nil {
								t.Fatalf("failed to append tar to want stargz: %v", err)
							}
							if _, err := sw.Close(); err != nil {
								t.Fatalf("failed to prepare want stargz: %v", err)
							}
							wantData := wantBuf.Bytes()
							want, err := Open(io.NewSectionReader(
								bytes.NewReader(wantData), 0, int64(len(wantData))),
								WithDecompressors(cl),
							)
							if err != nil {
								t.Fatalf("failed to parse the want stargz: %v", err)
							}

							// Prepare testing data
							rc, err := Build(compressBlob(t, tarBlob, srcCompression),
								WithChunkSize(tt.chunkSize), WithCompression(cl))
							if err != nil {
								t.Fatalf("failed to build stargz: %v", err)
							}
							defer rc.Close()
							gotBuf := new(bytes.Buffer)
							if _, err := io.Copy(gotBuf, rc); err != nil {
								t.Fatalf("failed to copy built stargz blob: %v", err)
							}
							gotData := gotBuf.Bytes()
							got, err := Open(io.NewSectionReader(
								bytes.NewReader(gotBuf.Bytes()), 0, int64(len(gotData))),
								WithDecompressors(cl),
							)
							if err != nil {
								t.Fatalf("failed to parse the got stargz: %v", err)
							}

							// Check DiffID is properly calculated
							rc.Close()
							diffID := rc.DiffID()
							wantDiffID := cl.DiffIDOf(t, gotData)
							if diffID.String() != wantDiffID {
								t.Errorf("DiffID = %q; want %q", diffID, wantDiffID)
							}

							// Compare as stargz
							if !isSameVersion(t, cl, wantData, gotData) {
								t.Errorf("built stargz hasn't same json")
								return
							}
							if !isSameEntries(t, want, got) {
								t.Errorf("built stargz isn't same as the original")
								return
							}

							// Compare as tar.gz
							if !isSameTarGz(t, cl, wantData, gotData) {
								t.Errorf("built stargz isn't same tar.gz")
								return
							}
						})
					}
				}
			}
		}
	}
}

func isSameTarGz(t *testing.T, controller TestingController, a, b []byte) bool {
	aGz, err := controller.Reader(bytes.NewReader(a))
	if err != nil {
		t.Fatalf("failed to read A")
	}
	defer aGz.Close()
	bGz, err := controller.Reader(bytes.NewReader(b))
	if err != nil {
		t.Fatalf("failed to read B")
	}
	defer bGz.Close()

	// Same as tar's Next() method but ignores landmarks and TOCJSON file
	next := func(r *tar.Reader) (h *tar.Header, err error) {
		for {
			if h, err = r.Next(); err != nil {
				return
			}
			if h.Name != PrefetchLandmark &&
				h.Name != NoPrefetchLandmark &&
				h.Name != TOCTarName {
				return
			}
		}
	}

	aTar := tar.NewReader(aGz)
	bTar := tar.NewReader(bGz)
	for {
		// Fetch and parse next header.
		aH, aErr := next(aTar)
		bH, bErr := next(bTar)
		if aErr != nil || bErr != nil {
			if aErr == io.EOF && bErr == io.EOF {
				break
			}
			t.Fatalf("Failed to parse tar file: A: %v, B: %v", aErr, bErr)
		}
		if !reflect.DeepEqual(aH, bH) {
			t.Logf("different header (A = %v; B = %v)", aH, bH)
			return false

		}
		aFile, err := ioutil.ReadAll(aTar)
		if err != nil {
			t.Fatal("failed to read tar payload of A")
		}
		bFile, err := ioutil.ReadAll(bTar)
		if err != nil {
			t.Fatal("failed to read tar payload of B")
		}
		if !bytes.Equal(aFile, bFile) {
			t.Logf("different tar payload (A = %q; B = %q)", string(a), string(b))
			return false
		}
	}

	return true
}

func isSameVersion(t *testing.T, controller TestingController, a, b []byte) bool {
	aJTOC, _, err := parseStargz(io.NewSectionReader(bytes.NewReader(a), 0, int64(len(a))), controller)
	if err != nil {
		t.Fatalf("failed to parse A: %v", err)
	}
	bJTOC, _, err := parseStargz(io.NewSectionReader(bytes.NewReader(b), 0, int64(len(b))), controller)
	if err != nil {
		t.Fatalf("failed to parse B: %v", err)
	}
	t.Logf("A: TOCJSON: %v", dumpTOCJSON(t, aJTOC))
	t.Logf("B: TOCJSON: %v", dumpTOCJSON(t, bJTOC))
	return aJTOC.Version == bJTOC.Version
}

func isSameEntries(t *testing.T, a, b *Reader) bool {
	aroot, ok := a.Lookup("")
	if !ok {
		t.Fatalf("failed to get root of A")
	}
	broot, ok := b.Lookup("")
	if !ok {
		t.Fatalf("failed to get root of B")
	}
	aEntry := stargzEntry{aroot, a}
	bEntry := stargzEntry{broot, b}
	return contains(t, aEntry, bEntry) && contains(t, bEntry, aEntry)
}

func compressBlob(t *testing.T, src *io.SectionReader, srcCompression int) *io.SectionReader {
	buf := new(bytes.Buffer)
	var w io.WriteCloser
	var err error
	if srcCompression == gzipType {
		w = gzip.NewWriter(buf)
	} else if srcCompression == zstdType {
		w, err = zstd.NewWriter(buf)
		if err != nil {
			t.Fatalf("failed to init zstd writer: %v", err)
		}
	} else {
		return src
	}
	src.Seek(0, io.SeekStart)
	if _, err := io.Copy(w, src); err != nil {
		t.Fatalf("failed to compress source")
	}
	if err := w.Close(); err != nil {
		t.Fatalf("failed to finalize compress source")
	}
	data := buf.Bytes()
	return io.NewSectionReader(bytes.NewReader(data), 0, int64(len(data)))

}

type stargzEntry struct {
	e *TOCEntry
	r *Reader
}

// contains checks if all child entries in "b" are also contained in "a".
// This function also checks if the files/chunks contain the same contents among "a" and "b".
func contains(t *testing.T, a, b stargzEntry) bool {
	ae, ar := a.e, a.r
	be, br := b.e, b.r
	t.Logf("Comparing: %q vs %q", ae.Name, be.Name)
	if !equalEntry(ae, be) {
		t.Logf("%q != %q: entry: a: %v, b: %v", ae.Name, be.Name, ae, be)
		return false
	}
	if ae.Type == "dir" {
		t.Logf("Directory: %q vs %q: %v vs %v", ae.Name, be.Name,
			allChildrenName(ae), allChildrenName(be))
		iscontain := true
		ae.ForeachChild(func(aBaseName string, aChild *TOCEntry) bool {
			// Walk through all files on this stargz file.

			if aChild.Name == PrefetchLandmark ||
				aChild.Name == NoPrefetchLandmark {
				return true // Ignore landmarks
			}

			// Ignore a TOCEntry of "./" (formated as "" by stargz lib) on root directory
			// because this points to the root directory itself.
			if aChild.Name == "" && ae.Name == "" {
				return true
			}

			bChild, ok := be.LookupChild(aBaseName)
			if !ok {
				t.Logf("%q (base: %q): not found in b: %v",
					ae.Name, aBaseName, allChildrenName(be))
				iscontain = false
				return false
			}

			childcontain := contains(t, stargzEntry{aChild, a.r}, stargzEntry{bChild, b.r})
			if !childcontain {
				t.Logf("%q != %q: non-equal dir", ae.Name, be.Name)
				iscontain = false
				return false
			}
			return true
		})
		return iscontain
	} else if ae.Type == "reg" {
		af, err := ar.OpenFile(ae.Name)
		if err != nil {
			t.Fatalf("failed to open file %q on A: %v", ae.Name, err)
		}
		bf, err := br.OpenFile(be.Name)
		if err != nil {
			t.Fatalf("failed to open file %q on B: %v", be.Name, err)
		}

		var nr int64
		for nr < ae.Size {
			abytes, anext, aok := readOffset(t, af, nr, a)
			bbytes, bnext, bok := readOffset(t, bf, nr, b)
			if !aok && !bok {
				break
			} else if !(aok && bok) || anext != bnext {
				t.Logf("%q != %q (offset=%d): chunk existence a=%v vs b=%v, anext=%v vs bnext=%v",
					ae.Name, be.Name, nr, aok, bok, anext, bnext)
				return false
			}
			nr = anext
			if !bytes.Equal(abytes, bbytes) {
				t.Logf("%q != %q: different contents %v vs %v",
					ae.Name, be.Name, string(abytes), string(bbytes))
				return false
			}
		}
		return true
	}

	return true
}

func allChildrenName(e *TOCEntry) (children []string) {
	e.ForeachChild(func(baseName string, _ *TOCEntry) bool {
		children = append(children, baseName)
		return true
	})
	return
}

func equalEntry(a, b *TOCEntry) bool {
	// Here, we selectively compare fileds that we are interested in.
	return a.Name == b.Name &&
		a.Type == b.Type &&
		a.Size == b.Size &&
		a.ModTime3339 == b.ModTime3339 &&
		a.Stat().ModTime().Equal(b.Stat().ModTime()) && // modTime     time.Time
		a.LinkName == b.LinkName &&
		a.Mode == b.Mode &&
		a.UID == b.UID &&
		a.GID == b.GID &&
		a.Uname == b.Uname &&
		a.Gname == b.Gname &&
		(a.Offset > 0) == (b.Offset > 0) &&
		(a.NextOffset() > 0) == (b.NextOffset() > 0) &&
		a.DevMajor == b.DevMajor &&
		a.DevMinor == b.DevMinor &&
		a.NumLink == b.NumLink &&
		reflect.DeepEqual(a.Xattrs, b.Xattrs) &&
		// chunk-related infomations aren't compared in this function.
		// ChunkOffset int64 `json:"chunkOffset,omitempty"`
		// ChunkSize   int64 `json:"chunkSize,omitempty"`
		// children map[string]*TOCEntry
		a.Digest == b.Digest
}

func readOffset(t *testing.T, r *io.SectionReader, offset int64, e stargzEntry) ([]byte, int64, bool) {
	ce, ok := e.r.ChunkEntryForOffset(e.e.Name, offset)
	if !ok {
		return nil, 0, false
	}
	data := make([]byte, ce.ChunkSize)
	t.Logf("Offset: %v, NextOffset: %v", ce.Offset, ce.NextOffset())
	n, err := r.ReadAt(data, ce.ChunkOffset)
	if err != nil {
		t.Fatalf("failed to read file payload of %q (offset:%d,size:%d): %v",
			e.e.Name, ce.ChunkOffset, ce.ChunkSize, err)
	}
	if int64(n) != ce.ChunkSize {
		t.Fatalf("unexpected copied data size %d; want %d",
			n, ce.ChunkSize)
	}
	return data[:n], offset + ce.ChunkSize, true
}

func dumpTOCJSON(t *testing.T, tocJSON *JTOC) string {
	jtocData, err := json.Marshal(*tocJSON)
	if err != nil {
		t.Fatalf("failed to marshal TOC JSON: %v", err)
	}
	buf := new(bytes.Buffer)
	if _, err := io.Copy(buf, bytes.NewReader(jtocData)); err != nil {
		t.Fatalf("failed to read toc json blob: %v", err)
	}
	return buf.String()
}

const chunkSize = 3

// type check func(t *testing.T, sgzData []byte, tocDigest digest.Digest, dgstMap map[string]digest.Digest, compressionLevel int)
type check func(t *testing.T, sgzData []byte, tocDigest digest.Digest, dgstMap map[string]digest.Digest, controller TestingController)

// testDigestAndVerify runs specified checks against sample stargz blobs.
func testDigestAndVerify(t *testing.T, controllers ...TestingController) {
	tests := []struct {
		name    string
		tarInit func(t *testing.T, dgstMap map[string]digest.Digest) (blob []tarEntry)
		checks  []check
	}{
		{
			name: "no-regfile",
			tarInit: func(t *testing.T, dgstMap map[string]digest.Digest) (blob []tarEntry) {
				return tarOf(
					dir("test/"),
				)
			},
			checks: []check{
				checkStargzTOC,
				checkVerifyTOC,
				checkVerifyInvalidStargzFail(buildTar(t, tarOf(
					dir("test2/"), // modified
				), allowedPrefix[0])),
			},
		},
		{
			name: "small-files",
			tarInit: func(t *testing.T, dgstMap map[string]digest.Digest) (blob []tarEntry) {
				return tarOf(
					regDigest(t, "baz.txt", "", dgstMap),
					regDigest(t, "foo.txt", "a", dgstMap),
					dir("test/"),
					regDigest(t, "test/bar.txt", "bbb", dgstMap),
				)
			},
			checks: []check{
				checkStargzTOC,
				checkVerifyTOC,
				checkVerifyInvalidStargzFail(buildTar(t, tarOf(
					file("baz.txt", ""),
					file("foo.txt", "M"), // modified
					dir("test/"),
					file("test/bar.txt", "bbb"),
				), allowedPrefix[0])),
				// checkVerifyInvalidTOCEntryFail("foo.txt"), // TODO
				checkVerifyBrokenContentFail("foo.txt"),
			},
		},
		{
			name: "big-files",
			tarInit: func(t *testing.T, dgstMap map[string]digest.Digest) (blob []tarEntry) {
				return tarOf(
					regDigest(t, "baz.txt", "bazbazbazbazbazbazbaz", dgstMap),
					regDigest(t, "foo.txt", "a", dgstMap),
					dir("test/"),
					regDigest(t, "test/bar.txt", "testbartestbar", dgstMap),
				)
			},
			checks: []check{
				checkStargzTOC,
				checkVerifyTOC,
				checkVerifyInvalidStargzFail(buildTar(t, tarOf(
					file("baz.txt", "bazbazbazMMMbazbazbaz"), // modified
					file("foo.txt", "a"),
					dir("test/"),
					file("test/bar.txt", "testbartestbar"),
				), allowedPrefix[0])),
				checkVerifyInvalidTOCEntryFail("test/bar.txt"),
				checkVerifyBrokenContentFail("test/bar.txt"),
			},
		},
		{
			name: "with-non-regfiles",
			tarInit: func(t *testing.T, dgstMap map[string]digest.Digest) (blob []tarEntry) {
				return tarOf(
					regDigest(t, "baz.txt", "bazbazbazbazbazbazbaz", dgstMap),
					regDigest(t, "foo.txt", "a", dgstMap),
					symlink("barlink", "test/bar.txt"),
					dir("test/"),
					regDigest(t, "test/bar.txt", "testbartestbar", dgstMap),
					dir("test2/"),
					link("test2/bazlink", "baz.txt"),
				)
			},
			checks: []check{
				checkStargzTOC,
				checkVerifyTOC,
				checkVerifyInvalidStargzFail(buildTar(t, tarOf(
					file("baz.txt", "bazbazbazbazbazbazbaz"),
					file("foo.txt", "a"),
					symlink("barlink", "test/bar.txt"),
					dir("test/"),
					file("test/bar.txt", "testbartestbar"),
					dir("test2/"),
					link("test2/bazlink", "foo.txt"), // modified
				), allowedPrefix[0])),
				checkVerifyInvalidTOCEntryFail("test/bar.txt"),
				checkVerifyBrokenContentFail("test/bar.txt"),
			},
		},
	}

	for _, tt := range tests {
		for _, srcCompression := range srcCompressions {
			srcCompression := srcCompression
			for _, cl := range controllers {
				cl := cl
				for _, prefix := range allowedPrefix {
					prefix := prefix
					for _, srcTarFormat := range []tar.Format{tar.FormatUSTAR, tar.FormatPAX, tar.FormatGNU} {
						srcTarFormat := srcTarFormat
						t.Run(tt.name+"-"+fmt.Sprintf("compression=%v,prefix=%q,format=%s", cl, prefix, srcTarFormat), func(t *testing.T) {
							// Get original tar file and chunk digests
							dgstMap := make(map[string]digest.Digest)
							tarBlob := buildTar(t, tt.tarInit(t, dgstMap), prefix, srcTarFormat)

							rc, err := Build(compressBlob(t, tarBlob, srcCompression),
								WithChunkSize(chunkSize), WithCompression(cl))
							if err != nil {
								t.Fatalf("failed to convert stargz: %v", err)
							}
							tocDigest := rc.TOCDigest()
							defer rc.Close()
							buf := new(bytes.Buffer)
							if _, err := io.Copy(buf, rc); err != nil {
								t.Fatalf("failed to copy built stargz blob: %v", err)
							}
							newStargz := buf.Bytes()
							// NoPrefetchLandmark is added during `Bulid`, which is expected behaviour.
							dgstMap[chunkID(NoPrefetchLandmark, 0, int64(len([]byte{landmarkContents})))] = digest.FromBytes([]byte{landmarkContents})

							for _, check := range tt.checks {
								check(t, newStargz, tocDigest, dgstMap, cl)
							}
						})
					}
				}
			}
		}
	}
}

// checkStargzTOC checks the TOC JSON of the passed stargz has the expected
// digest and contains valid chunks. It walks all entries in the stargz and
// checks all chunk digests stored to the TOC JSON match the actual contents.
func checkStargzTOC(t *testing.T, sgzData []byte, tocDigest digest.Digest, dgstMap map[string]digest.Digest, controller TestingController) {
	sgz, err := Open(
		io.NewSectionReader(bytes.NewReader(sgzData), 0, int64(len(sgzData))),
		WithDecompressors(controller),
	)
	if err != nil {
		t.Errorf("failed to parse converted stargz: %v", err)
		return
	}
	digestMapTOC, err := listDigests(io.NewSectionReader(
		bytes.NewReader(sgzData), 0, int64(len(sgzData))),
		controller,
	)
	if err != nil {
		t.Fatalf("failed to list digest: %v", err)
	}
	found := make(map[string]bool)
	for id := range dgstMap {
		found[id] = false
	}
	zr, err := controller.Reader(bytes.NewReader(sgzData))
	if err != nil {
		t.Fatalf("failed to decompress converted stargz: %v", err)
	}
	defer zr.Close()
	tr := tar.NewReader(zr)
	for {
		h, err := tr.Next()
		if err != nil {
			if err != io.EOF {
				t.Errorf("failed to read tar entry: %v", err)
				return
			}
			break
		}
		if h.Name == TOCTarName {
			// Check the digest of TOC JSON based on the actual contents
			// It's sure that TOC JSON exists in this archive because
			// Open succeeded.
			dgstr := digest.Canonical.Digester()
			if _, err := io.Copy(dgstr.Hash(), tr); err != nil {
				t.Fatalf("failed to calculate digest of TOC JSON: %v",
					err)
			}
			if dgstr.Digest() != tocDigest {
				t.Errorf("invalid TOC JSON %q; want %q", tocDigest, dgstr.Digest())
			}
			continue
		}
		if _, ok := sgz.Lookup(h.Name); !ok {
			t.Errorf("lost stargz entry %q in the converted TOC", h.Name)
			return
		}
		var n int64
		for n < h.Size {
			ce, ok := sgz.ChunkEntryForOffset(h.Name, n)
			if !ok {
				t.Errorf("lost chunk %q(offset=%d) in the converted TOC",
					h.Name, n)
				return
			}

			// Get the original digest to make sure the file contents are kept unchanged
			// from the original tar, during the whole conversion steps.
			id := chunkID(h.Name, n, ce.ChunkSize)
			want, ok := dgstMap[id]
			if !ok {
				t.Errorf("Unexpected chunk %q(offset=%d,size=%d): %v",
					h.Name, n, ce.ChunkSize, dgstMap)
				return
			}
			found[id] = true

			// Check the file contents
			dgstr := digest.Canonical.Digester()
			if _, err := io.CopyN(dgstr.Hash(), tr, ce.ChunkSize); err != nil {
				t.Fatalf("failed to calculate digest of %q (offset=%d,size=%d)",
					h.Name, n, ce.ChunkSize)
			}
			if want != dgstr.Digest() {
				t.Errorf("Invalid contents in converted stargz %q: %q; want %q",
					h.Name, dgstr.Digest(), want)
				return
			}

			// Check the digest stored in TOC JSON
			dgstTOC, ok := digestMapTOC[ce.Offset]
			if !ok {
				t.Errorf("digest of %q(offset=%d,size=%d,chunkOffset=%d) isn't registered",
					h.Name, ce.Offset, ce.ChunkSize, ce.ChunkOffset)
			}
			if want != dgstTOC {
				t.Errorf("Invalid digest in TOCEntry %q: %q; want %q",
					h.Name, dgstTOC, want)
				return
			}

			n += ce.ChunkSize
		}
	}

	for id, ok := range found {
		if !ok {
			t.Errorf("required chunk %q not found in the converted stargz: %v", id, found)
		}
	}
}

// checkVerifyTOC checks the verification works for the TOC JSON of the passed
// stargz. It walks all entries in the stargz and checks the verifications for
// all chunks work.
func checkVerifyTOC(t *testing.T, sgzData []byte, tocDigest digest.Digest, dgstMap map[string]digest.Digest, controller TestingController) {
	sgz, err := Open(
		io.NewSectionReader(bytes.NewReader(sgzData), 0, int64(len(sgzData))),
		WithDecompressors(controller),
	)
	if err != nil {
		t.Errorf("failed to parse converted stargz: %v", err)
		return
	}
	ev, err := sgz.VerifyTOC(tocDigest)
	if err != nil {
		t.Errorf("failed to verify stargz: %v", err)
		return
	}

	found := make(map[string]bool)
	for id := range dgstMap {
		found[id] = false
	}
	zr, err := controller.Reader(bytes.NewReader(sgzData))
	if err != nil {
		t.Fatalf("failed to decompress converted stargz: %v", err)
	}
	defer zr.Close()
	tr := tar.NewReader(zr)
	for {
		h, err := tr.Next()
		if err != nil {
			if err != io.EOF {
				t.Errorf("failed to read tar entry: %v", err)
				return
			}
			break
		}
		if h.Name == TOCTarName {
			continue
		}
		if _, ok := sgz.Lookup(h.Name); !ok {
			t.Errorf("lost stargz entry %q in the converted TOC", h.Name)
			return
		}
		var n int64
		for n < h.Size {
			ce, ok := sgz.ChunkEntryForOffset(h.Name, n)
			if !ok {
				t.Errorf("lost chunk %q(offset=%d) in the converted TOC",
					h.Name, n)
				return
			}

			v, err := ev.Verifier(ce)
			if err != nil {
				t.Errorf("failed to get verifier for %q(offset=%d)", h.Name, n)
			}

			found[chunkID(h.Name, n, ce.ChunkSize)] = true

			// Check the file contents
			if _, err := io.CopyN(v, tr, ce.ChunkSize); err != nil {
				t.Fatalf("failed to get chunk of %q (offset=%d,size=%d)",
					h.Name, n, ce.ChunkSize)
			}
			if !v.Verified() {
				t.Errorf("Invalid contents in converted stargz %q (should be succeeded)",
					h.Name)
				return
			}
			n += ce.ChunkSize
		}
	}

	for id, ok := range found {
		if !ok {
			t.Errorf("required chunk %q not found in the converted stargz: %v", id, found)
		}
	}
}

// checkVerifyInvalidTOCEntryFail checks if misconfigured TOC JSON can be
// detected during the verification and the verification returns an error.
func checkVerifyInvalidTOCEntryFail(filename string) check {
	return func(t *testing.T, sgzData []byte, tocDigest digest.Digest, dgstMap map[string]digest.Digest, controller TestingController) {
		funcs := map[string]rewriteFunc{
			"lost digest in a entry": func(t *testing.T, toc *JTOC, sgz *io.SectionReader) {
				var found bool
				for _, e := range toc.Entries {
					if cleanEntryName(e.Name) == filename {
						if e.Type != "reg" && e.Type != "chunk" {
							t.Fatalf("entry %q to break must be regfile or chunk", filename)
						}
						if e.ChunkDigest == "" {
							t.Fatalf("entry %q is already invalid", filename)
						}
						e.ChunkDigest = ""
						found = true
					}
				}
				if !found {
					t.Fatalf("rewrite target not found")
				}
			},
			"duplicated entry offset": func(t *testing.T, toc *JTOC, sgz *io.SectionReader) {
				var (
					sampleEntry *TOCEntry
					targetEntry *TOCEntry
				)
				for _, e := range toc.Entries {
					if e.Type == "reg" || e.Type == "chunk" {
						if cleanEntryName(e.Name) == filename {
							targetEntry = e
						} else {
							sampleEntry = e
						}
					}
				}
				if sampleEntry == nil {
					t.Fatalf("TOC must contain at least one regfile or chunk entry other than the rewrite target")
				}
				if targetEntry == nil {
					t.Fatalf("rewrite target not found")
				}
				targetEntry.Offset = sampleEntry.Offset
			},
		}

		for name, rFunc := range funcs {
			t.Run(name, func(t *testing.T) {
				newSgz, newTocDigest := rewriteTOCJSON(t, io.NewSectionReader(bytes.NewReader(sgzData), 0, int64(len(sgzData))), rFunc, controller)
				buf := new(bytes.Buffer)
				if _, err := io.Copy(buf, newSgz); err != nil {
					t.Fatalf("failed to get converted stargz")
				}
				isgz := buf.Bytes()

				sgz, err := Open(
					io.NewSectionReader(bytes.NewReader(isgz), 0, int64(len(isgz))),
					WithDecompressors(controller),
				)
				if err != nil {
					t.Fatalf("failed to parse converted stargz: %v", err)
					return
				}
				_, err = sgz.VerifyTOC(newTocDigest)
				if err == nil {
					t.Errorf("must fail for invalid TOC")
					return
				}
			})
		}
	}
}

// checkVerifyInvalidStargzFail checks if the verification detects that the
// given stargz file doesn't match to the expected digest and returns error.
func checkVerifyInvalidStargzFail(invalid *io.SectionReader) check {
	return func(t *testing.T, sgzData []byte, tocDigest digest.Digest, dgstMap map[string]digest.Digest, controller TestingController) {
		rc, err := Build(invalid, WithChunkSize(chunkSize), WithCompression(controller))
		if err != nil {
			t.Fatalf("failed to convert stargz: %v", err)
		}
		defer rc.Close()
		buf := new(bytes.Buffer)
		if _, err := io.Copy(buf, rc); err != nil {
			t.Fatalf("failed to copy built stargz blob: %v", err)
		}
		mStargz := buf.Bytes()

		sgz, err := Open(
			io.NewSectionReader(bytes.NewReader(mStargz), 0, int64(len(mStargz))),
			WithDecompressors(controller),
		)
		if err != nil {
			t.Fatalf("failed to parse converted stargz: %v", err)
			return
		}
		_, err = sgz.VerifyTOC(tocDigest)
		if err == nil {
			t.Errorf("must fail for invalid TOC")
			return
		}
	}
}

// checkVerifyBrokenContentFail checks if the verifier detects broken contents
// that doesn't match to the expected digest and returns error.
func checkVerifyBrokenContentFail(filename string) check {
	return func(t *testing.T, sgzData []byte, tocDigest digest.Digest, dgstMap map[string]digest.Digest, controller TestingController) {
		// Parse stargz file
		sgz, err := Open(
			io.NewSectionReader(bytes.NewReader(sgzData), 0, int64(len(sgzData))),
			WithDecompressors(controller),
		)
		if err != nil {
			t.Fatalf("failed to parse converted stargz: %v", err)
			return
		}
		ev, err := sgz.VerifyTOC(tocDigest)
		if err != nil {
			t.Fatalf("failed to verify stargz: %v", err)
			return
		}

		// Open the target file
		sr, err := sgz.OpenFile(filename)
		if err != nil {
			t.Fatalf("failed to open file %q", filename)
		}
		ce, ok := sgz.ChunkEntryForOffset(filename, 0)
		if !ok {
			t.Fatalf("lost chunk %q(offset=%d) in the converted TOC", filename, 0)
			return
		}
		if ce.ChunkSize == 0 {
			t.Fatalf("file mustn't be empty")
			return
		}
		data := make([]byte, ce.ChunkSize)
		if _, err := sr.ReadAt(data, ce.ChunkOffset); err != nil {
			t.Errorf("failed to get data of a chunk of %q(offset=%q)",
				filename, ce.ChunkOffset)
		}

		// Check the broken chunk (must fail)
		v, err := ev.Verifier(ce)
		if err != nil {
			t.Fatalf("failed to get verifier for %q", filename)
		}
		broken := append([]byte{^data[0]}, data[1:]...)
		if _, err := io.CopyN(v, bytes.NewReader(broken), ce.ChunkSize); err != nil {
			t.Fatalf("failed to get chunk of %q (offset=%d,size=%d)",
				filename, ce.ChunkOffset, ce.ChunkSize)
		}
		if v.Verified() {
			t.Errorf("verification must fail for broken file chunk %q(org:%q,broken:%q)",
				filename, data, broken)
		}
	}
}

func chunkID(name string, offset, size int64) string {
	return fmt.Sprintf("%s-%d-%d", cleanEntryName(name), offset, size)
}

type rewriteFunc func(t *testing.T, toc *JTOC, sgz *io.SectionReader)

func rewriteTOCJSON(t *testing.T, sgz *io.SectionReader, rewrite rewriteFunc, controller TestingController) (newSgz io.Reader, tocDigest digest.Digest) {
	decodedJTOC, jtocOffset, err := parseStargz(sgz, controller)
	if err != nil {
		t.Fatalf("failed to extract TOC JSON: %v", err)
	}

	rewrite(t, decodedJTOC, sgz)

	tocFooter, tocDigest, err := tocAndFooter(controller, decodedJTOC, jtocOffset)
	if err != nil {
		t.Fatalf("failed to create toc and footer: %v", err)
	}

	// Reconstruct stargz file with the modified TOC JSON
	if _, err := sgz.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("failed to reset the seek position of stargz: %v", err)
	}
	return io.MultiReader(
		io.LimitReader(sgz, jtocOffset), // Original stargz (before TOC JSON)
		tocFooter,                       // Rewritten TOC and footer
	), tocDigest
}

func listDigests(sgz *io.SectionReader, controller TestingController) (map[int64]digest.Digest, error) {
	decodedJTOC, _, err := parseStargz(sgz, controller)
	if err != nil {
		return nil, err
	}
	digestMap := make(map[int64]digest.Digest)
	for _, e := range decodedJTOC.Entries {
		if e.Type == "reg" || e.Type == "chunk" {
			if e.Type == "reg" && e.Size == 0 {
				continue // ignores empty file
			}
			if e.ChunkDigest == "" {
				return nil, fmt.Errorf("ChunkDigest of %q(off=%d) not found in TOC JSON",
					e.Name, e.Offset)
			}
			d, err := digest.Parse(e.ChunkDigest)
			if err != nil {
				return nil, err
			}
			digestMap[e.Offset] = d
		}
	}
	return digestMap, nil
}

func parseStargz(sgz *io.SectionReader, controller TestingController) (decodedJTOC *JTOC, jtocOffset int64, err error) {
	fSize := controller.FooterSize()
	footer := make([]byte, fSize)
	if _, err := sgz.ReadAt(footer, sgz.Size()-fSize); err != nil {
		return nil, 0, errors.Wrap(err, "error reading footer")
	}
	_, tocOffset, _, err := controller.ParseFooter(footer[positive(int64(len(footer))-fSize):])
	if err != nil {
		return nil, 0, errors.Wrapf(err, "failed to parse footer")
	}

	// Decode the TOC JSON
	tocReader := io.NewSectionReader(sgz, tocOffset, sgz.Size()-tocOffset-fSize)
	decodedJTOC, _, err = controller.ParseTOC(tocReader)
	if err != nil {
		return nil, 0, errors.Wrap(err, "failed to parse TOC")
	}
	return decodedJTOC, tocOffset, nil
}

func testWriteAndOpen(t *testing.T, controllers ...TestingController) {
	const content = "Some contents"
	invalidUtf8 := "\xff\xfe\xfd"

	xAttrFile := xAttr{"foo": "bar", "invalid-utf8": invalidUtf8}
	sampleOwner := owner{uid: 50, gid: 100}

	tests := []struct {
		name      string
		chunkSize int
		in        []tarEntry
		want      []stargzCheck
		wantNumGz int // expected number of streams

		wantNumGzLossLess  int // expected number of streams (> 0) in lossless mode if it's different from wantNumGz
		wantFailOnLossLess bool
	}{
		{
			name:              "empty",
			in:                tarOf(),
			wantNumGz:         2, // empty tar + TOC + footer
			wantNumGzLossLess: 3, // empty tar + TOC + footer
			want: checks(
				numTOCEntries(0),
			),
		},
		{
			name: "1dir_1empty_file",
			in: tarOf(
				dir("foo/"),
				file("foo/bar.txt", ""),
			),
			wantNumGz: 3, // dir, TOC, footer
			want: checks(
				numTOCEntries(2),
				hasDir("foo/"),
				hasFileLen("foo/bar.txt", 0),
				entryHasChildren("foo", "bar.txt"),
				hasFileDigest("foo/bar.txt", digestFor("")),
			),
		},
		{
			name: "1dir_1file",
			in: tarOf(
				dir("foo/"),
				file("foo/bar.txt", content, xAttrFile),
			),
			wantNumGz: 4, // var dir, foo.txt alone, TOC, footer
			want: checks(
				numTOCEntries(2),
				hasDir("foo/"),
				hasFileLen("foo/bar.txt", len(content)),
				hasFileDigest("foo/bar.txt", digestFor(content)),
				hasFileContentsRange("foo/bar.txt", 0, content),
				hasFileContentsRange("foo/bar.txt", 1, content[1:]),
				entryHasChildren("", "foo"),
				entryHasChildren("foo", "bar.txt"),
				hasFileXattrs("foo/bar.txt", "foo", "bar"),
				hasFileXattrs("foo/bar.txt", "invalid-utf8", invalidUtf8),
			),
		},
		{
			name: "2meta_2file",
			in: tarOf(
				dir("bar/", sampleOwner),
				dir("foo/", sampleOwner),
				file("foo/bar.txt", content, sampleOwner),
			),
			wantNumGz: 4, // both dirs, foo.txt alone, TOC, footer
			want: checks(
				numTOCEntries(3),
				hasDir("bar/"),
				hasDir("foo/"),
				hasFileLen("foo/bar.txt", len(content)),
				entryHasChildren("", "bar", "foo"),
				entryHasChildren("foo", "bar.txt"),
				hasChunkEntries("foo/bar.txt", 1),
				hasEntryOwner("bar/", sampleOwner),
				hasEntryOwner("foo/", sampleOwner),
				hasEntryOwner("foo/bar.txt", sampleOwner),
			),
		},
		{
			name: "3dir",
			in: tarOf(
				dir("bar/"),
				dir("foo/"),
				dir("foo/bar/"),
			),
			wantNumGz: 3, // 3 dirs, TOC, footer
			want: checks(
				hasDirLinkCount("bar/", 2),
				hasDirLinkCount("foo/", 3),
				hasDirLinkCount("foo/bar/", 2),
			),
		},
		{
			name: "symlink",
			in: tarOf(
				dir("foo/"),
				symlink("foo/bar", "../../x"),
			),
			wantNumGz: 3, // metas + TOC + footer
			want: checks(
				numTOCEntries(2),
				hasSymlink("foo/bar", "../../x"),
				entryHasChildren("", "foo"),
				entryHasChildren("foo", "bar"),
			),
		},
		{
			name:      "chunked_file",
			chunkSize: 4,
			in: tarOf(
				dir("foo/"),
				file("foo/big.txt", "This "+"is s"+"uch "+"a bi"+"g fi"+"le"),
			),
			wantNumGz: 9,
			want: checks(
				numTOCEntries(7), // 1 for foo dir, 6 for the foo/big.txt file
				hasDir("foo/"),
				hasFileLen("foo/big.txt", len("This is such a big file")),
				hasFileDigest("foo/big.txt", digestFor("This is such a big file")),
				hasFileContentsRange("foo/big.txt", 0, "This is such a big file"),
				hasFileContentsRange("foo/big.txt", 1, "his is such a big file"),
				hasFileContentsRange("foo/big.txt", 2, "is is such a big file"),
				hasFileContentsRange("foo/big.txt", 3, "s is such a big file"),
				hasFileContentsRange("foo/big.txt", 4, " is such a big file"),
				hasFileContentsRange("foo/big.txt", 5, "is such a big file"),
				hasFileContentsRange("foo/big.txt", 6, "s such a big file"),
				hasFileContentsRange("foo/big.txt", 7, " such a big file"),
				hasFileContentsRange("foo/big.txt", 8, "such a big file"),
				hasFileContentsRange("foo/big.txt", 9, "uch a big file"),
				hasFileContentsRange("foo/big.txt", 10, "ch a big file"),
				hasFileContentsRange("foo/big.txt", 11, "h a big file"),
				hasFileContentsRange("foo/big.txt", 12, " a big file"),
				hasFileContentsRange("foo/big.txt", len("This is such a big file")-1, ""),
				hasChunkEntries("foo/big.txt", 6),
			),
		},
		{
			name: "recursive",
			in: tarOf(
				dir("/", sampleOwner),
				dir("bar/", sampleOwner),
				dir("foo/", sampleOwner),
				file("foo/bar.txt", content, sampleOwner),
			),
			wantNumGz: 4, // dirs, bar.txt alone, TOC, footer
			want: checks(
				maxDepth(2), // 0: root directory, 1: "foo/", 2: "bar.txt"
			),
		},
		{
			name: "block_char_fifo",
			in: tarOf(
				tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
					return w.WriteHeader(&tar.Header{
						Name:     prefix + "b",
						Typeflag: tar.TypeBlock,
						Devmajor: 123,
						Devminor: 456,
						Format:   format,
					})
				}),
				tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
					return w.WriteHeader(&tar.Header{
						Name:     prefix + "c",
						Typeflag: tar.TypeChar,
						Devmajor: 111,
						Devminor: 222,
						Format:   format,
					})
				}),
				tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
					return w.WriteHeader(&tar.Header{
						Name:     prefix + "f",
						Typeflag: tar.TypeFifo,
						Format:   format,
					})
				}),
			),
			wantNumGz: 3,
			want: checks(
				lookupMatch("b", &TOCEntry{Name: "b", Type: "block", DevMajor: 123, DevMinor: 456, NumLink: 1}),
				lookupMatch("c", &TOCEntry{Name: "c", Type: "char", DevMajor: 111, DevMinor: 222, NumLink: 1}),
				lookupMatch("f", &TOCEntry{Name: "f", Type: "fifo", NumLink: 1}),
			),
		},
		{
			name: "modes",
			in: tarOf(
				dir("foo1/", 0755|os.ModeDir|os.ModeSetgid),
				file("foo1/bar1", content, 0700|os.ModeSetuid),
				file("foo1/bar2", content, 0755|os.ModeSetgid),
				dir("foo2/", 0755|os.ModeDir|os.ModeSticky),
				file("foo2/bar3", content, 0755|os.ModeSticky),
				dir("foo3/", 0755|os.ModeDir),
				file("foo3/bar4", content, os.FileMode(0700)),
				file("foo3/bar5", content, os.FileMode(0755)),
			),
			wantNumGz: 8, // dir, bar1 alone, bar2 alone + dir, bar3 alone + dir, bar4 alone, bar5 alone, TOC, footer
			want: checks(
				hasMode("foo1/", 0755|os.ModeDir|os.ModeSetgid),
				hasMode("foo1/bar1", 0700|os.ModeSetuid),
				hasMode("foo1/bar2", 0755|os.ModeSetgid),
				hasMode("foo2/", 0755|os.ModeDir|os.ModeSticky),
				hasMode("foo2/bar3", 0755|os.ModeSticky),
				hasMode("foo3/", 0755|os.ModeDir),
				hasMode("foo3/bar4", os.FileMode(0700)),
				hasMode("foo3/bar5", os.FileMode(0755)),
			),
		},
		{
			name: "lossy",
			in: tarOf(
				dir("bar/", sampleOwner),
				dir("foo/", sampleOwner),
				file("foo/bar.txt", content, sampleOwner),
				file(TOCTarName, "dummy"), // ignored by the writer. (lossless write returns error)
			),
			wantNumGz: 4, // both dirs, foo.txt alone, TOC, footer
			want: checks(
				numTOCEntries(3),
				hasDir("bar/"),
				hasDir("foo/"),
				hasFileLen("foo/bar.txt", len(content)),
				entryHasChildren("", "bar", "foo"),
				entryHasChildren("foo", "bar.txt"),
				hasChunkEntries("foo/bar.txt", 1),
				hasEntryOwner("bar/", sampleOwner),
				hasEntryOwner("foo/", sampleOwner),
				hasEntryOwner("foo/bar.txt", sampleOwner),
			),
			wantFailOnLossLess: true,
		},
	}

	for _, tt := range tests {
		for _, cl := range controllers {
			cl := cl
			for _, prefix := range allowedPrefix {
				prefix := prefix
				for _, srcTarFormat := range []tar.Format{tar.FormatUSTAR, tar.FormatPAX, tar.FormatGNU} {
					srcTarFormat := srcTarFormat
					for _, lossless := range []bool{true, false} {
						t.Run(tt.name+"-"+fmt.Sprintf("compression=%v,prefix=%q,lossless=%v,format=%s", cl, prefix, lossless, srcTarFormat), func(t *testing.T) {
							var tr io.Reader = buildTar(t, tt.in, prefix, srcTarFormat)
							origTarDgstr := digest.Canonical.Digester()
							tr = io.TeeReader(tr, origTarDgstr.Hash())
							var stargzBuf bytes.Buffer
							w := NewWriterWithCompressor(&stargzBuf, cl)
							w.ChunkSize = tt.chunkSize
							if lossless {
								err := w.AppendTarLossLess(tr)
								if tt.wantFailOnLossLess {
									if err != nil {
										return // expected to fail
									}
									t.Fatalf("Append wanted to fail on lossless")
								}
								if err != nil {
									t.Fatalf("Append(lossless): %v", err)
								}
							} else {
								if err := w.AppendTar(tr); err != nil {
									t.Fatalf("Append: %v", err)
								}
							}
							if _, err := w.Close(); err != nil {
								t.Fatalf("Writer.Close: %v", err)
							}
							b := stargzBuf.Bytes()

							if lossless {
								// Check if the result blob reserves original tar metadata
								rc, err := Unpack(io.NewSectionReader(bytes.NewReader(b), 0, int64(len(b))), cl)
								if err != nil {
									t.Errorf("failed to decompress blob: %v", err)
									return
								}
								defer rc.Close()
								resultDgstr := digest.Canonical.Digester()
								if _, err := io.Copy(resultDgstr.Hash(), rc); err != nil {
									t.Errorf("failed to read result decompressed blob: %v", err)
									return
								}
								if resultDgstr.Digest() != origTarDgstr.Digest() {
									t.Errorf("lossy compression occurred: digest=%v; want %v",
										resultDgstr.Digest(), origTarDgstr.Digest())
									return
								}
							}

							diffID := w.DiffID()
							wantDiffID := cl.DiffIDOf(t, b)
							if diffID != wantDiffID {
								t.Errorf("DiffID = %q; want %q", diffID, wantDiffID)
							}

							got := cl.CountStreams(t, b)
							wantNumGz := tt.wantNumGz
							if lossless && tt.wantNumGzLossLess > 0 {
								wantNumGz = tt.wantNumGzLossLess
							}
							if got != wantNumGz {
								t.Errorf("number of streams = %d; want %d", got, wantNumGz)
							}

							telemetry, checkCalled := newCalledTelemetry()
							r, err := Open(
								io.NewSectionReader(bytes.NewReader(b), 0, int64(len(b))),
								WithDecompressors(cl),
								WithTelemetry(telemetry),
							)
							if err != nil {
								t.Fatalf("stargz.Open: %v", err)
							}
							if err := checkCalled(); err != nil {
								t.Errorf("telemetry failure: %v", err)
							}
							for _, want := range tt.want {
								want.check(t, r)
							}
						})
					}
				}
			}
		}
	}
}

func newCalledTelemetry() (telemetry *Telemetry, check func() error) {
	var getFooterLatencyCalled bool
	var getTocLatencyCalled bool
	var deserializeTocLatencyCalled bool
	return &Telemetry{
			func(time.Time) { getFooterLatencyCalled = true },
			func(time.Time) { getTocLatencyCalled = true },
			func(time.Time) { deserializeTocLatencyCalled = true },
		}, func() error {
			var allErr []error
			if !getFooterLatencyCalled {
				allErr = append(allErr, fmt.Errorf("metrics GetFooterLatency isn't called"))
			}
			if !getTocLatencyCalled {
				allErr = append(allErr, fmt.Errorf("metrics GetTocLatency isn't called"))
			}
			if !deserializeTocLatencyCalled {
				allErr = append(allErr, fmt.Errorf("metrics DeserializeTocLatency isn't called"))
			}
			return errorutil.Aggregate(allErr)
		}
}

func digestFor(content string) string {
	sum := sha256.Sum256([]byte(content))
	return fmt.Sprintf("sha256:%x", sum)
}

type numTOCEntries int

func (n numTOCEntries) check(t *testing.T, r *Reader) {
	if r.toc == nil {
		t.Fatal("nil TOC")
	}
	if got, want := len(r.toc.Entries), int(n); got != want {
		t.Errorf("got %d TOC entries; want %d", got, want)
	}
	t.Logf("got TOC entries:")
	for i, ent := range r.toc.Entries {
		entj, _ := json.Marshal(ent)
		t.Logf("  [%d]: %s\n", i, entj)
	}
	if t.Failed() {
		t.FailNow()
	}
}

func checks(s ...stargzCheck) []stargzCheck { return s }

type stargzCheck interface {
	check(t *testing.T, r *Reader)
}

type stargzCheckFn func(*testing.T, *Reader)

func (f stargzCheckFn) check(t *testing.T, r *Reader) { f(t, r) }

func maxDepth(max int) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		e, ok := r.Lookup("")
		if !ok {
			t.Fatal("root directory not found")
		}
		d, err := getMaxDepth(t, e, 0, 10*max)
		if err != nil {
			t.Errorf("failed to get max depth (wanted %d): %v", max, err)
			return
		}
		if d != max {
			t.Errorf("invalid depth %d; want %d", d, max)
			return
		}
	})
}

func getMaxDepth(t *testing.T, e *TOCEntry, current, limit int) (max int, rErr error) {
	if current > limit {
		return -1, fmt.Errorf("walkMaxDepth: exceeds limit: current:%d > limit:%d",
			current, limit)
	}
	max = current
	e.ForeachChild(func(baseName string, ent *TOCEntry) bool {
		t.Logf("%q(basename:%q) is child of %q\n", ent.Name, baseName, e.Name)
		d, err := getMaxDepth(t, ent, current+1, limit)
		if err != nil {
			rErr = err
			return false
		}
		if d > max {
			max = d
		}
		return true
	})
	return
}

func hasFileLen(file string, wantLen int) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		for _, ent := range r.toc.Entries {
			if ent.Name == file {
				if ent.Type != "reg" {
					t.Errorf("file type of %q is %q; want \"reg\"", file, ent.Type)
				} else if ent.Size != int64(wantLen) {
					t.Errorf("file size of %q = %d; want %d", file, ent.Size, wantLen)
				}
				return
			}
		}
		t.Errorf("file %q not found", file)
	})
}

func hasFileXattrs(file, name, value string) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		for _, ent := range r.toc.Entries {
			if ent.Name == file {
				if ent.Type != "reg" {
					t.Errorf("file type of %q is %q; want \"reg\"", file, ent.Type)
				}
				if ent.Xattrs == nil {
					t.Errorf("file %q has no xattrs", file)
					return
				}
				valueFound, found := ent.Xattrs[name]
				if !found {
					t.Errorf("file %q has no xattr %q", file, name)
					return
				}
				if string(valueFound) != value {
					t.Errorf("file %q has xattr %q with value %q instead of %q", file, name, valueFound, value)
				}

				return
			}
		}
		t.Errorf("file %q not found", file)
	})
}

func hasFileDigest(file string, digest string) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		ent, ok := r.Lookup(file)
		if !ok {
			t.Fatalf("didn't find TOCEntry for file %q", file)
		}
		if ent.Digest != digest {
			t.Fatalf("Digest(%q) = %q, want %q", file, ent.Digest, digest)
		}
	})
}

func hasFileContentsRange(file string, offset int, want string) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		f, err := r.OpenFile(file)
		if err != nil {
			t.Fatal(err)
		}
		got := make([]byte, len(want))
		n, err := f.ReadAt(got, int64(offset))
		if err != nil {
			t.Fatalf("ReadAt(len %d, offset %d) = %v, %v", len(got), offset, n, err)
		}
		if string(got) != want {
			t.Fatalf("ReadAt(len %d, offset %d) = %q, want %q", len(got), offset, got, want)
		}
	})
}

func hasChunkEntries(file string, wantChunks int) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		ent, ok := r.Lookup(file)
		if !ok {
			t.Fatalf("no file for %q", file)
		}
		if ent.Type != "reg" {
			t.Fatalf("file %q has unexpected type %q; want reg", file, ent.Type)
		}
		chunks := r.getChunks(ent)
		if len(chunks) != wantChunks {
			t.Errorf("len(r.getChunks(%q)) = %d; want %d", file, len(chunks), wantChunks)
			return
		}
		f := chunks[0]

		var gotChunks []*TOCEntry
		var last *TOCEntry
		for off := int64(0); off < f.Size; off++ {
			e, ok := r.ChunkEntryForOffset(file, off)
			if !ok {
				t.Errorf("no ChunkEntryForOffset at %d", off)
				return
			}
			if last != e {
				gotChunks = append(gotChunks, e)
				last = e
			}
		}
		if !reflect.DeepEqual(chunks, gotChunks) {
			t.Errorf("gotChunks=%d, want=%d; contents mismatch", len(gotChunks), wantChunks)
		}

		// And verify the NextOffset
		for i := 0; i < len(gotChunks)-1; i++ {
			ci := gotChunks[i]
			cnext := gotChunks[i+1]
			if ci.NextOffset() != cnext.Offset {
				t.Errorf("chunk %d NextOffset %d != next chunk's Offset of %d", i, ci.NextOffset(), cnext.Offset)
			}
		}
	})
}

func entryHasChildren(dir string, want ...string) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		want := append([]string(nil), want...)
		var got []string
		ent, ok := r.Lookup(dir)
		if !ok {
			t.Fatalf("didn't find TOCEntry for dir node %q", dir)
		}
		for baseName := range ent.children {
			got = append(got, baseName)
		}
		sort.Strings(got)
		sort.Strings(want)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("children of %q = %q; want %q", dir, got, want)
		}
	})
}

func hasDir(file string) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		for _, ent := range r.toc.Entries {
			if ent.Name == cleanEntryName(file) {
				if ent.Type != "dir" {
					t.Errorf("file type of %q is %q; want \"dir\"", file, ent.Type)
				}
				return
			}
		}
		t.Errorf("directory %q not found", file)
	})
}

func hasDirLinkCount(file string, count int) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		for _, ent := range r.toc.Entries {
			if ent.Name == cleanEntryName(file) {
				if ent.Type != "dir" {
					t.Errorf("file type of %q is %q; want \"dir\"", file, ent.Type)
					return
				}
				if ent.NumLink != count {
					t.Errorf("link count of %q = %d; want %d", file, ent.NumLink, count)
				}
				return
			}
		}
		t.Errorf("directory %q not found", file)
	})
}

func hasMode(file string, mode os.FileMode) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		for _, ent := range r.toc.Entries {
			if ent.Name == cleanEntryName(file) {
				if ent.Stat().Mode() != mode {
					t.Errorf("invalid mode: got %v; want %v", ent.Stat().Mode(), mode)
					return
				}
				return
			}
		}
		t.Errorf("file %q not found", file)
	})
}

func hasSymlink(file, target string) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		for _, ent := range r.toc.Entries {
			if ent.Name == file {
				if ent.Type != "symlink" {
					t.Errorf("file type of %q is %q; want \"symlink\"", file, ent.Type)
				} else if ent.LinkName != target {
					t.Errorf("link target of symlink %q is %q; want %q", file, ent.LinkName, target)
				}
				return
			}
		}
		t.Errorf("symlink %q not found", file)
	})
}

func lookupMatch(name string, want *TOCEntry) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		e, ok := r.Lookup(name)
		if !ok {
			t.Fatalf("failed to Lookup entry %q", name)
		}
		if !reflect.DeepEqual(e, want) {
			t.Errorf("entry %q mismatch.\n got: %+v\nwant: %+v\n", name, e, want)
		}

	})
}

func hasEntryOwner(entry string, owner owner) stargzCheck {
	return stargzCheckFn(func(t *testing.T, r *Reader) {
		ent, ok := r.Lookup(strings.TrimSuffix(entry, "/"))
		if !ok {
			t.Errorf("entry %q not found", entry)
			return
		}
		if ent.UID != owner.uid || ent.GID != owner.gid {
			t.Errorf("entry %q has invalid owner (uid:%d, gid:%d) instead of (uid:%d, gid:%d)", entry, ent.UID, ent.GID, owner.uid, owner.gid)
			return
		}
	})
}

func tarOf(s ...tarEntry) []tarEntry { return s }

type tarEntry interface {
	appendTar(tw *tar.Writer, prefix string, format tar.Format) error
}

type tarEntryFunc func(*tar.Writer, string, tar.Format) error

func (f tarEntryFunc) appendTar(tw *tar.Writer, prefix string, format tar.Format) error {
	return f(tw, prefix, format)
}

func buildTar(t *testing.T, ents []tarEntry, prefix string, opts ...interface{}) *io.SectionReader {
	format := tar.FormatUnknown
	for _, opt := range opts {
		switch v := opt.(type) {
		case tar.Format:
			format = v
		default:
			panic(fmt.Errorf("unsupported opt for buildTar: %v", opt))
		}
	}
	buf := new(bytes.Buffer)
	tw := tar.NewWriter(buf)
	for _, ent := range ents {
		if err := ent.appendTar(tw, prefix, format); err != nil {
			t.Fatalf("building input tar: %v", err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Errorf("closing write of input tar: %v", err)
	}
	data := append(buf.Bytes(), make([]byte, 100)...) // append empty bytes at the tail to see lossless works
	return io.NewSectionReader(bytes.NewReader(data), 0, int64(len(data)))
}

func dir(name string, opts ...interface{}) tarEntry {
	return tarEntryFunc(func(tw *tar.Writer, prefix string, format tar.Format) error {
		var o owner
		mode := os.FileMode(0755)
		for _, opt := range opts {
			switch v := opt.(type) {
			case owner:
				o = v
			case os.FileMode:
				mode = v
			default:
				return errors.New("unsupported opt")
			}
		}
		if !strings.HasSuffix(name, "/") {
			panic(fmt.Sprintf("missing trailing slash in dir %q ", name))
		}
		tm, err := fileModeToTarMode(mode)
		if err != nil {
			return err
		}
		return tw.WriteHeader(&tar.Header{
			Typeflag: tar.TypeDir,
			Name:     prefix + name,
			Mode:     tm,
			Uid:      o.uid,
			Gid:      o.gid,
			Format:   format,
		})
	})
}

// xAttr are extended attributes to set on test files created with the file func.
type xAttr map[string]string

// owner is owner ot set on test files and directories with the file and dir functions.
type owner struct {
	uid int
	gid int
}

func file(name, contents string, opts ...interface{}) tarEntry {
	return tarEntryFunc(func(tw *tar.Writer, prefix string, format tar.Format) error {
		var xattrs xAttr
		var o owner
		mode := os.FileMode(0644)
		for _, opt := range opts {
			switch v := opt.(type) {
			case xAttr:
				xattrs = v
			case owner:
				o = v
			case os.FileMode:
				mode = v
			default:
				return errors.New("unsupported opt")
			}
		}
		if strings.HasSuffix(name, "/") {
			return fmt.Errorf("bogus trailing slash in file %q", name)
		}
		tm, err := fileModeToTarMode(mode)
		if err != nil {
			return err
		}
		if len(xattrs) > 0 {
			format = tar.FormatPAX // only PAX supports xattrs
		}
		if err := tw.WriteHeader(&tar.Header{
			Typeflag: tar.TypeReg,
			Name:     prefix + name,
			Mode:     tm,
			Xattrs:   xattrs,
			Size:     int64(len(contents)),
			Uid:      o.uid,
			Gid:      o.gid,
			Format:   format,
		}); err != nil {
			return err
		}
		_, err = io.WriteString(tw, contents)
		return err
	})
}

func symlink(name, target string) tarEntry {
	return tarEntryFunc(func(tw *tar.Writer, prefix string, format tar.Format) error {
		return tw.WriteHeader(&tar.Header{
			Typeflag: tar.TypeSymlink,
			Name:     prefix + name,
			Linkname: target,
			Mode:     0644,
			Format:   format,
		})
	})
}

func link(name string, linkname string) tarEntry {
	now := time.Now()
	return tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
		return w.WriteHeader(&tar.Header{
			Typeflag: tar.TypeLink,
			Name:     prefix + name,
			Linkname: linkname,
			ModTime:  now,
			Format:   format,
		})
	})
}

func chardev(name string, major, minor int64) tarEntry {
	now := time.Now()
	return tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
		return w.WriteHeader(&tar.Header{
			Typeflag: tar.TypeChar,
			Name:     prefix + name,
			Devmajor: major,
			Devminor: minor,
			ModTime:  now,
			Format:   format,
		})
	})
}

func blockdev(name string, major, minor int64) tarEntry {
	now := time.Now()
	return tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
		return w.WriteHeader(&tar.Header{
			Typeflag: tar.TypeBlock,
			Name:     prefix + name,
			Devmajor: major,
			Devminor: minor,
			ModTime:  now,
			Format:   format,
		})
	})
}
func fifo(name string) tarEntry {
	now := time.Now()
	return tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
		return w.WriteHeader(&tar.Header{
			Typeflag: tar.TypeFifo,
			Name:     prefix + name,
			ModTime:  now,
			Format:   format,
		})
	})
}

func prefetchLandmark() tarEntry {
	return tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
		if err := w.WriteHeader(&tar.Header{
			Name:     PrefetchLandmark,
			Typeflag: tar.TypeReg,
			Size:     int64(len([]byte{landmarkContents})),
			Format:   format,
		}); err != nil {
			return err
		}
		contents := []byte{landmarkContents}
		if _, err := io.CopyN(w, bytes.NewReader(contents), int64(len(contents))); err != nil {
			return err
		}
		return nil
	})
}

func noPrefetchLandmark() tarEntry {
	return tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
		if err := w.WriteHeader(&tar.Header{
			Name:     NoPrefetchLandmark,
			Typeflag: tar.TypeReg,
			Size:     int64(len([]byte{landmarkContents})),
			Format:   format,
		}); err != nil {
			return err
		}
		contents := []byte{landmarkContents}
		if _, err := io.CopyN(w, bytes.NewReader(contents), int64(len(contents))); err != nil {
			return err
		}
		return nil
	})
}

func regDigest(t *testing.T, name string, contentStr string, digestMap map[string]digest.Digest) tarEntry {
	if digestMap == nil {
		t.Fatalf("digest map mustn't be nil")
	}
	content := []byte(contentStr)

	var n int64
	for n < int64(len(content)) {
		size := int64(chunkSize)
		remain := int64(len(content)) - n
		if remain < size {
			size = remain
		}
		dgstr := digest.Canonical.Digester()
		if _, err := io.CopyN(dgstr.Hash(), bytes.NewReader(content[n:n+size]), size); err != nil {
			t.Fatalf("failed to calculate digest of %q (name=%q,offset=%d,size=%d)",
				string(content[n:n+size]), name, n, size)
		}
		digestMap[chunkID(name, n, size)] = dgstr.Digest()
		n += size
	}

	return tarEntryFunc(func(w *tar.Writer, prefix string, format tar.Format) error {
		if err := w.WriteHeader(&tar.Header{
			Typeflag: tar.TypeReg,
			Name:     prefix + name,
			Size:     int64(len(content)),
			Format:   format,
		}); err != nil {
			return err
		}
		if _, err := io.CopyN(w, bytes.NewReader(content), int64(len(content))); err != nil {
			return err
		}
		return nil
	})
}

func fileModeToTarMode(mode os.FileMode) (int64, error) {
	h, err := tar.FileInfoHeader(fileInfoOnlyMode(mode), "")
	if err != nil {
		return 0, err
	}
	return h.Mode, nil
}

// fileInfoOnlyMode is os.FileMode that populates only file mode.
type fileInfoOnlyMode os.FileMode

func (f fileInfoOnlyMode) Name() string       { return "" }
func (f fileInfoOnlyMode) Size() int64        { return 0 }
func (f fileInfoOnlyMode) Mode() os.FileMode  { return os.FileMode(f) }
func (f fileInfoOnlyMode) ModTime() time.Time { return time.Now() }
func (f fileInfoOnlyMode) IsDir() bool        { return os.FileMode(f).IsDir() }
func (f fileInfoOnlyMode) Sys() interface{}   { return nil }
