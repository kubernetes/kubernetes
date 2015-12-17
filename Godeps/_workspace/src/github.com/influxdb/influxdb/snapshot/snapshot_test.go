package snapshot_test

import (
	"bytes"
	"io"
	"io/ioutil"
	"reflect"
	"testing"
	"time"

	"github.com/influxdb/influxdb/snapshot"
)

// Ensure a manifest can be diff'd so that only newer files are retrieved.
func TestManifest_Diff(t *testing.T) {
	for i, tt := range []struct {
		s      *snapshot.Manifest
		other  *snapshot.Manifest
		result *snapshot.Manifest
	}{
		// 0. Mixed higher, lower, equal indices.
		{
			s: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // remove: older
				{Name: "b", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // remove: equal date
				{Name: "c", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // keep: newer
				{Name: "d", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // keep: newer
			}},
			other: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", ModTime: time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "b", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "c", ModTime: time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "d", ModTime: time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
			result: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "c", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "d", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
		},

		// 1. Files in other-only should not be added to diff.
		{
			s: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
			other: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", ModTime: time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "b", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
			result: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
		},

		// 2. Files in s-only should be added to diff.
		{
			s: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
			other: &snapshot.Manifest{Files: []snapshot.File{}},
			result: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
		},

		// 3. Empty snapshots should return empty diffs.
		{
			s:      &snapshot.Manifest{Files: []snapshot.File{}},
			other:  &snapshot.Manifest{Files: []snapshot.File{}},
			result: &snapshot.Manifest{Files: nil},
		},
	} {
		result := tt.s.Diff(tt.other)
		if !reflect.DeepEqual(tt.result, result) {
			t.Errorf("%d. mismatch:\n\nexp=%#v\n\ngot=%#v", i, tt.result, result)
		}
	}
}

// Ensure a snapshot can be merged so that the newest files from the two snapshots are returned.
func TestSnapshot_Merge(t *testing.T) {
	for i, tt := range []struct {
		s      *snapshot.Manifest
		other  *snapshot.Manifest
		result *snapshot.Manifest
	}{
		// 0. Mixed higher, lower, equal indices.
		{
			s: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", Size: 10, ModTime: time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "b", Size: 10, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // keep: same, first
				{Name: "c", Size: 10, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // keep: higher
				{Name: "e", Size: 10, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // keep: higher
			}},
			other: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", Size: 20, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // keep: higher
				{Name: "b", Size: 20, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "c", Size: 20, ModTime: time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "d", Size: 20, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}, // keep: new
				{Name: "e", Size: 20, ModTime: time.Date(1999, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
			result: &snapshot.Manifest{Files: []snapshot.File{
				{Name: "a", Size: 20, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "b", Size: 10, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "c", Size: 10, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "d", Size: 20, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
				{Name: "e", Size: 10, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
			}},
		},
	} {
		result := tt.s.Merge(tt.other)
		if !reflect.DeepEqual(tt.result, result) {
			t.Errorf("%d. mismatch:\n\nexp=%#v\n\ngot=%#v", i, tt.result, result)
		}
	}
}

// Ensure a writer can write a set of files to an archive
func TestWriter(t *testing.T) {
	// Create a new writer with a snapshot and file writers.
	sw := snapshot.NewWriter()
	sw.Manifest.Files = []snapshot.File{
		{Name: "meta", Size: 3, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
		{Name: "shards/1", Size: 5, ModTime: time.Date(2000, time.February, 1, 0, 0, 0, 0, time.UTC)},
	}
	sw.FileWriters["meta"] = &bufCloser{Buffer: *bytes.NewBufferString("foo")}
	sw.FileWriters["shards/1"] = &bufCloser{Buffer: *bytes.NewBufferString("55555")}

	// Write the snapshot to a buffer.
	var buf bytes.Buffer
	if _, err := sw.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}

	// Ensure file writers are closed as they're writing.
	if !sw.FileWriters["meta"].(*bufCloser).closed {
		t.Fatal("meta file writer not closed")
	} else if !sw.FileWriters["shards/1"].(*bufCloser).closed {
		t.Fatal("shards/1 file writer not closed")
	}

	// Close writer.
	if err := sw.Close(); err != nil {
		t.Fatal(err)
	}

	// Read snapshot from buffer.
	sr := snapshot.NewReader(&buf)

	// Read the manifest.
	if ss, err := sr.Manifest(); err != nil {
		t.Fatalf("unexpected error(manifest): %s", err)
	} else if !reflect.DeepEqual(sw.Manifest, ss) {
		t.Fatalf("manifest mismatch:\n\nexp=%#v\n\ngot=%#v", sw.Manifest, ss)
	}

	// Next should be the meta file.
	if f, err := sr.Next(); err != nil {
		t.Fatalf("unexpected error(meta): %s", err)
	} else if !reflect.DeepEqual(f, snapshot.File{Name: "meta", Size: 3, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}) {
		t.Fatalf("file mismatch(meta): %#v", f)
	} else if b := MustReadAll(sr); string(b) != `foo` {
		t.Fatalf("unexpected file(meta): %s", b)
	}

	// Next should be the shard file.
	if f, err := sr.Next(); err != nil {
		t.Fatalf("unexpected error(shards/1): %s", err)
	} else if !reflect.DeepEqual(f, snapshot.File{Name: "shards/1", Size: 5, ModTime: time.Date(2000, time.February, 1, 0, 0, 0, 0, time.UTC)}) {
		t.Fatalf("file mismatch(shards/1): %#v", f)
	} else if b := MustReadAll(sr); string(b) != `55555` {
		t.Fatalf("unexpected file(shards/1): %s", b)
	}

	// Check for end of snapshot.
	if _, err := sr.Next(); err != io.EOF {
		t.Fatalf("expected EOF: %s", err)
	}
}

// Ensure a writer closes unused file writers.
func TestWriter_CloseUnused(t *testing.T) {
	// Create a new writer with a manifest and file writers.
	sw := snapshot.NewWriter()
	sw.Manifest.Files = []snapshot.File{
		{Name: "meta", Size: 3},
	}
	sw.FileWriters["meta"] = &bufCloser{Buffer: *bytes.NewBufferString("foo")}
	sw.FileWriters["other"] = &bufCloser{Buffer: *bytes.NewBufferString("55555")}

	// Write the snapshot to a buffer.
	var buf bytes.Buffer
	if _, err := sw.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}

	// Ensure other writer is closed.
	// This should happen at the beginning of the write so that it doesn't have
	// to wait until the close of the whole writer.
	if !sw.FileWriters["other"].(*bufCloser).closed {
		t.Fatal("'other' file writer not closed")
	}
}

// Ensure a MultiReader can read from multiple snapshots.
func TestMultiReader(t *testing.T) {
	var sw *snapshot.Writer
	bufs := make([]bytes.Buffer, 2)

	// Snapshot #1
	sw = snapshot.NewWriter()
	sw.Manifest.Files = []snapshot.File{
		{Name: "meta", Size: 3, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
		{Name: "shards/1", Size: 5, ModTime: time.Date(2000, time.February, 1, 0, 0, 0, 0, time.UTC)},
	}
	sw.FileWriters["meta"] = &bufCloser{Buffer: *bytes.NewBufferString("foo")}
	sw.FileWriters["shards/1"] = &bufCloser{Buffer: *bytes.NewBufferString("55555")}
	if _, err := sw.WriteTo(&bufs[0]); err != nil {
		t.Fatal(err)
	} else if err = sw.Close(); err != nil {
		t.Fatal(err)
	}

	// Snapshot #2
	sw = snapshot.NewWriter()
	sw.Manifest.Files = []snapshot.File{
		{Name: "meta", Size: 3, ModTime: time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC)},
		{Name: "shards/2", Size: 6, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)},
	}
	sw.FileWriters["meta"] = &bufCloser{Buffer: *bytes.NewBufferString("bar")}
	sw.FileWriters["shards/2"] = &bufCloser{Buffer: *bytes.NewBufferString("666666")}
	if _, err := sw.WriteTo(&bufs[1]); err != nil {
		t.Fatal(err)
	} else if err = sw.Close(); err != nil {
		t.Fatal(err)
	}

	// Read and merge snapshots.
	ssr := snapshot.NewMultiReader(&bufs[0], &bufs[1])

	// Next should be the second meta file.
	if f, err := ssr.Next(); err != nil {
		t.Fatalf("unexpected error(meta): %s", err)
	} else if !reflect.DeepEqual(f, snapshot.File{Name: "meta", Size: 3, ModTime: time.Date(2001, time.January, 1, 0, 0, 0, 0, time.UTC)}) {
		t.Fatalf("file mismatch(meta): %#v", f)
	} else if b := MustReadAll(ssr); string(b) != `bar` {
		t.Fatalf("unexpected file(meta): %s", b)
	}

	// Next should be shards/1.
	if f, err := ssr.Next(); err != nil {
		t.Fatalf("unexpected error(shards/1): %s", err)
	} else if !reflect.DeepEqual(f, snapshot.File{Name: "shards/1", Size: 5, ModTime: time.Date(2000, time.February, 1, 0, 0, 0, 0, time.UTC)}) {
		t.Fatalf("file mismatch(shards/1): %#v", f)
	} else if b := MustReadAll(ssr); string(b) != `55555` {
		t.Fatalf("unexpected file(shards/1): %s", b)
	}

	// Next should be shards/2.
	if f, err := ssr.Next(); err != nil {
		t.Fatalf("unexpected error(shards/2): %s", err)
	} else if !reflect.DeepEqual(f, snapshot.File{Name: "shards/2", Size: 6, ModTime: time.Date(2000, time.January, 1, 0, 0, 0, 0, time.UTC)}) {
		t.Fatalf("file mismatch(shards/2): %#v", f)
	} else if b := MustReadAll(ssr); string(b) != `666666` {
		t.Fatalf("unexpected file(shards/2): %s", b)
	}

	// Check for end of snapshot.
	if _, err := ssr.Next(); err != io.EOF {
		t.Fatalf("expected EOF: %s", err)
	}
}

// bufCloser adds a Close() method to a bytes.Buffer
type bufCloser struct {
	bytes.Buffer
	closed bool
}

// Close marks the buffer as closed.
func (b *bufCloser) Close() error {
	b.closed = true
	return nil
}

// Reads all data from the reader. Panic on error.
func MustReadAll(r io.Reader) []byte {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		panic(err.Error())
	}
	return b
}
