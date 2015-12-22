package tsm1_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/influxdb/influxdb/tsdb/engine/tsm1"
)

func TestFileStore_Read(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"mem", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	// Search for an entry that exists in the second file
	values, err := fs.Read("cpu", time.Unix(1, 0))
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[1]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_SeekToAsc_FromStart(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 3.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	c := fs.KeyCursor("cpu")
	// Search for an entry that exists in the second file
	values, err := c.SeekTo(time.Unix(0, 0), true)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[0]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_SeekToAsc_Duplicate(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 3.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 4.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	c := fs.KeyCursor("cpu")
	// Search for an entry that exists in the second file
	values, err := c.SeekTo(time.Unix(0, 0), true)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[1]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %v", i, got, exp)
		}
	}

	// Check that calling Next will dedupe points
	values, err = c.Next(true)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}
	exp = data[3]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %v", i, got, exp)
		}
	}

}
func TestFileStore_SeekToAsc_BeforeStart(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(3, 0), 3.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	// Search for an entry that exists in the second file
	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(0, 0), true)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[0]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_SeekToAsc_Middle(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 1.0),
			tsm1.NewValue(time.Unix(2, 0), 2.0),
			tsm1.NewValue(time.Unix(3, 0), 3.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(4, 0), 4.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	// Search for an entry that exists in the second file
	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(3, 0), true)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[0]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_SeekToAsc_End(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 3.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(2, 0), true)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[2]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_SeekToDesc_FromStart(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 3.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	// Search for an entry that exists in the second file
	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(0, 0), false)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}
	exp := data[0]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_SeekToDesc_Duplicate(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 4.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 3.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	// Search for an entry that exists in the second file
	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(2, 0), false)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}
	exp := data[3]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %v", i, got, exp)
		}
	}

	// Check that calling Next will dedupe points
	values, err = c.Next(false)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}
	exp = data[1]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %v", i, got, exp)
		}
	}
}
func TestFileStore_SeekToDesc_AfterEnd(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(3, 0), 3.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(4, 0), false)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[2]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %v", i, got, exp)
		}
	}
}

func TestFileStore_SeekToDesc_Middle(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{
			tsm1.NewValue(time.Unix(2, 0), 2.0),
			tsm1.NewValue(time.Unix(3, 0), 3.0),
			tsm1.NewValue(time.Unix(4, 0), 4.0)},
		},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	// Search for an entry that exists in the second file
	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(3, 0), false)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[1]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_SeekToDesc_End(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(2, 0), 3.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	c := fs.KeyCursor("cpu")
	values, err := c.SeekTo(time.Unix(2, 0), false)
	if err != nil {
		t.Fatalf("unexpected error reading values: %v", err)
	}

	exp := data[2]
	if got, exp := len(values), len(exp.values); got != exp {
		t.Fatalf("value length mismatch: got %v, exp %v", got, exp)
	}

	for i, v := range exp.values {
		if got, exp := values[i].Value(), v.Value(); got != exp {
			t.Fatalf("read value mismatch(%d): got %v, exp %d", i, got, exp)
		}
	}
}

func TestFileStore_Open(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	// Create 3 TSM files...
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"mem", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
	}

	_, err := newFileDir(dir, data...)
	if err != nil {
		fatal(t, "creating test files", err)
	}

	fs := tsm1.NewFileStore(dir)
	if err := fs.Open(); err != nil {
		fatal(t, "opening file store", err)
	}
	defer fs.Close()

	if got, exp := fs.Count(), 3; got != exp {
		t.Fatalf("file count mismatch: got %v, exp %v", got, exp)
	}

	if got, exp := fs.CurrentGeneration(), 4; got != exp {
		t.Fatalf("current ID mismatch: got %v, exp %v", got, exp)
	}
}

func TestFileStore_Remove(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	// Create 3 TSM files...
	data := []keyValues{
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"mem", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
	}

	files, err := newFileDir(dir, data...)
	if err != nil {
		fatal(t, "creating test files", err)
	}

	fs := tsm1.NewFileStore(dir)
	if err := fs.Open(); err != nil {
		fatal(t, "opening file store", err)
	}
	defer fs.Close()

	if got, exp := fs.Count(), 3; got != exp {
		t.Fatalf("file count mismatch: got %v, exp %v", got, exp)
	}

	if got, exp := fs.CurrentGeneration(), 4; got != exp {
		t.Fatalf("current ID mismatch: got %v, exp %v", got, exp)
	}

	fs.Remove(files[2])

	if got, exp := fs.Count(), 2; got != exp {
		t.Fatalf("file count mismatch: got %v, exp %v", got, exp)
	}

	if got, exp := fs.CurrentGeneration(), 4; got != exp {
		t.Fatalf("current ID mismatch: got %v, exp %v", got, exp)
	}
}

func TestFileStore_Open_Deleted(t *testing.T) {
	dir := MustTempDir()
	defer os.RemoveAll(dir)

	// Create 3 TSM files...
	data := []keyValues{
		keyValues{"cpu,host=server2!~#!value", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu,host=server1!~#!value", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"mem,host=server1!~#!value", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
	}

	_, err := newFileDir(dir, data...)
	if err != nil {
		fatal(t, "creating test files", err)
	}

	fs := tsm1.NewFileStore(dir)
	if err := fs.Open(); err != nil {
		fatal(t, "opening file store", err)
	}
	defer fs.Close()

	if got, exp := len(fs.Keys()), 3; got != exp {
		t.Fatalf("file count mismatch: got %v, exp %v", got, exp)
	}

	if err := fs.Delete([]string{"cpu,host=server2!~#!value"}); err != nil {
		fatal(t, "deleting", err)
	}

	fs2 := tsm1.NewFileStore(dir)
	if err := fs2.Open(); err != nil {
		fatal(t, "opening file store", err)
	}
	defer fs2.Close()

	if got, exp := len(fs2.Keys()), 2; got != exp {
		t.Fatalf("file count mismatch: got %v, exp %v", got, exp)
	}
}

func TestFileStore_Delete(t *testing.T) {
	fs := tsm1.NewFileStore("")

	// Setup 3 files
	data := []keyValues{
		keyValues{"cpu,host=server2!~#!value", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
		keyValues{"cpu,host=server1!~#!value", []tsm1.Value{tsm1.NewValue(time.Unix(1, 0), 2.0)}},
		keyValues{"mem,host=server1!~#!value", []tsm1.Value{tsm1.NewValue(time.Unix(0, 0), 1.0)}},
	}

	files, err := newFiles(data...)
	if err != nil {
		t.Fatalf("unexpected error creating files: %v", err)
	}

	fs.Add(files...)

	keys := fs.Keys()
	if got, exp := len(keys), 3; got != exp {
		t.Fatalf("key length mismatch: got %v, exp %v", got, exp)
	}

	if err := fs.Delete([]string{"cpu,host=server2!~#!value"}); err != nil {
		fatal(t, "deleting", err)
	}

	keys = fs.Keys()
	if got, exp := len(keys), 2; got != exp {
		t.Fatalf("key length mismatch: got %v, exp %v", got, exp)
	}
}

func newFileDir(dir string, values ...keyValues) ([]string, error) {
	var files []string

	id := 1
	for _, v := range values {
		f := MustTempFile(dir)
		w, err := tsm1.NewTSMWriter(f)
		if err != nil {
			return nil, err
		}

		if err := w.Write(v.key, v.values); err != nil {
			return nil, err
		}

		if err := w.WriteIndex(); err != nil {
			return nil, err
		}

		if err := f.Close(); err != nil {
			return nil, err
		}
		newName := filepath.Join(filepath.Dir(f.Name()), tsmFileName(id))
		if err := os.Rename(f.Name(), newName); err != nil {
			return nil, err
		}
		id++

		files = append(files, newName)
	}
	return files, nil

}

func newFiles(values ...keyValues) ([]tsm1.TSMFile, error) {
	var files []tsm1.TSMFile

	for _, v := range values {
		var b bytes.Buffer
		w, err := tsm1.NewTSMWriter(&b)
		if err != nil {
			return nil, err
		}

		if err := w.Write(v.key, v.values); err != nil {
			return nil, err
		}

		if err := w.WriteIndex(); err != nil {
			return nil, err
		}

		r, err := tsm1.NewTSMReader(bytes.NewReader(b.Bytes()))
		if err != nil {
			return nil, err
		}
		files = append(files, r)
	}
	return files, nil
}

type keyValues struct {
	key    string
	values []tsm1.Value
}

func MustTempDir() string {
	dir, err := ioutil.TempDir("", "tsm1-test")
	if err != nil {
		panic(fmt.Sprintf("failed to create temp dir: %v", err))
	}
	return dir
}

func MustTempFile(dir string) *os.File {
	f, err := ioutil.TempFile(dir, "tsm1test")
	if err != nil {
		panic(fmt.Sprintf("failed to create temp file: %v", err))
	}
	return f
}

func fatal(t *testing.T, msg string, err error) {
	t.Fatalf("unexpected error %v: %v", msg, err)
}

func tsmFileName(id int) string {
	return fmt.Sprintf("%09d-%09d.tsm", id, 1)
}
