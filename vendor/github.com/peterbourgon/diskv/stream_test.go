package diskv

import (
	"bytes"
	"io/ioutil"
	"testing"
)

func TestBasicStreamCaching(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 1024,
	})
	defer d.EraseAll()

	input := "a1b2c3"
	key, writeBuf, sync := "a", bytes.NewBufferString(input), true
	if err := d.WriteStream(key, writeBuf, sync); err != nil {
		t.Fatal(err)
	}

	if d.isCached(key) {
		t.Fatalf("'%s' cached, but shouldn't be (yet)", key)
	}

	rc, err := d.ReadStream(key, false)
	if err != nil {
		t.Fatal(err)
	}

	readBuf, err := ioutil.ReadAll(rc)
	if err != nil {
		t.Fatal(err)
	}

	if !cmpBytes(readBuf, []byte(input)) {
		t.Fatalf("'%s' != '%s'", string(readBuf), input)
	}

	if !d.isCached(key) {
		t.Fatalf("'%s' isn't cached, but should be", key)
	}
}

func TestReadStreamDirect(t *testing.T) {
	var (
		basePath = "test-data"
	)
	dWrite := New(Options{
		BasePath:     basePath,
		CacheSizeMax: 0,
	})
	defer dWrite.EraseAll()
	dRead := New(Options{
		BasePath:     basePath,
		CacheSizeMax: 1024,
	})

	// Write
	key, val1, val2 := "a", []byte(`1234567890`), []byte(`aaaaaaaaaa`)
	if err := dWrite.Write(key, val1); err != nil {
		t.Fatalf("during first write: %s", err)
	}

	// First, caching read.
	val, err := dRead.Read(key)
	if err != nil {
		t.Fatalf("during initial read: %s", err)
	}
	t.Logf("read 1: %s => %s", key, string(val))
	if !cmpBytes(val1, val) {
		t.Errorf("expected %q, got %q", string(val1), string(val))
	}
	if !dRead.isCached(key) {
		t.Errorf("%q should be cached, but isn't", key)
	}

	// Write a different value.
	if err := dWrite.Write(key, val2); err != nil {
		t.Fatalf("during second write: %s", err)
	}

	// Second read, should hit cache and get the old value.
	val, err = dRead.Read(key)
	if err != nil {
		t.Fatalf("during second (cache-hit) read: %s", err)
	}
	t.Logf("read 2: %s => %s", key, string(val))
	if !cmpBytes(val1, val) {
		t.Errorf("expected %q, got %q", string(val1), string(val))
	}

	// Third, direct read, should get the updated value.
	rc, err := dRead.ReadStream(key, true)
	if err != nil {
		t.Fatalf("during third (direct) read, ReadStream: %s", err)
	}
	defer rc.Close()
	val, err = ioutil.ReadAll(rc)
	if err != nil {
		t.Fatalf("during third (direct) read, ReadAll: %s", err)
	}
	t.Logf("read 3: %s => %s", key, string(val))
	if !cmpBytes(val2, val) {
		t.Errorf("expected %q, got %q", string(val1), string(val))
	}

	// Fourth read, should hit cache and get the new value.
	val, err = dRead.Read(key)
	if err != nil {
		t.Fatalf("during fourth (cache-hit) read: %s", err)
	}
	t.Logf("read 4: %s => %s", key, string(val))
	if !cmpBytes(val2, val) {
		t.Errorf("expected %q, got %q", string(val1), string(val))
	}
}
