package diskv

import (
	"bytes"
	"testing"
	"time"
)

func cmpBytes(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (d *Diskv) isCached(key string) bool {
	d.RLock()
	defer d.RUnlock()
	_, ok := d.cache[key]
	return ok
}

func TestWriteReadErase(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 1024,
	})
	defer d.EraseAll()
	k, v := "a", []byte{'b'}
	if err := d.Write(k, v); err != nil {
		t.Fatalf("write: %s", err)
	}
	if readVal, err := d.Read(k); err != nil {
		t.Fatalf("read: %s", err)
	} else if bytes.Compare(v, readVal) != 0 {
		t.Fatalf("read: expected %s, got %s", v, readVal)
	}
	if err := d.Erase(k); err != nil {
		t.Fatalf("erase: %s", err)
	}
}

func TestWRECache(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 1024,
	})
	defer d.EraseAll()
	k, v := "xxx", []byte{' ', ' ', ' '}
	if d.isCached(k) {
		t.Fatalf("key cached before Write and Read")
	}
	if err := d.Write(k, v); err != nil {
		t.Fatalf("write: %s", err)
	}
	if d.isCached(k) {
		t.Fatalf("key cached before Read")
	}
	if readVal, err := d.Read(k); err != nil {
		t.Fatalf("read: %s", err)
	} else if bytes.Compare(v, readVal) != 0 {
		t.Fatalf("read: expected %s, got %s", v, readVal)
	}
	for i := 0; i < 10 && !d.isCached(k); i++ {
		time.Sleep(10 * time.Millisecond)
	}
	if !d.isCached(k) {
		t.Fatalf("key not cached after Read")
	}
	if err := d.Erase(k); err != nil {
		t.Fatalf("erase: %s", err)
	}
	if d.isCached(k) {
		t.Fatalf("key cached after Erase")
	}
}

func TestStrings(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 1024,
	})
	defer d.EraseAll()

	keys := map[string]bool{"a": false, "b": false, "c": false, "d": false}
	v := []byte{'1'}
	for k := range keys {
		if err := d.Write(k, v); err != nil {
			t.Fatalf("write: %s: %s", k, err)
		}
	}

	for k := range d.Keys(nil) {
		if _, present := keys[k]; present {
			t.Logf("got: %s", k)
			keys[k] = true
		} else {
			t.Fatalf("strings() returns unknown key: %s", k)
		}
	}

	for k, found := range keys {
		if !found {
			t.Errorf("never got %s", k)
		}
	}
}

func TestZeroByteCache(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 0,
	})
	defer d.EraseAll()

	k, v := "a", []byte{'1', '2', '3'}
	if err := d.Write(k, v); err != nil {
		t.Fatalf("Write: %s", err)
	}

	if d.isCached(k) {
		t.Fatalf("key cached, expected not-cached")
	}

	if _, err := d.Read(k); err != nil {
		t.Fatalf("Read: %s", err)
	}

	if d.isCached(k) {
		t.Fatalf("key cached, expected not-cached")
	}
}

func TestOneByteCache(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 1,
	})
	defer d.EraseAll()

	k1, k2, v1, v2 := "a", "b", []byte{'1'}, []byte{'1', '2'}
	if err := d.Write(k1, v1); err != nil {
		t.Fatal(err)
	}

	if v, err := d.Read(k1); err != nil {
		t.Fatal(err)
	} else if !cmpBytes(v, v1) {
		t.Fatalf("Read: expected %s, got %s", string(v1), string(v))
	}

	for i := 0; i < 10 && !d.isCached(k1); i++ {
		time.Sleep(10 * time.Millisecond)
	}
	if !d.isCached(k1) {
		t.Fatalf("expected 1-byte value to be cached, but it wasn't")
	}

	if err := d.Write(k2, v2); err != nil {
		t.Fatal(err)
	}
	if _, err := d.Read(k2); err != nil {
		t.Fatalf("--> %s", err)
	}

	for i := 0; i < 10 && (!d.isCached(k1) || d.isCached(k2)); i++ {
		time.Sleep(10 * time.Millisecond) // just wait for lazy-cache
	}
	if !d.isCached(k1) {
		t.Fatalf("1-byte value was uncached for no reason")
	}

	if d.isCached(k2) {
		t.Fatalf("2-byte value was cached, but cache max size is 1")
	}
}

func TestStaleCache(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 1,
	})
	defer d.EraseAll()

	k, first, second := "a", "first", "second"
	if err := d.Write(k, []byte(first)); err != nil {
		t.Fatal(err)
	}

	v, err := d.Read(k)
	if err != nil {
		t.Fatal(err)
	}
	if string(v) != first {
		t.Errorf("expected '%s', got '%s'", first, v)
	}

	if err := d.Write(k, []byte(second)); err != nil {
		t.Fatal(err)
	}

	v, err = d.Read(k)
	if err != nil {
		t.Fatal(err)
	}

	if string(v) != second {
		t.Errorf("expected '%s', got '%s'", second, v)
	}
}

func TestHas(t *testing.T) {
	d := New(Options{
		BasePath:     "test-data",
		CacheSizeMax: 1024,
	})
	defer d.EraseAll()

	for k, v := range map[string]string{
		"a":      "1",
		"foo":    "2",
		"012345": "3",
	} {
		d.Write(k, []byte(v))
	}

	d.Read("foo") // cache one of them
	if !d.isCached("foo") {
		t.Errorf("'foo' didn't get cached")
	}

	for _, tuple := range []struct {
		key      string
		expected bool
	}{
		{"a", true},
		{"b", false},
		{"foo", true},
		{"bar", false},
		{"01234", false},
		{"012345", true},
		{"0123456", false},
	} {
		if expected, got := tuple.expected, d.Has(tuple.key); expected != got {
			t.Errorf("Has(%s): expected %v, got %v", tuple.key, expected, got)
		}
	}
}
