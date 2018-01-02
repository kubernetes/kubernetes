package diskv

import (
	"bytes"
	"io/ioutil"
	"sync"
	"testing"
	"time"
)

// ReadStream from cache shouldn't panic on a nil dereference from a nonexistent
// Compression :)
func TestIssue2A(t *testing.T) {
	d := New(Options{
		BasePath:     "test-issue-2a",
		Transform:    func(string) []string { return []string{} },
		CacheSizeMax: 1024,
	})
	defer d.EraseAll()

	input := "abcdefghijklmnopqrstuvwxy"
	key, writeBuf, sync := "a", bytes.NewBufferString(input), false
	if err := d.WriteStream(key, writeBuf, sync); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ {
		began := time.Now()
		rc, err := d.ReadStream(key, false)
		if err != nil {
			t.Fatal(err)
		}
		buf, err := ioutil.ReadAll(rc)
		if err != nil {
			t.Fatal(err)
		}
		if !cmpBytes(buf, []byte(input)) {
			t.Fatalf("read #%d: '%s' != '%s'", i+1, string(buf), input)
		}
		rc.Close()
		t.Logf("read #%d in %s", i+1, time.Since(began))
	}
}

// ReadStream on a key that resolves to a directory should return an error.
func TestIssue2B(t *testing.T) {
	blockTransform := func(s string) []string {
		transformBlockSize := 3
		sliceSize := len(s) / transformBlockSize
		pathSlice := make([]string, sliceSize)
		for i := 0; i < sliceSize; i++ {
			from, to := i*transformBlockSize, (i*transformBlockSize)+transformBlockSize
			pathSlice[i] = s[from:to]
		}
		return pathSlice
	}

	d := New(Options{
		BasePath:     "test-issue-2b",
		Transform:    blockTransform,
		CacheSizeMax: 0,
	})
	defer d.EraseAll()

	v := []byte{'1', '2', '3'}
	if err := d.Write("abcabc", v); err != nil {
		t.Fatal(err)
	}

	_, err := d.ReadStream("abc", false)
	if err == nil {
		t.Fatal("ReadStream('abc') should return error")
	}
	t.Logf("ReadStream('abc') returned error: %v", err)
}

// Ensure ReadStream with direct=true isn't racy.
func TestIssue17(t *testing.T) {
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
		CacheSizeMax: 50,
	})

	cases := map[string]string{
		"a": `1234567890`,
		"b": `2345678901`,
		"c": `3456789012`,
		"d": `4567890123`,
		"e": `5678901234`,
	}

	for k, v := range cases {
		if err := dWrite.Write(k, []byte(v)); err != nil {
			t.Fatalf("during write: %s", err)
		}
		dRead.Read(k) // ensure it's added to cache
	}

	var wg sync.WaitGroup
	start := make(chan struct{})
	for k, v := range cases {
		wg.Add(1)
		go func(k, v string) {
			<-start
			dRead.ReadStream(k, true)
			wg.Done()
		}(k, v)
	}
	close(start)
	wg.Wait()
}
