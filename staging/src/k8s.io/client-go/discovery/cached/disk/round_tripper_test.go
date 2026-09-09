/*
Copyright 2017 The Kubernetes Authors.

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

package disk

import (
	"bytes"
	"crypto/sha256"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/peterbourgon/diskv"
	"github.com/stretchr/testify/assert"
)

// copied from k8s.io/client-go/transport/round_trippers_test.go
type testRoundTripper struct {
	Request  *http.Request
	Response *http.Response
	Err      error
}

func (rt *testRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.Request = req
	return rt.Response, rt.Err
}

func BenchmarkDiskCache(b *testing.B) {
	cacheDir, err := os.MkdirTemp("", "cache-rt")
	if err != nil {
		b.Fatal(err)
	}
	defer os.RemoveAll(cacheDir)

	d := diskv.New(diskv.Options{
		PathPerm: os.FileMode(0750),
		FilePerm: os.FileMode(0660),
		BasePath: cacheDir,
		TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
	})

	k := "localhost:8080/apis/batch/v1.json"
	v, err := os.ReadFile("../../testdata/apis/batch/v1.json")
	if err != nil {
		b.Fatal(err)
	}

	c := sumDiskCache{disk: d}

	for n := 0; n < b.N; n++ {
		c.Set(k, v)
		c.Get(k)
		c.Delete(k)
	}
}

func TestCacheRoundTripper(t *testing.T) {
	rt := &testRoundTripper{}
	cacheDir, err := os.MkdirTemp("", "cache-rt")
	defer os.RemoveAll(cacheDir)
	if err != nil {
		t.Fatal(err)
	}
	cache := newCacheRoundTripper(cacheDir, rt)

	// First call, caches the response
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Host: "localhost"},
	}
	rt.Response = &http.Response{
		Header:     http.Header{"ETag": []string{`"123456"`}},
		Body:       io.NopCloser(bytes.NewReader([]byte("Content"))),
		StatusCode: http.StatusOK,
	}
	resp, err := cache.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	content, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "Content" {
		t.Errorf(`Expected Body to be "Content", got %q`, string(content))
	}

	// Second call, returns cached response
	req = &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Host: "localhost"},
	}
	rt.Response = &http.Response{
		StatusCode: http.StatusNotModified,
		Body:       io.NopCloser(bytes.NewReader([]byte("Other Content"))),
	}

	resp, err = cache.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}

	// Read body and make sure we have the initial content
	content, err = io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "Content" {
		t.Errorf("Invalid content read from cache %q", string(content))
	}
}

func TestCacheRoundTripperPathPerm(t *testing.T) {
	assert := assert.New(t)

	rt := &testRoundTripper{}
	cacheDir, err := os.MkdirTemp("", "cache-rt")
	os.RemoveAll(cacheDir)
	defer os.RemoveAll(cacheDir)

	if err != nil {
		t.Fatal(err)
	}
	cache := newCacheRoundTripper(cacheDir, rt)

	// First call, caches the response
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Host: "localhost"},
	}
	rt.Response = &http.Response{
		Header:     http.Header{"ETag": []string{`"123456"`}},
		Body:       io.NopCloser(bytes.NewReader([]byte("Content"))),
		StatusCode: http.StatusOK,
	}
	resp, err := cache.RoundTrip(req)
	if err != nil {
		t.Fatal(err)
	}
	content, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "Content" {
		t.Errorf(`Expected Body to be "Content", got %q`, string(content))
	}

	err = filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			assert.Equal(os.FileMode(0750), info.Mode().Perm())
		} else {
			assert.Equal(os.FileMode(0660), info.Mode().Perm())
		}
		return nil
	})
	assert.NoError(err)
}

func TestSumDiskCache(t *testing.T) {
	assert := assert.New(t)

	// Ensure that we'll return a cache miss if the backing file doesn't exist.
	t.Run("NoSuchKey", func(t *testing.T) {
		cacheDir, err := os.MkdirTemp("", "cache-test")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(cacheDir)
		d := diskv.New(diskv.Options{BasePath: cacheDir, TempDir: filepath.Join(cacheDir, ".diskv-temp")})
		c := &sumDiskCache{disk: d}

		key := "testing"

		got, ok := c.Get(key)
		assert.False(ok)
		assert.Equal([]byte{}, got)
	})

	// Ensure that we'll return a cache miss if the backing file is empty.
	t.Run("EmptyFile", func(t *testing.T) {
		cacheDir, err := os.MkdirTemp("", "cache-test")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(cacheDir)
		d := diskv.New(diskv.Options{BasePath: cacheDir, TempDir: filepath.Join(cacheDir, ".diskv-temp")})
		c := &sumDiskCache{disk: d}

		key := "testing"

		f, err := os.Create(filepath.Join(cacheDir, sanitize(key)))
		if err != nil {
			t.Fatal(err)
		}
		f.Close()

		got, ok := c.Get(key)
		assert.False(ok)
		assert.Equal([]byte{}, got)
	})

	// Ensure that we'll return a cache miss if the backing has an invalid
	// checksum.
	t.Run("InvalidChecksum", func(t *testing.T) {
		cacheDir, err := os.MkdirTemp("", "cache-test")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(cacheDir)
		d := diskv.New(diskv.Options{BasePath: cacheDir, TempDir: filepath.Join(cacheDir, ".diskv-temp")})
		c := &sumDiskCache{disk: d}

		key := "testing"
		value := []byte("testing")
		mismatchedValue := []byte("testink")
		sum := sha256.Sum256(value)

		// Create a file with the sum of 'value' followed by the bytes of
		// 'mismatchedValue'.
		f, err := os.Create(filepath.Join(cacheDir, sanitize(key)))
		if err != nil {
			t.Fatal(err)
		}
		f.Write(sum[:])
		f.Write(mismatchedValue)
		f.Close()

		// The mismatched checksum should result in a cache miss.
		got, ok := c.Get(key)
		assert.False(ok)
		assert.Equal([]byte{}, got)
	})

	// Ensure that our disk cache will happily cache over the top of an existing
	// value. We depend on this behaviour to recover from corrupted cache
	// entries. When Get detects a bad checksum it will return a cache miss.
	// This should cause httpcache to fall back to its underlying transport and
	// to subsequently cache the new value, overwriting the corrupt one.
	t.Run("OverwriteExistingKey", func(t *testing.T) {
		cacheDir, err := os.MkdirTemp("", "cache-test")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(cacheDir)
		d := diskv.New(diskv.Options{BasePath: cacheDir, TempDir: filepath.Join(cacheDir, ".diskv-temp")})
		c := &sumDiskCache{disk: d}

		key := "testing"
		value := []byte("cool value!")

		// Write a value.
		c.Set(key, value)
		got, ok := c.Get(key)

		// Ensure we can read back what we wrote.
		assert.True(ok)
		assert.Equal(value, got)

		differentValue := []byte("I'm different!")

		// Write a different value.
		c.Set(key, differentValue)
		got, ok = c.Get(key)

		// Ensure we can read back the different value.
		assert.True(ok)
		assert.Equal(differentValue, got)
	})

	// Ensure that deleting a key does in fact delete it.
	t.Run("DeleteKey", func(t *testing.T) {
		cacheDir, err := os.MkdirTemp("", "cache-test")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(cacheDir)
		d := diskv.New(diskv.Options{BasePath: cacheDir, TempDir: filepath.Join(cacheDir, ".diskv-temp")})
		c := &sumDiskCache{disk: d}

		key := "testing"
		value := []byte("coolValue")

		c.Set(key, value)

		// Ensure we successfully set the value.
		got, ok := c.Get(key)
		assert.True(ok)
		assert.Equal(value, got)

		c.Delete(key)

		// Ensure the value is gone.
		got, ok = c.Get(key)
		assert.False(ok)
		assert.Equal([]byte{}, got)

		// Ensure that deleting a non-existent value is a no-op.
		c.Delete(key)
	})
}
