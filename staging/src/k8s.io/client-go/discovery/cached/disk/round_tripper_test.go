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
	"crypto/sha256"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/bartventer/httpcache/store/acceptance"
	"github.com/bartventer/httpcache/store/driver"
	"github.com/peterbourgon/diskv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	require.NoError(b, err)
	defer os.RemoveAll(cacheDir)

	d := diskv.New(diskv.Options{
		PathPerm: os.FileMode(0750),
		FilePerm: os.FileMode(0660),
		BasePath: cacheDir,
		TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
	})

	k := "localhost:8080/apis/batch/v1.json"
	v, err := os.ReadFile("../../testdata/apis/batch/v1.json")
	require.NoError(b, err)

	c := sumDiskCache{disk: d}

	for b.Loop() {
		c.Set(k, v)
		c.Get(k)
		c.Delete(k)
	}
}

func setupSumDiskCache(t *testing.T) (conn *sumDiskCache, cleanup func()) {
	t.Helper()
	cacheDir := t.TempDir()
	d, err := openSumDiskCache(cacheDir)
	require.NoError(t, err, "Failed to create sumDiskCache")
	cleanup = func() {} //noop; t.TempDir() handles cleanup
	return d, cleanup
}

func TestCacheRoundTripper(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		etag := r.Header.Get("X-Test-ETag-Hint")
		w.Header().Set("ETag", etag)
		w.Header().Set("Vary", "Accept-Encoding")
		w.Header().Set("Cache-Control", "public, max-age=3600")
		w.WriteHeader(http.StatusOK)
		switch etag {
		case "W/1":
			w.Write([]byte("Content"))
		case "W/2":
			w.Header().Set("Content-Encoding", "gzip")
			w.Write([]byte("Content with gzip"))
		}
	}))
	defer server.Close()

	cacheDir, err := os.MkdirTemp("", "cache-rt")
	require.NoError(t, err, "Failed to create cache directory")
	os.RemoveAll(cacheDir)
	defer os.RemoveAll(cacheDir)
	cache := newCacheRoundTripper(cacheDir, http.DefaultTransport)

	type args struct {
		reqHeader http.Header
	}

	tests := []struct {
		name                             string
		args                             args
		wantStatusCode                   int
		wantXCacheStatus, wantXFromCache string
		wantBody                         string
	}{
		{
			name: "Cache Miss",
			args: args{
				reqHeader: http.Header{
					"X-Test-ETag-Hint": []string{"W/1"},
				},
			},
			wantStatusCode:   http.StatusOK,
			wantXCacheStatus: "MISS",
			wantXFromCache:   "",
			wantBody:         "Content",
		},
		{
			name: "Cache Miss Different Vary Header",
			args: args{
				reqHeader: http.Header{
					"X-Test-ETag-Hint": []string{"W/2"},
					"Accept-Encoding":  []string{"gzip"},
				},
			},
			wantStatusCode:   http.StatusOK,
			wantXCacheStatus: "MISS",
			wantXFromCache:   "",
			wantBody:         "Content with gzip",
		},
		{
			name: "Cache Hit",
			args: args{
				reqHeader: http.Header{
					"If-None-Match": []string{"W/1"},
				},
			},
			wantStatusCode:   http.StatusOK,
			wantXCacheStatus: "HIT",
			wantXFromCache:   "1",
			wantBody:         "Content",
		},

		{
			name: "Cache Hit Different Vary Header",
			args: args{
				reqHeader: http.Header{
					"If-None-Match":   []string{"W/2"},
					"Accept-Encoding": []string{"gzip"},
				},
			},
			wantStatusCode:   http.StatusOK,
			wantXCacheStatus: "HIT",
			wantXFromCache:   "1",
			wantBody:         "Content with gzip",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest(http.MethodGet, server.URL, nil)
			maps.Copy(req.Header, tt.args.reqHeader)

			resp, err := cache.RoundTrip(req)
			require.NoError(t, err, "RoundTrip failed")
			defer resp.Body.Close()

			content, err := io.ReadAll(resp.Body)
			require.NoError(t, err, "Failed to read response body")

			assert.Equal(t, tt.wantStatusCode, resp.StatusCode)
			assert.Equal(t, tt.wantXCacheStatus, resp.Header.Get("X-Httpcache-Status"))
			assert.Equal(t, tt.wantXFromCache, resp.Header.Get("X-From-Cache"))
			assert.Equal(t, tt.wantBody, string(content))
		})
	}

	t.Run("Cache Directory Permissions", func(t *testing.T) {
		err := filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() {
				assert.Equal(t, os.FileMode(0750), info.Mode().Perm())
			} else {
				assert.Equal(t, os.FileMode(0660), info.Mode().Perm())
			}
			return nil
		})
		assert.NoError(t, err)
	})
}

func TestSumDiskCache(t *testing.T) {
	t.Run("Acceptance", func(t *testing.T) {
		acceptance.Run(t, acceptance.FactoryFunc(func() (driver.Conn, func()) {
			return setupSumDiskCache(t)
		}))
	})

	// Below are edge cases specific to sumDiskCache that are not covered by
	// the acceptance tests.
	t.Run("EmptyFile", func(t *testing.T) {
		c, cleanup := setupSumDiskCache(t)
		t.Cleanup(cleanup)

		key := "testing"
		f, err := os.Create(filepath.Join(c.disk.BasePath, sanitize(key)))
		require.NoError(t, err, "Failed to create cache file")
		f.Close()

		got, err := c.Get(key)
		assert.ErrorIs(t, err, errInvalidCacheFile)
		assert.Equal(t, []byte{}, got)
	})

	t.Run("InvalidChecksum", func(t *testing.T) {
		c, cleanup := setupSumDiskCache(t)
		t.Cleanup(cleanup)

		key := "testing"
		value := []byte("testing")
		mismatchedValue := []byte("testink")
		sum := sha256.Sum256(value)

		f, err := os.Create(filepath.Join(c.disk.BasePath, sanitize(key)))
		require.NoError(t, err, "Failed to create cache file")
		f.Write(sum[:])
		f.Write(mismatchedValue)
		f.Close()

		// The mismatched checksum should result in a cache miss.
		got, err := c.Get(key)
		assert.ErrorIs(t, err, errChecksumMismatch)
		assert.Equal(t, []byte{}, got)
	})
}
