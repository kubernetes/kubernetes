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
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"

	"github.com/bartventer/httpcache"
	"github.com/bartventer/httpcache/store"
	"github.com/bartventer/httpcache/store/driver"
	"github.com/peterbourgon/diskv"
	"k8s.io/klog/v2"
)

const sumDiskScheme = "sumdisk"

func init() {
	// Workaround for kube-openapi's incorrect use of "UTC" in HTTP date headers.
	// bartventer/httpcache allows handling "UTC" in Expires headers by
	// setting HTTPCACHE_ALLOW_UTC_DATETIMEFORMAT=1. This is necessary until
	// kube-openapi uses "GMT" as required by RFC 9110 Section 5.6.7.
	//
	// References:
	//   - https://www.rfc-editor.org/rfc/rfc9110#section-5.6.7
	//   - https://github.com/bartventer/httpcache/releases/tag/v0.9.2
	//   - https://github.com/kubernetes/kube-openapi/blob/9bd5c66d9911c53f5aedb8595fde9c229ca56703/pkg/handler3/handler.go#L281
	//
	// Context:
	// Per RFC 9110 Section 5.6.7, HTTP date fields MUST use the IMF-fixdate format:
	//   IMF-fixdate = day-name "," SP date1 SP time-of-day SP GMT
	//   GMT = %s"GMT"
	// And "When a sender generates a field that contains one or more timestamps
	// defined as HTTP-date, the sender MUST generate those timestamps in the
	// IMF-fixdate format."
	//
	// While both GMT and UTC represent the same time (Coordinated Universal Time),
	// the RFC requires the literal string "GMT" in HTTP-date fields. kube-openapi
	// incorrectly generates "UTC", violating RFC 9110.
	//
	// TODO: Remove once kube-openapi uses GMT in Expires headers.
	_ = os.Setenv("HTTPCACHE_ALLOW_UTC_DATETIMEFORMAT", "1")

	store.Register(sumDiskScheme, driver.DriverFunc(func(u *url.URL) (driver.Conn, error) {
		return openSumDiskCacheFromURL(u)
	}))
}

func openSumDiskCacheFromURL(u *url.URL) (*sumDiskCache, error) {
	cacheDir := u.Query().Get("cachedir")
	return openSumDiskCache(cacheDir)
}

func openSumDiskCache(cacheDir string) (*sumDiskCache, error) {
	return &sumDiskCache{
		disk: diskv.New(diskv.Options{
			PathPerm: os.FileMode(0750),
			FilePerm: os.FileMode(0660),
			BasePath: cacheDir,
			TempDir:  filepath.Join(cacheDir, ".diskv-temp"),
		}),
	}, nil
}

func newCacheRoundTripper(cacheDir string, upstream http.RoundTripper) http.RoundTripper {
	dsn := sumDiskScheme + "://?cachedir=" + url.QueryEscape(cacheDir)
	return &cancelableTransport{httpcache.NewTransport(dsn, httpcache.WithUpstream(upstream))}
}

type cancelableTransport struct{ rt http.RoundTripper }

func (rt *cancelableTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return rt.rt.RoundTrip(req)
}

func (rt *cancelableTransport) CancelRequest(req *http.Request) {
	type canceler interface {
		CancelRequest(*http.Request)
	}
	if cr, ok := rt.rt.(canceler); ok {
		cr.CancelRequest(req)
	} else {
		klog.Errorf("CancelRequest not implemented by %T", rt.rt)
	}
}

func (rt *cancelableTransport) WrappedRoundTripper() http.RoundTripper { return rt.rt }

// sumDiskCache is a cache backend for github.com/bartventer/httpcache that
// uses [diskv] with SHA256 checksums for integrity verification. Files contain
// a 32-byte SHA256 prefix followed by the response data, and are named using
// hashed keys. Relies on filesystem's eventual sync rather than immediate
// disk sync for better performance.
//
// See https://github.com/kubernetes/kubernetes/issues/110753 for more.
type sumDiskCache struct {
	disk *diskv.Diskv
}

var (
	errInvalidCacheFile = errors.New("disk cache: invalid cache file, must contain at least 32 bytes for SHA256 checksum")
	errChecksumMismatch = errors.New("disk cache: checksum mismatch")
)

// Get returns the response bytes for the given key. If the key does not
// exist, or if the file is too short to contain a SHA256 checksum, or if the
// checksum does not match, it returns an error.
func (c *sumDiskCache) Get(key string) ([]byte, error) {
	b, err := c.disk.Read(sanitize(key))
	switch {
	case err != nil:
		if errors.Is(err, os.ErrNotExist) {
			err = errors.Join(err, driver.ErrNotExist)
		}
		return []byte{}, err
	case len(b) < sha256.Size:
		return []byte{}, fmt.Errorf("%w: got %d bytes", errInvalidCacheFile, len(b))
	}

	response := b[sha256.Size:]
	want := b[:sha256.Size] // The first 32 bytes of the file should be the SHA256 sum.
	got := sha256.Sum256(response)
	if !bytes.Equal(want, got[:]) {
		return []byte{}, fmt.Errorf("%w: got %x, want %x", errChecksumMismatch, got, want)
	}
	return response, nil
}

// Set writes the response to a file on disk. The filename will be the SHA256
// sum of the key. The file will contain a SHA256 sum of the response bytes,
// followed by said response bytes.
func (c *sumDiskCache) Set(key string, response []byte) error {
	s := sha256.Sum256(response)
	data := make([]byte, sha256.Size+len(response))
	copy(data, s[:])
	copy(data[sha256.Size:], response)
	return c.disk.Write(sanitize(key), data)
}

func (c *sumDiskCache) Delete(key string) error {
	err := c.disk.Erase(sanitize(key))
	if errors.Is(err, os.ErrNotExist) {
		err = errors.Join(err, driver.ErrNotExist)
	}
	return err
}

// Sanitize an httpcache key such that it can be used as a diskv key, which must
// be a valid filename. The httpcache key will either be the requested URL (if
// the request method was GET) or "<method> <url>" for other methods, per the
// httpcache.cacheKey function.
func sanitize(key string) string {
	// These keys are not sensitive. We use sha256 to avoid a (potentially
	// malicious) collision causing the wrong cache data to be written or
	// accessed.
	return fmt.Sprintf("%x", sha256.Sum256([]byte(key)))
}
