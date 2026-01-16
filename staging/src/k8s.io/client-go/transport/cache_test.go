/*
Copyright 2015 The Kubernetes Authors.

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

package transport

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"runtime"
	"sync"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

func TestTLSConfigKey(t *testing.T) {
	// Make sure config fields that don't affect the tls config don't affect the cache key
	identicalConfigurations := map[string]*Config{
		"empty":          {},
		"basic":          {Username: "bob", Password: "password"},
		"bearer":         {BearerToken: "token"},
		"user agent":     {UserAgent: "useragent"},
		"transport":      {Transport: http.DefaultTransport},
		"wrap transport": {WrapTransport: func(http.RoundTripper) http.RoundTripper { return nil }},
	}
	for nameA, valueA := range identicalConfigurations {
		for nameB, valueB := range identicalConfigurations {
			keyA, canCache, err := tlsConfigKey(valueA)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameA, err)
				continue
			}
			if !canCache {
				t.Errorf("Unexpected canCache=false")
				continue
			}
			keyB, canCache, err := tlsConfigKey(valueB)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameB, err)
				continue
			}
			if !canCache {
				t.Errorf("Unexpected canCache=false")
				continue
			}
			if keyA != keyB {
				t.Errorf("Expected identical cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
				continue
			}
			if keyA != (tlsCacheKey{}) {
				t.Errorf("Expected empty cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
				continue
			}
		}
	}

	// Make sure config fields that affect the tls config affect the cache key
	dialer := net.Dialer{}
	getCert := &GetCertHolder{GetCert: func() (*tls.Certificate, error) { return nil, nil }}
	uniqueConfigurations := map[string]*Config{
		"proxy":    {Proxy: func(request *http.Request) (*url.URL, error) { return nil, nil }},
		"no tls":   {},
		"dialer":   {DialHolder: &DialHolder{Dial: dialer.DialContext}},
		"dialer2":  {DialHolder: &DialHolder{Dial: func(ctx context.Context, network, address string) (net.Conn, error) { return nil, nil }}},
		"insecure": {TLS: TLSConfig{Insecure: true}},
		"cadata 1": {TLS: TLSConfig{CAData: []byte{1}}},
		"cadata 2": {TLS: TLSConfig{CAData: []byte{2}}},
		"cert 1, key 1": {
			TLS: TLSConfig{
				CertData: []byte{1},
				KeyData:  []byte{1},
			},
		},
		"cert 1, key 1, servername 1": {
			TLS: TLSConfig{
				CertData:   []byte{1},
				KeyData:    []byte{1},
				ServerName: "1",
			},
		},
		"cert 1, key 1, servername 2": {
			TLS: TLSConfig{
				CertData:   []byte{1},
				KeyData:    []byte{1},
				ServerName: "2",
			},
		},
		"cert 1, key 2": {
			TLS: TLSConfig{
				CertData: []byte{1},
				KeyData:  []byte{2},
			},
		},
		"cert 2, key 1": {
			TLS: TLSConfig{
				CertData: []byte{2},
				KeyData:  []byte{1},
			},
		},
		"cert 2, key 2": {
			TLS: TLSConfig{
				CertData: []byte{2},
				KeyData:  []byte{2},
			},
		},
		"cadata 1, cert 1, key 1": {
			TLS: TLSConfig{
				CAData:   []byte{1},
				CertData: []byte{1},
				KeyData:  []byte{1},
			},
		},
		"getCert1": {
			TLS: TLSConfig{
				KeyData:       []byte{1},
				GetCertHolder: getCert,
			},
		},
		"getCert2": {
			TLS: TLSConfig{
				KeyData:       []byte{1},
				GetCertHolder: &GetCertHolder{GetCert: func() (*tls.Certificate, error) { return nil, nil }},
			},
		},
		"getCert1, key 2": {
			TLS: TLSConfig{
				KeyData:       []byte{2},
				GetCertHolder: getCert,
			},
		},
		"http2, http1.1": {TLS: TLSConfig{NextProtos: []string{"h2", "http/1.1"}}},
		"http1.1-only":   {TLS: TLSConfig{NextProtos: []string{"http/1.1"}}},
	}
	for nameA, valueA := range uniqueConfigurations {
		for nameB, valueB := range uniqueConfigurations {
			keyA, canCacheA, err := tlsConfigKey(valueA)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameA, err)
				continue
			}
			keyB, canCacheB, err := tlsConfigKey(valueB)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameB, err)
				continue
			}

			shouldCacheA := valueA.Proxy == nil
			if shouldCacheA != canCacheA {
				t.Error("Unexpected canCache=false for " + nameA)
			}

			configIsNotEmpty := !reflect.DeepEqual(*valueA, Config{})
			if keyA == (tlsCacheKey{}) && shouldCacheA && configIsNotEmpty {
				t.Errorf("Expected non-empty cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
				continue
			}

			// Make sure we get the same key on the same config
			if nameA == nameB {
				if keyA != keyB {
					t.Errorf("Expected identical cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
				}
				if canCacheA != canCacheB {
					t.Errorf("Expected identical canCache %q and %q, got:\n\t%v\n\t%v", nameA, nameB, canCacheA, canCacheB)
				}
				continue
			}

			if canCacheA && canCacheB {
				if keyA == keyB {
					t.Errorf("Expected unique cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
					continue
				}
			}
		}
	}
}

func TestCacheLeak(t *testing.T) {
	requireCacheLen(t, tlsCache, 0) // clean start

	// manually create some transports that have some overlap
	// these 3 calls result in 2 transports in the cache
	rt1, err := New(&Config{TLS: TLSConfig{ServerName: "1"}})
	if err != nil {
		t.Fatal(err)
	}
	rt2, err := New(&Config{TLS: TLSConfig{ServerName: "2"}})
	if err != nil {
		t.Fatal(err)
	}
	rt3, err := New(&Config{TLS: TLSConfig{ServerName: "1"}})
	if err != nil {
		t.Fatal(err)
	}

	requireCacheLen(t, tlsCache, 2) // rt1 and rt2 (rt3 is the same as rt1)

	var wg wait.Group
	var d net.Dialer
	var rts []http.RoundTripper
	var rtsLock sync.Mutex
	for i := range 1_000 { // outer loop forces cache miss via dialer
		dh := &DialHolder{Dial: d.DialContext}
		for range i%7 + 1 { // inner loop exercises each cache value having 1 to N references
			wg.Start(func() {
				rt, err := New(&Config{DialHolder: dh})
				if err != nil {
					panic(err)
				}
				rtsLock.Lock()
				rts = append(rts, rt) // keep a live reference to the round tripper
				rtsLock.Unlock()
			})
		}
	}
	wg.Wait()

	requireCacheLen(t, tlsCache, 1_000+2) // rts and rt1 and rt2 (rt3 is the same as rt1)

	runtime.KeepAlive(rts) // prevent round trippers from being GC'd too early

	pollCacheSizeWithGC(t, tlsCache, 2) // rt1 and rt2 (rt3 is the same as rt1)

	runtime.KeepAlive(rt1)
	runtime.KeepAlive(rt2)
	runtime.KeepAlive(rt3)

	pollCacheSizeWithGC(t, tlsCache, 0)
}

func requireCacheLen(t *testing.T, c *tlsTransportCache, want int) {
	t.Helper()

	if cacheLen(c) != want {
		t.Fatalf("cache len %d, want %d", cacheLen(c), want)
	}
}

func cacheLen(c *tlsTransportCache) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	return len(c.transports)
}

func pollCacheSizeWithGC(t *testing.T, c *tlsTransportCache, want int) {
	t.Helper()

	if err := wait.PollUntilContextTimeout(t.Context(), 10*time.Millisecond, 10*time.Second, true, func(_ context.Context) (done bool, _ error) {
		runtime.GC() // run the garbage collector so the cleanups run
		return cacheLen(c) == want, nil
	}); err != nil {
		t.Fatalf("cache len %d, want %d: %v", cacheLen(c), want, err)
	}

	for range 3 { // make sure the cache size is stable even when more GC's happen
		runtime.GC()
	}
	requireCacheLen(t, c, want)
}
