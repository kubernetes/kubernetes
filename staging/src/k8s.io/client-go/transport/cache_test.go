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
	"net"
	"net/http"
	"testing"
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
			keyA, err := tlsConfigKey(valueA)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameA, err)
				continue
			}
			keyB, err := tlsConfigKey(valueB)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameB, err)
				continue
			}
			if keyA != keyB {
				t.Errorf("Expected identical cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
				continue
			}
		}
	}

	// Make sure config fields that affect the tls config affect the cache key
	uniqueConfigurations := map[string]*Config{
		"no tls":   {},
		"dialer":   {Dial: net.Dial},
		"dialer2":  {Dial: func(network, address string) (net.Conn, error) { return nil, nil }},
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
	}
	for nameA, valueA := range uniqueConfigurations {
		for nameB, valueB := range uniqueConfigurations {
			keyA, err := tlsConfigKey(valueA)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameA, err)
				continue
			}
			keyB, err := tlsConfigKey(valueB)
			if err != nil {
				t.Errorf("Unexpected error for %q: %v", nameB, err)
				continue
			}

			// Make sure we get the same key on the same config
			if nameA == nameB {
				if keyA != keyB {
					t.Errorf("Expected identical cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
				}
				continue
			}

			if keyA == keyB {
				t.Errorf("Expected unique cache keys for %q and %q, got:\n\t%s\n\t%s", nameA, nameB, keyA, keyB)
				continue
			}
		}
	}
}
