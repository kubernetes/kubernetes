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

package flag

import (
	"crypto/tls"
	"fmt"
	"go/importer"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStrToUInt16(t *testing.T) {
	tests := []struct {
		flag           []string
		expected       []uint16
		expectedToFail bool
	}{
		{
			// Happy case
			flag:           []string{"TLS_RSA_WITH_RC4_128_SHA", "TLS_RSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_RSA_WITH_RC4_128_SHA", "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA"},
			expected:       []uint16{tls.TLS_RSA_WITH_RC4_128_SHA, tls.TLS_RSA_WITH_AES_128_CBC_SHA, tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA, tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA},
			expectedToFail: false,
		},
		{
			// One flag only
			flag:           []string{"TLS_RSA_WITH_RC4_128_SHA"},
			expected:       []uint16{tls.TLS_RSA_WITH_RC4_128_SHA},
			expectedToFail: false,
		},
		{
			// Empty flag
			flag:           []string{},
			expected:       nil,
			expectedToFail: false,
		},
		{
			// Duplicated flag
			flag:           []string{"TLS_RSA_WITH_RC4_128_SHA", "TLS_RSA_WITH_AES_128_CBC_SHA", "TLS_ECDHE_RSA_WITH_RC4_128_SHA", "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", "TLS_RSA_WITH_RC4_128_SHA"},
			expected:       []uint16{tls.TLS_RSA_WITH_RC4_128_SHA, tls.TLS_RSA_WITH_AES_128_CBC_SHA, tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA, tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA, tls.TLS_RSA_WITH_RC4_128_SHA},
			expectedToFail: false,
		},
		{
			// Invalid flag
			flag:           []string{"foo"},
			expected:       nil,
			expectedToFail: true,
		},
	}

	for i, test := range tests {
		uIntFlags, err := TLSCipherSuites(test.flag)
		if test.expectedToFail {
			assert.NotNil(t, err, "%d: expecting error, got %+v", i, err)
		} else {
			assert.Nil(t, err, "%d: unexpected error %s", i, err)
			assert.ElementsMatchf(t, test.expected, uIntFlags, "%d: expected %+v, got %+v", i, test.expected, uIntFlags)
		}
	}
}

func TestConstantMaps(t *testing.T) {
	pkg, err := importer.Default().Import("crypto/tls")
	if err != nil {
		fmt.Printf("error: %s\n", err.Error())
		return
	}
	discoveredVersions := map[string]bool{}
	discoveredCiphers := map[string]bool{}
	for _, declName := range pkg.Scope().Names() {
		if strings.HasPrefix(declName, "VersionTLS") {
			discoveredVersions[declName] = true
		}
		if strings.HasPrefix(declName, "TLS_RSA_") || strings.HasPrefix(declName, "TLS_ECDHE_") {
			discoveredCiphers[declName] = true
		}
	}

	for k := range discoveredCiphers {
		_, ok := ciphers[k]
		assert.True(t, ok, "discovered cipher tls.%s not in ciphers map", k)
	}
	for k := range ciphers {
		_, ok := discoveredCiphers[k]
		assert.True(t, ok, "ciphers map has %s not in tls package", k)
	}
	for k := range discoveredVersions {
		_, ok := versions[k]
		assert.True(t, ok, "discovered version tls.%s not in version map", k)
	}
	for k := range versions {
		_, ok := discoveredVersions[k]
		assert.True(t, ok, "versions map has %s not in tls package", k)
	}
}
