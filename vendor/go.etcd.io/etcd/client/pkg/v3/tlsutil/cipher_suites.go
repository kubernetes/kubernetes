// Copyright 2018 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tlsutil

import "crypto/tls"

// GetCipherSuite returns the corresponding cipher suite,
// and boolean value if it is supported.
func GetCipherSuite(s string) (uint16, bool) {
	for _, c := range tls.CipherSuites() {
		if s == c.Name {
			return c.ID, true
		}
	}
	for _, c := range tls.InsecureCipherSuites() {
		if s == c.Name {
			return c.ID, true
		}
	}
	switch s {
	case "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305":
		return tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256, true
	case "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305":
		return tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256, true
	}
	return 0, false
}
