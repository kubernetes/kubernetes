// +build go1.14

/*
Copyright 2020 The Kubernetes Authors.

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
)

func init() {
	// support official IANA names as well on go1.14
	ciphers["TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256"] = tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	ciphers["TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256"] = tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
}
