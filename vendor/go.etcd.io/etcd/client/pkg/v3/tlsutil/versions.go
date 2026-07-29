// Copyright 2023 The etcd Authors
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

import (
	"crypto/tls"
	"fmt"
)

type TLSVersion string

// Constants for TLS versions.
const (
	TLSVersionDefault TLSVersion = ""
	TLSVersion12      TLSVersion = "TLS1.2"
	TLSVersion13      TLSVersion = "TLS1.3"
)

// GetTLSVersion returns the corresponding tls.Version or error.
func GetTLSVersion(version string) (uint16, error) {
	var v uint16

	switch version {
	case string(TLSVersionDefault):
		v = 0 // 0 means let Go decide.
	case string(TLSVersion12):
		v = tls.VersionTLS12
	case string(TLSVersion13):
		v = tls.VersionTLS13
	default:
		return 0, fmt.Errorf("unexpected TLS version %q (must be one of: TLS1.2, TLS1.3)", version)
	}

	return v, nil
}
