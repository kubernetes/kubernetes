// Copyright 2016 The Linux Foundation
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

package identity

import (
	_ "crypto/sha256" // side-effect to install impls, sha256
	_ "crypto/sha512" // side-effect to install impls, sha384/sh512

	"io"

	digest "github.com/opencontainers/go-digest"
)

// FromReader consumes the content of rd until io.EOF, returning canonical
// digest.
func FromReader(rd io.Reader) (digest.Digest, error) {
	return digest.Canonical.FromReader(rd)
}

// FromBytes digests the input and returns a Digest.
func FromBytes(p []byte) digest.Digest {
	return digest.Canonical.FromBytes(p)
}

// FromString digests the input and returns a Digest.
func FromString(s string) digest.Digest {
	return digest.Canonical.FromString(s)
}
