/*
 *
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package halfconn

import (
	"fmt"
	"hash"

	"golang.org/x/crypto/hkdf"
)

// hkdfExpander is the interface for the HKDF expansion function; see
// https://tools.ietf.org/html/rfc5869 for details. its use in TLS 1.3 is
// specified in https://tools.ietf.org/html/rfc8446#section-7.2
type hkdfExpander interface {
	// expand takes a secret, a label, and the output length in bytes, and
	// returns the resulting expanded key.
	expand(secret, label []byte, length int) ([]byte, error)
}

// defaultHKDFExpander is the default HKDF expander which uses Go's crypto/hkdf
// for HKDF expansion.
type defaultHKDFExpander struct {
	h func() hash.Hash
}

// newDefaultHKDFExpander creates an instance of the default HKDF expander
// using the given hash function.
func newDefaultHKDFExpander(h func() hash.Hash) hkdfExpander {
	return &defaultHKDFExpander{h: h}
}

func (d *defaultHKDFExpander) expand(secret, label []byte, length int) ([]byte, error) {
	outBuf := make([]byte, length)
	n, err := hkdf.Expand(d.h, secret, label).Read(outBuf)
	if err != nil {
		return nil, fmt.Errorf("hkdf.Expand.Read failed with error: %v", err)
	}
	if n < length {
		return nil, fmt.Errorf("hkdf.Expand.Read returned unexpected length, got %d, want %d", n, length)
	}
	return outBuf, nil
}
