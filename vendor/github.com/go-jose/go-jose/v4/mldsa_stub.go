//go:build !go1.27

/*-
 * Copyright 2026 The Go JOSE Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// This file provides the ML-DSA seam for toolchains without crypto/mldsa, which
// was added in Go 1.27. Every hook reports that it did not recognize the key, so
// callers fall through to their existing unsupported-key/algorithm error.
//
// Each identifier here must have a counterpart in mldsa.go.

package jose

func mldsaKeyInfo(_ interface{}) (isPublic bool, ok bool) { return false, false }

func mldsaPublicOf(_ interface{}) (interface{}, bool) { return nil, false }

func mldsaSigner(_ SignatureAlgorithm, _ interface{}) (recipientSigInfo, bool, error) {
	return recipientSigInfo{}, false, nil
}

func mldsaVerifier(_ interface{}) (payloadVerifier, bool, error) { return nil, false, nil }

func mldsaRawJWK(_ interface{}) (*rawJSONWebKey, bool, error) { return nil, false, nil }

func mldsaParseJWK(_ *rawJSONWebKey, _ interface{}) (interface{}, error) {
	return nil, ErrUnsupportedKeyType
}

func mldsaThumbprintInput(_ interface{}) (string, bool, error) { return "", false, nil }
