/*-
 * Copyright 2014 Square Inc.
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

package jose

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"math/big"
	"regexp"
)

// Reset random reader to original value
func resetRandReader() {
	randReader = rand.Reader
}

// Build big int from hex-encoded string. Strips whitespace (for testing).
func fromHexInt(base16 string) *big.Int {
	re := regexp.MustCompile(`\s+`)
	val, ok := new(big.Int).SetString(re.ReplaceAllString(base16, ""), 16)
	if !ok {
		panic("Invalid test data")
	}
	return val
}

// Build big int from base64-encoded string. Strips whitespace (for testing).
func fromBase64Int(encoded string) *big.Int {
	re := regexp.MustCompile(`\s+`)
	val, err := base64.RawURLEncoding.DecodeString(re.ReplaceAllString(encoded, ""))
	if err != nil {
		panic("Invalid test data: " + err.Error())
	}
	return new(big.Int).SetBytes(val)
}

// Decode hex-encoded string into byte array. Strips whitespace (for testing).
func fromHexBytes(base16 string) []byte {
	re := regexp.MustCompile(`\s+`)
	val, err := hex.DecodeString(re.ReplaceAllString(base16, ""))
	if err != nil {
		panic("Invalid test data")
	}
	return val
}

// Decode base64-encoded string into byte array. Strips whitespace (for testing).
func fromBase64Bytes(b64 string) []byte {
	re := regexp.MustCompile(`\s+`)
	val, err := base64.StdEncoding.DecodeString(re.ReplaceAllString(b64, ""))
	if err != nil {
		panic("Invalid test data")
	}
	return val
}
