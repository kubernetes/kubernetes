// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v1

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"strconv"
	"strings"
)

// Hash is an unqualified digest of some content, e.g. sha256:deadbeef
type Hash struct {
	// Algorithm holds the algorithm used to compute the hash.
	Algorithm string

	// Hex holds the hex portion of the content hash.
	Hex string
}

// String reverses NewHash returning the string-form of the hash.
func (h Hash) String() string {
	return fmt.Sprintf("%s:%s", h.Algorithm, h.Hex)
}

// NewHash validates the input string is a hash and returns a strongly type Hash object.
func NewHash(s string) (Hash, error) {
	h := Hash{}
	if err := h.parse(s); err != nil {
		return Hash{}, err
	}
	return h, nil
}

// MarshalJSON implements json.Marshaler
func (h Hash) MarshalJSON() ([]byte, error) {
	return json.Marshal(h.String())
}

// UnmarshalJSON implements json.Unmarshaler
func (h *Hash) UnmarshalJSON(data []byte) error {
	s, err := strconv.Unquote(string(data))
	if err != nil {
		return err
	}
	return h.parse(s)
}

// MarshalText implements encoding.TextMarshaler. This is required to use
// v1.Hash as a key in a map when marshalling JSON.
func (h Hash) MarshalText() (text []byte, err error) {
	return []byte(h.String()), nil
}

// UnmarshalText implements encoding.TextUnmarshaler. This is required to use
// v1.Hash as a key in a map when unmarshalling JSON.
func (h *Hash) UnmarshalText(text []byte) error {
	return h.parse(string(text))
}

// Hasher returns a hash.Hash for the named algorithm (e.g. "sha256")
func Hasher(name string) (hash.Hash, error) {
	switch name {
	case "sha256":
		return sha256.New(), nil
	default:
		return nil, fmt.Errorf("unsupported hash: %q", name)
	}
}

func (h *Hash) parse(unquoted string) error {
	parts := strings.Split(unquoted, ":")
	if len(parts) != 2 {
		return fmt.Errorf("cannot parse hash: %q", unquoted)
	}

	rest := strings.TrimLeft(parts[1], "0123456789abcdef")
	if len(rest) != 0 {
		return fmt.Errorf("found non-hex character in hash: %c", rest[0])
	}

	hasher, err := Hasher(parts[0])
	if err != nil {
		return err
	}
	// Compare the hex to the expected size (2 hex characters per byte)
	if len(parts[1]) != hasher.Size()*2 {
		return fmt.Errorf("wrong number of hex digits for %s: %s", parts[0], parts[1])
	}

	h.Algorithm = parts[0]
	h.Hex = parts[1]
	return nil
}

// SHA256 computes the Hash of the provided io.Reader's content.
func SHA256(r io.Reader) (Hash, int64, error) {
	hasher := sha256.New()
	n, err := io.Copy(hasher, r)
	if err != nil {
		return Hash{}, 0, err
	}
	return Hash{
		Algorithm: "sha256",
		Hex:       hex.EncodeToString(hasher.Sum(make([]byte, 0, hasher.Size()))),
	}, n, nil
}
