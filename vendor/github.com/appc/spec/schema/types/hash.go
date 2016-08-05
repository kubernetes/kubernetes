// Copyright 2015 The appc Authors
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

package types

import (
	"crypto/sha512"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
)

const (
	maxHashSize = (sha512.Size / 2) + len("sha512-")
)

// Hash encodes a hash specified in a string of the form:
//    "<type>-<value>"
// for example
//    "sha512-06c733b1838136838e6d2d3e8fa5aea4c7905e92[...]"
// Valid types are currently:
//  * sha512
type Hash struct {
	typ string
	Val string
}

func NewHash(s string) (*Hash, error) {
	elems := strings.Split(s, "-")
	if len(elems) != 2 {
		return nil, errors.New("badly formatted hash string")
	}
	nh := Hash{
		typ: elems[0],
		Val: elems[1],
	}
	if err := nh.assertValid(); err != nil {
		return nil, err
	}
	return &nh, nil
}

func (h Hash) String() string {
	return fmt.Sprintf("%s-%s", h.typ, h.Val)
}

func (h *Hash) Set(s string) error {
	nh, err := NewHash(s)
	if err == nil {
		*h = *nh
	}
	return err
}

func (h Hash) Empty() bool {
	return reflect.DeepEqual(h, Hash{})
}

func (h Hash) assertValid() error {
	switch h.typ {
	case "sha512":
	case "":
		return fmt.Errorf("unexpected empty hash type")
	default:
		return fmt.Errorf("unrecognized hash type: %v", h.typ)
	}
	if h.Val == "" {
		return fmt.Errorf("unexpected empty hash value")
	}
	return nil
}

func (h *Hash) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	nh, err := NewHash(s)
	if err != nil {
		return err
	}
	*h = *nh
	return nil
}

func (h Hash) MarshalJSON() ([]byte, error) {
	if err := h.assertValid(); err != nil {
		return nil, err
	}
	return json.Marshal(h.String())
}

func NewHashSHA512(b []byte) *Hash {
	h := sha512.New()
	h.Write(b)
	nh, _ := NewHash(fmt.Sprintf("sha512-%x", h.Sum(nil)))
	return nh
}

func ShortHash(hash string) string {
	if len(hash) > maxHashSize {
		return hash[:maxHashSize]
	}
	return hash
}
