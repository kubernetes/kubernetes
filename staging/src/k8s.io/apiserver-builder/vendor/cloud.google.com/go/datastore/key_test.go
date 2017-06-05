// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datastore

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"testing"

	"golang.org/x/net/context"
)

func TestNamespace(t *testing.T) {
	c := context.Background()
	k := NewIncompleteKey(c, "foo", nil)
	if got, want := k.Namespace(), ""; got != want {
		t.Errorf("No namespace, k.Namespace() = %q, want %q", got, want)
	}

	c = WithNamespace(c, "gopherspace")
	k = NewIncompleteKey(c, "foo", nil)
	if got, want := k.Namespace(), "gopherspace"; got != want {
		t.Errorf("No namespace, k.Namespace() = %q, want %q", got, want)
	}
}

func TestParent(t *testing.T) {
	c := context.Background()
	k := NewIncompleteKey(c, "foo", nil)
	par := NewKey(c, "foomum", "", 1248, nil)
	k.SetParent(par)
	if got := k.Parent(); got != par {
		t.Errorf("k.Parent() = %v; want %v", got, par)
	}
}

func TestEqual(t *testing.T) {
	c := context.Background()
	cN := WithNamespace(c, "gopherspace")

	testCases := []struct {
		x, y  *Key
		equal bool
	}{
		{
			x:     nil,
			y:     nil,
			equal: true,
		},
		{
			x:     NewKey(c, "kindA", "", 0, nil),
			y:     NewIncompleteKey(c, "kindA", nil),
			equal: true,
		},
		{
			x:     NewKey(c, "kindA", "nameA", 0, nil),
			y:     NewKey(c, "kindA", "nameA", 0, nil),
			equal: true,
		},
		{
			x:     NewKey(cN, "kindA", "nameA", 0, nil),
			y:     NewKey(cN, "kindA", "nameA", 0, nil),
			equal: true,
		},
		{
			x:     NewKey(c, "kindA", "", 1337, NewKey(c, "kindX", "nameX", 0, nil)),
			y:     NewKey(c, "kindA", "", 1337, NewKey(c, "kindX", "nameX", 0, nil)),
			equal: true,
		},
		{
			x:     NewKey(c, "kindA", "nameA", 0, nil),
			y:     NewKey(c, "kindB", "nameA", 0, nil),
			equal: false,
		},
		{
			x:     NewKey(c, "kindA", "nameA", 0, nil),
			y:     NewKey(c, "kindA", "nameB", 0, nil),
			equal: false,
		},
		{
			x:     NewKey(c, "kindA", "nameA", 0, nil),
			y:     NewKey(c, "kindA", "", 1337, nil),
			equal: false,
		},
		{
			x:     NewKey(c, "kindA", "nameA", 0, nil),
			y:     NewKey(cN, "kindA", "nameA", 0, nil),
			equal: false,
		},
		{
			x:     NewKey(c, "kindA", "", 1337, NewKey(c, "kindX", "nameX", 0, nil)),
			y:     NewKey(c, "kindA", "", 1337, NewKey(c, "kindY", "nameX", 0, nil)),
			equal: false,
		},
		{
			x:     NewKey(c, "kindA", "", 1337, NewKey(c, "kindX", "nameX", 0, nil)),
			y:     NewKey(c, "kindA", "", 1337, nil),
			equal: false,
		},
	}

	for _, tt := range testCases {
		if got := tt.x.Equal(tt.y); got != tt.equal {
			t.Errorf("Equal(%v, %v) = %t; want %t", tt.x, tt.y, got, tt.equal)
		}
		if got := tt.y.Equal(tt.x); got != tt.equal {
			t.Errorf("Equal(%v, %v) = %t; want %t", tt.y, tt.x, got, tt.equal)
		}
	}
}

func TestEncoding(t *testing.T) {
	c := context.Background()
	cN := WithNamespace(c, "gopherspace")

	testCases := []struct {
		k     *Key
		valid bool
	}{
		{
			k:     nil,
			valid: false,
		},
		{
			k:     NewKey(c, "", "", 0, nil),
			valid: false,
		},
		{
			k:     NewKey(c, "kindA", "", 0, nil),
			valid: true,
		},
		{
			k:     NewKey(cN, "kindA", "", 0, nil),
			valid: true,
		},
		{
			k:     NewKey(c, "kindA", "nameA", 0, nil),
			valid: true,
		},
		{
			k:     NewKey(c, "kindA", "", 1337, nil),
			valid: true,
		},
		{
			k:     NewKey(c, "kindA", "nameA", 1337, nil),
			valid: false,
		},
		{
			k:     NewKey(c, "kindA", "", 0, NewKey(c, "kindB", "nameB", 0, nil)),
			valid: true,
		},
		{
			k:     NewKey(c, "kindA", "", 0, NewKey(c, "kindB", "", 0, nil)),
			valid: false,
		},
		{
			k:     NewKey(c, "kindA", "", 0, NewKey(cN, "kindB", "nameB", 0, nil)),
			valid: false,
		},
	}

	for _, tt := range testCases {
		if got := tt.k.valid(); got != tt.valid {
			t.Errorf("valid(%v) = %t; want %t", tt.k, got, tt.valid)
		}

		// Check encoding/decoding for valid keys.
		if !tt.valid {
			continue
		}
		enc := tt.k.Encode()
		dec, err := DecodeKey(enc)
		if err != nil {
			t.Errorf("DecodeKey(%q) from %v: %v", enc, tt.k, err)
			continue
		}
		if !tt.k.Equal(dec) {
			t.Logf("Proto: %s", keyToProto(tt.k))
			t.Errorf("Decoded key %v not equal to %v", dec, tt.k)
		}

		b, err := json.Marshal(tt.k)
		if err != nil {
			t.Errorf("json.Marshal(%v): %v", tt.k, err)
			continue
		}
		key := &Key{}
		if err := json.Unmarshal(b, key); err != nil {
			t.Errorf("json.Unmarshal(%s) for key %v: %v", b, tt.k, err)
			continue
		}
		if !tt.k.Equal(key) {
			t.Errorf("JSON decoded key %v not equal to %v", dec, tt.k)
		}

		buf := &bytes.Buffer{}
		gobEnc := gob.NewEncoder(buf)
		if err := gobEnc.Encode(tt.k); err != nil {
			t.Errorf("gobEnc.Encode(%v): %v", tt.k, err)
			continue
		}
		gobDec := gob.NewDecoder(buf)
		key = &Key{}
		if err := gobDec.Decode(key); err != nil {
			t.Errorf("gobDec.Decode() for key %v: %v", tt.k, err)
		}
		if !tt.k.Equal(key) {
			t.Errorf("gob decoded key %v not equal to %v", dec, tt.k)
		}
	}
}

func TestInvalidKeyDecode(t *testing.T) {
	// Check that decoding an invalid key returns an err and doesn't panic.
	enc := NewKey(context.Background(), "Kind", "Foo", 0, nil).Encode()

	invalid := []string{
		"",
		"Laboratorio",
		enc + "Junk",
		enc[:len(enc)-4],
	}
	for _, enc := range invalid {
		key, err := DecodeKey(enc)
		if err == nil || key != nil {
			t.Errorf("DecodeKey(%q) = %v, %v; want nil, error", enc, key, err)
		}
	}
}
