// Copyright 2011 Google Inc. All Rights Reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"testing"

	"golang.org/x/net/context"

	"google.golang.org/appengine/internal"
)

func TestKeyEncoding(t *testing.T) {
	testCases := []struct {
		desc string
		key  *Key
		exp  string
	}{
		{
			desc: "A simple key with an int ID",
			key: &Key{
				kind:  "Person",
				intID: 1,
				appID: "glibrary",
			},
			exp: "aghnbGlicmFyeXIMCxIGUGVyc29uGAEM",
		},
		{
			desc: "A simple key with a string ID",
			key: &Key{
				kind:     "Graph",
				stringID: "graph:7-day-active",
				appID:    "glibrary",
			},
			exp: "aghnbGlicmFyeXIdCxIFR3JhcGgiEmdyYXBoOjctZGF5LWFjdGl2ZQw",
		},
		{
			desc: "A key with a parent",
			key: &Key{
				kind:  "WordIndex",
				intID: 1033,
				parent: &Key{
					kind:  "WordIndex",
					intID: 1020032,
					appID: "glibrary",
				},
				appID: "glibrary",
			},
			exp: "aghnbGlicmFyeXIhCxIJV29yZEluZGV4GIChPgwLEglXb3JkSW5kZXgYiQgM",
		},
	}
	for _, tc := range testCases {
		enc := tc.key.Encode()
		if enc != tc.exp {
			t.Errorf("%s: got %q, want %q", tc.desc, enc, tc.exp)
		}

		key, err := DecodeKey(tc.exp)
		if err != nil {
			t.Errorf("%s: failed decoding key: %v", tc.desc, err)
			continue
		}
		if !key.Equal(tc.key) {
			t.Errorf("%s: decoded key %v, want %v", tc.desc, key, tc.key)
		}
	}
}

func TestKeyGob(t *testing.T) {
	k := &Key{
		kind:  "Gopher",
		intID: 3,
		parent: &Key{
			kind:     "Mom",
			stringID: "narwhal",
			appID:    "gopher-con",
		},
		appID: "gopher-con",
	}

	buf := new(bytes.Buffer)
	if err := gob.NewEncoder(buf).Encode(k); err != nil {
		t.Fatalf("gob encode failed: %v", err)
	}

	k2 := new(Key)
	if err := gob.NewDecoder(buf).Decode(k2); err != nil {
		t.Fatalf("gob decode failed: %v", err)
	}
	if !k2.Equal(k) {
		t.Errorf("gob round trip of %v produced %v", k, k2)
	}
}

func TestNilKeyGob(t *testing.T) {
	type S struct {
		Key *Key
	}
	s1 := new(S)

	buf := new(bytes.Buffer)
	if err := gob.NewEncoder(buf).Encode(s1); err != nil {
		t.Fatalf("gob encode failed: %v", err)
	}

	s2 := new(S)
	if err := gob.NewDecoder(buf).Decode(s2); err != nil {
		t.Fatalf("gob decode failed: %v", err)
	}
	if s2.Key != nil {
		t.Errorf("gob round trip of nil key produced %v", s2.Key)
	}
}

func TestKeyJSON(t *testing.T) {
	k := &Key{
		kind:  "Gopher",
		intID: 2,
		parent: &Key{
			kind:     "Mom",
			stringID: "narwhal",
			appID:    "gopher-con",
		},
		appID: "gopher-con",
	}
	exp := `"` + k.Encode() + `"`

	buf, err := json.Marshal(k)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	if s := string(buf); s != exp {
		t.Errorf("JSON encoding of key %v: got %q, want %q", k, s, exp)
	}

	k2 := new(Key)
	if err := json.Unmarshal(buf, k2); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if !k2.Equal(k) {
		t.Errorf("JSON round trip of %v produced %v", k, k2)
	}
}

func TestNilKeyJSON(t *testing.T) {
	type S struct {
		Key *Key
	}
	s1 := new(S)

	buf, err := json.Marshal(s1)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	s2 := new(S)
	if err := json.Unmarshal(buf, s2); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if s2.Key != nil {
		t.Errorf("JSON round trip of nil key produced %v", s2.Key)
	}
}

func TestIncompleteKeyWithParent(t *testing.T) {
	c := internal.WithAppIDOverride(context.Background(), "s~some-app")

	// fadduh is a complete key.
	fadduh := NewKey(c, "Person", "", 1, nil)
	if fadduh.Incomplete() {
		t.Fatalf("fadduh is incomplete")
	}

	// robert is an incomplete key with fadduh as a parent.
	robert := NewIncompleteKey(c, "Person", fadduh)
	if !robert.Incomplete() {
		t.Fatalf("robert is complete")
	}

	// Both should be valid keys.
	if !fadduh.valid() {
		t.Errorf("fadduh is invalid: %v", fadduh)
	}
	if !robert.valid() {
		t.Errorf("robert is invalid: %v", robert)
	}
}

func TestNamespace(t *testing.T) {
	key := &Key{
		kind:      "Person",
		intID:     1,
		appID:     "s~some-app",
		namespace: "mynamespace",
	}
	if g, w := key.Namespace(), "mynamespace"; g != w {
		t.Errorf("key.Namespace() = %q, want %q", g, w)
	}
}
