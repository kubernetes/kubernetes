// Copyright 2014 The rkt Authors
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

package imagestore

import (
	"bytes"
	"database/sql"
	"encoding/hex"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/coreos/rkt/pkg/aci"
	"github.com/coreos/rkt/pkg/aci/acitest"
	"github.com/coreos/rkt/pkg/multicall"

	"github.com/appc/spec/schema/types"
)

const tstprefix = "store-test"

func init() {
	multicall.MaybeExec()
}

func TestBlobStore(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer s.Close()
	for _, valueStr := range []string{
		"I am a manually placed object",
	} {
		s.stores[blobType].Write(types.NewHashSHA512([]byte(valueStr)).String(), []byte(valueStr))
	}

	s.Dump(false)
}

func TestResolveKey(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer s.Close()

	// Return a hash key buffer from a hex string
	str2key := func(s string) *bytes.Buffer {
		k, _ := hex.DecodeString(s)
		return bytes.NewBufferString(keyToString(k))
	}

	// Set up store (use key == data for simplicity)
	data := []*bytes.Buffer{
		str2key("12345678900000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
		str2key("abcdefabc00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
		str2key("abcabcabc00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
		str2key("abc01234500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
		str2key("67147019a5b56f5e2ee01e989a8aa4787f56b8445960be2d8678391cf111009bc0780f31001fd181a2b61507547aee4caa44cda4b8bdb238d0e4ba830069ed2c"),
	}
	for _, d := range data {
		// Save aciinfo
		err := s.db.Do(func(tx *sql.Tx) error {
			aciinfo := &ACIInfo{
				BlobKey:    d.String(),
				Name:       "example.com/app",
				ImportTime: time.Now(),
			}
			return WriteACIInfo(tx, aciinfo)
		})
		if err != nil {
			t.Fatalf("error writing to store: %v", err)
		}
	}

	// Full key already - should return short version of the full key
	fkl := "sha512-67147019a5b56f5e2ee01e989a8aa4787f56b8445960be2d8678391cf111009bc0780f31001fd181a2b61507547aee4caa44cda4b8bdb238d0e4ba830069ed2c"
	fks := "sha512-67147019a5b56f5e2ee01e989a8aa4787f56b8445960be2d8678391cf111009b"
	for _, k := range []string{fkl, fks} {
		key, err := s.ResolveKey(k)
		if key != fks {
			t.Errorf("expected ResolveKey to return unaltered short key, but got %q", key)
		}
		if err != nil {
			t.Errorf("expected err=nil, got %v", err)
		}
	}

	// Unambiguous prefix match
	k, err := s.ResolveKey("sha512-123")
	if k != "sha512-1234567890000000000000000000000000000000000000000000000000000000" {
		t.Errorf("expected %q, got %q", "sha512-1234567890000000000000000000000000000000000000000000000000000000", k)
	}
	if err != nil {
		t.Errorf("expected err=nil, got %v", err)
	}

	// Ambiguous prefix match
	k, err = s.ResolveKey("sha512-abc")
	if k != "" {
		t.Errorf("expected %q, got %q", "", k)
	}
	if err == nil {
		t.Errorf("expected non-nil error!")
	}

	// wrong key prefix
	k, err = s.ResolveKey("badprefix-1")
	expectedErr := "wrong key prefix"
	if err == nil {
		t.Errorf("expected non-nil error!")
	}
	if err.Error() != expectedErr {
		t.Errorf("expected err=%q, got %q", expectedErr, err)
	}

	// image ID too short
	k, err = s.ResolveKey("sha512-1")
	expectedErr = "image ID too short"
	if err == nil {
		t.Errorf("expected non-nil error!")
	}
	if err.Error() != expectedErr {
		t.Errorf("expected err=%q, got %q", expectedErr, err)
	}
}

func TestGetImageManifest(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer s.Close()

	imj, err := acitest.ImageManifestString(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	aci, err := aci.NewACI(dir, imj, nil)
	if err != nil {
		t.Fatalf("error creating test tar: %v", err)
	}
	// Rewind the ACI
	if _, err := aci.Seek(0, 0); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	key, err := s.WriteACI(aci, ACIFetchInfo{Latest: false})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	wanted := "example.com/test01"
	im, err := s.GetImageManifest(key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if im.Name.String() != wanted {
		t.Errorf("expected im with name: %s, got: %s", wanted, im.Name.String())
	}

	// test nonexistent key
	im, err = s.GetImageManifest("sha512-aaaaaaaaaaaaaaaaa")
	if err == nil {
		t.Fatalf("expected non-nil error!")
	}
}

func TestGetAci(t *testing.T) {
	type test struct {
		name     types.ACIdentifier
		labels   types.Labels
		expected int // the aci index to expect or -1 if not result expected,
	}

	type acidef struct {
		imj    string
		latest bool
	}

	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	defer s.Close()

	tests := []struct {
		acidefs []acidef
		tests   []test
	}{
		{
			[]acidef{
				{
					`{
						"acKind": "ImageManifest",
						"acVersion": "0.8.10",
						"name": "example.com/test01"
					}`,
					false,
				},
				{
					`{
						"acKind": "ImageManifest",
						"acVersion": "0.8.10",
						"name": "example.com/test02",
						"labels": [
							{
								"name": "version",
								"value": "1.0.0"
							}
						]
					}`,
					true,
				},
				{
					`{
						"acKind": "ImageManifest",
						"acVersion": "0.8.10",
						"name": "example.com/test02",
						"labels": [
							{
								"name": "version",
								"value": "2.0.0"
							}
						]
					}`,
					false,
				},
			},
			[]test{
				{
					"example.com/nonexistentaci",
					types.Labels{},
					-1,
				},
				{
					"example.com/test01",
					types.Labels{},
					0,
				},
				{
					"example.com/test02",
					// Workaround for https://github.com/golang/go/issues/6820 :
					// `go vet` does not correctly detect types.Labels as a container
					[]types.Label{
						{
							Name:  "version",
							Value: "1.0.0",
						},
					},
					1,
				},
				{
					"example.com/test02",
					[]types.Label{
						{
							Name:  "version",
							Value: "2.0.0",
						},
					},
					2,
				},
				{
					"example.com/test02",
					types.Labels{},
					1,
				},
			},
		},
	}

	for _, tt := range tests {
		var keys []string
		// Create ACIs
		for _, ad := range tt.acidefs {
			aci, err := aci.NewACI(dir, ad.imj, nil)
			if err != nil {
				t.Fatalf("error creating test tar: %v", err)
			}

			// Rewind the ACI
			if _, err := aci.Seek(0, 0); err != nil {
				t.Fatalf("unexpected error %v", err)
			}

			key, err := s.WriteACI(aci, ACIFetchInfo{Latest: ad.latest})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			keys = append(keys, key)
		}

		for _, test := range tt.tests {
			key, err := s.GetACI(test.name, test.labels)
			if test.expected == -1 {
				if err == nil {
					t.Fatalf("Expected no key for name %s, got %s", test.name, key)
				}

			} else {
				if err != nil {
					t.Fatalf("unexpected error on GetACI for name %s, labels: %v: %v", test.name, test.labels, err)
				}
				if keys[test.expected] != key {
					t.Errorf("expected key: %s, got %s. GetACI with name: %s, labels: %v", key, keys[test.expected], test.name, test.labels)
				}
			}
		}
	}
}

func TestRemoveACI(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer s.Close()

	imj, err := acitest.ImageManifestString(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	aciFile, err := aci.NewACI(dir, imj, nil)
	if err != nil {
		t.Fatalf("error creating test tar: %v", err)
	}
	// Rewind the ACI
	if _, err := aciFile.Seek(0, 0); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	key, err := s.WriteACI(aciFile, ACIFetchInfo{Latest: false})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	aciURL := "http://example.com/test01.aci"
	// Create our first Remote, and simulate Store() to create row in the table
	na := NewRemote(aciURL, "")
	na.BlobKey = key
	s.WriteRemote(na)

	err = s.RemoveACI(key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify that no remote for the specified key exists
	_, err = s.GetRemote(aciURL)
	if err != ErrRemoteNotFound {
		t.Fatalf("unexpected error: %v", err)
	}

	// Try to remove a non-existent key
	err = s.RemoveACI("sha512-aaaaaaaaaaaaaaaaa")
	if err == nil {
		t.Fatalf("expected error")
	}

	aciFile, err = aci.NewACI(dir, imj, nil)
	if err != nil {
		t.Fatalf("error creating test tar: %v", err)
	}
	// Rewind the ACI
	if _, err := aciFile.Seek(0, 0); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	key, err = s.WriteACI(aciFile, ACIFetchInfo{Latest: false})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	aciURL = "http://example.com/test02.aci"
	// Create our first Remote, and simulate Store() to create row in the table
	na = NewRemote(aciURL, "")
	na.BlobKey = key
	s.WriteRemote(na)

	err = os.Remove(filepath.Join(dir, "blob", blockTransform(key)[0], blockTransform(key)[1], key))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = s.RemoveACI(key)
	if err == nil {
		t.Fatalf("expected error: %v", err)
	}
	if _, ok := err.(*StoreRemovalError); !ok {
		t.Fatalf("expected StoreRemovalError got: %v", err)
	}
}
