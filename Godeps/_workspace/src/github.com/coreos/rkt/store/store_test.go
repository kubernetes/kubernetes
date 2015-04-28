// Copyright 2014 CoreOS, Inc.
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

package store

import (
	"archive/tar"
	"bytes"
	"database/sql"
	"encoding/hex"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/pkg/aci"
)

const tstprefix = "store-test"

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
				AppName:    "example.com/app",
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

	// key too short
	k, err = s.ResolveKey("sha512-1")
	expectedErr = "key too short"
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

	imj := `{
			"acKind": "ImageManifest",
			"acVersion": "0.5.4",
			"name": "example.com/test01"
		}`

	aci, err := aci.NewACI(dir, imj, nil)
	if err != nil {
		t.Fatalf("error creating test tar: %v", err)
	}
	// Rewind the ACI
	if _, err := aci.Seek(0, 0); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	key, err := s.WriteACI(aci, false)
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

	// test unexistent key
	im, err = s.GetImageManifest("sha512-aaaaaaaaaaaaaaaaa")
	if err == nil {
		t.Fatalf("expected non-nil error!")
	}
}

func TestGetAci(t *testing.T) {
	type test struct {
		name     types.ACName
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

	tests := []struct {
		acidefs []acidef
		tests   []test
	}{
		{
			[]acidef{
				{
					`{
						"acKind": "ImageManifest",
						"acVersion": "0.1.1",
						"name": "example.com/test01"
					}`,
					false,
				},
				{
					`{
						"acKind": "ImageManifest",
						"acVersion": "0.1.1",
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
						"acVersion": "0.1.1",
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
					"example.com/unexistentaci",
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
					types.Labels{
						{
							Name:  "version",
							Value: "1.0.0",
						},
					},
					1,
				},
				{
					"example.com/test02",
					types.Labels{
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
		keys := []string{}
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

			key, err := s.WriteACI(aci, ad.latest)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			keys = append(keys, key)
		}

		for _, test := range tt.tests {
			key, err := s.GetACI(test.name, test.labels)
			if test.expected == -1 {
				if err == nil {
					t.Fatalf("Expected no key for appName %s, got %s", test.name, key)
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

func TestTreeStore(t *testing.T) {
	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.5.4",
		    "name": "example.com/test01"
		}
	`

	entries := []*aci.ACIEntry{
		// An empty dir
		{
			Header: &tar.Header{
				Name:     "rootfs/a",
				Typeflag: tar.TypeDir,
			},
		},
		{
			Contents: "hello",
			Header: &tar.Header{
				Name: "hello.txt",
				Size: 5,
			},
		},
		{
			Header: &tar.Header{
				Name:     "rootfs/link.txt",
				Linkname: "rootfs/hello.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
		// dangling symlink
		{
			Header: &tar.Header{
				Name:     "rootfs/link2.txt",
				Linkname: "rootfs/missingfile.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
		{
			Header: &tar.Header{
				Name:     "rootfs/fifo",
				Typeflag: tar.TypeFifo,
			},
		},
	}
	aci, err := aci.NewACI(dir, imj, entries)
	if err != nil {
		t.Fatalf("error creating test tar: %v", err)
	}
	defer aci.Close()

	// Rewind the ACI
	if _, err := aci.Seek(0, 0); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	// Import the new ACI
	key, err := s.WriteACI(aci, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Ask the store to render the treestore
	err = s.RenderTreeStore(key, false)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify image Hash. Should be the same.
	err = s.CheckTreeStore(key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Change a file permission
	rootfs := s.GetTreeStoreRootFS(key)
	err = os.Chmod(filepath.Join(rootfs, "a"), 0600)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify image Hash. Should be different
	err = s.CheckTreeStore(key)
	if err == nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// rebuild the tree
	err = s.RenderTreeStore(key, true)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Add a file
	rootfs = s.GetTreeStoreRootFS(key)
	err = ioutil.WriteFile(filepath.Join(rootfs, "newfile"), []byte("newfile"), 0644)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify image Hash. Should be different
	err = s.CheckTreeStore(key)
	if err == nil {
		t.Fatalf("unexpected error: %v", err)
	}

}
