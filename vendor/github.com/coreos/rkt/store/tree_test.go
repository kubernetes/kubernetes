// Copyright 2015 The rkt Authors
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
	"io/ioutil"
	"os"
	"testing"

	"github.com/coreos/rkt/pkg/aci"
	"github.com/coreos/rkt/pkg/sys"
)

func treeStoreWriteACI(dir string, s *Store) (string, error) {
	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.7.4",
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
		return "", err
	}
	defer aci.Close()

	// Rewind the ACI
	if _, err := aci.Seek(0, 0); err != nil {
		return "", err
	}

	// Import the new ACI
	key, err := s.WriteACI(aci, false)
	if err != nil {
		return "", err
	}
	return key, nil
}

func TestTreeStoreWrite(t *testing.T) {
	if !sys.HasChrootCapability() {
		t.Skipf("chroot capability not available. Disabling test.")
	}

	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	key, err := treeStoreWriteACI(dir, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	id := "treestoreid01"

	// Ask the store to render the treestore
	_, err = s.treestore.Write(id, key, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify image Hash. Should be the same.
	_, err = s.treestore.Check(id)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTreeStoreRemove(t *testing.T) {
	if !sys.HasChrootCapability() {
		t.Skipf("chroot capability not available. Disabling test.")
	}

	dir, err := ioutil.TempDir("", tstprefix)
	if err != nil {
		t.Fatalf("error creating tempdir: %v", err)
	}
	defer os.RemoveAll(dir)
	s, err := NewStore(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	key, err := treeStoreWriteACI(dir, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	id := "treestoreid01"

	// Test non existent dir
	err = s.treestore.Remove(id, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Test rendered tree
	_, err = s.treestore.Write(id, key, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = s.treestore.Remove(id, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
