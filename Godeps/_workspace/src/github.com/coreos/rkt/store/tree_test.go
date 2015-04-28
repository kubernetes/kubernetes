package store

import (
	"archive/tar"
	"io/ioutil"
	"os"
	"testing"

	"github.com/coreos/rkt/pkg/aci"
)

func treeStoreWriteACI(dir string, s *Store) (string, error) {
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

	// Ask the store to render the treestore
	err = s.treestore.Write(key, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify image Hash. Should be the same.
	err = s.treestore.Check(key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTreeStoreRemove(t *testing.T) {
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

	// Test non existent dir
	err = s.treestore.Remove(key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Test rendered tree
	err = s.treestore.Write(key, s)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = s.treestore.Remove(key)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
