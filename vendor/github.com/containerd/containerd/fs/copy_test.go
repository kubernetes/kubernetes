package fs

import (
	"io/ioutil"
	"os"
	"testing"

	_ "crypto/sha256"

	"github.com/containerd/containerd/fs/fstest"
	"github.com/pkg/errors"
)

// TODO: Create copy directory which requires privilege
//  chown
//  mknod
//  setxattr fstest.SetXAttr("/home", "trusted.overlay.opaque", "y"),

func TestCopyDirectory(t *testing.T) {
	apply := fstest.Apply(
		fstest.CreateDir("/etc/", 0755),
		fstest.CreateFile("/etc/hosts", []byte("localhost 127.0.0.1"), 0644),
		fstest.Link("/etc/hosts", "/etc/hosts.allow"),
		fstest.CreateDir("/usr/local/lib", 0755),
		fstest.CreateFile("/usr/local/lib/libnothing.so", []byte{0x00, 0x00}, 0755),
		fstest.Symlink("libnothing.so", "/usr/local/lib/libnothing.so.2"),
		fstest.CreateDir("/home", 0755),
	)

	if err := testCopy(apply); err != nil {
		t.Fatalf("Copy test failed: %+v", err)
	}
}

// This test used to fail because link-no-nothing.txt would be copied first,
// then file operations in dst during the CopyDir would follow the symlink and
// fail.
func TestCopyDirectoryWithLocalSymlink(t *testing.T) {
	apply := fstest.Apply(
		fstest.CreateFile("nothing.txt", []byte{0x00, 0x00}, 0755),
		fstest.Symlink("nothing.txt", "link-no-nothing.txt"),
	)

	if err := testCopy(apply); err != nil {
		t.Fatalf("Copy test failed: %+v", err)
	}
}

func testCopy(apply fstest.Applier) error {
	t1, err := ioutil.TempDir("", "test-copy-src-")
	if err != nil {
		return errors.Wrap(err, "failed to create temporary directory")
	}
	defer os.RemoveAll(t1)

	t2, err := ioutil.TempDir("", "test-copy-dst-")
	if err != nil {
		return errors.Wrap(err, "failed to create temporary directory")
	}
	defer os.RemoveAll(t2)

	if err := apply.Apply(t1); err != nil {
		return errors.Wrap(err, "failed to apply changes")
	}

	if err := CopyDir(t2, t1); err != nil {
		return errors.Wrap(err, "failed to copy")
	}

	return fstest.CheckDirectoryEqual(t1, t2)
}
