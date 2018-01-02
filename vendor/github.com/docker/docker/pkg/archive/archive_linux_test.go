package archive

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/docker/docker/pkg/system"
	"golang.org/x/sys/unix"
)

// setupOverlayTestDir creates files in a directory with overlay whiteouts
// Tree layout
// .
// ├── d1     # opaque, 0700
// │   └── f1 # empty file, 0600
// ├── d2     # opaque, 0750
// │   └── f1 # empty file, 0660
// └── d3     # 0700
//     └── f1 # whiteout, 0644
func setupOverlayTestDir(t *testing.T, src string) {
	// Create opaque directory containing single file and permission 0700
	if err := os.Mkdir(filepath.Join(src, "d1"), 0700); err != nil {
		t.Fatal(err)
	}

	if err := system.Lsetxattr(filepath.Join(src, "d1"), "trusted.overlay.opaque", []byte("y"), 0); err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(filepath.Join(src, "d1", "f1"), []byte{}, 0600); err != nil {
		t.Fatal(err)
	}

	// Create another opaque directory containing single file but with permission 0750
	if err := os.Mkdir(filepath.Join(src, "d2"), 0750); err != nil {
		t.Fatal(err)
	}

	if err := system.Lsetxattr(filepath.Join(src, "d2"), "trusted.overlay.opaque", []byte("y"), 0); err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile(filepath.Join(src, "d2", "f1"), []byte{}, 0660); err != nil {
		t.Fatal(err)
	}

	// Create regular directory with deleted file
	if err := os.Mkdir(filepath.Join(src, "d3"), 0700); err != nil {
		t.Fatal(err)
	}

	if err := system.Mknod(filepath.Join(src, "d3", "f1"), unix.S_IFCHR, 0); err != nil {
		t.Fatal(err)
	}
}

func checkOpaqueness(t *testing.T, path string, opaque string) {
	xattrOpaque, err := system.Lgetxattr(path, "trusted.overlay.opaque")
	if err != nil {
		t.Fatal(err)
	}
	if string(xattrOpaque) != opaque {
		t.Fatalf("Unexpected opaque value: %q, expected %q", string(xattrOpaque), opaque)
	}

}

func checkOverlayWhiteout(t *testing.T, path string) {
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	statT, ok := stat.Sys().(*syscall.Stat_t)
	if !ok {
		t.Fatalf("Unexpected type: %t, expected *syscall.Stat_t", stat.Sys())
	}
	if statT.Rdev != 0 {
		t.Fatalf("Non-zero device number for whiteout")
	}
}

func checkFileMode(t *testing.T, path string, perm os.FileMode) {
	stat, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if stat.Mode() != perm {
		t.Fatalf("Unexpected file mode for %s: %o, expected %o", path, stat.Mode(), perm)
	}
}

func TestOverlayTarUntar(t *testing.T) {
	oldmask, err := system.Umask(0)
	if err != nil {
		t.Fatal(err)
	}
	defer system.Umask(oldmask)

	src, err := ioutil.TempDir("", "docker-test-overlay-tar-src")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(src)

	setupOverlayTestDir(t, src)

	dst, err := ioutil.TempDir("", "docker-test-overlay-tar-dst")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dst)

	options := &TarOptions{
		Compression:    Uncompressed,
		WhiteoutFormat: OverlayWhiteoutFormat,
	}
	archive, err := TarWithOptions(src, options)
	if err != nil {
		t.Fatal(err)
	}
	defer archive.Close()

	if err := Untar(archive, dst, options); err != nil {
		t.Fatal(err)
	}

	checkFileMode(t, filepath.Join(dst, "d1"), 0700|os.ModeDir)
	checkFileMode(t, filepath.Join(dst, "d2"), 0750|os.ModeDir)
	checkFileMode(t, filepath.Join(dst, "d3"), 0700|os.ModeDir)
	checkFileMode(t, filepath.Join(dst, "d1", "f1"), 0600)
	checkFileMode(t, filepath.Join(dst, "d2", "f1"), 0660)
	checkFileMode(t, filepath.Join(dst, "d3", "f1"), os.ModeCharDevice|os.ModeDevice)

	checkOpaqueness(t, filepath.Join(dst, "d1"), "y")
	checkOpaqueness(t, filepath.Join(dst, "d2"), "y")
	checkOpaqueness(t, filepath.Join(dst, "d3"), "")
	checkOverlayWhiteout(t, filepath.Join(dst, "d3", "f1"))
}

func TestOverlayTarAUFSUntar(t *testing.T) {
	oldmask, err := system.Umask(0)
	if err != nil {
		t.Fatal(err)
	}
	defer system.Umask(oldmask)

	src, err := ioutil.TempDir("", "docker-test-overlay-tar-src")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(src)

	setupOverlayTestDir(t, src)

	dst, err := ioutil.TempDir("", "docker-test-overlay-tar-dst")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dst)

	archive, err := TarWithOptions(src, &TarOptions{
		Compression:    Uncompressed,
		WhiteoutFormat: OverlayWhiteoutFormat,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer archive.Close()

	if err := Untar(archive, dst, &TarOptions{
		Compression:    Uncompressed,
		WhiteoutFormat: AUFSWhiteoutFormat,
	}); err != nil {
		t.Fatal(err)
	}

	checkFileMode(t, filepath.Join(dst, "d1"), 0700|os.ModeDir)
	checkFileMode(t, filepath.Join(dst, "d1", WhiteoutOpaqueDir), 0700)
	checkFileMode(t, filepath.Join(dst, "d2"), 0750|os.ModeDir)
	checkFileMode(t, filepath.Join(dst, "d2", WhiteoutOpaqueDir), 0750)
	checkFileMode(t, filepath.Join(dst, "d3"), 0700|os.ModeDir)
	checkFileMode(t, filepath.Join(dst, "d1", "f1"), 0600)
	checkFileMode(t, filepath.Join(dst, "d2", "f1"), 0660)
	checkFileMode(t, filepath.Join(dst, "d3", WhiteoutPrefix+"f1"), 0600)
}
