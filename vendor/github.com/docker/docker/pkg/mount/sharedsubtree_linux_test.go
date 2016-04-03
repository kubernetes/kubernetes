// +build linux

package mount

import (
	"os"
	"path"
	"syscall"
	"testing"
)

// nothing is propagated in or out
func TestSubtreePrivate(t *testing.T) {
	tmp := path.Join(os.TempDir(), "mount-tests")
	if err := os.MkdirAll(tmp, 0777); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	var (
		sourceDir   = path.Join(tmp, "source")
		targetDir   = path.Join(tmp, "target")
		outside1Dir = path.Join(tmp, "outside1")
		outside2Dir = path.Join(tmp, "outside2")

		outside1Path      = path.Join(outside1Dir, "file.txt")
		outside2Path      = path.Join(outside2Dir, "file.txt")
		outside1CheckPath = path.Join(targetDir, "a", "file.txt")
		outside2CheckPath = path.Join(sourceDir, "b", "file.txt")
	)
	if err := os.MkdirAll(path.Join(sourceDir, "a"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(path.Join(sourceDir, "b"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(targetDir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outside1Dir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outside2Dir, 0777); err != nil {
		t.Fatal(err)
	}

	if err := createFile(outside1Path); err != nil {
		t.Fatal(err)
	}
	if err := createFile(outside2Path); err != nil {
		t.Fatal(err)
	}

	// mount the shared directory to a target
	if err := Mount(sourceDir, targetDir, "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()

	// next, make the target private
	if err := MakePrivate(targetDir); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()

	// mount in an outside path to a mounted path inside the _source_
	if err := Mount(outside1Dir, path.Join(sourceDir, "a"), "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(path.Join(sourceDir, "a")); err != nil {
			t.Fatal(err)
		}
	}()

	// check that this file _does_not_ show in the _target_
	if _, err := os.Stat(outside1CheckPath); err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	} else if err == nil {
		t.Fatalf("%q should not be visible, but is", outside1CheckPath)
	}

	// next mount outside2Dir into the _target_
	if err := Mount(outside2Dir, path.Join(targetDir, "b"), "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(path.Join(targetDir, "b")); err != nil {
			t.Fatal(err)
		}
	}()

	// check that this file _does_not_ show in the _source_
	if _, err := os.Stat(outside2CheckPath); err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	} else if err == nil {
		t.Fatalf("%q should not be visible, but is", outside2CheckPath)
	}
}

// Testing that when a target is a shared mount,
// then child mounts propagate to the source
func TestSubtreeShared(t *testing.T) {
	tmp := path.Join(os.TempDir(), "mount-tests")
	if err := os.MkdirAll(tmp, 0777); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	var (
		sourceDir  = path.Join(tmp, "source")
		targetDir  = path.Join(tmp, "target")
		outsideDir = path.Join(tmp, "outside")

		outsidePath     = path.Join(outsideDir, "file.txt")
		sourceCheckPath = path.Join(sourceDir, "a", "file.txt")
	)

	if err := os.MkdirAll(path.Join(sourceDir, "a"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(targetDir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outsideDir, 0777); err != nil {
		t.Fatal(err)
	}

	if err := createFile(outsidePath); err != nil {
		t.Fatal(err)
	}

	// mount the source as shared
	if err := MakeShared(sourceDir); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(sourceDir); err != nil {
			t.Fatal(err)
		}
	}()

	// mount the shared directory to a target
	if err := Mount(sourceDir, targetDir, "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()

	// mount in an outside path to a mounted path inside the target
	if err := Mount(outsideDir, path.Join(targetDir, "a"), "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(path.Join(targetDir, "a")); err != nil {
			t.Fatal(err)
		}
	}()

	// NOW, check that the file from the outside directory is avaible in the source directory
	if _, err := os.Stat(sourceCheckPath); err != nil {
		t.Fatal(err)
	}
}

// testing that mounts to a shared source show up in the slave target,
// and that mounts into a slave target do _not_ show up in the shared source
func TestSubtreeSharedSlave(t *testing.T) {
	tmp := path.Join(os.TempDir(), "mount-tests")
	if err := os.MkdirAll(tmp, 0777); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	var (
		sourceDir   = path.Join(tmp, "source")
		targetDir   = path.Join(tmp, "target")
		outside1Dir = path.Join(tmp, "outside1")
		outside2Dir = path.Join(tmp, "outside2")

		outside1Path      = path.Join(outside1Dir, "file.txt")
		outside2Path      = path.Join(outside2Dir, "file.txt")
		outside1CheckPath = path.Join(targetDir, "a", "file.txt")
		outside2CheckPath = path.Join(sourceDir, "b", "file.txt")
	)
	if err := os.MkdirAll(path.Join(sourceDir, "a"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(path.Join(sourceDir, "b"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(targetDir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outside1Dir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(outside2Dir, 0777); err != nil {
		t.Fatal(err)
	}

	if err := createFile(outside1Path); err != nil {
		t.Fatal(err)
	}
	if err := createFile(outside2Path); err != nil {
		t.Fatal(err)
	}

	// mount the source as shared
	if err := MakeShared(sourceDir); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(sourceDir); err != nil {
			t.Fatal(err)
		}
	}()

	// mount the shared directory to a target
	if err := Mount(sourceDir, targetDir, "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()

	// next, make the target slave
	if err := MakeSlave(targetDir); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()

	// mount in an outside path to a mounted path inside the _source_
	if err := Mount(outside1Dir, path.Join(sourceDir, "a"), "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(path.Join(sourceDir, "a")); err != nil {
			t.Fatal(err)
		}
	}()

	// check that this file _does_ show in the _target_
	if _, err := os.Stat(outside1CheckPath); err != nil {
		t.Fatal(err)
	}

	// next mount outside2Dir into the _target_
	if err := Mount(outside2Dir, path.Join(targetDir, "b"), "none", "bind,rw"); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(path.Join(targetDir, "b")); err != nil {
			t.Fatal(err)
		}
	}()

	// check that this file _does_not_ show in the _source_
	if _, err := os.Stat(outside2CheckPath); err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	} else if err == nil {
		t.Fatalf("%q should not be visible, but is", outside2CheckPath)
	}
}

func TestSubtreeUnbindable(t *testing.T) {
	tmp := path.Join(os.TempDir(), "mount-tests")
	if err := os.MkdirAll(tmp, 0777); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	var (
		sourceDir = path.Join(tmp, "source")
		targetDir = path.Join(tmp, "target")
	)
	if err := os.MkdirAll(sourceDir, 0777); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(targetDir, 0777); err != nil {
		t.Fatal(err)
	}

	// next, make the source unbindable
	if err := MakeUnbindable(sourceDir); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := Unmount(sourceDir); err != nil {
			t.Fatal(err)
		}
	}()

	// then attempt to mount it to target. It should fail
	if err := Mount(sourceDir, targetDir, "none", "bind,rw"); err != nil && err != syscall.EINVAL {
		t.Fatal(err)
	} else if err == nil {
		t.Fatalf("%q should not have been bindable", sourceDir)
	}
	defer func() {
		if err := Unmount(targetDir); err != nil {
			t.Fatal(err)
		}
	}()
}

func createFile(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	f.WriteString("hello world!")
	return f.Close()
}
