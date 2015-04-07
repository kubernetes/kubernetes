package archive

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"sort"
	"testing"
	"time"
)

func max(x, y int) int {
	if x >= y {
		return x
	}
	return y
}

func copyDir(src, dst string) error {
	cmd := exec.Command("cp", "-a", src, dst)
	if err := cmd.Run(); err != nil {
		return err
	}
	return nil
}

type FileType uint32

const (
	Regular FileType = iota
	Dir
	Symlink
)

type FileData struct {
	filetype    FileType
	path        string
	contents    string
	permissions os.FileMode
}

func createSampleDir(t *testing.T, root string) {
	files := []FileData{
		{Regular, "file1", "file1\n", 0600},
		{Regular, "file2", "file2\n", 0666},
		{Regular, "file3", "file3\n", 0404},
		{Regular, "file4", "file4\n", 0600},
		{Regular, "file5", "file5\n", 0600},
		{Regular, "file6", "file6\n", 0600},
		{Regular, "file7", "file7\n", 0600},
		{Dir, "dir1", "", 0740},
		{Regular, "dir1/file1-1", "file1-1\n", 01444},
		{Regular, "dir1/file1-2", "file1-2\n", 0666},
		{Dir, "dir2", "", 0700},
		{Regular, "dir2/file2-1", "file2-1\n", 0666},
		{Regular, "dir2/file2-2", "file2-2\n", 0666},
		{Dir, "dir3", "", 0700},
		{Regular, "dir3/file3-1", "file3-1\n", 0666},
		{Regular, "dir3/file3-2", "file3-2\n", 0666},
		{Dir, "dir4", "", 0700},
		{Regular, "dir4/file3-1", "file4-1\n", 0666},
		{Regular, "dir4/file3-2", "file4-2\n", 0666},
		{Symlink, "symlink1", "target1", 0666},
		{Symlink, "symlink2", "target2", 0666},
	}

	now := time.Now()
	for _, info := range files {
		p := path.Join(root, info.path)
		if info.filetype == Dir {
			if err := os.MkdirAll(p, info.permissions); err != nil {
				t.Fatal(err)
			}
		} else if info.filetype == Regular {
			if err := ioutil.WriteFile(p, []byte(info.contents), info.permissions); err != nil {
				t.Fatal(err)
			}
		} else if info.filetype == Symlink {
			if err := os.Symlink(info.contents, p); err != nil {
				t.Fatal(err)
			}
		}

		if info.filetype != Symlink {
			// Set a consistent ctime, atime for all files and dirs
			if err := os.Chtimes(p, now, now); err != nil {
				t.Fatal(err)
			}
		}
	}
}

// Create an directory, copy it, make sure we report no changes between the two
func TestChangesDirsEmpty(t *testing.T) {
	src, err := ioutil.TempDir("", "docker-changes-test")
	if err != nil {
		t.Fatal(err)
	}
	createSampleDir(t, src)
	dst := src + "-copy"
	if err := copyDir(src, dst); err != nil {
		t.Fatal(err)
	}
	changes, err := ChangesDirs(dst, src)
	if err != nil {
		t.Fatal(err)
	}

	if len(changes) != 0 {
		t.Fatalf("Reported changes for identical dirs: %v", changes)
	}
	os.RemoveAll(src)
	os.RemoveAll(dst)
}

func mutateSampleDir(t *testing.T, root string) {
	// Remove a regular file
	if err := os.RemoveAll(path.Join(root, "file1")); err != nil {
		t.Fatal(err)
	}

	// Remove a directory
	if err := os.RemoveAll(path.Join(root, "dir1")); err != nil {
		t.Fatal(err)
	}

	// Remove a symlink
	if err := os.RemoveAll(path.Join(root, "symlink1")); err != nil {
		t.Fatal(err)
	}

	// Rewrite a file
	if err := ioutil.WriteFile(path.Join(root, "file2"), []byte("fileNN\n"), 0777); err != nil {
		t.Fatal(err)
	}

	// Replace a file
	if err := os.RemoveAll(path.Join(root, "file3")); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(root, "file3"), []byte("fileMM\n"), 0404); err != nil {
		t.Fatal(err)
	}

	// Touch file
	if err := os.Chtimes(path.Join(root, "file4"), time.Now().Add(time.Second), time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}

	// Replace file with dir
	if err := os.RemoveAll(path.Join(root, "file5")); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(path.Join(root, "file5"), 0666); err != nil {
		t.Fatal(err)
	}

	// Create new file
	if err := ioutil.WriteFile(path.Join(root, "filenew"), []byte("filenew\n"), 0777); err != nil {
		t.Fatal(err)
	}

	// Create new dir
	if err := os.MkdirAll(path.Join(root, "dirnew"), 0766); err != nil {
		t.Fatal(err)
	}

	// Create a new symlink
	if err := os.Symlink("targetnew", path.Join(root, "symlinknew")); err != nil {
		t.Fatal(err)
	}

	// Change a symlink
	if err := os.RemoveAll(path.Join(root, "symlink2")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("target2change", path.Join(root, "symlink2")); err != nil {
		t.Fatal(err)
	}

	// Replace dir with file
	if err := os.RemoveAll(path.Join(root, "dir2")); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(path.Join(root, "dir2"), []byte("dir2\n"), 0777); err != nil {
		t.Fatal(err)
	}

	// Touch dir
	if err := os.Chtimes(path.Join(root, "dir3"), time.Now().Add(time.Second), time.Now().Add(time.Second)); err != nil {
		t.Fatal(err)
	}
}

func TestChangesDirsMutated(t *testing.T) {
	src, err := ioutil.TempDir("", "docker-changes-test")
	if err != nil {
		t.Fatal(err)
	}
	createSampleDir(t, src)
	dst := src + "-copy"
	if err := copyDir(src, dst); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(src)
	defer os.RemoveAll(dst)

	mutateSampleDir(t, dst)

	changes, err := ChangesDirs(dst, src)
	if err != nil {
		t.Fatal(err)
	}

	sort.Sort(changesByPath(changes))

	expectedChanges := []Change{
		{"/dir1", ChangeDelete},
		{"/dir2", ChangeModify},
		{"/dir3", ChangeModify},
		{"/dirnew", ChangeAdd},
		{"/file1", ChangeDelete},
		{"/file2", ChangeModify},
		{"/file3", ChangeModify},
		{"/file4", ChangeModify},
		{"/file5", ChangeModify},
		{"/filenew", ChangeAdd},
		{"/symlink1", ChangeDelete},
		{"/symlink2", ChangeModify},
		{"/symlinknew", ChangeAdd},
	}

	for i := 0; i < max(len(changes), len(expectedChanges)); i++ {
		if i >= len(expectedChanges) {
			t.Fatalf("unexpected change %s\n", changes[i].String())
		}
		if i >= len(changes) {
			t.Fatalf("no change for expected change %s\n", expectedChanges[i].String())
		}
		if changes[i].Path == expectedChanges[i].Path {
			if changes[i] != expectedChanges[i] {
				t.Fatalf("Wrong change for %s, expected %s, got %s\n", changes[i].Path, changes[i].String(), expectedChanges[i].String())
			}
		} else if changes[i].Path < expectedChanges[i].Path {
			t.Fatalf("unexpected change %s\n", changes[i].String())
		} else {
			t.Fatalf("no change for expected change %s != %s\n", expectedChanges[i].String(), changes[i].String())
		}
	}
}

func TestApplyLayer(t *testing.T) {
	src, err := ioutil.TempDir("", "docker-changes-test")
	if err != nil {
		t.Fatal(err)
	}
	createSampleDir(t, src)
	defer os.RemoveAll(src)
	dst := src + "-copy"
	if err := copyDir(src, dst); err != nil {
		t.Fatal(err)
	}
	mutateSampleDir(t, dst)
	defer os.RemoveAll(dst)

	changes, err := ChangesDirs(dst, src)
	if err != nil {
		t.Fatal(err)
	}

	layer, err := ExportChanges(dst, changes)
	if err != nil {
		t.Fatal(err)
	}

	layerCopy, err := NewTempArchive(layer, "")
	if err != nil {
		t.Fatal(err)
	}

	if _, err := ApplyLayer(src, layerCopy); err != nil {
		t.Fatal(err)
	}

	changes2, err := ChangesDirs(src, dst)
	if err != nil {
		t.Fatal(err)
	}

	if len(changes2) != 0 {
		t.Fatalf("Unexpected differences after reapplying mutation: %v", changes2)
	}
}
