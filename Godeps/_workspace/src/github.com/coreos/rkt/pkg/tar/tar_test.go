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

package tar

import (
	"archive/tar"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

type testTarEntry struct {
	header   *tar.Header
	contents string
}

func newTestTar(entries []*testTarEntry) (string, error) {
	t, err := ioutil.TempFile("", "test-tar")
	if err != nil {
		return "", err
	}
	defer t.Close()
	tw := tar.NewWriter(t)
	for _, entry := range entries {
		// Add default mode
		if entry.header.Mode == 0 {
			if entry.header.Typeflag == tar.TypeDir {
				entry.header.Mode = 0755
			} else {
				entry.header.Mode = 0644
			}
		}
		// Add calling user uid and gid or tests will fail
		entry.header.Uid = os.Getuid()
		entry.header.Gid = os.Getgid()
		if err := tw.WriteHeader(entry.header); err != nil {
			return "", err
		}
		if _, err := io.WriteString(tw, entry.contents); err != nil {
			return "", err
		}
	}
	if err := tw.Close(); err != nil {
		return "", err
	}
	return t.Name(), nil
}

type fileInfo struct {
	path     string
	typeflag byte
	size     int64
	contents string
	mode     os.FileMode
}

func fileInfoSliceToMap(slice []*fileInfo) map[string]*fileInfo {
	fim := make(map[string]*fileInfo, len(slice))
	for _, fi := range slice {
		fim[fi.path] = fi
	}
	return fim
}

func checkExpectedFiles(dir string, expectedFiles map[string]*fileInfo) error {
	files := make(map[string]*fileInfo)
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		fm := info.Mode()
		if path == dir {
			return nil
		}
		relpath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		switch {
		case fm.IsRegular():
			files[relpath] = &fileInfo{path: relpath, typeflag: tar.TypeReg, size: info.Size(), mode: info.Mode().Perm()}
		case info.IsDir():
			files[relpath] = &fileInfo{path: relpath, typeflag: tar.TypeDir, mode: info.Mode().Perm()}
		case fm&os.ModeSymlink != 0:
			files[relpath] = &fileInfo{path: relpath, typeflag: tar.TypeSymlink, mode: info.Mode()}
		default:
			return fmt.Errorf("file mode not handled: %v", fm)
		}

		return nil
	})
	if err != nil {
		return err
	}

	// Set defaults for not specified expected file mode
	for _, ef := range expectedFiles {
		if ef.mode == 0 {
			if ef.typeflag == tar.TypeDir {
				ef.mode = 0755
			} else {
				ef.mode = 0644
			}
		}
	}

	for _, ef := range expectedFiles {
		_, ok := files[ef.path]
		if !ok {
			return fmt.Errorf("Expected file %q not in files", ef.path)
		}

	}

	for _, file := range files {
		ef, ok := expectedFiles[file.path]
		if !ok {
			return fmt.Errorf("file %q not in expectedFiles", file.path)
		}
		if ef.typeflag != file.typeflag {
			return fmt.Errorf("file %q: file type differs: wanted: %d, got: %d", file.path, ef.typeflag, file.typeflag)
		}
		if ef.typeflag == tar.TypeReg {
			if ef.size != file.size {
				return fmt.Errorf("file %q: size differs: wanted %d, wanted: %d", file.path, ef.size, file.size)
			}
			if ef.contents != "" {
				buf, err := ioutil.ReadFile(filepath.Join(dir, file.path))
				if err != nil {
					return fmt.Errorf("unexpected error: %v", err)
				}
				if string(buf) != ef.contents {
					return fmt.Errorf("unexpected contents, wanted: %s, got: %s", ef.contents, buf)
				}

			}
		}
		// Check modes but ignore symlinks
		if ef.mode != file.mode && ef.typeflag != tar.TypeSymlink {
			return fmt.Errorf("file %q: mode differs: wanted %#o, got: %#o", file.path, ef.mode, file.mode)
		}

	}
	return nil
}

func TestExtractTarInsecureSymlink(t *testing.T) {
	entries := []*testTarEntry{
		{
			contents: "hello",
			header: &tar.Header{
				Name: "hello.txt",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "link.txt",
				Linkname: "hello.txt",
				Typeflag: tar.TypeSymlink,
			},
		},
	}
	insecureSymlinkEntries := append(entries, &testTarEntry{
		header: &tar.Header{
			Name:     "../etc/secret.conf",
			Linkname: "secret.conf",
			Typeflag: tar.TypeSymlink,
		},
	})
	insecureHardlinkEntries := append(entries, &testTarEntry{
		header: &tar.Header{
			Name:     "../etc/secret.conf",
			Linkname: "secret.conf",
			Typeflag: tar.TypeLink,
		},
	})
	for _, entries := range [][]*testTarEntry{insecureSymlinkEntries, insecureHardlinkEntries} {
		testTarPath, err := newTestTar(entries)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		defer os.Remove(testTarPath)
		containerTar, err := os.Open(testTarPath)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		tr := tar.NewReader(containerTar)
		tmpdir, err := ioutil.TempDir("", "rkt-temp-dir")
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		os.RemoveAll(tmpdir)
		err = os.MkdirAll(tmpdir, 0755)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		defer os.RemoveAll(tmpdir)
		err = ExtractTar(tr, tmpdir, false, nil)
		if _, ok := err.(insecureLinkError); !ok {
			t.Errorf("expected insecureSymlinkError error")
		}
	}
}

func TestExtractTarFolders(t *testing.T) {
	entries := []*testTarEntry{
		{
			contents: "foo",
			header: &tar.Header{
				Name: "deep/folder/foo.txt",
				Size: 3,
			},
		},
		{
			header: &tar.Header{
				Name:     "deep/folder/",
				Typeflag: tar.TypeDir,
				Mode:     int64(0747),
			},
		},
		{
			contents: "bar",
			header: &tar.Header{
				Name: "deep/folder/bar.txt",
				Size: 3,
			},
		},
		{
			header: &tar.Header{
				Name:     "deep/folder2/symlink.txt",
				Typeflag: tar.TypeSymlink,
				Linkname: "deep/folder/foo.txt",
			},
		},
		{
			header: &tar.Header{
				Name:     "deep/folder2/",
				Typeflag: tar.TypeDir,
				Mode:     int64(0747),
			},
		},
		{
			contents: "bar",
			header: &tar.Header{
				Name: "deep/folder2/bar.txt",
				Size: 3,
			},
		},
		{
			header: &tar.Header{
				Name:     "deep/deep/folder",
				Typeflag: tar.TypeDir,
				Mode:     int64(0755),
			},
		},
		{
			header: &tar.Header{
				Name:     "deep/deep/",
				Typeflag: tar.TypeDir,
				Mode:     int64(0747),
			},
		},
	}

	testTarPath, err := newTestTar(entries)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.Remove(testTarPath)
	containerTar, err := os.Open(testTarPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	tr := tar.NewReader(containerTar)
	tmpdir, err := ioutil.TempDir("", "rkt-temp-dir")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	os.RemoveAll(tmpdir)
	err = os.MkdirAll(tmpdir, 0755)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.RemoveAll(tmpdir)
	err = ExtractTar(tr, tmpdir, false, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	matches, err := filepath.Glob(filepath.Join(tmpdir, "deep/folder/*.txt"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(matches) != 2 {
		t.Errorf("unexpected number of files found: %d, wanted 2", len(matches))
	}
	matches, err = filepath.Glob(filepath.Join(tmpdir, "deep/folder2/*.txt"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(matches) != 2 {
		t.Errorf("unexpected number of files found: %d, wanted 2", len(matches))
	}

	dirInfo, err := os.Lstat(filepath.Join(tmpdir, "deep/folder"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if dirInfo.Mode().Perm() != os.FileMode(0747) {
		t.Errorf("unexpected dir mode: %s", dirInfo.Mode())
	}
	dirInfo, err = os.Lstat(filepath.Join(tmpdir, "deep/deep"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if dirInfo.Mode().Perm() != os.FileMode(0747) {
		t.Errorf("unexpected dir mode: %s", dirInfo.Mode())
	}
}

func TestExtractTarFileToBuf(t *testing.T) {
	entries := []*testTarEntry{
		{
			header: &tar.Header{
				Name:     "folder/",
				Typeflag: tar.TypeDir,
				Mode:     int64(0747),
			},
		},
		{
			contents: "foo",
			header: &tar.Header{
				Name: "folder/foo.txt",
				Size: 3,
			},
		},
		{
			contents: "bar",
			header: &tar.Header{
				Name: "folder/bar.txt",
				Size: 3,
			},
		},
		{
			header: &tar.Header{
				Name:     "folder/symlink.txt",
				Typeflag: tar.TypeSymlink,
				Linkname: "folder/foo.txt",
			},
		},
	}
	testTarPath, err := newTestTar(entries)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(testTarPath)
	containerTar, err := os.Open(testTarPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	tr := tar.NewReader(containerTar)
	buf, err := ExtractFileFromTar(tr, "folder/foo.txt")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(buf) != "foo" {
		t.Errorf("unexpected contents, wanted: %s, got: %s", "foo", buf)
	}
	containerTar.Close()

	containerTar, err = os.Open(testTarPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	tr = tar.NewReader(containerTar)
	buf, err = ExtractFileFromTar(tr, "folder/symlink.txt")
	if err == nil {
		t.Errorf("expected error")
	}
	containerTar.Close()
}

func TestExtractTarPWL(t *testing.T) {
	entries := []*testTarEntry{
		{
			header: &tar.Header{
				Name:     "folder/",
				Typeflag: tar.TypeDir,
				Mode:     int64(0747),
			},
		},
		{
			contents: "foo",
			header: &tar.Header{
				Name: "folder/foo.txt",
				Size: 3,
			},
		},
		{
			contents: "bar",
			header: &tar.Header{
				Name: "folder/bar.txt",
				Size: 3,
			},
		},
		{
			header: &tar.Header{
				Name:     "folder/symlink.txt",
				Typeflag: tar.TypeSymlink,
				Linkname: "folder/foo.txt",
			},
		},
	}
	testTarPath, err := newTestTar(entries)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.Remove(testTarPath)
	containerTar, err := os.Open(testTarPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	tr := tar.NewReader(containerTar)
	tmpdir, err := ioutil.TempDir("", "rkt-temp-dir")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	pwl := make(PathWhitelistMap)
	pwl["folder/foo.txt"] = struct{}{}
	err = ExtractTar(tr, tmpdir, false, pwl)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	matches, err := filepath.Glob(filepath.Join(tmpdir, "folder/*.txt"))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(matches) != 1 {
		t.Errorf("unexpected number of files found: %d, wanted 1", len(matches))
	}
}

func TestExtractTarOverwrite(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "rkt-temp-dir")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	entries := []*testTarEntry{
		{
			contents: "hello",
			header: &tar.Header{
				Name: "hello.txt",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "afolder",
				Typeflag: tar.TypeDir,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "afolder/hello.txt",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "afile",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "folder01",
				Typeflag: tar.TypeDir,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "folder01/file01",
				Size: 5,
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "filesymlinked",
				Size: 5,
			},
		},
		{
			header: &tar.Header{
				Name:     "linktofile",
				Linkname: "filesymlinked",
				Typeflag: tar.TypeSymlink,
			},
		},

		{
			header: &tar.Header{
				Name:     "dirsymlinked",
				Typeflag: tar.TypeDir,
			},
		},
		{
			header: &tar.Header{
				Name:     "linktodir",
				Linkname: "dirsymlinked",
				Typeflag: tar.TypeSymlink,
			},
		},
	}

	testTarPath, err := newTestTar(entries)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.Remove(testTarPath)
	containerTar, err := os.Open(testTarPath)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	tr := tar.NewReader(containerTar)
	err = ExtractTar(tr, tmpdir, false, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Now overwrite:
	// a file with a new file
	// a dir with a file
	entries = []*testTarEntry{
		{
			contents: "newhello",
			header: &tar.Header{
				Name: "hello.txt",
				Size: 8,
			},
		},
		// Now this is a file
		{
			contents: "nowafile",
			header: &tar.Header{
				Name:     "afolder",
				Typeflag: tar.TypeReg,
				Size:     8,
			},
		},
		// Now this is a dir
		{
			header: &tar.Header{
				Name:     "afile",
				Typeflag: tar.TypeDir,
			},
		},
		// Overwrite symlink to a file with a regular file
		// the linked file shouldn't be removed
		{
			contents: "filereplacingsymlink",
			header: &tar.Header{
				Name:     "linktofile",
				Typeflag: tar.TypeReg,
				Size:     20,
			},
		},
		// Overwrite symlink to a dir with a regular file
		// the linked directory and all its contents shouldn't be
		// removed
		{
			contents: "filereplacingsymlink",
			header: &tar.Header{
				Name:     "linktodir",
				Typeflag: tar.TypeReg,
				Size:     20,
			},
		},
		// folder01 already exists and shouldn't be removed (keeping folder01/file01)
		{
			header: &tar.Header{
				Name:     "folder01",
				Typeflag: tar.TypeDir,
				Mode:     int64(0755),
			},
		},
		{
			contents: "hello",
			header: &tar.Header{
				Name: "folder01/file02",
				Size: 5,
				Mode: int64(0644),
			},
		},
	}
	testTarPath, err = newTestTar(entries)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.Remove(testTarPath)
	containerTar, err = os.Open(testTarPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	tr = tar.NewReader(containerTar)
	err = ExtractTar(tr, tmpdir, true, nil)

	expectedFiles := []*fileInfo{
		&fileInfo{path: "hello.txt", typeflag: tar.TypeReg, size: 8, contents: "newhello"},
		&fileInfo{path: "linktofile", typeflag: tar.TypeReg, size: 20},
		&fileInfo{path: "linktodir", typeflag: tar.TypeReg, size: 20},
		&fileInfo{path: "afolder", typeflag: tar.TypeReg, size: 8},
		&fileInfo{path: "dirsymlinked", typeflag: tar.TypeDir},
		&fileInfo{path: "afile", typeflag: tar.TypeDir},
		&fileInfo{path: "filesymlinked", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "folder01", typeflag: tar.TypeDir},
		&fileInfo{path: "folder01/file01", typeflag: tar.TypeReg, size: 5},
		&fileInfo{path: "folder01/file02", typeflag: tar.TypeReg, size: 5},
	}

	err = checkExpectedFiles(tmpdir, fileInfoSliceToMap(expectedFiles))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestExtractTarTimes(t *testing.T) {

	// Do not set ns as tar has second precision
	time1 := time.Unix(100000, 0)
	time2 := time.Unix(200000, 0)
	time3 := time.Unix(300000, 0)
	entries := []*testTarEntry{
		{
			header: &tar.Header{
				Name:     "folder/",
				Typeflag: tar.TypeDir,
				Mode:     int64(0747),
				ModTime:  time1,
			},
		},
		{
			contents: "foo",
			header: &tar.Header{
				Name:    "folder/foo.txt",
				Size:    3,
				ModTime: time2,
			},
		},
		{
			header: &tar.Header{
				Name:     "folder/symlink.txt",
				Typeflag: tar.TypeSymlink,
				Linkname: "folder/foo.txt",
				ModTime:  time3,
			},
		},
	}

	testTarPath, err := newTestTar(entries)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.Remove(testTarPath)
	containerTar, err := os.Open(testTarPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	tr := tar.NewReader(containerTar)
	tmpdir, err := ioutil.TempDir("", "rkt-temp-dir")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	os.RemoveAll(tmpdir)
	err = os.MkdirAll(tmpdir, 0755)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	err = ExtractTar(tr, tmpdir, false, nil)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	err = checkTime(filepath.Join(tmpdir, "folder/"), time1)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	err = checkTime(filepath.Join(tmpdir, "folder/foo.txt"), time2)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	//Check only (by now) on linux
	if runtime.GOOS == "linux" {
		err = checkTime(filepath.Join(tmpdir, "folder/symlink.txt"), time3)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	}
}

func checkTime(path string, time time.Time) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.ModTime() != time {
		return fmt.Errorf("%s: info.ModTime: %s, different from expected time: %s", path, info.ModTime(), time)
	}
	return nil
}
