// Copyright © 2014 Steve Francia <spf@spf13.com>.
// Copyright 2009 The Go Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package afero

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"testing"
)

var testName = "test.txt"
var Fss = []Fs{&MemMapFs{}, &OsFs{}}

var testRegistry map[Fs][]string = make(map[Fs][]string)

func testDir(fs Fs) string {
	name, err := TempDir(fs, "", "afero")
	if err != nil {
		panic(fmt.Sprint("unable to work with test dir", err))
	}
	testRegistry[fs] = append(testRegistry[fs], name)

	return name
}

func tmpFile(fs Fs) File {
	x, err := TempFile(fs, "", "afero")

	if err != nil {
		panic(fmt.Sprint("unable to work with temp file", err))
	}

	testRegistry[fs] = append(testRegistry[fs], x.Name())

	return x
}

//Read with length 0 should not return EOF.
func TestRead0(t *testing.T) {
	for _, fs := range Fss {
		f := tmpFile(fs)
		defer f.Close()
		f.WriteString("Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

		var b []byte
		// b := make([]byte, 0)
		n, err := f.Read(b)
		if n != 0 || err != nil {
			t.Errorf("%v: Read(0) = %d, %v, want 0, nil", fs.Name(), n, err)
		}
		f.Seek(0, 0)
		b = make([]byte, 100)
		n, err = f.Read(b)
		if n <= 0 || err != nil {
			t.Errorf("%v: Read(100) = %d, %v, want >0, nil", fs.Name(), n, err)
		}
	}
}

func TestOpenFile(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		tmp := testDir(fs)
		path := filepath.Join(tmp, testName)

		f, err := fs.OpenFile(path, os.O_RDWR|os.O_CREATE, 0600)
		if err != nil {
			t.Error(fs.Name(), "OpenFile (O_CREATE) failed:", err)
			continue
		}
		io.WriteString(f, "initial")
		f.Close()

		f, err = fs.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0600)
		if err != nil {
			t.Error(fs.Name(), "OpenFile (O_APPEND) failed:", err)
			continue
		}
		io.WriteString(f, "|append")
		f.Close()

		f, err = fs.OpenFile(path, os.O_RDONLY, 0600)
		contents, _ := ioutil.ReadAll(f)
		expectedContents := "initial|append"
		if string(contents) != expectedContents {
			t.Errorf("%v: appending, expected '%v', got: '%v'", fs.Name(), expectedContents, string(contents))
		}
		f.Close()

		f, err = fs.OpenFile(path, os.O_RDWR|os.O_TRUNC, 0600)
		if err != nil {
			t.Error(fs.Name(), "OpenFile (O_TRUNC) failed:", err)
			continue
		}
		contents, _ = ioutil.ReadAll(f)
		if string(contents) != "" {
			t.Errorf("%v: expected truncated file, got: '%v'", fs.Name(), string(contents))
		}
		f.Close()
	}
}

func TestCreate(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		tmp := testDir(fs)
		path := filepath.Join(tmp, testName)

		f, err := fs.Create(path)
		if err != nil {
			t.Error(fs.Name(), "Create failed:", err)
			f.Close()
			continue
		}
		io.WriteString(f, "initial")
		f.Close()

		f, err = fs.Create(path)
		if err != nil {
			t.Error(fs.Name(), "Create failed:", err)
			f.Close()
			continue
		}
		secondContent := "second create"
		io.WriteString(f, secondContent)
		f.Close()

		f, err = fs.Open(path)
		if err != nil {
			t.Error(fs.Name(), "Open failed:", err)
			f.Close()
			continue
		}
		buf, err := ReadAll(f)
		if err != nil {
			t.Error(fs.Name(), "ReadAll failed:", err)
			f.Close()
			continue
		}
		if string(buf) != secondContent {
			t.Error(fs.Name(), "Content should be", "\""+secondContent+"\" but is \""+string(buf)+"\"")
			f.Close()
			continue
		}
		f.Close()
	}
}

func TestMemFileRead(t *testing.T) {
	f := tmpFile(new(MemMapFs))
	// f := MemFileCreate("testfile")
	f.WriteString("abcd")
	f.Seek(0, 0)
	b := make([]byte, 8)
	n, err := f.Read(b)
	if n != 4 {
		t.Errorf("didn't read all bytes: %v %v %v", n, err, b)
	}
	if err != nil {
		t.Errorf("err is not nil: %v %v %v", n, err, b)
	}
	n, err = f.Read(b)
	if n != 0 {
		t.Errorf("read more bytes: %v %v %v", n, err, b)
	}
	if err != io.EOF {
		t.Errorf("error is not EOF: %v %v %v", n, err, b)
	}
}

func TestRename(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		tDir := testDir(fs)
		from := filepath.Join(tDir, "/renamefrom")
		to := filepath.Join(tDir, "/renameto")
		exists := filepath.Join(tDir, "/renameexists")
		file, err := fs.Create(from)
		if err != nil {
			t.Fatalf("%s: open %q failed: %v", fs.Name(), to, err)
		}
		if err = file.Close(); err != nil {
			t.Errorf("%s: close %q failed: %v", fs.Name(), to, err)
		}
		file, err = fs.Create(exists)
		if err != nil {
			t.Fatalf("%s: open %q failed: %v", fs.Name(), to, err)
		}
		if err = file.Close(); err != nil {
			t.Errorf("%s: close %q failed: %v", fs.Name(), to, err)
		}
		err = fs.Rename(from, to)
		if err != nil {
			t.Fatalf("%s: rename %q, %q failed: %v", fs.Name(), to, from, err)
		}
		file, err = fs.Create(from)
		if err != nil {
			t.Fatalf("%s: open %q failed: %v", fs.Name(), to, err)
		}
		if err = file.Close(); err != nil {
			t.Errorf("%s: close %q failed: %v", fs.Name(), to, err)
		}
		err = fs.Rename(from, exists)
		if err != nil {
			t.Errorf("%s: rename %q, %q failed: %v", fs.Name(), exists, from, err)
		}
		names, err := readDirNames(fs, tDir)
		if err != nil {
			t.Errorf("%s: readDirNames error: %v", fs.Name(), err)
		}
		found := false
		for _, e := range names {
			if e == "renamefrom" {
				t.Error("File is still called renamefrom")
			}
			if e == "renameto" {
				found = true
			}
		}
		if !found {
			t.Error("File was not renamed to renameto")
		}

		_, err = fs.Stat(to)
		if err != nil {
			t.Errorf("%s: stat %q failed: %v", fs.Name(), to, err)
		}
	}
}

func TestRemove(t *testing.T) {
	for _, fs := range Fss {

		x, err := TempFile(fs, "", "afero")
		if err != nil {
			t.Error(fmt.Sprint("unable to work with temp file", err))
		}

		path := x.Name()
		x.Close()

		tDir := filepath.Dir(path)

		err = fs.Remove(path)
		if err != nil {
			t.Errorf("%v: Remove() failed: %v", fs.Name(), err)
			continue
		}

		_, err = fs.Stat(path)
		if !os.IsNotExist(err) {
			t.Errorf("%v: Remove() didn't remove file", fs.Name())
			continue
		}

		// Deleting non-existent file should raise error
		err = fs.Remove(path)
		if !os.IsNotExist(err) {
			t.Errorf("%v: Remove() didn't raise error for non-existent file", fs.Name())
		}

		f, err := fs.Open(tDir)
		if err != nil {
			t.Error("TestDir should still exist:", err)
		}

		names, err := f.Readdirnames(-1)
		if err != nil {
			t.Error("Readdirnames failed:", err)
		}

		for _, e := range names {
			if e == testName {
				t.Error("File was not removed from parent directory")
			}
		}
	}
}

func TestTruncate(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		// f := newFile("TestTruncate", fs, t)
		f := tmpFile(fs)
		defer f.Close()

		checkSize(t, f, 0)
		f.Write([]byte("hello, world\n"))
		checkSize(t, f, 13)
		f.Truncate(10)
		checkSize(t, f, 10)
		f.Truncate(1024)
		checkSize(t, f, 1024)
		f.Truncate(0)
		checkSize(t, f, 0)
		_, err := f.Write([]byte("surprise!"))
		if err == nil {
			checkSize(t, f, 13+9) // wrote at offset past where hello, world was.
		}
	}
}

func TestSeek(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		f := tmpFile(fs)
		defer f.Close()

		const data = "hello, world\n"
		io.WriteString(f, data)

		type test struct {
			in     int64
			whence int
			out    int64
		}
		var tests = []test{
			{0, 1, int64(len(data))},
			{0, 0, 0},
			{5, 0, 5},
			{0, 2, int64(len(data))},
			{0, 0, 0},
			{-1, 2, int64(len(data)) - 1},
			{1 << 33, 0, 1 << 33},
			{1 << 33, 2, 1<<33 + int64(len(data))},
		}
		for i, tt := range tests {
			off, err := f.Seek(tt.in, tt.whence)
			if off != tt.out || err != nil {
				if e, ok := err.(*os.PathError); ok && e.Err == syscall.EINVAL && tt.out > 1<<32 {
					// Reiserfs rejects the big seeks.
					// http://code.google.com/p/go/issues/detail?id=91
					break
				}
				t.Errorf("#%d: Seek(%v, %v) = %v, %v want %v, nil", i, tt.in, tt.whence, off, err, tt.out)
			}
		}
	}
}

func TestReadAt(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		f := tmpFile(fs)
		defer f.Close()

		const data = "hello, world\n"
		io.WriteString(f, data)

		b := make([]byte, 5)
		n, err := f.ReadAt(b, 7)
		if err != nil || n != len(b) {
			t.Fatalf("ReadAt 7: %d, %v", n, err)
		}
		if string(b) != "world" {
			t.Fatalf("ReadAt 7: have %q want %q", string(b), "world")
		}
	}
}

func TestWriteAt(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		// f := newFile("TestWriteAt", fs, t)
		f := tmpFile(fs)
		defer f.Close()

		const data = "hello, world\n"
		io.WriteString(f, data)

		n, err := f.WriteAt([]byte("WORLD"), 7)
		if err != nil || n != 5 {
			t.Fatalf("WriteAt 7: %d, %v", n, err)
		}

		f2, err := fs.Open(f.Name())
		defer f2.Close()
		buf := new(bytes.Buffer)
		buf.ReadFrom(f2)
		b := buf.Bytes()
		if err != nil {
			t.Fatalf("%v: ReadFile %s: %v", fs.Name(), f.Name(), err)
		}
		if string(b) != "hello, WORLD\n" {
			t.Fatalf("after write: have %q want %q", string(b), "hello, WORLD\n")
		}

	}
}

func setupTestDir(t *testing.T, fs Fs) string {
	path := testDir(fs)
	return setupTestFiles(t, fs, path)
}

func setupTestDirRoot(t *testing.T, fs Fs) string {
	path := testDir(fs)
	setupTestFiles(t, fs, path)
	return path
}

func setupTestDirReusePath(t *testing.T, fs Fs, path string) string {
	testRegistry[fs] = append(testRegistry[fs], path)
	return setupTestFiles(t, fs, path)
}

func setupTestFiles(t *testing.T, fs Fs, path string) string {
	testSubDir := filepath.Join(path, "more", "subdirectories", "for", "testing", "we")
	err := fs.MkdirAll(testSubDir, 0700)
	if err != nil && !os.IsExist(err) {
		t.Fatal(err)
	}

	f, err := fs.Create(filepath.Join(testSubDir, "testfile1"))
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("Testfile 1 content")
	f.Close()

	f, err = fs.Create(filepath.Join(testSubDir, "testfile2"))
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("Testfile 2 content")
	f.Close()

	f, err = fs.Create(filepath.Join(testSubDir, "testfile3"))
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("Testfile 3 content")
	f.Close()

	f, err = fs.Create(filepath.Join(testSubDir, "testfile4"))
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString("Testfile 4 content")
	f.Close()
	return testSubDir
}

func TestReaddirnames(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		testSubDir := setupTestDir(t, fs)
		tDir := filepath.Dir(testSubDir)

		root, err := fs.Open(tDir)
		if err != nil {
			t.Fatal(fs.Name(), tDir, err)
		}
		defer root.Close()

		namesRoot, err := root.Readdirnames(-1)
		if err != nil {
			t.Fatal(fs.Name(), namesRoot, err)
		}

		sub, err := fs.Open(testSubDir)
		if err != nil {
			t.Fatal(err)
		}
		defer sub.Close()

		namesSub, err := sub.Readdirnames(-1)
		if err != nil {
			t.Fatal(fs.Name(), namesSub, err)
		}

		findNames(fs, t, tDir, testSubDir, namesRoot, namesSub)
	}
}

func TestReaddirSimple(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		testSubDir := setupTestDir(t, fs)
		tDir := filepath.Dir(testSubDir)

		root, err := fs.Open(tDir)
		if err != nil {
			t.Fatal(err)
		}
		defer root.Close()

		rootInfo, err := root.Readdir(1)
		if err != nil {
			t.Log(myFileInfo(rootInfo))
			t.Error(err)
		}

		rootInfo, err = root.Readdir(5)
		if err != io.EOF {
			t.Log(myFileInfo(rootInfo))
			t.Error(err)
		}

		sub, err := fs.Open(testSubDir)
		if err != nil {
			t.Fatal(err)
		}
		defer sub.Close()

		subInfo, err := sub.Readdir(5)
		if err != nil {
			t.Log(myFileInfo(subInfo))
			t.Error(err)
		}
	}
}

func TestReaddir(t *testing.T) {
	defer removeAllTestFiles(t)
	for num := 0; num < 6; num++ {
		outputs := make([]string, len(Fss))
		infos := make([]string, len(Fss))
		for i, fs := range Fss {
			testSubDir := setupTestDir(t, fs)
			//tDir := filepath.Dir(testSubDir)
			root, err := fs.Open(testSubDir)
			if err != nil {
				t.Fatal(err)
			}
			defer root.Close()

			for j := 0; j < 6; j++ {
				info, err := root.Readdir(num)
				outputs[i] += fmt.Sprintf("%v  Error: %v\n", myFileInfo(info), err)
				infos[i] += fmt.Sprintln(len(info), err)
			}
		}

		fail := false
		for i, o := range infos {
			if i == 0 {
				continue
			}
			if o != infos[i-1] {
				fail = true
				break
			}
		}
		if fail {
			t.Log("Readdir outputs not equal for Readdir(", num, ")")
			for i, o := range outputs {
				t.Log(Fss[i].Name())
				t.Log(o)
			}
			t.Fail()
		}
	}
}

type myFileInfo []os.FileInfo

func (m myFileInfo) String() string {
	out := "Fileinfos:\n"
	for _, e := range m {
		out += "  " + e.Name() + "\n"
	}
	return out
}

func TestReaddirAll(t *testing.T) {
	defer removeAllTestFiles(t)
	for _, fs := range Fss {
		testSubDir := setupTestDir(t, fs)
		tDir := filepath.Dir(testSubDir)

		root, err := fs.Open(tDir)
		if err != nil {
			t.Fatal(err)
		}
		defer root.Close()

		rootInfo, err := root.Readdir(-1)
		if err != nil {
			t.Fatal(err)
		}
		var namesRoot = []string{}
		for _, e := range rootInfo {
			namesRoot = append(namesRoot, e.Name())
		}

		sub, err := fs.Open(testSubDir)
		if err != nil {
			t.Fatal(err)
		}
		defer sub.Close()

		subInfo, err := sub.Readdir(-1)
		if err != nil {
			t.Fatal(err)
		}
		var namesSub = []string{}
		for _, e := range subInfo {
			namesSub = append(namesSub, e.Name())
		}

		findNames(fs, t, tDir, testSubDir, namesRoot, namesSub)
	}
}

func findNames(fs Fs, t *testing.T, tDir, testSubDir string, root, sub []string) {
	var foundRoot bool
	for _, e := range root {
		f, err := fs.Open(filepath.Join(tDir, e))
		if err != nil {
			t.Error("Open", filepath.Join(tDir, e), ":", err)
		}
		defer f.Close()

		if equal(e, "we") {
			foundRoot = true
		}
	}
	if !foundRoot {
		t.Logf("Names root: %v", root)
		t.Logf("Names sub: %v", sub)
		t.Error("Didn't find subdirectory we")
	}

	var found1, found2 bool
	for _, e := range sub {
		f, err := fs.Open(filepath.Join(testSubDir, e))
		if err != nil {
			t.Error("Open", filepath.Join(testSubDir, e), ":", err)
		}
		defer f.Close()

		if equal(e, "testfile1") {
			found1 = true
		}
		if equal(e, "testfile2") {
			found2 = true
		}
	}

	if !found1 {
		t.Logf("Names root: %v", root)
		t.Logf("Names sub: %v", sub)
		t.Error("Didn't find testfile1")
	}
	if !found2 {
		t.Logf("Names root: %v", root)
		t.Logf("Names sub: %v", sub)
		t.Error("Didn't find testfile2")
	}
}

func removeAllTestFiles(t *testing.T) {
	for fs, list := range testRegistry {
		for _, path := range list {
			if err := fs.RemoveAll(path); err != nil {
				t.Error(fs.Name(), err)
			}
		}
	}
	testRegistry = make(map[Fs][]string)
}

func newFile(testName string, fs Fs, t *testing.T) (f File) {
	// Use a local file system, not NFS.
	// On Unix, override $TMPDIR in case the user
	// has it set to an NFS-mounted directory.
	dir := ""
	if runtime.GOOS != "windows" {
		dir = "/tmp"
	}
	fs.MkdirAll(dir, 0777)
	f, err := fs.Create(path.Join(dir, testName))
	if err != nil {
		t.Fatalf("%v: open %s: %s", fs.Name(), testName, err)
	}
	return f
}

func writeFile(t *testing.T, fs Fs, fname string, flag int, text string) string {
	f, err := fs.OpenFile(fname, flag, 0666)
	if err != nil {
		t.Fatalf("Unable to Open file %q for writing: %v", fname, err)
	}
	n, err := io.WriteString(f, text)
	if err != nil {
		t.Fatalf("WriteString: %d, %v", n, err)
	}
	f.Close()
	data, err := ioutil.ReadFile(fname)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	return string(data)
}

func equal(name1, name2 string) (r bool) {
	switch runtime.GOOS {
	case "windows":
		r = strings.ToLower(name1) == strings.ToLower(name2)
	default:
		r = name1 == name2
	}
	return
}

func checkSize(t *testing.T, f File, size int64) {
	dir, err := f.Stat()
	if err != nil {
		t.Fatalf("Stat %q (looking for size %d): %s", f.Name(), size, err)
	}
	if dir.Size() != size {
		t.Errorf("Stat %q: size %d want %d", f.Name(), dir.Size(), size)
	}
}
