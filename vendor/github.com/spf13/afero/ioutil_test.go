// ©2015 The Go Authors
// Copyright ©2015 Steve Francia <spf@spf13.com>
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

package afero

import "testing"

func checkSizePath(t *testing.T, path string, size int64) {
	dir, err := testFS.Stat(path)
	if err != nil {
		t.Fatalf("Stat %q (looking for size %d): %s", path, size, err)
	}
	if dir.Size() != size {
		t.Errorf("Stat %q: size %d want %d", path, dir.Size(), size)
	}
}

func TestReadFile(t *testing.T) {
	testFS = &MemMapFs{}
	fsutil := &Afero{Fs: testFS}

	testFS.Create("this_exists.go")
	filename := "rumpelstilzchen"
	contents, err := fsutil.ReadFile(filename)
	if err == nil {
		t.Fatalf("ReadFile %s: error expected, none found", filename)
	}

	filename = "this_exists.go"
	contents, err = fsutil.ReadFile(filename)
	if err != nil {
		t.Fatalf("ReadFile %s: %v", filename, err)
	}

	checkSizePath(t, filename, int64(len(contents)))
}

func TestWriteFile(t *testing.T) {
	testFS = &MemMapFs{}
	fsutil := &Afero{Fs: testFS}
	f, err := fsutil.TempFile("", "ioutil-test")
	if err != nil {
		t.Fatal(err)
	}
	filename := f.Name()
	data := "Programming today is a race between software engineers striving to " +
		"build bigger and better idiot-proof programs, and the Universe trying " +
		"to produce bigger and better idiots. So far, the Universe is winning."

	if err := fsutil.WriteFile(filename, []byte(data), 0644); err != nil {
		t.Fatalf("WriteFile %s: %v", filename, err)
	}

	contents, err := fsutil.ReadFile(filename)
	if err != nil {
		t.Fatalf("ReadFile %s: %v", filename, err)
	}

	if string(contents) != data {
		t.Fatalf("contents = %q\nexpected = %q", string(contents), data)
	}

	// cleanup
	f.Close()
	testFS.Remove(filename) // ignore error
}

func TestReadDir(t *testing.T) {
	testFS = &MemMapFs{}
	testFS.Mkdir("/i-am-a-dir", 0777)
	testFS.Create("/this_exists.go")
	dirname := "rumpelstilzchen"
	_, err := ReadDir(testFS, dirname)
	if err == nil {
		t.Fatalf("ReadDir %s: error expected, none found", dirname)
	}

	dirname = ".."
	list, err := ReadDir(testFS, dirname)
	if err != nil {
		t.Fatalf("ReadDir %s: %v", dirname, err)
	}

	foundFile := false
	foundSubDir := false
	for _, dir := range list {
		switch {
		case !dir.IsDir() && dir.Name() == "this_exists.go":
			foundFile = true
		case dir.IsDir() && dir.Name() == "i-am-a-dir":
			foundSubDir = true
		}
	}
	if !foundFile {
		t.Fatalf("ReadDir %s: this_exists.go file not found", dirname)
	}
	if !foundSubDir {
		t.Fatalf("ReadDir %s: i-am-a-dir directory not found", dirname)
	}
}
