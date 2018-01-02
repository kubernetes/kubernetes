// Copyright ©2015 Steve Francia <spf@spf13.com>
// Portions Copyright ©2015 The Hugo Authors
//
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
//

package afero

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

var testFS = new(MemMapFs)

func TestDirExists(t *testing.T) {
	type test struct {
		input    string
		expected bool
	}

	// First create a couple directories so there is something in the filesystem
	//testFS := new(MemMapFs)
	testFS.MkdirAll("/foo/bar", 0777)

	data := []test{
		{".", true},
		{"./", true},
		{"..", true},
		{"../", true},
		{"./..", true},
		{"./../", true},
		{"/foo/", true},
		{"/foo", true},
		{"/foo/bar", true},
		{"/foo/bar/", true},
		{"/", true},
		{"/some-really-random-directory-name", false},
		{"/some/really/random/directory/name", false},
		{"./some-really-random-local-directory-name", false},
		{"./some/really/random/local/directory/name", false},
	}

	for i, d := range data {
		exists, _ := DirExists(testFS, filepath.FromSlash(d.input))
		if d.expected != exists {
			t.Errorf("Test %d %q failed. Expected %t got %t", i, d.input, d.expected, exists)
		}
	}
}

func TestIsDir(t *testing.T) {
	testFS = new(MemMapFs)

	type test struct {
		input    string
		expected bool
	}
	data := []test{
		{"./", true},
		{"/", true},
		{"./this-directory-does-not-existi", false},
		{"/this-absolute-directory/does-not-exist", false},
	}

	for i, d := range data {

		exists, _ := IsDir(testFS, d.input)
		if d.expected != exists {
			t.Errorf("Test %d failed. Expected %t got %t", i, d.expected, exists)
		}
	}
}

func TestIsEmpty(t *testing.T) {
	testFS = new(MemMapFs)

	zeroSizedFile, _ := createZeroSizedFileInTempDir()
	defer deleteFileInTempDir(zeroSizedFile)
	nonZeroSizedFile, _ := createNonZeroSizedFileInTempDir()
	defer deleteFileInTempDir(nonZeroSizedFile)
	emptyDirectory, _ := createEmptyTempDir()
	defer deleteTempDir(emptyDirectory)
	nonEmptyZeroLengthFilesDirectory, _ := createTempDirWithZeroLengthFiles()
	defer deleteTempDir(nonEmptyZeroLengthFilesDirectory)
	nonEmptyNonZeroLengthFilesDirectory, _ := createTempDirWithNonZeroLengthFiles()
	defer deleteTempDir(nonEmptyNonZeroLengthFilesDirectory)
	nonExistentFile := os.TempDir() + "/this-file-does-not-exist.txt"
	nonExistentDir := os.TempDir() + "/this/direcotry/does/not/exist/"

	fileDoesNotExist := fmt.Errorf("%q path does not exist", nonExistentFile)
	dirDoesNotExist := fmt.Errorf("%q path does not exist", nonExistentDir)

	type test struct {
		input          string
		expectedResult bool
		expectedErr    error
	}

	data := []test{
		{zeroSizedFile.Name(), true, nil},
		{nonZeroSizedFile.Name(), false, nil},
		{emptyDirectory, true, nil},
		{nonEmptyZeroLengthFilesDirectory, false, nil},
		{nonEmptyNonZeroLengthFilesDirectory, false, nil},
		{nonExistentFile, false, fileDoesNotExist},
		{nonExistentDir, false, dirDoesNotExist},
	}
	for i, d := range data {
		exists, err := IsEmpty(testFS, d.input)
		if d.expectedResult != exists {
			t.Errorf("Test %d %q failed exists. Expected result %t got %t", i, d.input, d.expectedResult, exists)
		}
		if d.expectedErr != nil {
			if d.expectedErr.Error() != err.Error() {
				t.Errorf("Test %d failed with err. Expected %q(%#v) got %q(%#v)", i, d.expectedErr, d.expectedErr, err, err)
			}
		} else {
			if d.expectedErr != err {
				t.Errorf("Test %d failed. Expected error %q(%#v) got %q(%#v)", i, d.expectedErr, d.expectedErr, err, err)
			}
		}
	}
}

func TestReaderContains(t *testing.T) {
	for i, this := range []struct {
		v1     string
		v2     [][]byte
		expect bool
	}{
		{"abc", [][]byte{[]byte("a")}, true},
		{"abc", [][]byte{[]byte("b")}, true},
		{"abcdefg", [][]byte{[]byte("efg")}, true},
		{"abc", [][]byte{[]byte("d")}, false},
		{"abc", [][]byte{[]byte("d"), []byte("e")}, false},
		{"abc", [][]byte{[]byte("d"), []byte("a")}, true},
		{"abc", [][]byte{[]byte("b"), []byte("e")}, true},
		{"", nil, false},
		{"", [][]byte{[]byte("a")}, false},
		{"a", [][]byte{[]byte("")}, false},
		{"", [][]byte{[]byte("")}, false}} {
		result := readerContainsAny(strings.NewReader(this.v1), this.v2...)
		if result != this.expect {
			t.Errorf("[%d] readerContains: got %t but expected %t", i, result, this.expect)
		}
	}

	if readerContainsAny(nil, []byte("a")) {
		t.Error("readerContains with nil reader")
	}

	if readerContainsAny(nil, nil) {
		t.Error("readerContains with nil arguments")
	}
}

func createZeroSizedFileInTempDir() (File, error) {
	filePrefix := "_path_test_"
	f, e := TempFile(testFS, "", filePrefix) // dir is os.TempDir()
	if e != nil {
		// if there was an error no file was created.
		// => no requirement to delete the file
		return nil, e
	}
	return f, nil
}

func createNonZeroSizedFileInTempDir() (File, error) {
	f, err := createZeroSizedFileInTempDir()
	if err != nil {
		// no file ??
	}
	byteString := []byte("byteString")
	err = WriteFile(testFS, f.Name(), byteString, 0644)
	if err != nil {
		// delete the file
		deleteFileInTempDir(f)
		return nil, err
	}
	return f, nil
}

func deleteFileInTempDir(f File) {
	err := testFS.Remove(f.Name())
	if err != nil {
		// now what?
	}
}

func createEmptyTempDir() (string, error) {
	dirPrefix := "_dir_prefix_"
	d, e := TempDir(testFS, "", dirPrefix) // will be in os.TempDir()
	if e != nil {
		// no directory to delete - it was never created
		return "", e
	}
	return d, nil
}

func createTempDirWithZeroLengthFiles() (string, error) {
	d, dirErr := createEmptyTempDir()
	if dirErr != nil {
		//now what?
	}
	filePrefix := "_path_test_"
	_, fileErr := TempFile(testFS, d, filePrefix) // dir is os.TempDir()
	if fileErr != nil {
		// if there was an error no file was created.
		// but we need to remove the directory to clean-up
		deleteTempDir(d)
		return "", fileErr
	}
	// the dir now has one, zero length file in it
	return d, nil

}

func createTempDirWithNonZeroLengthFiles() (string, error) {
	d, dirErr := createEmptyTempDir()
	if dirErr != nil {
		//now what?
	}
	filePrefix := "_path_test_"
	f, fileErr := TempFile(testFS, d, filePrefix) // dir is os.TempDir()
	if fileErr != nil {
		// if there was an error no file was created.
		// but we need to remove the directory to clean-up
		deleteTempDir(d)
		return "", fileErr
	}
	byteString := []byte("byteString")
	fileErr = WriteFile(testFS, f.Name(), byteString, 0644)
	if fileErr != nil {
		// delete the file
		deleteFileInTempDir(f)
		// also delete the directory
		deleteTempDir(d)
		return "", fileErr
	}

	// the dir now has one, zero length file in it
	return d, nil

}

func TestExists(t *testing.T) {
	zeroSizedFile, _ := createZeroSizedFileInTempDir()
	defer deleteFileInTempDir(zeroSizedFile)
	nonZeroSizedFile, _ := createNonZeroSizedFileInTempDir()
	defer deleteFileInTempDir(nonZeroSizedFile)
	emptyDirectory, _ := createEmptyTempDir()
	defer deleteTempDir(emptyDirectory)
	nonExistentFile := os.TempDir() + "/this-file-does-not-exist.txt"
	nonExistentDir := os.TempDir() + "/this/direcotry/does/not/exist/"

	type test struct {
		input          string
		expectedResult bool
		expectedErr    error
	}

	data := []test{
		{zeroSizedFile.Name(), true, nil},
		{nonZeroSizedFile.Name(), true, nil},
		{emptyDirectory, true, nil},
		{nonExistentFile, false, nil},
		{nonExistentDir, false, nil},
	}
	for i, d := range data {
		exists, err := Exists(testFS, d.input)
		if d.expectedResult != exists {
			t.Errorf("Test %d failed. Expected result %t got %t", i, d.expectedResult, exists)
		}
		if d.expectedErr != err {
			t.Errorf("Test %d failed. Expected %q got %q", i, d.expectedErr, err)
		}
	}

}

func TestSafeWriteToDisk(t *testing.T) {
	emptyFile, _ := createZeroSizedFileInTempDir()
	defer deleteFileInTempDir(emptyFile)
	tmpDir, _ := createEmptyTempDir()
	defer deleteTempDir(tmpDir)

	randomString := "This is a random string!"
	reader := strings.NewReader(randomString)

	fileExists := fmt.Errorf("%v already exists", emptyFile.Name())

	type test struct {
		filename    string
		expectedErr error
	}

	now := time.Now().Unix()
	nowStr := strconv.FormatInt(now, 10)
	data := []test{
		{emptyFile.Name(), fileExists},
		{tmpDir + "/" + nowStr, nil},
	}

	for i, d := range data {
		e := SafeWriteReader(testFS, d.filename, reader)
		if d.expectedErr != nil {
			if d.expectedErr.Error() != e.Error() {
				t.Errorf("Test %d failed. Expected error %q but got %q", i, d.expectedErr.Error(), e.Error())
			}
		} else {
			if d.expectedErr != e {
				t.Errorf("Test %d failed. Expected %q but got %q", i, d.expectedErr, e)
			}
			contents, _ := ReadFile(testFS, d.filename)
			if randomString != string(contents) {
				t.Errorf("Test %d failed. Expected contents %q but got %q", i, randomString, string(contents))
			}
		}
		reader.Seek(0, 0)
	}
}

func TestWriteToDisk(t *testing.T) {
	emptyFile, _ := createZeroSizedFileInTempDir()
	defer deleteFileInTempDir(emptyFile)
	tmpDir, _ := createEmptyTempDir()
	defer deleteTempDir(tmpDir)

	randomString := "This is a random string!"
	reader := strings.NewReader(randomString)

	type test struct {
		filename    string
		expectedErr error
	}

	now := time.Now().Unix()
	nowStr := strconv.FormatInt(now, 10)
	data := []test{
		{emptyFile.Name(), nil},
		{tmpDir + "/" + nowStr, nil},
	}

	for i, d := range data {
		e := WriteReader(testFS, d.filename, reader)
		if d.expectedErr != e {
			t.Errorf("Test %d failed. WriteToDisk Error Expected %q but got %q", i, d.expectedErr, e)
		}
		contents, e := ReadFile(testFS, d.filename)
		if e != nil {
			t.Errorf("Test %d failed. Could not read file %s. Reason: %s\n", i, d.filename, e)
		}
		if randomString != string(contents) {
			t.Errorf("Test %d failed. Expected contents %q but got %q", i, randomString, string(contents))
		}
		reader.Seek(0, 0)
	}
}

func TestGetTempDir(t *testing.T) {
	dir := os.TempDir()
	if FilePathSeparator != dir[len(dir)-1:] {
		dir = dir + FilePathSeparator
	}
	testDir := "hugoTestFolder" + FilePathSeparator
	tests := []struct {
		input    string
		expected string
	}{
		{"", dir},
		{testDir + "  Foo bar  ", dir + testDir + "  Foo bar  " + FilePathSeparator},
		{testDir + "Foo.Bar/foo_Bar-Foo", dir + testDir + "Foo.Bar/foo_Bar-Foo" + FilePathSeparator},
		{testDir + "fOO,bar:foo%bAR", dir + testDir + "fOObarfoo%bAR" + FilePathSeparator},
		{testDir + "FOo/BaR.html", dir + testDir + "FOo/BaR.html" + FilePathSeparator},
		{testDir + "трям/трям", dir + testDir + "трям/трям" + FilePathSeparator},
		{testDir + "은행", dir + testDir + "은행" + FilePathSeparator},
		{testDir + "Банковский кассир", dir + testDir + "Банковский кассир" + FilePathSeparator},
	}

	for _, test := range tests {
		output := GetTempDir(new(MemMapFs), test.input)
		if output != test.expected {
			t.Errorf("Expected %#v, got %#v\n", test.expected, output)
		}
	}
}

// This function is very dangerous. Don't use it.
func deleteTempDir(d string) {
	err := os.RemoveAll(d)
	if err != nil {
		// now what?
	}
}

func TestFullBaseFsPath(t *testing.T) {
	type dirSpec struct {
		Dir1, Dir2, Dir3 string
	}
	dirSpecs := []dirSpec{
		dirSpec{Dir1: "/", Dir2: "/", Dir3: "/"},
		dirSpec{Dir1: "/", Dir2: "/path2", Dir3: "/"},
		dirSpec{Dir1: "/path1/dir", Dir2: "/path2/dir/", Dir3: "/path3/dir"},
		dirSpec{Dir1: "C:/path1", Dir2: "path2/dir", Dir3: "/path3/dir/"},
	}

	for _, ds := range dirSpecs {
		memFs := NewMemMapFs()
		level1Fs := NewBasePathFs(memFs, ds.Dir1)
		level2Fs := NewBasePathFs(level1Fs, ds.Dir2)
		level3Fs := NewBasePathFs(level2Fs, ds.Dir3)

		type spec struct {
			BaseFs       Fs
			FileName     string
			ExpectedPath string
		}
		specs := []spec{
			spec{BaseFs: level3Fs, FileName: "f.txt", ExpectedPath: filepath.Join(ds.Dir1, ds.Dir2, ds.Dir3, "f.txt")},
			spec{BaseFs: level3Fs, FileName: "", ExpectedPath: filepath.Join(ds.Dir1, ds.Dir2, ds.Dir3, "")},
			spec{BaseFs: level2Fs, FileName: "f.txt", ExpectedPath: filepath.Join(ds.Dir1, ds.Dir2, "f.txt")},
			spec{BaseFs: level2Fs, FileName: "", ExpectedPath: filepath.Join(ds.Dir1, ds.Dir2, "")},
			spec{BaseFs: level1Fs, FileName: "f.txt", ExpectedPath: filepath.Join(ds.Dir1, "f.txt")},
			spec{BaseFs: level1Fs, FileName: "", ExpectedPath: filepath.Join(ds.Dir1, "")},
		}

		for _, s := range specs {
			if actualPath := FullBaseFsPath(s.BaseFs.(*BasePathFs), s.FileName); actualPath != s.ExpectedPath {
				t.Errorf("Expected \n%s got \n%s", s.ExpectedPath, actualPath)
			}
		}
	}
}
