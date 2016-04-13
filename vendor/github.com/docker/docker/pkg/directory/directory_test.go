package directory

import (
	"io/ioutil"
	"os"
	"testing"
)

// Size of an empty directory should be 0
func TestSizeEmpty(t *testing.T) {
	var dir string
	var err error
	if dir, err = ioutil.TempDir(os.TempDir(), "testSizeEmptyDirectory"); err != nil {
		t.Fatalf("failed to create directory: %s", err)
	}

	var size int64
	if size, _ = Size(dir); size != 0 {
		t.Fatalf("empty directory has size: %d", size)
	}
}

// Size of a directory with one empty file should be 0
func TestSizeEmptyFile(t *testing.T) {
	var dir string
	var err error
	if dir, err = ioutil.TempDir(os.TempDir(), "testSizeEmptyFile"); err != nil {
		t.Fatalf("failed to create directory: %s", err)
	}

	var file *os.File
	if file, err = ioutil.TempFile(dir, "file"); err != nil {
		t.Fatalf("failed to create file: %s", err)
	}

	var size int64
	if size, _ = Size(file.Name()); size != 0 {
		t.Fatalf("directory with one file has size: %d", size)
	}
}

// Size of a directory with one 5-byte file should be 5
func TestSizeNonemptyFile(t *testing.T) {
	var dir string
	var err error
	if dir, err = ioutil.TempDir(os.TempDir(), "testSizeNonemptyFile"); err != nil {
		t.Fatalf("failed to create directory: %s", err)
	}

	var file *os.File
	if file, err = ioutil.TempFile(dir, "file"); err != nil {
		t.Fatalf("failed to create file: %s", err)
	}

	d := []byte{97, 98, 99, 100, 101}
	file.Write(d)

	var size int64
	if size, _ = Size(file.Name()); size != 5 {
		t.Fatalf("directory with one 5-byte file has size: %d", size)
	}
}

// Size of a directory with one empty directory should be 0
func TestSizeNestedDirectoryEmpty(t *testing.T) {
	var dir string
	var err error
	if dir, err = ioutil.TempDir(os.TempDir(), "testSizeNestedDirectoryEmpty"); err != nil {
		t.Fatalf("failed to create directory: %s", err)
	}
	if dir, err = ioutil.TempDir(dir, "nested"); err != nil {
		t.Fatalf("failed to create nested directory: %s", err)
	}

	var size int64
	if size, _ = Size(dir); size != 0 {
		t.Fatalf("directory with one empty directory has size: %d", size)
	}
}

// Test directory with 1 file and 1 empty directory
func TestSizeFileAndNestedDirectoryEmpty(t *testing.T) {
	var dir string
	var err error
	if dir, err = ioutil.TempDir(os.TempDir(), "testSizeFileAndNestedDirectoryEmpty"); err != nil {
		t.Fatalf("failed to create directory: %s", err)
	}
	if dir, err = ioutil.TempDir(dir, "nested"); err != nil {
		t.Fatalf("failed to create nested directory: %s", err)
	}

	var file *os.File
	if file, err = ioutil.TempFile(dir, "file"); err != nil {
		t.Fatalf("failed to create file: %s", err)
	}

	d := []byte{100, 111, 99, 107, 101, 114}
	file.Write(d)

	var size int64
	if size, _ = Size(dir); size != 6 {
		t.Fatalf("directory with 6-byte file and empty directory has size: %d", size)
	}
}

// Test directory with 1 file and 1 non-empty directory
func TestSizeFileAndNestedDirectoryNonempty(t *testing.T) {
	var dir, dirNested string
	var err error
	if dir, err = ioutil.TempDir(os.TempDir(), "TestSizeFileAndNestedDirectoryNonempty"); err != nil {
		t.Fatalf("failed to create directory: %s", err)
	}
	if dirNested, err = ioutil.TempDir(dir, "nested"); err != nil {
		t.Fatalf("failed to create nested directory: %s", err)
	}

	var file *os.File
	if file, err = ioutil.TempFile(dir, "file"); err != nil {
		t.Fatalf("failed to create file: %s", err)
	}

	data := []byte{100, 111, 99, 107, 101, 114}
	file.Write(data)

	var nestedFile *os.File
	if nestedFile, err = ioutil.TempFile(dirNested, "file"); err != nil {
		t.Fatalf("failed to create file in nested directory: %s", err)
	}

	nestedData := []byte{100, 111, 99, 107, 101, 114}
	nestedFile.Write(nestedData)

	var size int64
	if size, _ = Size(dir); size != 12 {
		t.Fatalf("directory with 6-byte file and nested directory with 6-byte file has size: %d", size)
	}
}
