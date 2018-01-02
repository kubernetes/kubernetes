package pidfile

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestNewAndRemove(t *testing.T) {
	dir, err := ioutil.TempDir(os.TempDir(), "test-pidfile")
	if err != nil {
		t.Fatal("Could not create test directory")
	}

	path := filepath.Join(dir, "testfile")
	file, err := New(path)
	if err != nil {
		t.Fatal("Could not create test file", err)
	}

	_, err = New(path)
	if err == nil {
		t.Fatal("Test file creation not blocked")
	}

	if err := file.Remove(); err != nil {
		t.Fatal("Could not delete created test file")
	}
}

func TestRemoveInvalidPath(t *testing.T) {
	file := PIDFile{path: filepath.Join("foo", "bar")}

	if err := file.Remove(); err == nil {
		t.Fatal("Non-existing file doesn't give an error on delete")
	}
}
