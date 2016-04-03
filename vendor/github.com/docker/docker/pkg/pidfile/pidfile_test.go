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

	file, err := New(filepath.Join(dir, "testfile"))
	if err != nil {
		t.Fatal("Could not create test file", err)
	}

	if err := file.Remove(); err != nil {
		t.Fatal("Could not delete created test file")
	}
}

func TestRemoveInvalidPath(t *testing.T) {
	file := PidFile{path: filepath.Join("foo", "bar")}

	if err := file.Remove(); err == nil {
		t.Fatal("Non-existing file doesn't give an error on delete")
	}
}
