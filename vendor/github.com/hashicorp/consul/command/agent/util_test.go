package agent

import (
	"io/ioutil"
	"os"
	"runtime"
	"testing"
	"time"
)

func TestAEScale(t *testing.T) {
	intv := time.Minute
	if v := aeScale(intv, 100); v != intv {
		t.Fatalf("Bad: %v", v)
	}
	if v := aeScale(intv, 200); v != 2*intv {
		t.Fatalf("Bad: %v", v)
	}
	if v := aeScale(intv, 1000); v != 4*intv {
		t.Fatalf("Bad: %v", v)
	}
	if v := aeScale(intv, 10000); v != 8*intv {
		t.Fatalf("Bad: %v", v)
	}
}

func TestStringHash(t *testing.T) {
	in := "hello world"
	expected := "5eb63bbbe01eeed093cb22bb8f5acdc3"

	if out := stringHash(in); out != expected {
		t.Fatalf("bad: %s", out)
	}
}

func TestSetFilePermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.SkipNow()
	}
	tempFile, err := ioutil.TempFile("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	path := tempFile.Name()
	defer os.Remove(path)

	// Bad UID fails
	if err := setFilePermissions(path, UnixSocketPermissions{Usr: "%"}); err == nil {
		t.Fatalf("should fail")
	}

	// Bad GID fails
	if err := setFilePermissions(path, UnixSocketPermissions{Grp: "%"}); err == nil {
		t.Fatalf("should fail")
	}

	// Bad mode fails
	if err := setFilePermissions(path, UnixSocketPermissions{Perms: "%"}); err == nil {
		t.Fatalf("should fail")
	}

	// Allows omitting user/group/mode
	if err := setFilePermissions(path, UnixSocketPermissions{}); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Doesn't change mode if not given
	if err := os.Chmod(path, 0700); err != nil {
		t.Fatalf("err: %s", err)
	}
	if err := setFilePermissions(path, UnixSocketPermissions{}); err != nil {
		t.Fatalf("err: %s", err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if fi.Mode().String() != "-rwx------" {
		t.Fatalf("bad: %s", fi.Mode())
	}

	// Changes mode if given
	if err := setFilePermissions(path, UnixSocketPermissions{Perms: "0777"}); err != nil {
		t.Fatalf("err: %s", err)
	}
	fi, err = os.Stat(path)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if fi.Mode().String() != "-rwxrwxrwx" {
		t.Fatalf("bad: %s", fi.Mode())
	}
}
