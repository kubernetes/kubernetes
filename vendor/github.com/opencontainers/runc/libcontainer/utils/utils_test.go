package utils

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/sys/unix"
)

func TestGenerateName(t *testing.T) {
	name, err := GenerateRandomName("veth", 5)
	if err != nil {
		t.Fatal(err)
	}

	expected := 5 + len("veth")
	if len(name) != expected {
		t.Fatalf("expected name to be %d chars but received %d", expected, len(name))
	}

	name, err = GenerateRandomName("veth", 65)
	if err != nil {
		t.Fatal(err)
	}

	expected = 64 + len("veth")
	if len(name) != expected {
		t.Fatalf("expected name to be %d chars but received %d", expected, len(name))
	}
}

var labelTest = []struct {
	labels        []string
	query         string
	expectedValue string
}{
	{[]string{"bundle=/path/to/bundle"}, "bundle", "/path/to/bundle"},
	{[]string{"test=a", "test=b"}, "bundle", ""},
	{[]string{"bundle=a", "test=b", "bundle=c"}, "bundle", "a"},
	{[]string{"", "test=a", "bundle=b"}, "bundle", "b"},
	{[]string{"test", "bundle=a"}, "bundle", "a"},
	{[]string{"test=a", "bundle="}, "bundle", ""},
}

func TestSearchLabels(t *testing.T) {
	for _, tt := range labelTest {
		if v := SearchLabels(tt.labels, tt.query); v != tt.expectedValue {
			t.Errorf("expected value '%s' for query '%s'; got '%s'", tt.expectedValue, tt.query, v)
		}
	}
}

func TestResolveRootfs(t *testing.T) {
	dir := "rootfs"
	os.Mkdir(dir, 0600)
	defer os.Remove(dir)

	path, err := ResolveRootfs(dir)
	if err != nil {
		t.Fatal(err)
	}
	pwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if path != fmt.Sprintf("%s/%s", pwd, "rootfs") {
		t.Errorf("expected rootfs to be abs and was %s", path)
	}
}

func TestResolveRootfsWithSymlink(t *testing.T) {
	dir := "rootfs"
	tmpDir, _ := filepath.EvalSymlinks(os.TempDir())
	os.Symlink(tmpDir, dir)
	defer os.Remove(dir)

	path, err := ResolveRootfs(dir)
	if err != nil {
		t.Fatal(err)
	}

	if path != tmpDir {
		t.Errorf("expected rootfs to be the real path %s and was %s", path, os.TempDir())
	}
}

func TestResolveRootfsWithNonExistingDir(t *testing.T) {
	_, err := ResolveRootfs("foo")
	if err == nil {
		t.Error("expected error to happen but received nil")
	}
}

func TestExitStatus(t *testing.T) {
	status := unix.WaitStatus(0)
	ex := ExitStatus(status)
	if ex != 0 {
		t.Errorf("expected exit status to equal 0 and received %d", ex)
	}
}

func TestExitStatusSignaled(t *testing.T) {
	status := unix.WaitStatus(2)
	ex := ExitStatus(status)
	if ex != 130 {
		t.Errorf("expected exit status to equal 130 and received %d", ex)
	}
}

func TestWriteJSON(t *testing.T) {
	person := struct {
		Name string
		Age  int
	}{
		Name: "Alice",
		Age:  30,
	}

	var b bytes.Buffer
	err := WriteJSON(&b, person)
	if err != nil {
		t.Fatal(err)
	}

	expected := `{"Name":"Alice","Age":30}`
	if b.String() != expected {
		t.Errorf("expected to write %s but was %s", expected, b.String())
	}
}

func TestCleanPath(t *testing.T) {
	path := CleanPath("")
	if path != "" {
		t.Errorf("expected to receive empty string and received %s", path)
	}

	path = CleanPath("rootfs")
	if path != "rootfs" {
		t.Errorf("expected to receive 'rootfs' and received %s", path)
	}

	path = CleanPath("../../../var")
	if path != "var" {
		t.Errorf("expected to receive 'var' and received %s", path)
	}

	path = CleanPath("/../../../var")
	if path != "/var" {
		t.Errorf("expected to receive '/var' and received %s", path)
	}
}
