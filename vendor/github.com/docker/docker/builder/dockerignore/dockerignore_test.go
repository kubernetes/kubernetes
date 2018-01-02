package dockerignore

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
)

func TestReadAll(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "dockerignore-test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	di, err := ReadAll(nil)
	if err != nil {
		t.Fatalf("Expected not to have error, got %v", err)
	}

	if diLen := len(di); diLen != 0 {
		t.Fatalf("Expected to have zero dockerignore entry, got %d", diLen)
	}

	diName := filepath.Join(tmpDir, ".dockerignore")
	content := fmt.Sprintf("test1\n/test2\n/a/file/here\n\nlastfile\n# this is a comment\n! /inverted/abs/path\n!\n! \n")
	err = ioutil.WriteFile(diName, []byte(content), 0777)
	if err != nil {
		t.Fatal(err)
	}

	diFd, err := os.Open(diName)
	if err != nil {
		t.Fatal(err)
	}
	defer diFd.Close()

	di, err = ReadAll(diFd)
	if err != nil {
		t.Fatal(err)
	}

	if len(di) != 7 {
		t.Fatalf("Expected 5 entries, got %v", len(di))
	}
	if di[0] != "test1" {
		t.Fatal("First element is not test1")
	}
	if di[1] != "test2" { // according to https://docs.docker.com/engine/reference/builder/#dockerignore-file, /foo/bar should be treated as foo/bar
		t.Fatal("Second element is not test2")
	}
	if di[2] != "a/file/here" { // according to https://docs.docker.com/engine/reference/builder/#dockerignore-file, /foo/bar should be treated as foo/bar
		t.Fatal("Third element is not a/file/here")
	}
	if di[3] != "lastfile" {
		t.Fatal("Fourth element is not lastfile")
	}
	if di[4] != "!inverted/abs/path" {
		t.Fatal("Fifth element is not !inverted/abs/path")
	}
	if di[5] != "!" {
		t.Fatalf("Sixth element is not !, but %s", di[5])
	}
	if di[6] != "!" {
		t.Fatalf("Sixth element is not !, but %s", di[6])
	}
}
