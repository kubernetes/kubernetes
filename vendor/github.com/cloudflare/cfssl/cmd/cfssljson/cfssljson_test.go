package main

import (
	"testing"
)

func TestReadFile(t *testing.T) {
	_, err := readFile("-")
	if err != nil {
		t.Fatal(err)
	}

	file, err := readFile("./testdata/test.txt")
	if err != nil {
		t.Fatal(err)
	}
	if string(file) != "This is a test file" {
		t.Fatal("File not read correctly")
	}
}
