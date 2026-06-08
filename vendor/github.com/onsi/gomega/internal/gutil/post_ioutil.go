//go:build go1.16
// +build go1.16

// Package gutil is a replacement for ioutil, which should not be used in new
// code as of Go 1.16. With Go 1.16 and higher, this implementation
// uses the ioutil replacement functions in "io" and "os" with some
// Gomega specifics. This means that we should not get deprecation warnings
// for ioutil when they are added.
package gutil

import (
	"io"
	"os"
)

func NopCloser(r io.Reader) io.ReadCloser {
	return io.NopCloser(r)
}

func ReadAll(r io.Reader) ([]byte, error) {
	return io.ReadAll(r)
}

func ReadDir(dirname string) ([]string, error) {
	entries, err := os.ReadDir(dirname)
	if err != nil {
		return nil, err
	}

	var names []string
	for _, entry := range entries {
		names = append(names, entry.Name())
	}

	return names, nil
}

func ReadFile(filename string) ([]byte, error) {
	return os.ReadFile(filename)
}

func MkdirTemp(dir, pattern string) (string, error) {
	return os.MkdirTemp(dir, pattern)
}

func WriteFile(filename string, data []byte) error {
	return os.WriteFile(filename, data, 0644)
}
