//go:build !go1.16
// +build !go1.16

// Package gutil is a replacement for ioutil, which should not be used in new
// code as of Go 1.16. With Go 1.15 and lower, this implementation
// uses the ioutil functions, meaning that although Gomega is not officially
// supported on these versions, it is still likely to work.
package gutil

import (
	"io"
	"io/ioutil"
)

func NopCloser(r io.Reader) io.ReadCloser {
	return ioutil.NopCloser(r)
}

func ReadAll(r io.Reader) ([]byte, error) {
	return ioutil.ReadAll(r)
}

func ReadDir(dirname string) ([]string, error) {
	files, err := ioutil.ReadDir(dirname)
	if err != nil {
		return nil, err
	}

	var names []string
	for _, file := range files {
		names = append(names, file.Name())
	}

	return names, nil
}

func ReadFile(filename string) ([]byte, error) {
	return ioutil.ReadFile(filename)
}

func MkdirTemp(dir, pattern string) (string, error) {
	return ioutil.TempDir(dir, pattern)
}

func WriteFile(filename string, data []byte) error {
	return ioutil.WriteFile(filename, data, 0644)
}
