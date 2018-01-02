package main

// Borrowed from Concourse: https://github.com/concourse/atc/blob/master/atccmd/file_flag.go

import (
	"fmt"
	"os"
	"path/filepath"
)

// FileFlag is a flag for passing a path to a file on disk. The file is
// expected to be a file, not a directory, that actually exists.
type FileFlag string

// UnmarshalFlag implements go-flag's Unmarshaler interface
func (f *FileFlag) UnmarshalFlag(value string) error {
	stat, err := os.Stat(value)
	if err != nil {
		return err
	}

	if stat.IsDir() {
		return fmt.Errorf("path '%s' is a directory, not a file", value)
	}

	abs, err := filepath.Abs(value)
	if err != nil {
		return err
	}

	*f = FileFlag(abs)

	return nil
}

// Path is the path to the file
func (f FileFlag) Path() string {
	return string(f)
}
