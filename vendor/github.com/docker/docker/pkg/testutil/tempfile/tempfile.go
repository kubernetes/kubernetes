package tempfile

import (
	"io/ioutil"
	"os"

	"github.com/stretchr/testify/require"
)

// TempFile is a temporary file that can be used with unit tests. TempFile
// reduces the boilerplate setup required in each test case by handling
// setup errors.
type TempFile struct {
	File *os.File
}

// NewTempFile returns a new temp file with contents
func NewTempFile(t require.TestingT, prefix string, content string) *TempFile {
	file, err := ioutil.TempFile("", prefix+"-")
	require.NoError(t, err)

	_, err = file.Write([]byte(content))
	require.NoError(t, err)
	file.Close()
	return &TempFile{File: file}
}

// Name returns the filename
func (f *TempFile) Name() string {
	return f.File.Name()
}

// Remove removes the file
func (f *TempFile) Remove() {
	os.Remove(f.Name())
}

// TempDir is a temporary directory that can be used with unit tests. TempDir
// reduces the boilerplate setup required in each test case by handling
// setup errors.
type TempDir struct {
	Path string
}

// NewTempDir returns a new temp file with contents
func NewTempDir(t require.TestingT, prefix string) *TempDir {
	path, err := ioutil.TempDir("", prefix+"-")
	require.NoError(t, err)

	return &TempDir{Path: path}
}

// Remove removes the file
func (f *TempDir) Remove() {
	os.Remove(f.Path)
}
