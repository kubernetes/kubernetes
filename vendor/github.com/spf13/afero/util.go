// Copyright ©2015 Steve Francia <spf@spf13.com>
// Portions Copyright ©2015 The Hugo Authors
// Portions Copyright 2016-present Bjørn Erik Pedersen <bjorn.erik.pedersen@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package afero

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode"

	"golang.org/x/text/transform"
	"golang.org/x/text/unicode/norm"
)

// Filepath separator defined by os.Separator.
const FilePathSeparator = string(filepath.Separator)

// Takes a reader and a path and writes the content
func (a Afero) WriteReader(path string, r io.Reader) (err error) {
	return WriteReader(a.Fs, path, r)
}

func WriteReader(fs Fs, path string, r io.Reader) (err error) {
	dir, _ := filepath.Split(path)
	ospath := filepath.FromSlash(dir)

	if ospath != "" {
		err = fs.MkdirAll(ospath, 0777) // rwx, rw, r
		if err != nil {
			if err != os.ErrExist {
				return err
			}
		}
	}

	file, err := fs.Create(path)
	if err != nil {
		return
	}
	defer file.Close()

	_, err = io.Copy(file, r)
	return
}

// Same as WriteReader but checks to see if file/directory already exists.
func (a Afero) SafeWriteReader(path string, r io.Reader) (err error) {
	return SafeWriteReader(a.Fs, path, r)
}

func SafeWriteReader(fs Fs, path string, r io.Reader) (err error) {
	dir, _ := filepath.Split(path)
	ospath := filepath.FromSlash(dir)

	if ospath != "" {
		err = fs.MkdirAll(ospath, 0777) // rwx, rw, r
		if err != nil {
			return
		}
	}

	exists, err := Exists(fs, path)
	if err != nil {
		return
	}
	if exists {
		return fmt.Errorf("%v already exists", path)
	}

	file, err := fs.Create(path)
	if err != nil {
		return
	}
	defer file.Close()

	_, err = io.Copy(file, r)
	return
}

func (a Afero) GetTempDir(subPath string) string {
	return GetTempDir(a.Fs, subPath)
}

// GetTempDir returns the default temp directory with trailing slash
// if subPath is not empty then it will be created recursively with mode 777 rwx rwx rwx
func GetTempDir(fs Fs, subPath string) string {
	addSlash := func(p string) string {
		if FilePathSeparator != p[len(p)-1:] {
			p = p + FilePathSeparator
		}
		return p
	}
	dir := addSlash(os.TempDir())

	if subPath != "" {
		// preserve windows backslash :-(
		if FilePathSeparator == "\\" {
			subPath = strings.Replace(subPath, "\\", "____", -1)
		}
		dir = dir + UnicodeSanitize((subPath))
		if FilePathSeparator == "\\" {
			dir = strings.Replace(dir, "____", "\\", -1)
		}

		if exists, _ := Exists(fs, dir); exists {
			return addSlash(dir)
		}

		err := fs.MkdirAll(dir, 0777)
		if err != nil {
			panic(err)
		}
		dir = addSlash(dir)
	}
	return dir
}

// Rewrite string to remove non-standard path characters
func UnicodeSanitize(s string) string {
	source := []rune(s)
	target := make([]rune, 0, len(source))

	for _, r := range source {
		if unicode.IsLetter(r) ||
			unicode.IsDigit(r) ||
			unicode.IsMark(r) ||
			r == '.' ||
			r == '/' ||
			r == '\\' ||
			r == '_' ||
			r == '-' ||
			r == '%' ||
			r == ' ' ||
			r == '#' {
			target = append(target, r)
		}
	}

	return string(target)
}

// Transform characters with accents into plain forms.
func NeuterAccents(s string) string {
	t := transform.Chain(norm.NFD, transform.RemoveFunc(isMn), norm.NFC)
	result, _, _ := transform.String(t, string(s))

	return result
}

func isMn(r rune) bool {
	return unicode.Is(unicode.Mn, r) // Mn: nonspacing marks
}

func (a Afero) FileContainsBytes(filename string, subslice []byte) (bool, error) {
	return FileContainsBytes(a.Fs, filename, subslice)
}

// Check if a file contains a specified byte slice.
func FileContainsBytes(fs Fs, filename string, subslice []byte) (bool, error) {
	f, err := fs.Open(filename)
	if err != nil {
		return false, err
	}
	defer f.Close()

	return readerContainsAny(f, subslice), nil
}

func (a Afero) FileContainsAnyBytes(filename string, subslices [][]byte) (bool, error) {
	return FileContainsAnyBytes(a.Fs, filename, subslices)
}

// Check if a file contains any of the specified byte slices.
func FileContainsAnyBytes(fs Fs, filename string, subslices [][]byte) (bool, error) {
	f, err := fs.Open(filename)
	if err != nil {
		return false, err
	}
	defer f.Close()

	return readerContainsAny(f, subslices...), nil
}

// readerContains reports whether any of the subslices is within r.
func readerContainsAny(r io.Reader, subslices ...[]byte) bool {

	if r == nil || len(subslices) == 0 {
		return false
	}

	largestSlice := 0

	for _, sl := range subslices {
		if len(sl) > largestSlice {
			largestSlice = len(sl)
		}
	}

	if largestSlice == 0 {
		return false
	}

	bufflen := largestSlice * 4
	halflen := bufflen / 2
	buff := make([]byte, bufflen)
	var err error
	var n, i int

	for {
		i++
		if i == 1 {
			n, err = io.ReadAtLeast(r, buff[:halflen], halflen)
		} else {
			if i != 2 {
				// shift left to catch overlapping matches
				copy(buff[:], buff[halflen:])
			}
			n, err = io.ReadAtLeast(r, buff[halflen:], halflen)
		}

		if n > 0 {
			for _, sl := range subslices {
				if bytes.Contains(buff, sl) {
					return true
				}
			}
		}

		if err != nil {
			break
		}
	}
	return false
}

func (a Afero) DirExists(path string) (bool, error) {
	return DirExists(a.Fs, path)
}

// DirExists checks if a path exists and is a directory.
func DirExists(fs Fs, path string) (bool, error) {
	fi, err := fs.Stat(path)
	if err == nil && fi.IsDir() {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func (a Afero) IsDir(path string) (bool, error) {
	return IsDir(a.Fs, path)
}

// IsDir checks if a given path is a directory.
func IsDir(fs Fs, path string) (bool, error) {
	fi, err := fs.Stat(path)
	if err != nil {
		return false, err
	}
	return fi.IsDir(), nil
}

func (a Afero) IsEmpty(path string) (bool, error) {
	return IsEmpty(a.Fs, path)
}

// IsEmpty checks if a given file or directory is empty.
func IsEmpty(fs Fs, path string) (bool, error) {
	if b, _ := Exists(fs, path); !b {
		return false, fmt.Errorf("%q path does not exist", path)
	}
	fi, err := fs.Stat(path)
	if err != nil {
		return false, err
	}
	if fi.IsDir() {
		f, err := fs.Open(path)
		if err != nil {
			return false, err
		}
		defer f.Close()
		list, err := f.Readdir(-1)
		return len(list) == 0, nil
	}
	return fi.Size() == 0, nil
}

func (a Afero) Exists(path string) (bool, error) {
	return Exists(a.Fs, path)
}

// Check if a file or directory exists.
func Exists(fs Fs, path string) (bool, error) {
	_, err := fs.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

func FullBaseFsPath(basePathFs *BasePathFs, relativePath string) string {
	combinedPath := filepath.Join(basePathFs.path, relativePath)
	if parent, ok := basePathFs.source.(*BasePathFs); ok {
		return FullBaseFsPath(parent, combinedPath)
	}

	return combinedPath
}
