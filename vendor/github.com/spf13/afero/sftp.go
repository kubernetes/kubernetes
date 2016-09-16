// Copyright © 2015 Jerry Jacobs <jerry.jacobs@xor-gate.org>.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package afero

import (
	"os"
	"time"

	"github.com/spf13/afero/sftp"

	"github.com/pkg/sftp"
)

// SftpFs is a Fs implementation that uses functions provided by the sftp package.
//
// For details in any method, check the documentation of the sftp package
// (github.com/pkg/sftp).
type SftpFs struct{
	SftpClient  *sftp.Client
}

func (s SftpFs) Name() string { return "SftpFs" }

func (s SftpFs) Create(name string) (File, error) {
	f, err := sftpfs.FileCreate(s.SftpClient, name)
	return f, err
}

func (s SftpFs) Mkdir(name string, perm os.FileMode) error {
	err := s.SftpClient.Mkdir(name)
	if err != nil {
		return err
	}
	return s.SftpClient.Chmod(name, perm)
}

func (s SftpFs) MkdirAll(path string, perm os.FileMode) error {
	// Fast path: if we can tell whether path is a directory or file, stop with success or error.
	dir, err := s.Stat(path)
	if err == nil {
		if dir.IsDir() {
			return nil
		}
		return err
	}

	// Slow path: make sure parent exists and then call Mkdir for path.
	i := len(path)
	for i > 0 && os.IsPathSeparator(path[i-1]) { // Skip trailing path separator.
		i--
	}

	j := i
	for j > 0 && !os.IsPathSeparator(path[j-1]) { // Scan backward over element.
		j--
	}

	if j > 1 {
		// Create parent
		err = s.MkdirAll(path[0:j-1], perm)
		if err != nil {
			return err
		}
	}

	// Parent now exists; invoke Mkdir and use its result.
	err = s.Mkdir(path, perm)
	if err != nil {
		// Handle arguments like "foo/." by
		// double-checking that directory doesn't exist.
		dir, err1 := s.Lstat(path)
		if err1 == nil && dir.IsDir() {
			return nil
		}
		return err
	}
	return nil
}

func (s SftpFs) Open(name string) (File, error) {
	f, err := sftpfs.FileOpen(s.SftpClient, name)
	return f, err
}

func (s SftpFs) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	return nil,nil
}

func (s SftpFs) Remove(name string) error {
	return s.SftpClient.Remove(name)
}

func (s SftpFs) RemoveAll(path string) error {
	// TODO have a look at os.RemoveAll
	// https://github.com/golang/go/blob/master/src/os/path.go#L66
	return nil
}

func (s SftpFs) Rename(oldname, newname string) error {
	return s.SftpClient.Rename(oldname, newname)
}

func (s SftpFs) Stat(name string) (os.FileInfo, error) {
	return s.SftpClient.Stat(name)
}

func (s SftpFs) Lstat(p string) (os.FileInfo, error) {
	return s.SftpClient.Lstat(p)
}

func (s SftpFs) Chmod(name string, mode os.FileMode) error {
	return s.SftpClient.Chmod(name, mode)
}

func (s SftpFs) Chtimes(name string, atime time.Time, mtime time.Time) error {
	return s.SftpClient.Chtimes(name, atime, mtime)
}
