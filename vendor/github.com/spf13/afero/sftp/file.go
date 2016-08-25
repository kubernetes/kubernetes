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

package sftpfs

import (
	"os"
	"github.com/pkg/sftp"
)

type File struct {
	fd *sftp.File
}

func FileOpen(s *sftp.Client, name string) (*File, error) {
	fd, err := s.Open(name)
	if err != nil {
		return &File{}, err
	}
	return &File{fd: fd}, nil
}

func FileCreate(s *sftp.Client, name string) (*File, error) {
	fd, err := s.Create(name)
	if err != nil {
		return &File{}, err
	}
	return &File{fd: fd}, nil
}

func (f *File) Close() error {
	return f.fd.Close()
}

func (f *File) Name() string {
	return f.fd.Name()
}

func (f *File) Stat() (os.FileInfo, error) {
	return f.fd.Stat()
}

func (f *File) Sync() error {
	return nil
}

func (f *File) Truncate(size int64) error {
	return f.fd.Truncate(size)
}

func (f *File) Read(b []byte) (n int, err error) {
	return f.fd.Read(b)
}

// TODO
func (f *File) ReadAt(b []byte, off int64) (n int, err error) {
	return 0,nil
}

// TODO
func (f *File) Readdir(count int) (res []os.FileInfo, err error) {
	return nil,nil
}

// TODO
func (f *File) Readdirnames(n int) (names []string, err error) {
	return nil,nil
}

func (f *File) Seek(offset int64, whence int) (int64, error) {
	return f.fd.Seek(offset, whence)
}

func (f *File) Write(b []byte) (n int, err error) {
	return f.fd.Write(b)
}

// TODO
func (f *File) WriteAt(b []byte, off int64) (n int, err error) {
	return 0,nil
}

func (f *File) WriteString(s string) (ret int, err error) {
	return f.fd.Write([]byte(s))
}
