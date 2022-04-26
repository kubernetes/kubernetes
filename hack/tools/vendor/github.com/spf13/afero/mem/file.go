// Copyright © 2015 Steve Francia <spf@spf13.com>.
// Copyright 2013 tsuru authors. All rights reserved.
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

package mem

import (
	"bytes"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

const FilePathSeparator = string(filepath.Separator)

type File struct {
	// atomic requires 64-bit alignment for struct field access
	at           int64
	readDirCount int64
	closed       bool
	readOnly     bool
	fileData     *FileData
}

func NewFileHandle(data *FileData) *File {
	return &File{fileData: data}
}

func NewReadOnlyFileHandle(data *FileData) *File {
	return &File{fileData: data, readOnly: true}
}

func (f File) Data() *FileData {
	return f.fileData
}

type FileData struct {
	sync.Mutex
	name    string
	data    []byte
	memDir  Dir
	dir     bool
	mode    os.FileMode
	modtime time.Time
	uid     int
	gid     int
}

func (d *FileData) Name() string {
	d.Lock()
	defer d.Unlock()
	return d.name
}

func CreateFile(name string) *FileData {
	return &FileData{name: name, mode: os.ModeTemporary, modtime: time.Now()}
}

func CreateDir(name string) *FileData {
	return &FileData{name: name, memDir: &DirMap{}, dir: true}
}

func ChangeFileName(f *FileData, newname string) {
	f.Lock()
	f.name = newname
	f.Unlock()
}

func SetMode(f *FileData, mode os.FileMode) {
	f.Lock()
	f.mode = mode
	f.Unlock()
}

func SetModTime(f *FileData, mtime time.Time) {
	f.Lock()
	setModTime(f, mtime)
	f.Unlock()
}

func setModTime(f *FileData, mtime time.Time) {
	f.modtime = mtime
}

func SetUID(f *FileData, uid int) {
	f.Lock()
	f.uid = uid
	f.Unlock()
}

func SetGID(f *FileData, gid int) {
	f.Lock()
	f.gid = gid
	f.Unlock()
}

func GetFileInfo(f *FileData) *FileInfo {
	return &FileInfo{f}
}

func (f *File) Open() error {
	atomic.StoreInt64(&f.at, 0)
	atomic.StoreInt64(&f.readDirCount, 0)
	f.fileData.Lock()
	f.closed = false
	f.fileData.Unlock()
	return nil
}

func (f *File) Close() error {
	f.fileData.Lock()
	f.closed = true
	if !f.readOnly {
		setModTime(f.fileData, time.Now())
	}
	f.fileData.Unlock()
	return nil
}

func (f *File) Name() string {
	return f.fileData.Name()
}

func (f *File) Stat() (os.FileInfo, error) {
	return &FileInfo{f.fileData}, nil
}

func (f *File) Sync() error {
	return nil
}

func (f *File) Readdir(count int) (res []os.FileInfo, err error) {
	if !f.fileData.dir {
		return nil, &os.PathError{Op: "readdir", Path: f.fileData.name, Err: errors.New("not a dir")}
	}
	var outLength int64

	f.fileData.Lock()
	files := f.fileData.memDir.Files()[f.readDirCount:]
	if count > 0 {
		if len(files) < count {
			outLength = int64(len(files))
		} else {
			outLength = int64(count)
		}
		if len(files) == 0 {
			err = io.EOF
		}
	} else {
		outLength = int64(len(files))
	}
	f.readDirCount += outLength
	f.fileData.Unlock()

	res = make([]os.FileInfo, outLength)
	for i := range res {
		res[i] = &FileInfo{files[i]}
	}

	return res, err
}

func (f *File) Readdirnames(n int) (names []string, err error) {
	fi, err := f.Readdir(n)
	names = make([]string, len(fi))
	for i, f := range fi {
		_, names[i] = filepath.Split(f.Name())
	}
	return names, err
}

func (f *File) Read(b []byte) (n int, err error) {
	f.fileData.Lock()
	defer f.fileData.Unlock()
	if f.closed == true {
		return 0, ErrFileClosed
	}
	if len(b) > 0 && int(f.at) == len(f.fileData.data) {
		return 0, io.EOF
	}
	if int(f.at) > len(f.fileData.data) {
		return 0, io.ErrUnexpectedEOF
	}
	if len(f.fileData.data)-int(f.at) >= len(b) {
		n = len(b)
	} else {
		n = len(f.fileData.data) - int(f.at)
	}
	copy(b, f.fileData.data[f.at:f.at+int64(n)])
	atomic.AddInt64(&f.at, int64(n))
	return
}

func (f *File) ReadAt(b []byte, off int64) (n int, err error) {
	prev := atomic.LoadInt64(&f.at)
	atomic.StoreInt64(&f.at, off)
	n, err = f.Read(b)
	atomic.StoreInt64(&f.at, prev)
	return
}

func (f *File) Truncate(size int64) error {
	if f.closed == true {
		return ErrFileClosed
	}
	if f.readOnly {
		return &os.PathError{Op: "truncate", Path: f.fileData.name, Err: errors.New("file handle is read only")}
	}
	if size < 0 {
		return ErrOutOfRange
	}
	f.fileData.Lock()
	defer f.fileData.Unlock()
	if size > int64(len(f.fileData.data)) {
		diff := size - int64(len(f.fileData.data))
		f.fileData.data = append(f.fileData.data, bytes.Repeat([]byte{00}, int(diff))...)
	} else {
		f.fileData.data = f.fileData.data[0:size]
	}
	setModTime(f.fileData, time.Now())
	return nil
}

func (f *File) Seek(offset int64, whence int) (int64, error) {
	if f.closed == true {
		return 0, ErrFileClosed
	}
	switch whence {
	case io.SeekStart:
		atomic.StoreInt64(&f.at, offset)
	case io.SeekCurrent:
		atomic.AddInt64(&f.at, offset)
	case io.SeekEnd:
		atomic.StoreInt64(&f.at, int64(len(f.fileData.data))+offset)
	}
	return f.at, nil
}

func (f *File) Write(b []byte) (n int, err error) {
	if f.closed == true {
		return 0, ErrFileClosed
	}
	if f.readOnly {
		return 0, &os.PathError{Op: "write", Path: f.fileData.name, Err: errors.New("file handle is read only")}
	}
	n = len(b)
	cur := atomic.LoadInt64(&f.at)
	f.fileData.Lock()
	defer f.fileData.Unlock()
	diff := cur - int64(len(f.fileData.data))
	var tail []byte
	if n+int(cur) < len(f.fileData.data) {
		tail = f.fileData.data[n+int(cur):]
	}
	if diff > 0 {
		f.fileData.data = append(f.fileData.data, append(bytes.Repeat([]byte{00}, int(diff)), b...)...)
		f.fileData.data = append(f.fileData.data, tail...)
	} else {
		f.fileData.data = append(f.fileData.data[:cur], b...)
		f.fileData.data = append(f.fileData.data, tail...)
	}
	setModTime(f.fileData, time.Now())

	atomic.AddInt64(&f.at, int64(n))
	return
}

func (f *File) WriteAt(b []byte, off int64) (n int, err error) {
	atomic.StoreInt64(&f.at, off)
	return f.Write(b)
}

func (f *File) WriteString(s string) (ret int, err error) {
	return f.Write([]byte(s))
}

func (f *File) Info() *FileInfo {
	return &FileInfo{f.fileData}
}

type FileInfo struct {
	*FileData
}

// Implements os.FileInfo
func (s *FileInfo) Name() string {
	s.Lock()
	_, name := filepath.Split(s.name)
	s.Unlock()
	return name
}
func (s *FileInfo) Mode() os.FileMode {
	s.Lock()
	defer s.Unlock()
	return s.mode
}
func (s *FileInfo) ModTime() time.Time {
	s.Lock()
	defer s.Unlock()
	return s.modtime
}
func (s *FileInfo) IsDir() bool {
	s.Lock()
	defer s.Unlock()
	return s.dir
}
func (s *FileInfo) Sys() interface{} { return nil }
func (s *FileInfo) Size() int64 {
	if s.IsDir() {
		return int64(42)
	}
	s.Lock()
	defer s.Unlock()
	return int64(len(s.data))
}

var (
	ErrFileClosed        = errors.New("File is closed")
	ErrOutOfRange        = errors.New("Out of range")
	ErrTooLarge          = errors.New("Too large")
	ErrFileNotFound      = os.ErrNotExist
	ErrFileExists        = os.ErrExist
	ErrDestinationExists = os.ErrExist
)
