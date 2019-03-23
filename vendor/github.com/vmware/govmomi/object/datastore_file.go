/*
Copyright (c) 2016-2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package object

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"sync"
	"time"

	"github.com/vmware/govmomi/vim25/soap"
)

// DatastoreFile implements io.Reader, io.Seeker and io.Closer interfaces for datastore file access.
type DatastoreFile struct {
	d    Datastore
	ctx  context.Context
	name string

	buf    io.Reader
	body   io.ReadCloser
	length int64
	offset struct {
		read, seek int64
	}
}

// Open opens the named file relative to the Datastore.
func (d Datastore) Open(ctx context.Context, name string) (*DatastoreFile, error) {
	return &DatastoreFile{
		d:      d,
		name:   name,
		length: -1,
		ctx:    ctx,
	}, nil
}

// Read reads up to len(b) bytes from the DatastoreFile.
func (f *DatastoreFile) Read(b []byte) (int, error) {
	if f.offset.read != f.offset.seek {
		// A Seek() call changed the offset, we need to issue a new GET
		_ = f.Close()

		f.offset.read = f.offset.seek
	} else if f.buf != nil {
		// f.buf + f behaves like an io.MultiReader
		n, err := f.buf.Read(b)
		if err == io.EOF {
			f.buf = nil // buffer has been drained
		}
		if n > 0 {
			return n, nil
		}
	}

	body, err := f.get()
	if err != nil {
		return 0, err
	}

	n, err := body.Read(b)

	f.offset.read += int64(n)
	f.offset.seek += int64(n)

	return n, err
}

// Close closes the DatastoreFile.
func (f *DatastoreFile) Close() error {
	var err error

	if f.body != nil {
		err = f.body.Close()
		f.body = nil
	}

	f.buf = nil

	return err
}

// Seek sets the offset for the next Read on the DatastoreFile.
func (f *DatastoreFile) Seek(offset int64, whence int) (int64, error) {
	switch whence {
	case io.SeekStart:
	case io.SeekCurrent:
		offset += f.offset.seek
	case io.SeekEnd:
		if f.length < 0 {
			_, err := f.Stat()
			if err != nil {
				return 0, err
			}
		}
		offset += f.length
	default:
		return 0, errors.New("Seek: invalid whence")
	}

	// allow negative SeekStart for initial Range request
	if offset < 0 {
		return 0, errors.New("Seek: invalid offset")
	}

	f.offset.seek = offset

	return offset, nil
}

type fileStat struct {
	file   *DatastoreFile
	header http.Header
}

func (s *fileStat) Name() string {
	return path.Base(s.file.name)
}

func (s *fileStat) Size() int64 {
	return s.file.length
}

func (s *fileStat) Mode() os.FileMode {
	return 0
}

func (s *fileStat) ModTime() time.Time {
	return time.Now() // no Last-Modified
}

func (s *fileStat) IsDir() bool {
	return false
}

func (s *fileStat) Sys() interface{} {
	return s.header
}

func statusError(res *http.Response) error {
	if res.StatusCode == http.StatusNotFound {
		return os.ErrNotExist
	}
	return errors.New(res.Status)
}

// Stat returns the os.FileInfo interface describing file.
func (f *DatastoreFile) Stat() (os.FileInfo, error) {
	// TODO: consider using Datastore.Stat() instead
	u, p, err := f.d.downloadTicket(f.ctx, f.name, &soap.Download{Method: "HEAD"})
	if err != nil {
		return nil, err
	}

	res, err := f.d.Client().DownloadRequest(f.ctx, u, p)
	if err != nil {
		return nil, err
	}

	if res.StatusCode != http.StatusOK {
		return nil, statusError(res)
	}

	f.length = res.ContentLength

	return &fileStat{f, res.Header}, nil
}

func (f *DatastoreFile) get() (io.Reader, error) {
	if f.body != nil {
		return f.body, nil
	}

	u, p, err := f.d.downloadTicket(f.ctx, f.name, nil)
	if err != nil {
		return nil, err
	}

	if f.offset.read != 0 {
		p.Headers = map[string]string{
			"Range": fmt.Sprintf("bytes=%d-", f.offset.read),
		}
	}

	res, err := f.d.Client().DownloadRequest(f.ctx, u, p)
	if err != nil {
		return nil, err
	}

	switch res.StatusCode {
	case http.StatusOK:
		f.length = res.ContentLength
	case http.StatusPartialContent:
		var start, end int
		cr := res.Header.Get("Content-Range")
		_, err = fmt.Sscanf(cr, "bytes %d-%d/%d", &start, &end, &f.length)
		if err != nil {
			f.length = -1
		}
	case http.StatusRequestedRangeNotSatisfiable:
		// ok: Read() will return io.EOF
	default:
		return nil, statusError(res)
	}

	if f.length < 0 {
		_ = res.Body.Close()
		return nil, errors.New("unable to determine file size")
	}

	f.body = res.Body

	return f.body, nil
}

func lastIndexLines(s []byte, line *int, include func(l int, m string) bool) (int64, bool) {
	i := len(s) - 1
	done := false

	for i > 0 {
		o := bytes.LastIndexByte(s[:i], '\n')
		if o < 0 {
			break
		}

		msg := string(s[o+1 : i+1])
		if !include(*line, msg) {
			done = true
			break
		} else {
			i = o
			*line++
		}
	}

	return int64(i), done
}

// Tail seeks to the position of the last N lines of the file.
func (f *DatastoreFile) Tail(n int) error {
	return f.TailFunc(n, func(line int, _ string) bool { return n > line })
}

// TailFunc will seek backwards in the datastore file until it hits a line that does
// not satisfy the supplied `include` function.
func (f *DatastoreFile) TailFunc(lines int, include func(line int, message string) bool) error {
	// Read the file in reverse using bsize chunks
	const bsize = int64(1024 * 16)

	fsize, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		return err
	}

	if lines == 0 {
		return nil
	}

	chunk := int64(-1)

	buf := bytes.NewBuffer(make([]byte, 0, bsize))
	line := 0

	for {
		var eof bool
		var pos int64

		nread := bsize

		offset := chunk * bsize
		remain := fsize + offset

		if remain < 0 {
			if pos, err = f.Seek(0, io.SeekStart); err != nil {
				return err
			}

			nread = bsize + remain
			eof = true
		} else {
			if pos, err = f.Seek(offset, io.SeekEnd); err != nil {
				return err
			}
		}

		if _, err = io.CopyN(buf, f, nread); err != nil {
			if err != io.EOF {
				return err
			}
		}

		b := buf.Bytes()
		idx, done := lastIndexLines(b, &line, include)

		if done {
			if chunk == -1 {
				// We found all N lines in the last chunk of the file.
				// The seek offset is also now at the current end of file.
				// Save this buffer to avoid another GET request when Read() is called.
				buf.Next(int(idx + 1))
				f.buf = buf
				return nil
			}

			if _, err = f.Seek(pos+idx+1, io.SeekStart); err != nil {
				return err
			}

			break
		}

		if eof {
			if remain < 0 {
				// We found < N lines in the entire file, so seek to the start.
				_, _ = f.Seek(0, io.SeekStart)
			}
			break
		}

		chunk--
		buf.Reset()
	}

	return nil
}

type followDatastoreFile struct {
	r *DatastoreFile
	c chan struct{}
	i time.Duration
	o sync.Once
}

// Read reads up to len(b) bytes from the DatastoreFile being followed.
// This method will block until data is read, an error other than io.EOF is returned or Close() is called.
func (f *followDatastoreFile) Read(p []byte) (int, error) {
	offset := f.r.offset.seek
	stop := false

	for {
		n, err := f.r.Read(p)
		if err != nil && err == io.EOF {
			_ = f.r.Close() // GET request body has been drained.
			if stop {
				return n, err
			}
			err = nil
		}

		if n > 0 {
			return n, err
		}

		select {
		case <-f.c:
			// Wake up and stop polling once the body has been drained
			stop = true
		case <-time.After(f.i):
		}

		info, serr := f.r.Stat()
		if serr != nil {
			// Return EOF rather than 404 if the file goes away
			if serr == os.ErrNotExist {
				_ = f.r.Close()
				return 0, io.EOF
			}
			return 0, serr
		}

		if info.Size() < offset {
			// assume file has be truncated
			offset, err = f.r.Seek(0, io.SeekStart)
			if err != nil {
				return 0, err
			}
		}
	}
}

// Close will stop Follow polling and close the underlying DatastoreFile.
func (f *followDatastoreFile) Close() error {
	f.o.Do(func() { close(f.c) })
	return nil
}

// Follow returns an io.ReadCloser to stream the file contents as data is appended.
func (f *DatastoreFile) Follow(interval time.Duration) io.ReadCloser {
	return &followDatastoreFile{
		r: f,
		c: make(chan struct{}),
		i: interval,
	}
}
