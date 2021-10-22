/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package hgfs

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"io"
	"io/ioutil"
	"log"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vmware/govmomi/toolbox/vix"
)

// ArchiveScheme is the default scheme used to register the archive FileHandler
var ArchiveScheme = "archive"

// ArchiveHandler implements a FileHandler for transferring directories.
type ArchiveHandler struct {
	Read  func(*url.URL, *tar.Reader) error
	Write func(*url.URL, *tar.Writer) error
}

// NewArchiveHandler returns a FileHandler implementation for transferring directories using gzip'd tar files.
func NewArchiveHandler() FileHandler {
	return &ArchiveHandler{
		Read:  archiveRead,
		Write: archiveWrite,
	}
}

// Stat implements FileHandler.Stat
func (*ArchiveHandler) Stat(u *url.URL) (os.FileInfo, error) {
	switch u.Query().Get("format") {
	case "", "tar", "tgz":
		// ok
	default:
		log.Printf("unknown archive format: %q", u)
		return nil, vix.Error(vix.InvalidArg)
	}

	return &archive{
		name: u.Path,
		size: math.MaxInt64,
	}, nil
}

// Open implements FileHandler.Open
func (h *ArchiveHandler) Open(u *url.URL, mode int32) (File, error) {
	switch mode {
	case OpenModeReadOnly:
		return h.newArchiveFromGuest(u)
	case OpenModeWriteOnly:
		return h.newArchiveToGuest(u)
	default:
		return nil, os.ErrNotExist
	}
}

// archive implements the hgfs.File and os.FileInfo interfaces.
type archive struct {
	name string
	size int64
	done func() error

	io.Reader
	io.Writer
}

// Name implementation of the os.FileInfo interface method.
func (a *archive) Name() string {
	return a.name
}

// Size implementation of the os.FileInfo interface method.
func (a *archive) Size() int64 {
	return a.size
}

// Mode implementation of the os.FileInfo interface method.
func (a *archive) Mode() os.FileMode {
	return 0600
}

// ModTime implementation of the os.FileInfo interface method.
func (a *archive) ModTime() time.Time {
	return time.Now()
}

// IsDir implementation of the os.FileInfo interface method.
func (a *archive) IsDir() bool {
	return false
}

// Sys implementation of the os.FileInfo interface method.
func (a *archive) Sys() interface{} {
	return nil
}

// The trailer is required since TransferFromGuest requires a Content-Length,
// which toolbox doesn't know ahead of time as the gzip'd tarball never touches the disk.
// HTTP clients need to be aware of this and stop reading when they see the 2nd gzip header.
var gzipHeader = []byte{0x1f, 0x8b, 0x08} // rfc1952 {ID1, ID2, CM}

var gzipTrailer = true

// newArchiveFromGuest returns an hgfs.File implementation to read a directory as a gzip'd tar.
func (h *ArchiveHandler) newArchiveFromGuest(u *url.URL) (File, error) {
	r, w := io.Pipe()

	a := &archive{
		name:   u.Path,
		done:   r.Close,
		Reader: r,
		Writer: w,
	}

	var z io.Writer = w
	var c io.Closer = ioutil.NopCloser(nil)

	switch u.Query().Get("format") {
	case "tgz":
		gz := gzip.NewWriter(w)
		z = gz
		c = gz
	}

	tw := tar.NewWriter(z)

	go func() {
		err := h.Write(u, tw)

		_ = tw.Close()
		_ = c.Close()
		if gzipTrailer {
			_, _ = w.Write(gzipHeader)
		}
		_ = w.CloseWithError(err)
	}()

	return a, nil
}

// newArchiveToGuest returns an hgfs.File implementation to expand a gzip'd tar into a directory.
func (h *ArchiveHandler) newArchiveToGuest(u *url.URL) (File, error) {
	r, w := io.Pipe()

	buf := bufio.NewReader(r)

	a := &archive{
		name:   u.Path,
		Reader: buf,
		Writer: w,
	}

	var cerr error
	var wg sync.WaitGroup

	a.done = func() error {
		_ = w.Close()
		// We need to wait for unpack to finish to complete its work
		// and to propagate the error if any to Close.
		wg.Wait()
		return cerr
	}

	wg.Add(1)
	go func() {
		defer wg.Done()

		c := func() error {
			// Drain the pipe of tar trailer data (two null blocks)
			if cerr == nil {
				_, _ = io.Copy(ioutil.Discard, a.Reader)
			}
			return nil
		}

		header, _ := buf.Peek(len(gzipHeader))

		if bytes.Equal(header, gzipHeader) {
			gz, err := gzip.NewReader(a.Reader)
			if err != nil {
				_ = r.CloseWithError(err)
				cerr = err
				return
			}

			c = gz.Close
			a.Reader = gz
		}

		tr := tar.NewReader(a.Reader)

		cerr = h.Read(u, tr)

		_ = c()
		_ = r.CloseWithError(cerr)
	}()

	return a, nil
}

func (a *archive) Close() error {
	return a.done()
}

// archiveRead writes the contents of the given tar.Reader to the given directory.
func archiveRead(u *url.URL, tr *tar.Reader) error {
	for {
		header, err := tr.Next()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}

		name := filepath.Join(u.Path, header.Name)
		mode := os.FileMode(header.Mode)

		switch header.Typeflag {
		case tar.TypeDir:
			err = os.MkdirAll(name, mode)
		case tar.TypeReg:
			_ = os.MkdirAll(filepath.Dir(name), 0755)

			var f *os.File

			f, err = os.OpenFile(name, os.O_CREATE|os.O_RDWR|os.O_TRUNC, mode)
			if err == nil {
				_, cerr := io.Copy(f, tr)
				err = f.Close()
				if cerr != nil {
					err = cerr
				}
			}
		case tar.TypeSymlink:
			err = os.Symlink(header.Linkname, name)
		}

		// TODO: Uid/Gid may not be meaningful here without some mapping.
		// The other option to consider would be making use of the guest auth user ID.
		// os.Lchown(name, header.Uid, header.Gid)

		if err != nil {
			return err
		}
	}
}

// archiveWrite writes the contents of the given source directory to the given tar.Writer.
func archiveWrite(u *url.URL, tw *tar.Writer) error {
	info, err := os.Stat(u.Path)
	if err != nil {
		return err
	}

	// Note that the VMX will trim any trailing slash.  For example:
	// "/foo/bar/?prefix=bar/" will end up here as "/foo/bar/?prefix=bar"
	// Escape to avoid this: "/for/bar/?prefix=bar%2F"
	prefix := u.Query().Get("prefix")

	dir := u.Path

	f := func(file string, fi os.FileInfo, err error) error {
		if err != nil {
			return filepath.SkipDir
		}

		name := strings.TrimPrefix(file, dir)
		name = strings.TrimPrefix(name, "/")

		if name == "" {
			return nil // this is u.Path itself (which may or may not have a trailing "/")
		}

		if prefix != "" {
			name = prefix + name
		}

		header, _ := tar.FileInfoHeader(fi, name)

		header.Name = name

		if header.Typeflag == tar.TypeDir {
			header.Name += "/"
		}

		var f *os.File

		if header.Typeflag == tar.TypeReg && fi.Size() != 0 {
			f, err = os.Open(file)
			if err != nil {
				if os.IsPermission(err) {
					return nil
				}
				return err
			}
		}

		_ = tw.WriteHeader(header)

		if f != nil {
			_, err = io.Copy(tw, f)
			_ = f.Close()
		}

		return err
	}

	if info.IsDir() {
		return filepath.Walk(u.Path, f)
	}

	dir = filepath.Dir(dir)

	return f(u.Path, info, nil)
}
