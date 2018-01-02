// Copyright 2015 The rkt Authors
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

package image

import (
	"crypto/sha512"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/coreos/rkt/pkg/lock"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/hashicorp/errwrap"

	"github.com/coreos/ioprogress"
)

// writeSyncer is an interface that wraps io.Writer and a Sync method.
type writeSyncer interface {
	io.Writer
	Sync() error
}

// readSeekCloser is an interface that wraps io.ReadSeeker and
// io.Closer
type readSeekCloser interface {
	io.ReadSeeker
	io.Closer
}

type nopReadSeekCloser struct {
	io.ReadSeeker
}

func (nopReadSeekCloser) Close() error { return nil }

// NopReadSeekCloser wraps the given ReadSeeker
// and returns one that does nothing when Close() is being invoked.
func NopReadSeekCloser(rs io.ReadSeeker) readSeekCloser {
	return nopReadSeekCloser{rs}
}

// getIoProgressReader returns a reader that wraps the HTTP response
// body, so it prints a pretty progress bar when reading data from it.
func getIoProgressReader(label string, res *http.Response) io.Reader {
	prefix := "Downloading " + label
	fmtBytesSize := 18
	barSize := int64(80 - len(prefix) - fmtBytesSize)
	bar := ioprogress.DrawTextFormatBarForW(barSize, os.Stderr)
	fmtfunc := func(progress, total int64) string {
		// Content-Length is set to -1 when unknown.
		if total == -1 {
			return fmt.Sprintf(
				"%s: %v of an unknown total size",
				prefix,
				ioprogress.ByteUnitStr(progress),
			)
		}
		return fmt.Sprintf(
			"%s: %s %s",
			prefix,
			bar(progress, total),
			ioprogress.DrawTextFormatBytes(progress, total),
		)
	}
	return &ioprogress.Reader{
		Reader:       res.Body,
		Size:         res.ContentLength,
		DrawFunc:     ioprogress.DrawTerminalf(os.Stderr, fmtfunc),
		DrawInterval: time.Second,
	}
}

// removeOnClose is a wrapper around os.File that removes the file
// when closing it. removeOnClose implements a readSeekCloser
// interface.
type removeOnClose struct {
	// File is a wrapped os.File
	File *os.File
}

func (f *removeOnClose) Read(p []byte) (int, error) {
	return f.File.Read(p)
}

func (f *removeOnClose) Seek(offset int64, whence int) (int64, error) {
	return f.File.Seek(offset, whence)
}

// Close closes the file and then removes it from disk. No error is
// returned if the file did not exist at the point of removal.
func (f *removeOnClose) Close() error {
	if f == nil || f.File == nil {
		return nil
	}
	name := f.File.Name()
	if err := f.File.Close(); err != nil {
		return err
	}
	if err := os.Remove(name); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// getTmpROC returns a removeOnClose instance wrapping a temporary
// file provided by the passed store. The actual file name is based on
// a hash of the passed path.
func getTmpROC(s *imagestore.Store, path string) (*removeOnClose, error) {
	h := sha512.New()
	h.Write([]byte(path))
	pathHash := s.HashToKey(h)

	tmp, err := s.TmpNamedFile(pathHash)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error setting up temporary file"), err)
	}

	// let's lock the file to avoid concurrent writes to the temporary file, it
	// will go away when removing the temp file
	_, err = lock.TryExclusiveLock(tmp.Name(), lock.RegFile)
	if err != nil {
		if err != lock.ErrLocked {
			return nil, errwrap.Wrap(errors.New("failed to lock temporary file"), err)
		}
		log.Printf("another rkt instance is downloading this file, waiting...")
		_, err = lock.ExclusiveLock(tmp.Name(), lock.RegFile)
		if err != nil {
			return nil, errwrap.Wrap(errors.New("failed to lock temporary file"), err)
		}
	}

	return &removeOnClose{File: tmp}, nil
}
