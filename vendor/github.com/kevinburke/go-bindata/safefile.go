// Copyright 2013 Dmitry Chestnykh. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Some modifications made: API's were made internal to the package. os.Rename
// hack for old Windows/Plan 9 builds removed; if you are using those systems,
// you won't get atomic rewrites, sorry.

package bindata

import (
	"crypto/rand"
	"encoding/base32"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// ErrAlreadyCommitted error is returned when calling Commit on a file that
// has been already successfully committed.
var errAlreadyCommitted = errors.New("file already committed")

type safefileFile struct {
	*os.File
	origName    string
	closeFunc   func(*safefileFile) error
	isClosed    bool // if true, temporary file has been closed, but not renamed
	isCommitted bool // if true, the file has been successfully committed
}

func makeTempName(origname, prefix string) (tempname string, err error) {
	origname = filepath.Clean(origname)
	if len(origname) == 0 || origname[len(origname)-1] == filepath.Separator {
		return "", os.ErrInvalid
	}
	// Generate 10 random bytes.
	// This gives 80 bits of entropy, good enough
	// for making temporary file name unpredictable.
	var rnd [10]byte
	if _, err := rand.Read(rnd[:]); err != nil {
		return "", err
	}
	name := prefix + "-" + strings.ToLower(base32.StdEncoding.EncodeToString(rnd[:])) + ".tmp"
	return filepath.Join(filepath.Dir(origname), name), nil
}

// safefileCreate creates a temporary file in the same directory as filename,
// which will be renamed to the given filename when calling Commit.
func safefileCreate(filename string, perm os.FileMode) (*safefileFile, error) {
	for {
		tempname, err := makeTempName(filename, "sf")
		if err != nil {
			return nil, err
		}
		f, err := os.OpenFile(tempname, os.O_RDWR|os.O_CREATE|os.O_EXCL, perm)
		if err != nil {
			if os.IsExist(err) {
				continue
			}
			return nil, err
		}
		return &safefileFile{
			File:      f,
			origName:  filename,
			closeFunc: closeUncommitted,
		}, nil
	}
}

// OrigName returns the original filename given to Create.
func (f *safefileFile) OrigName() string {
	return f.origName
}

// Close closes temporary file and removes it.
// If the file has been committed, Close is no-op.
func (f *safefileFile) Close() error {
	return f.closeFunc(f)
}

func closeUncommitted(f *safefileFile) error {
	err0 := f.File.Close()
	err1 := os.Remove(f.Name())
	f.closeFunc = closeAgainError
	if err0 != nil {
		return err0
	}
	return err1
}

func closeAfterFailedRename(f *safefileFile) error {
	// Remove temporary file.
	//
	// The note from Commit function applies here too, as we may be
	// removing a different file. However, since we rely on our temporary
	// names being unpredictable, this should not be a concern.
	f.closeFunc = closeAgainError
	return os.Remove(f.Name())
}

func closeCommitted(f *safefileFile) error {
	// noop
	return nil
}

func closeAgainError(f *safefileFile) error {
	return os.ErrInvalid
}

// Commit safely commits data into the original file by syncing temporary
// file to disk, closing it and renaming to the original file name.
//
// In case of success, the temporary file is closed and no longer exists
// on disk. It is safe to call Close after Commit: the operation will do
// nothing.
//
// In case of error, the temporary file is still opened and exists on disk;
// it must be closed by callers by calling Close or by trying to commit again.

// Note that when trying to Commit again after a failed Commit when the file
// has been closed, but not renamed to its original name (the new commit will
// try again to rename it), safefile cannot guarantee that the temporary file
// has not been changed, or that it is the same temporary file we were dealing
// with.  However, since the temporary name is unpredictable, it is unlikely
// that this happened accidentally. If complete atomicity is needed, do not
// Commit again after error, write the file again.
func (f *safefileFile) Commit() error {
	if f.isCommitted {
		return errAlreadyCommitted
	}
	if !f.isClosed {
		// Sync to disk.
		err := f.Sync()
		if err != nil {
			return err
		}
		// Close underlying os.File.
		err = f.File.Close()
		if err != nil {
			return err
		}
		f.isClosed = true
	}
	err := os.Rename(f.Name(), f.origName)
	if err != nil {
		f.closeFunc = closeAfterFailedRename
		return err
	}
	f.closeFunc = closeCommitted
	f.isCommitted = true
	return nil
}

// WriteFile is a safe analog of ioutil.WriteFile.
func safefileWriteFile(filename string, data []byte, perm os.FileMode) error {
	f, err := safefileCreate(filename, perm)
	if err != nil {
		return err
	}
	defer f.Close()
	n, err := f.Write(data)
	if err != nil {
		return err
	}
	if err == nil && n < len(data) {
		err = io.ErrShortWrite
		return err
	}
	return f.Commit()
}
