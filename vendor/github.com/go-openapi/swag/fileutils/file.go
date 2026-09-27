// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import "mime/multipart"

// File represents an uploaded file.
//
// Data holds the payload, and Header the multipart metadata.
// File implements [io.ReadCloser] by delegating both methods to Data.
//
// The zero File is not usable: [File.Read] and [File.Close] both panic when Data is nil.
type File struct {
	Data   multipart.File
	Header *multipart.FileHeader
}

// Read reads bytes from the payload.
func (f *File) Read(p []byte) (n int, err error) {
	return f.Data.Read(p)
}

// Close closes the payload.
func (f *File) Close() error {
	return f.Data.Close()
}
