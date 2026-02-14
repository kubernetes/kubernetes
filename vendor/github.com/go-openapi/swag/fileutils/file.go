// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import "mime/multipart"

// File represents an uploaded file.
type File struct {
	Data   multipart.File
	Header *multipart.FileHeader
}

// Read bytes from the file
func (f *File) Read(p []byte) (n int, err error) {
	return f.Data.Read(p)
}

// Close the file
func (f *File) Close() error {
	return f.Data.Close()
}
