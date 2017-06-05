package driver

import "time"

// FileInfo returns information about a given path. Inspired by os.FileInfo,
// it elides the base name method for a full path instead.
type FileInfo interface {
	// Path provides the full path of the target of this file info.
	Path() string

	// Size returns current length in bytes of the file. The return value can
	// be used to write to the end of the file at path. The value is
	// meaningless if IsDir returns true.
	Size() int64

	// ModTime returns the modification time for the file. For backends that
	// don't have a modification time, the creation time should be returned.
	ModTime() time.Time

	// IsDir returns true if the path is a directory.
	IsDir() bool
}

// NOTE(stevvooe): The next two types, FileInfoFields and FileInfoInternal
// should only be used by storagedriver implementations. They should moved to
// a "driver" package, similar to database/sql.

// FileInfoFields provides the exported fields for implementing FileInfo
// interface in storagedriver implementations. It should be used with
// InternalFileInfo.
type FileInfoFields struct {
	// Path provides the full path of the target of this file info.
	Path string

	// Size is current length in bytes of the file. The value of this field
	// can be used to write to the end of the file at path. The value is
	// meaningless if IsDir is set to true.
	Size int64

	// ModTime returns the modification time for the file. For backends that
	// don't have a modification time, the creation time should be returned.
	ModTime time.Time

	// IsDir returns true if the path is a directory.
	IsDir bool
}

// FileInfoInternal implements the FileInfo interface. This should only be
// used by storagedriver implementations that don't have a specialized
// FileInfo type.
type FileInfoInternal struct {
	FileInfoFields
}

var _ FileInfo = FileInfoInternal{}
var _ FileInfo = &FileInfoInternal{}

// Path provides the full path of the target of this file info.
func (fi FileInfoInternal) Path() string {
	return fi.FileInfoFields.Path
}

// Size returns current length in bytes of the file. The return value can
// be used to write to the end of the file at path. The value is
// meaningless if IsDir returns true.
func (fi FileInfoInternal) Size() int64 {
	return fi.FileInfoFields.Size
}

// ModTime returns the modification time for the file. For backends that
// don't have a modification time, the creation time should be returned.
func (fi FileInfoInternal) ModTime() time.Time {
	return fi.FileInfoFields.ModTime
}

// IsDir returns true if the path is a directory.
func (fi FileInfoInternal) IsDir() bool {
	return fi.FileInfoFields.IsDir
}
