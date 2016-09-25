// +build windows

package system

import (
	"os"
	"time"
)

// StatT type contains status of a file. It contains metadata
// like name, permission, size, etc about a file.
type StatT struct {
	name    string
	size    int64
	mode    os.FileMode
	modTime time.Time
	isDir   bool
}

// Name returns file's name.
func (s StatT) Name() string {
	return s.name
}

// Size returns file's size.
func (s StatT) Size() int64 {
	return s.size
}

// Mode returns file's permission mode.
func (s StatT) Mode() os.FileMode {
	return s.mode
}

// ModTime returns file's last modification time.
func (s StatT) ModTime() time.Time {
	return s.modTime
}

// IsDir returns whether file is actually a directory.
func (s StatT) IsDir() bool {
	return s.isDir
}
