package rice

import (
	"os"
	"time"

	"github.com/GeertJohan/go.rice/embedded"
)

// re-type to make exported methods invisible to user (godoc)
// they're not required for the user
// embeddedDirInfo implements os.FileInfo
type embeddedDirInfo embedded.EmbeddedDir

// Name returns the base name of the directory
// (implementing os.FileInfo)
func (ed *embeddedDirInfo) Name() string {
	return ed.Filename
}

// Size always returns 0
// (implementing os.FileInfo)
func (ed *embeddedDirInfo) Size() int64 {
	return 0
}

// Mode returns the file mode bits
// (implementing os.FileInfo)
func (ed *embeddedDirInfo) Mode() os.FileMode {
	return os.FileMode(0555 | os.ModeDir) // dr-xr-xr-x
}

// ModTime returns the modification time
// (implementing os.FileInfo)
func (ed *embeddedDirInfo) ModTime() time.Time {
	return ed.DirModTime
}

// IsDir returns the abbreviation for Mode().IsDir() (always true)
// (implementing os.FileInfo)
func (ed *embeddedDirInfo) IsDir() bool {
	return true
}

// Sys returns the underlying data source (always nil)
// (implementing os.FileInfo)
func (ed *embeddedDirInfo) Sys() interface{} {
	return nil
}

// re-type to make exported methods invisible to user (godoc)
// they're not required for the user
// embeddedFileInfo implements os.FileInfo
type embeddedFileInfo embedded.EmbeddedFile

// Name returns the base name of the file
// (implementing os.FileInfo)
func (ef *embeddedFileInfo) Name() string {
	return ef.Filename
}

// Size returns the length in bytes for regular files; system-dependent for others
// (implementing os.FileInfo)
func (ef *embeddedFileInfo) Size() int64 {
	return int64(len(ef.Content))
}

// Mode returns the file mode bits
// (implementing os.FileInfo)
func (ef *embeddedFileInfo) Mode() os.FileMode {
	return os.FileMode(0555) // r-xr-xr-x
}

// ModTime returns the modification time
// (implementing os.FileInfo)
func (ef *embeddedFileInfo) ModTime() time.Time {
	return ef.FileModTime
}

// IsDir returns the abbreviation for Mode().IsDir() (always false)
// (implementing os.FileInfo)
func (ef *embeddedFileInfo) IsDir() bool {
	return false
}

// Sys returns the underlying data source (always nil)
// (implementing os.FileInfo)
func (ef *embeddedFileInfo) Sys() interface{} {
	return nil
}
