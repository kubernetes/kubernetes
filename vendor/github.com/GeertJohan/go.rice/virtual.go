package rice

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/GeertJohan/go.rice/embedded"
)

//++ TODO: IDEA: merge virtualFile and virtualDir, this decreases work done by rice.File

// Error indicating some function is not implemented yet (but available to satisfy an interface)
var ErrNotImplemented = errors.New("not implemented yet")

// virtualFile is a 'stateful' virtual file.
// virtualFile wraps an *EmbeddedFile for a call to Box.Open() and virtualizes 'read cursor' (offset) and 'closing'.
// virtualFile is only internally visible and should be exposed through rice.File
type virtualFile struct {
	*embedded.EmbeddedFile       // the actual embedded file, embedded to obtain methods
	offset                 int64 // read position on the virtual file
	closed                 bool  // closed when true
}

// create a new virtualFile for given EmbeddedFile
func newVirtualFile(ef *embedded.EmbeddedFile) *virtualFile {
	vf := &virtualFile{
		EmbeddedFile: ef,
		offset:       0,
		closed:       false,
	}
	return vf
}

//++ TODO check for nil pointers in all these methods. When so: return os.PathError with Err: os.ErrInvalid

func (vf *virtualFile) close() error {
	if vf.closed {
		return &os.PathError{
			Op:   "close",
			Path: vf.EmbeddedFile.Filename,
			Err:  errors.New("already closed"),
		}
	}
	vf.EmbeddedFile = nil
	vf.closed = true
	return nil
}

func (vf *virtualFile) stat() (os.FileInfo, error) {
	if vf.closed {
		return nil, &os.PathError{
			Op:   "stat",
			Path: vf.EmbeddedFile.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}
	return (*embeddedFileInfo)(vf.EmbeddedFile), nil
}

func (vf *virtualFile) readdir(count int) ([]os.FileInfo, error) {
	if vf.closed {
		return nil, &os.PathError{
			Op:   "readdir",
			Path: vf.EmbeddedFile.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}
	//TODO: return proper error for a readdir() call on a file
	return nil, ErrNotImplemented
}

func (vf *virtualFile) read(bts []byte) (int, error) {
	if vf.closed {
		return 0, &os.PathError{
			Op:   "read",
			Path: vf.EmbeddedFile.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}

	end := vf.offset + int64(len(bts))

	if end >= int64(len(vf.Content)) {
		// end of file, so return what we have + EOF
		n := copy(bts, vf.Content[vf.offset:])
		vf.offset = 0
		return n, io.EOF
	}

	n := copy(bts, vf.Content[vf.offset:end])
	vf.offset += int64(n)
	return n, nil

}

func (vf *virtualFile) seek(offset int64, whence int) (int64, error) {
	if vf.closed {
		return 0, &os.PathError{
			Op:   "seek",
			Path: vf.EmbeddedFile.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}
	var e error

	//++ TODO: check if this is correct implementation for seek
	switch whence {
	case os.SEEK_SET:
		//++ check if new offset isn't out of bounds, set e when it is, then break out of switch
		vf.offset = offset
	case os.SEEK_CUR:
		//++ check if new offset isn't out of bounds, set e when it is, then break out of switch
		vf.offset += offset
	case os.SEEK_END:
		//++ check if new offset isn't out of bounds, set e when it is, then break out of switch
		vf.offset = int64(len(vf.EmbeddedFile.Content)) - offset
	}

	if e != nil {
		return 0, &os.PathError{
			Op:   "seek",
			Path: vf.Filename,
			Err:  e,
		}
	}

	return vf.offset, nil
}

// virtualDir is a 'stateful' virtual directory.
// virtualDir wraps an *EmbeddedDir for a call to Box.Open() and virtualizes 'closing'.
// virtualDir is only internally visible and should be exposed through rice.File
type virtualDir struct {
	*embedded.EmbeddedDir
	offset int // readdir position on the directory
	closed bool
}

// create a new virtualDir for given EmbeddedDir
func newVirtualDir(ed *embedded.EmbeddedDir) *virtualDir {
	vd := &virtualDir{
		EmbeddedDir: ed,
		offset:      0,
		closed:      false,
	}
	return vd
}

func (vd *virtualDir) close() error {
	//++ TODO: needs sync mutex?
	if vd.closed {
		return &os.PathError{
			Op:   "close",
			Path: vd.EmbeddedDir.Filename,
			Err:  errors.New("already closed"),
		}
	}
	vd.closed = true
	return nil
}

func (vd *virtualDir) stat() (os.FileInfo, error) {
	if vd.closed {
		return nil, &os.PathError{
			Op:   "stat",
			Path: vd.EmbeddedDir.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}
	return (*embeddedDirInfo)(vd.EmbeddedDir), nil
}

func (vd *virtualDir) readdir(n int) (fi []os.FileInfo, err error) {

	if vd.closed {
		return nil, &os.PathError{
			Op:   "readdir",
			Path: vd.EmbeddedDir.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}

	// Build up the array of our contents
	var files []os.FileInfo

	// Add the child directories
	for _, child := range vd.ChildDirs {
		child.Filename = filepath.Base(child.Filename)
		files = append(files, (*embeddedDirInfo)(child))
	}

	// Add the child files
	for _, child := range vd.ChildFiles {
		child.Filename = filepath.Base(child.Filename)
		files = append(files, (*embeddedFileInfo)(child))
	}

	// Sort it by filename (lexical order)
	sort.Sort(SortByName(files))

	// Return all contents if that's what is requested
	if n <= 0 {
		vd.offset = 0
		return files, nil
	}

	// If user has requested past the end of our list
	// return what we can and send an EOF
	if vd.offset+n >= len(files) {
		offset := vd.offset
		vd.offset = 0
		return files[offset:], io.EOF
	}

	offset := vd.offset
	vd.offset += n
	return files[offset : offset+n], nil

}

func (vd *virtualDir) read(bts []byte) (int, error) {
	if vd.closed {
		return 0, &os.PathError{
			Op:   "read",
			Path: vd.EmbeddedDir.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}
	return 0, &os.PathError{
		Op:   "read",
		Path: vd.EmbeddedDir.Filename,
		Err:  errors.New("is a directory"),
	}
}

func (vd *virtualDir) seek(offset int64, whence int) (int64, error) {
	if vd.closed {
		return 0, &os.PathError{
			Op:   "seek",
			Path: vd.EmbeddedDir.Filename,
			Err:  errors.New("bad file descriptor"),
		}
	}
	return 0, &os.PathError{
		Op:   "seek",
		Path: vd.Filename,
		Err:  errors.New("is a directory"),
	}
}
