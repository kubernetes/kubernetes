package afero

import (
	"io"
	"os"
	"path/filepath"
	"syscall"
)

// The UnionFile implements the afero.File interface and will be returned
// when reading a directory present at least in the overlay or opening a file
// for writing.
//
// The calls to
// Readdir() and Readdirnames() merge the file os.FileInfo / names from the
// base and the overlay - for files present in both layers, only those
// from the overlay will be used.
//
// When opening files for writing (Create() / OpenFile() with the right flags)
// the operations will be done in both layers, starting with the overlay. A
// successful read in the overlay will move the cursor position in the base layer
// by the number of bytes read.
type UnionFile struct {
	base  File
	layer File
	off   int
	files []os.FileInfo
}

func (f *UnionFile) Close() error {
	// first close base, so we have a newer timestamp in the overlay. If we'd close
	// the overlay first, we'd get a cacheStale the next time we access this file
	// -> cache would be useless ;-)
	if f.base != nil {
		f.base.Close()
	}
	if f.layer != nil {
		return f.layer.Close()
	}
	return BADFD
}

func (f *UnionFile) Read(s []byte) (int, error) {
	if f.layer != nil {
		n, err := f.layer.Read(s)
		if (err == nil || err == io.EOF) && f.base != nil {
			// advance the file position also in the base file, the next
			// call may be a write at this position (or a seek with SEEK_CUR)
			if _, seekErr := f.base.Seek(int64(n), os.SEEK_CUR); seekErr != nil {
				// only overwrite err in case the seek fails: we need to
				// report an eventual io.EOF to the caller
				err = seekErr
			}
		}
		return n, err
	}
	if f.base != nil {
		return f.base.Read(s)
	}
	return 0, BADFD
}

func (f *UnionFile) ReadAt(s []byte, o int64) (int, error) {
	if f.layer != nil {
		n, err := f.layer.ReadAt(s, o)
		if (err == nil || err == io.EOF) && f.base != nil {
			_, err = f.base.Seek(o+int64(n), os.SEEK_SET)
		}
		return n, err
	}
	if f.base != nil {
		return f.base.ReadAt(s, o)
	}
	return 0, BADFD
}

func (f *UnionFile) Seek(o int64, w int) (pos int64, err error) {
	if f.layer != nil {
		pos, err = f.layer.Seek(o, w)
		if (err == nil || err == io.EOF) && f.base != nil {
			_, err = f.base.Seek(o, w)
		}
		return pos, err
	}
	if f.base != nil {
		return f.base.Seek(o, w)
	}
	return 0, BADFD
}

func (f *UnionFile) Write(s []byte) (n int, err error) {
	if f.layer != nil {
		n, err = f.layer.Write(s)
		if err == nil && f.base != nil { // hmm, do we have fixed size files where a write may hit the EOF mark?
			_, err = f.base.Write(s)
		}
		return n, err
	}
	if f.base != nil {
		return f.base.Write(s)
	}
	return 0, BADFD
}

func (f *UnionFile) WriteAt(s []byte, o int64) (n int, err error) {
	if f.layer != nil {
		n, err = f.layer.WriteAt(s, o)
		if err == nil && f.base != nil {
			_, err = f.base.WriteAt(s, o)
		}
		return n, err
	}
	if f.base != nil {
		return f.base.WriteAt(s, o)
	}
	return 0, BADFD
}

func (f *UnionFile) Name() string {
	if f.layer != nil {
		return f.layer.Name()
	}
	return f.base.Name()
}

// Readdir will weave the two directories together and
// return a single view of the overlayed directories
func (f *UnionFile) Readdir(c int) (ofi []os.FileInfo, err error) {
	if f.off == 0 {
		var files = make(map[string]os.FileInfo)
		var rfi []os.FileInfo
		if f.layer != nil {
			rfi, err = f.layer.Readdir(-1)
			if err != nil {
				return nil, err
			}
			for _, fi := range rfi {
				files[fi.Name()] = fi
			}
		}

		if f.base != nil {
			rfi, err = f.base.Readdir(-1)
			if err != nil {
				return nil, err
			}
			for _, fi := range rfi {
				if _, exists := files[fi.Name()]; !exists {
					files[fi.Name()] = fi
				}
			}
		}
		for _, fi := range files {
			f.files = append(f.files, fi)
		}
	}
	if c == -1 {
		return f.files[f.off:], nil
	}
	defer func() { f.off += c }()
	return f.files[f.off:c], nil
}

func (f *UnionFile) Readdirnames(c int) ([]string, error) {
	rfi, err := f.Readdir(c)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, fi := range rfi {
		names = append(names, fi.Name())
	}
	return names, nil
}

func (f *UnionFile) Stat() (os.FileInfo, error) {
	if f.layer != nil {
		return f.layer.Stat()
	}
	if f.base != nil {
		return f.base.Stat()
	}
	return nil, BADFD
}

func (f *UnionFile) Sync() (err error) {
	if f.layer != nil {
		err = f.layer.Sync()
		if err == nil && f.base != nil {
			err = f.base.Sync()
		}
		return err
	}
	if f.base != nil {
		return f.base.Sync()
	}
	return BADFD
}

func (f *UnionFile) Truncate(s int64) (err error) {
	if f.layer != nil {
		err = f.layer.Truncate(s)
		if err == nil && f.base != nil {
			err = f.base.Truncate(s)
		}
		return err
	}
	if f.base != nil {
		return f.base.Truncate(s)
	}
	return BADFD
}

func (f *UnionFile) WriteString(s string) (n int, err error) {
	if f.layer != nil {
		n, err = f.layer.WriteString(s)
		if err == nil && f.base != nil {
			_, err = f.base.WriteString(s)
		}
		return n, err
	}
	if f.base != nil {
		return f.base.WriteString(s)
	}
	return 0, BADFD
}

func copyToLayer(base Fs, layer Fs, name string) error {
	bfh, err := base.Open(name)
	if err != nil {
		return err
	}
	defer bfh.Close()

	// First make sure the directory exists
	exists, err := Exists(layer, filepath.Dir(name))
	if err != nil {
		return err
	}
	if !exists {
		err = layer.MkdirAll(filepath.Dir(name), 0777) // FIXME?
		if err != nil {
			return err
		}
	}

	// Create the file on the overlay
	lfh, err := layer.Create(name)
	if err != nil {
		return err
	}
	n, err := io.Copy(lfh, bfh)
	if err != nil {
		// If anything fails, clean up the file
		layer.Remove(name)
		lfh.Close()
		return err
	}

	bfi, err := bfh.Stat()
	if err != nil || bfi.Size() != n {
		layer.Remove(name)
		lfh.Close()
		return syscall.EIO
	}

	err = lfh.Close()
	if err != nil {
		layer.Remove(name)
		lfh.Close()
		return err
	}
	return layer.Chtimes(name, bfi.ModTime(), bfi.ModTime())
}
