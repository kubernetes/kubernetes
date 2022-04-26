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
	Base   File
	Layer  File
	Merger DirsMerger
	off    int
	files  []os.FileInfo
}

func (f *UnionFile) Close() error {
	// first close base, so we have a newer timestamp in the overlay. If we'd close
	// the overlay first, we'd get a cacheStale the next time we access this file
	// -> cache would be useless ;-)
	if f.Base != nil {
		f.Base.Close()
	}
	if f.Layer != nil {
		return f.Layer.Close()
	}
	return BADFD
}

func (f *UnionFile) Read(s []byte) (int, error) {
	if f.Layer != nil {
		n, err := f.Layer.Read(s)
		if (err == nil || err == io.EOF) && f.Base != nil {
			// advance the file position also in the base file, the next
			// call may be a write at this position (or a seek with SEEK_CUR)
			if _, seekErr := f.Base.Seek(int64(n), os.SEEK_CUR); seekErr != nil {
				// only overwrite err in case the seek fails: we need to
				// report an eventual io.EOF to the caller
				err = seekErr
			}
		}
		return n, err
	}
	if f.Base != nil {
		return f.Base.Read(s)
	}
	return 0, BADFD
}

func (f *UnionFile) ReadAt(s []byte, o int64) (int, error) {
	if f.Layer != nil {
		n, err := f.Layer.ReadAt(s, o)
		if (err == nil || err == io.EOF) && f.Base != nil {
			_, err = f.Base.Seek(o+int64(n), os.SEEK_SET)
		}
		return n, err
	}
	if f.Base != nil {
		return f.Base.ReadAt(s, o)
	}
	return 0, BADFD
}

func (f *UnionFile) Seek(o int64, w int) (pos int64, err error) {
	if f.Layer != nil {
		pos, err = f.Layer.Seek(o, w)
		if (err == nil || err == io.EOF) && f.Base != nil {
			_, err = f.Base.Seek(o, w)
		}
		return pos, err
	}
	if f.Base != nil {
		return f.Base.Seek(o, w)
	}
	return 0, BADFD
}

func (f *UnionFile) Write(s []byte) (n int, err error) {
	if f.Layer != nil {
		n, err = f.Layer.Write(s)
		if err == nil && f.Base != nil { // hmm, do we have fixed size files where a write may hit the EOF mark?
			_, err = f.Base.Write(s)
		}
		return n, err
	}
	if f.Base != nil {
		return f.Base.Write(s)
	}
	return 0, BADFD
}

func (f *UnionFile) WriteAt(s []byte, o int64) (n int, err error) {
	if f.Layer != nil {
		n, err = f.Layer.WriteAt(s, o)
		if err == nil && f.Base != nil {
			_, err = f.Base.WriteAt(s, o)
		}
		return n, err
	}
	if f.Base != nil {
		return f.Base.WriteAt(s, o)
	}
	return 0, BADFD
}

func (f *UnionFile) Name() string {
	if f.Layer != nil {
		return f.Layer.Name()
	}
	return f.Base.Name()
}

// DirsMerger is how UnionFile weaves two directories together.
// It takes the FileInfo slices from the layer and the base and returns a
// single view.
type DirsMerger func(lofi, bofi []os.FileInfo) ([]os.FileInfo, error)

var defaultUnionMergeDirsFn = func(lofi, bofi []os.FileInfo) ([]os.FileInfo, error) {
	var files = make(map[string]os.FileInfo)

	for _, fi := range lofi {
		files[fi.Name()] = fi
	}

	for _, fi := range bofi {
		if _, exists := files[fi.Name()]; !exists {
			files[fi.Name()] = fi
		}
	}

	rfi := make([]os.FileInfo, len(files))

	i := 0
	for _, fi := range files {
		rfi[i] = fi
		i++
	}

	return rfi, nil

}

// Readdir will weave the two directories together and
// return a single view of the overlayed directories.
// At the end of the directory view, the error is io.EOF if c > 0.
func (f *UnionFile) Readdir(c int) (ofi []os.FileInfo, err error) {
	var merge DirsMerger = f.Merger
	if merge == nil {
		merge = defaultUnionMergeDirsFn
	}

	if f.off == 0 {
		var lfi []os.FileInfo
		if f.Layer != nil {
			lfi, err = f.Layer.Readdir(-1)
			if err != nil {
				return nil, err
			}
		}

		var bfi []os.FileInfo
		if f.Base != nil {
			bfi, err = f.Base.Readdir(-1)
			if err != nil {
				return nil, err
			}

		}
		merged, err := merge(lfi, bfi)
		if err != nil {
			return nil, err
		}
		f.files = append(f.files, merged...)
	}
	files := f.files[f.off:]

	if c <= 0 {
		return files, nil
	}

	if len(files) == 0 {
		return nil, io.EOF
	}

	if c > len(files) {
		c = len(files)
	}

	defer func() { f.off += c }()
	return files[:c], nil
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
	if f.Layer != nil {
		return f.Layer.Stat()
	}
	if f.Base != nil {
		return f.Base.Stat()
	}
	return nil, BADFD
}

func (f *UnionFile) Sync() (err error) {
	if f.Layer != nil {
		err = f.Layer.Sync()
		if err == nil && f.Base != nil {
			err = f.Base.Sync()
		}
		return err
	}
	if f.Base != nil {
		return f.Base.Sync()
	}
	return BADFD
}

func (f *UnionFile) Truncate(s int64) (err error) {
	if f.Layer != nil {
		err = f.Layer.Truncate(s)
		if err == nil && f.Base != nil {
			err = f.Base.Truncate(s)
		}
		return err
	}
	if f.Base != nil {
		return f.Base.Truncate(s)
	}
	return BADFD
}

func (f *UnionFile) WriteString(s string) (n int, err error) {
	if f.Layer != nil {
		n, err = f.Layer.WriteString(s)
		if err == nil && f.Base != nil {
			_, err = f.Base.WriteString(s)
		}
		return n, err
	}
	if f.Base != nil {
		return f.Base.WriteString(s)
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
