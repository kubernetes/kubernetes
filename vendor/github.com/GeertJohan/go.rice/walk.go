package rice

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Walk is like filepath.Walk()
// Visit http://golang.org/pkg/path/filepath/#Walk for more information
func (b *Box) Walk(path string, walkFn filepath.WalkFunc) error {

	pathFile, err := b.Open(path)
	if err != nil {
		return err
	}
	defer pathFile.Close()

	pathInfo, err := pathFile.Stat()
	if err != nil {
		return err
	}

	if b.IsAppended() || b.IsEmbedded() {
		return b.walk(path, pathInfo, walkFn)
	}

	// We don't have any embedded or appended box so use live filesystem mode
	return filepath.Walk(b.absolutePath+string(os.PathSeparator)+path, func(path string, info os.FileInfo, err error) error {

		// Strip out the box name from the returned paths
		path = strings.TrimPrefix(path, b.absolutePath+string(os.PathSeparator))
		return walkFn(path, info, err)

	})

}

// walk recursively descends path.
// See walk() in $GOROOT/src/pkg/path/filepath/path.go
func (b *Box) walk(path string, info os.FileInfo, walkFn filepath.WalkFunc) error {

	err := walkFn(path, info, nil)
	if err != nil {
		if info.IsDir() && err == filepath.SkipDir {
			return nil
		}
		return err
	}

	if !info.IsDir() {
		return nil
	}

	names, err := b.readDirNames(path)
	if err != nil {
		return walkFn(path, info, err)
	}

	for _, name := range names {

		filename := filepath.Join(path, name)
		fileObject, err := b.Open(filename)
		if err != nil {
			return err
		}
		defer fileObject.Close()

		fileInfo, err := fileObject.Stat()
		if err != nil {
			if err := walkFn(filename, fileInfo, err); err != nil && err != filepath.SkipDir {
				return err
			}
		} else {
			err = b.walk(filename, fileInfo, walkFn)
			if err != nil {
				if !fileInfo.IsDir() || err != filepath.SkipDir {
					return err
				}
			}
		}
	}

	return nil

}

// readDirNames reads the directory named by path and returns a sorted list of directory entries.
// See readDirNames() in $GOROOT/pkg/path/filepath/path.go
func (b *Box) readDirNames(path string) ([]string, error) {

	f, err := b.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	if !stat.IsDir() {
		return nil, nil
	}

	infos, err := f.Readdir(0)
	if err != nil {
		return nil, err
	}

	var names []string

	for _, info := range infos {
		names = append(names, info.Name())
	}

	sort.Strings(names)
	return names, nil

}
