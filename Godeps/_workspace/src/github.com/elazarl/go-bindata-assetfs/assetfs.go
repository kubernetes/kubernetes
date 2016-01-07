package assetfs

import (
	"bytes"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"time"
)

var (
	fileTimestamp = time.Now()
)

// FakeFile implements os.FileInfo interface for a given path and size
type FakeFile struct {
	// Path is the path of this file
	Path string
	// Dir marks of the path is a directory
	Dir bool
	// Len is the length of the fake file, zero if it is a directory
	Len int64
}

func (f *FakeFile) Name() string {
	_, name := filepath.Split(f.Path)
	return name
}

func (f *FakeFile) Mode() os.FileMode {
	mode := os.FileMode(0644)
	if f.Dir {
		return mode | os.ModeDir
	}
	return mode
}

func (f *FakeFile) ModTime() time.Time {
	return fileTimestamp
}

func (f *FakeFile) Size() int64 {
	return f.Len
}

func (f *FakeFile) IsDir() bool {
	return f.Mode().IsDir()
}

func (f *FakeFile) Sys() interface{} {
	return nil
}

// AssetFile implements http.File interface for a no-directory file with content
type AssetFile struct {
	*bytes.Reader
	io.Closer
	FakeFile
}

func NewAssetFile(name string, content []byte) *AssetFile {
	return &AssetFile{
		bytes.NewReader(content),
		ioutil.NopCloser(nil),
		FakeFile{name, false, int64(len(content))}}
}

func (f *AssetFile) Readdir(count int) ([]os.FileInfo, error) {
	return nil, errors.New("not a directory")
}

func (f *AssetFile) Size() int64 {
	return f.FakeFile.Size()
}

func (f *AssetFile) Stat() (os.FileInfo, error) {
	return f, nil
}

// AssetDirectory implements http.File interface for a directory
type AssetDirectory struct {
	AssetFile
	ChildrenRead int
	Children     []os.FileInfo
}

func NewAssetDirectory(name string, children []string, fs *AssetFS) *AssetDirectory {
	fileinfos := make([]os.FileInfo, 0, len(children))
	for _, child := range children {
		_, err := fs.AssetDir(filepath.Join(name, child))
		fileinfos = append(fileinfos, &FakeFile{child, err == nil, 0})
	}
	return &AssetDirectory{
		AssetFile{
			bytes.NewReader(nil),
			ioutil.NopCloser(nil),
			FakeFile{name, true, 0},
		},
		0,
		fileinfos}
}

func (f *AssetDirectory) Readdir(count int) ([]os.FileInfo, error) {
	if count <= 0 {
		return f.Children, nil
	}
	if f.ChildrenRead+count > len(f.Children) {
		count = len(f.Children) - f.ChildrenRead
	}
	rv := f.Children[f.ChildrenRead : f.ChildrenRead+count]
	f.ChildrenRead += count
	return rv, nil
}

func (f *AssetDirectory) Stat() (os.FileInfo, error) {
	return f, nil
}

// AssetFS implements http.FileSystem, allowing
// embedded files to be served from net/http package.
type AssetFS struct {
	// Asset should return content of file in path if exists
	Asset func(path string) ([]byte, error)
	// AssetDir should return list of files in the path
	AssetDir func(path string) ([]string, error)
	// Prefix would be prepended to http requests
	Prefix string
}

func (fs *AssetFS) Open(name string) (http.File, error) {
	name = path.Join(fs.Prefix, name)
	if len(name) > 0 && name[0] == '/' {
		name = name[1:]
	}
	if b, err := fs.Asset(name); err == nil {
		return NewAssetFile(name, b), nil
	}
	if children, err := fs.AssetDir(name); err == nil {
		return NewAssetDirectory(name, children, fs), nil
	} else {
		return nil, err
	}
}
