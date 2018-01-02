/*
Copyright 2014 The Camlistore Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package gcs registers a Google Cloud Storage filesystem at the
// well-known /gcs/ filesystem path if the current machine is running
// on Google Compute Engine.
//
// It was initially only meant for small files, and as such, it can only
// read files smaller than 1MB for now.
package gcs // import "go4.org/wkfs/gcs"

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"strings"
	"time"

	"go4.org/wkfs"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/cloud"
	"google.golang.org/cloud/compute/metadata"
	"google.golang.org/cloud/storage"
)

// Max size for all files read, because we use a bytes.Reader as our file
// reader, instead of storage.NewReader. This is because we get all wkfs.File
// methods for free by embedding a bytes.Reader. This filesystem was only supposed
// to be for configuration data only, so this is ok for now.
const maxSize = 1 << 20

func init() {
	if !metadata.OnGCE() {
		return
	}
	hc, err := google.DefaultClient(oauth2.NoContext)
	if err != nil {
		registerBrokenFS(fmt.Errorf("could not get http client for context: %v", err))
		return
	}
	projID, err := metadata.ProjectID()
	if projID == "" || err != nil {
		registerBrokenFS(fmt.Errorf("could not get GCE project ID: %v", err))
		return
	}
	ctx := cloud.NewContext(projID, hc)
	sc, err := storage.NewClient(ctx)
	if err != nil {
		registerBrokenFS(fmt.Errorf("could not get cloud storage client: %v", err))
		return
	}
	wkfs.RegisterFS("/gcs/", &gcsFS{
		ctx: ctx,
		sc:  sc,
	})
}

type gcsFS struct {
	ctx context.Context
	sc  *storage.Client
	err error // sticky error
}

func registerBrokenFS(err error) {
	wkfs.RegisterFS("/gcs/", &gcsFS{
		err: err,
	})
}

func (fs *gcsFS) parseName(name string) (bucket, fileName string, err error) {
	if fs.err != nil {
		return "", "", fs.err
	}
	name = strings.TrimPrefix(name, "/gcs/")
	i := strings.Index(name, "/")
	if i < 0 {
		return name, "", nil
	}
	return name[:i], name[i+1:], nil
}

// Open opens the named file for reading. It returns an error if the file size
// is larger than 1 << 20.
func (fs *gcsFS) Open(name string) (wkfs.File, error) {
	bucket, fileName, err := fs.parseName(name)
	if err != nil {
		return nil, err
	}
	obj := fs.sc.Bucket(bucket).Object(fileName)
	attrs, err := obj.Attrs(fs.ctx)
	if err != nil {
		return nil, err
	}
	size := attrs.Size
	if size > maxSize {
		return nil, fmt.Errorf("file %s too large (%d bytes) for /gcs/ filesystem", name, size)
	}
	rc, err := obj.NewReader(fs.ctx)
	if err != nil {
		return nil, err
	}
	defer rc.Close()

	slurp, err := ioutil.ReadAll(io.LimitReader(rc, size))
	if err != nil {
		return nil, err
	}
	return &file{
		name:   name,
		Reader: bytes.NewReader(slurp),
	}, nil
}

func (fs *gcsFS) Stat(name string) (os.FileInfo, error) { return fs.Lstat(name) }
func (fs *gcsFS) Lstat(name string) (os.FileInfo, error) {
	bucket, fileName, err := fs.parseName(name)
	if err != nil {
		return nil, err
	}
	attrs, err := fs.sc.Bucket(bucket).Object(fileName).Attrs(fs.ctx)
	if err == storage.ErrObjectNotExist {
		return nil, os.ErrNotExist
	}
	if err != nil {
		return nil, err
	}
	return &statInfo{
		name: attrs.Name,
		size: attrs.Size,
	}, nil
}

func (fs *gcsFS) MkdirAll(path string, perm os.FileMode) error { return nil }

func (fs *gcsFS) OpenFile(name string, flag int, perm os.FileMode) (wkfs.FileWriter, error) {
	bucket, fileName, err := fs.parseName(name)
	if err != nil {
		return nil, err
	}
	switch flag {
	case os.O_WRONLY | os.O_CREATE | os.O_EXCL:
	case os.O_WRONLY | os.O_CREATE | os.O_TRUNC:
	default:
		return nil, fmt.Errorf("Unsupported OpenFlag flag mode %d on Google Cloud Storage", flag)
	}
	if flag&os.O_EXCL != 0 {
		if _, err := fs.Stat(name); err == nil {
			return nil, os.ErrExist
		}
	}
	// TODO(mpl): consider adding perm to the object's ObjectAttrs.Metadata
	return fs.sc.Bucket(bucket).Object(fileName).NewWriter(fs.ctx), nil
}

type statInfo struct {
	name    string
	size    int64
	isDir   bool
	modtime time.Time
}

func (si *statInfo) IsDir() bool        { return si.isDir }
func (si *statInfo) ModTime() time.Time { return si.modtime }
func (si *statInfo) Mode() os.FileMode  { return 0644 }
func (si *statInfo) Name() string       { return path.Base(si.name) }
func (si *statInfo) Size() int64        { return si.size }
func (si *statInfo) Sys() interface{}   { return nil }

type file struct {
	name string
	*bytes.Reader
}

func (*file) Close() error   { return nil }
func (f *file) Name() string { return path.Base(f.name) }
func (f *file) Stat() (os.FileInfo, error) {
	panic("Stat not implemented on /gcs/ files yet")
}
