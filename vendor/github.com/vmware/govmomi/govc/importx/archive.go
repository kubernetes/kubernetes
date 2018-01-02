/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package importx

import (
	"archive/tar"
	"context"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"

	"github.com/vmware/govmomi/ovf"
)

// ArchiveFlag doesn't register any flags;
// only encapsulates some common archive related functionality.
type ArchiveFlag struct {
	Archive
}

func newArchiveFlag(ctx context.Context) (*ArchiveFlag, context.Context) {
	return &ArchiveFlag{}, ctx
}

func (f *ArchiveFlag) Register(ctx context.Context, fs *flag.FlagSet) {
}

func (f *ArchiveFlag) Process(ctx context.Context) error {
	return nil
}

func (f *ArchiveFlag) ReadOvf(fpath string) ([]byte, error) {
	r, _, err := f.Archive.Open(fpath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	return ioutil.ReadAll(r)
}

func (f *ArchiveFlag) ReadEnvelope(fpath string) (*ovf.Envelope, error) {
	if fpath == "" {
		return &ovf.Envelope{}, nil
	}

	r, _, err := f.Open(fpath)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	e, err := ovf.Unmarshal(r)
	if err != nil {
		return nil, fmt.Errorf("failed to parse ovf: %s", err.Error())
	}

	return e, nil
}

type Archive interface {
	Open(string) (io.ReadCloser, int64, error)
}

type TapeArchive struct {
	path string
}

type TapeArchiveEntry struct {
	io.Reader
	f *os.File
}

func (t *TapeArchiveEntry) Close() error {
	return t.f.Close()
}

func (t *TapeArchive) Open(name string) (io.ReadCloser, int64, error) {
	f, err := os.Open(t.path)
	if err != nil {
		return nil, 0, err
	}

	r := tar.NewReader(f)

	for {
		h, err := r.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, 0, err
		}

		matched, err := path.Match(name, path.Base(h.Name))
		if err != nil {
			return nil, 0, err
		}

		if matched {
			return &TapeArchiveEntry{r, f}, h.Size, nil
		}
	}

	_ = f.Close()

	return nil, 0, os.ErrNotExist
}

type FileArchive struct {
	path string
}

func (t *FileArchive) Open(name string) (io.ReadCloser, int64, error) {
	fpath := name
	if name != t.path {
		fpath = filepath.Join(filepath.Dir(t.path), name)
	}

	s, err := os.Stat(fpath)
	if err != nil {
		return nil, 0, err
	}

	f, err := os.Open(fpath)
	if err != nil {
		return nil, 0, err
	}

	return f, s.Size(), nil
}
