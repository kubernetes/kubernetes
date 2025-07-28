// Copyright 2023 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package registry

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"

	v1 "github.com/google/go-containerregistry/pkg/v1"
)

type diskHandler struct {
	dir string
}

func NewDiskBlobHandler(dir string) BlobHandler { return &diskHandler{dir: dir} }

func (m *diskHandler) blobHashPath(h v1.Hash) string {
	return filepath.Join(m.dir, h.Algorithm, h.Hex)
}

func (m *diskHandler) Stat(_ context.Context, _ string, h v1.Hash) (int64, error) {
	fi, err := os.Stat(m.blobHashPath(h))
	if errors.Is(err, os.ErrNotExist) {
		return 0, errNotFound
	} else if err != nil {
		return 0, err
	}
	return fi.Size(), nil
}
func (m *diskHandler) Get(_ context.Context, _ string, h v1.Hash) (io.ReadCloser, error) {
	return os.Open(m.blobHashPath(h))
}
func (m *diskHandler) Put(_ context.Context, _ string, h v1.Hash, rc io.ReadCloser) error {
	// Put the temp file in the same directory to avoid cross-device problems
	// during the os.Rename.  The filenames cannot conflict.
	f, err := os.CreateTemp(m.dir, "upload-*")
	if err != nil {
		return err
	}

	if err := func() error {
		defer f.Close()
		_, err := io.Copy(f, rc)
		return err
	}(); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(m.dir, h.Algorithm), os.ModePerm); err != nil {
		return err
	}
	return os.Rename(f.Name(), m.blobHashPath(h))
}
func (m *diskHandler) Delete(_ context.Context, _ string, h v1.Hash) error {
	return os.Remove(m.blobHashPath(h))
}
