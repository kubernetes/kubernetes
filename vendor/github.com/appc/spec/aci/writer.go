// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package aci

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"io"
	"time"

	"github.com/appc/spec/schema"
)

// ArchiveWriter writes App Container Images. Users wanting to create an ACI or
// should create an ArchiveWriter and add files to it; the ACI will be written
// to the underlying tar.Writer
type ArchiveWriter interface {
	AddFile(hdr *tar.Header, r io.Reader) error
	Close() error
}

type imageArchiveWriter struct {
	*tar.Writer
	am *schema.ImageManifest
}

// NewImageWriter creates a new ArchiveWriter which will generate an App
// Container Image based on the given manifest and write it to the given
// tar.Writer
func NewImageWriter(am schema.ImageManifest, w *tar.Writer) ArchiveWriter {
	aw := &imageArchiveWriter{
		w,
		&am,
	}
	return aw
}

func (aw *imageArchiveWriter) AddFile(hdr *tar.Header, r io.Reader) error {
	err := aw.Writer.WriteHeader(hdr)
	if err != nil {
		return err
	}

	if r != nil {
		_, err := io.Copy(aw.Writer, r)
		if err != nil {
			return err
		}
	}

	return nil
}

func (aw *imageArchiveWriter) addFileNow(path string, contents []byte) error {
	buf := bytes.NewBuffer(contents)
	now := time.Now()
	hdr := tar.Header{
		Name:       path,
		Mode:       0644,
		Uid:        0,
		Gid:        0,
		Size:       int64(buf.Len()),
		ModTime:    now,
		Typeflag:   tar.TypeReg,
		Uname:      "root",
		Gname:      "root",
		ChangeTime: now,
	}
	return aw.AddFile(&hdr, buf)
}

func (aw *imageArchiveWriter) addManifest(name string, m json.Marshaler) error {
	out, err := m.MarshalJSON()
	if err != nil {
		return err
	}
	return aw.addFileNow(name, out)
}

func (aw *imageArchiveWriter) Close() error {
	if err := aw.addManifest(ManifestFile, aw.am); err != nil {
		return err
	}
	return aw.Writer.Close()
}
