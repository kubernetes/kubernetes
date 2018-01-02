// Copyright 2014 The rkt Authors
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

// Package aci implements helper functions for working with ACIs
package aci

import (
	"archive/tar"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"time"

	"github.com/appc/spec/aci"
	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
	"golang.org/x/crypto/openpgp"
)

type ACIEntry struct {
	Header   *tar.Header
	Contents string
}

type imageArchiveWriter struct {
	*tar.Writer
	am *schema.ImageManifest
}

// NewImageWriter creates a new ArchiveWriter which will generate an App
// Container Image based on the given manifest and write it to the given
// tar.Writer
// TODO(sgotti) this is a copy of appc/spec/aci.imageArchiveWriter with
// addFileNow changed to create the file with the current user. needed for
// testing as non root user.
func NewImageWriter(am schema.ImageManifest, w *tar.Writer) aci.ArchiveWriter {
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
		Uid:        os.Getuid(),
		Gid:        os.Getgid(),
		Size:       int64(buf.Len()),
		ModTime:    now,
		Typeflag:   tar.TypeReg,
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
	if err := aw.addManifest(aci.ManifestFile, aw.am); err != nil {
		return err
	}
	return aw.Writer.Close()
}

// NewBasicACI creates a new ACI in the given directory with the given name.
// Used for testing.
func NewBasicACI(dir string, name string) (*os.File, error) {
	manifest := schema.ImageManifest{
		ACKind:    schema.ImageManifestKind,
		ACVersion: schema.AppContainerVersion,
		Name:      types.ACIdentifier(name),
	}

	b, err := manifest.MarshalJSON()
	if err != nil {
		return nil, err
	}

	return NewACI(dir, string(b), nil)
}

// NewACI creates a new ACI in the given directory with the given image
// manifest and entries.
// Used for testing.
func NewACI(dir string, manifest string, entries []*ACIEntry) (*os.File, error) {
	var im schema.ImageManifest
	if err := im.UnmarshalJSON([]byte(manifest)); err != nil {
		return nil, errwrap.Wrap(errors.New("invalid image manifest"), err)
	}

	tf, err := ioutil.TempFile(dir, "")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tf.Name())

	tw := tar.NewWriter(tf)
	aw := NewImageWriter(im, tw)

	for _, entry := range entries {
		// Add default mode
		if entry.Header.Mode == 0 {
			if entry.Header.Typeflag == tar.TypeDir {
				entry.Header.Mode = 0755
			} else {
				entry.Header.Mode = 0644
			}
		}
		// Add calling user uid and gid or tests will fail
		entry.Header.Uid = os.Getuid()
		entry.Header.Gid = os.Getgid()
		sr := strings.NewReader(entry.Contents)
		if err := aw.AddFile(entry.Header, sr); err != nil {
			return nil, err
		}
	}

	if err := aw.Close(); err != nil {
		return nil, err
	}
	return tf, nil
}

// NewDetachedSignature creates a new openpgp armored detached signature for the given ACI
// signed with armoredPrivateKey.
func NewDetachedSignature(armoredPrivateKey string, aci io.Reader) (io.Reader, error) {
	entityList, err := openpgp.ReadArmoredKeyRing(bytes.NewBufferString(armoredPrivateKey))
	if err != nil {
		return nil, err
	}
	if len(entityList) < 1 {
		return nil, errors.New("empty entity list")
	}
	signature := &bytes.Buffer{}
	if err := openpgp.ArmoredDetachSign(signature, entityList[0], aci, nil); err != nil {
		return nil, err
	}
	return signature, nil
}
