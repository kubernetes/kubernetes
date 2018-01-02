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

/*

Image Layout

The on-disk layout of an app container is straightforward.
It includes a rootfs with all of the files that will exist in the root of the app and a manifest describing the image.
The layout MUST contain an image manifest.

/manifest
/rootfs/
/rootfs/usr/bin/mysql

*/

import (
	"archive/tar"
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

const (
	// Path to manifest file inside the layout
	ManifestFile = "manifest"
	// Path to rootfs directory inside the layout
	RootfsDir = "rootfs"
)

type ErrOldVersion struct {
	version types.SemVer
}

func (e ErrOldVersion) Error() string {
	return fmt.Sprintf("ACVersion too old. Found major version %v, expected %v", e.version.Major, schema.AppContainerVersion.Major)
}

var (
	ErrNoRootFS   = errors.New("no rootfs found in layout")
	ErrNoManifest = errors.New("no image manifest found in layout")
)

// ValidateLayout takes a directory and validates that the layout of the directory
// matches that expected by the Application Container Image format.
// If any errors are encountered during the validation, it will abort and
// return the first one.
func ValidateLayout(dir string) error {
	fi, err := os.Stat(dir)
	if err != nil {
		return fmt.Errorf("error accessing layout: %v", err)
	}
	if !fi.IsDir() {
		return fmt.Errorf("given path %q is not a directory", dir)
	}
	var flist []string
	var imOK, rfsOK bool
	var im io.Reader
	walkLayout := func(fpath string, fi os.FileInfo, err error) error {
		rpath, err := filepath.Rel(dir, fpath)
		if err != nil {
			return err
		}
		switch rpath {
		case ".":
		case ManifestFile:
			im, err = os.Open(fpath)
			if err != nil {
				return err
			}
			imOK = true
		case RootfsDir:
			if !fi.IsDir() {
				return errors.New("rootfs is not a directory")
			}
			rfsOK = true
		default:
			flist = append(flist, rpath)
		}
		return nil
	}
	if err := filepath.Walk(dir, walkLayout); err != nil {
		return err
	}
	return validate(imOK, im, rfsOK, flist)
}

// ValidateArchive takes a *tar.Reader and validates that the layout of the
// filesystem the reader encapsulates matches that expected by the
// Application Container Image format.  If any errors are encountered during
// the validation, it will abort and return the first one.
func ValidateArchive(tr *tar.Reader) error {
	var fseen map[string]bool = make(map[string]bool)
	var imOK, rfsOK bool
	var im bytes.Buffer
Tar:
	for {
		hdr, err := tr.Next()
		switch {
		case err == nil:
		case err == io.EOF:
			break Tar
		default:
			return err
		}
		name := filepath.Clean(hdr.Name)
		switch name {
		case ".":
		case ManifestFile:
			_, err := io.Copy(&im, tr)
			if err != nil {
				return err
			}
			imOK = true
		case RootfsDir:
			if !hdr.FileInfo().IsDir() {
				return fmt.Errorf("rootfs is not a directory")
			}
			rfsOK = true
		default:
			if _, seen := fseen[name]; seen {
				return fmt.Errorf("duplicate file entry in archive: %s", name)
			}
			fseen[name] = true
		}
	}
	var flist []string
	for key := range fseen {
		flist = append(flist, key)
	}
	return validate(imOK, &im, rfsOK, flist)
}

func validate(imOK bool, im io.Reader, rfsOK bool, files []string) error {
	defer func() {
		if rc, ok := im.(io.Closer); ok {
			rc.Close()
		}
	}()
	if !imOK {
		return ErrNoManifest
	}
	if !rfsOK {
		return ErrNoRootFS
	}
	b, err := ioutil.ReadAll(im)
	if err != nil {
		return fmt.Errorf("error reading image manifest: %v", err)
	}
	var a schema.ImageManifest
	if err := a.UnmarshalJSON(b); err != nil {
		return fmt.Errorf("image manifest validation failed: %v", err)
	}
	if a.ACVersion.LessThanMajor(schema.AppContainerVersion) {
		return ErrOldVersion{
			version: a.ACVersion,
		}
	}
	for _, f := range files {
		if !strings.HasPrefix(f, "rootfs") {
			return fmt.Errorf("unrecognized file path in layout: %q", f)
		}
	}
	return nil
}
