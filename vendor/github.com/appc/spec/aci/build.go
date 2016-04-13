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
	"io"
	"os"
	"path/filepath"

	"github.com/appc/spec/pkg/tarheader"
)

// TarHeaderWalkFunc is the type of the function which allows setting tar
// headers or filtering out tar entries when building an ACI. It will be
// applied to every entry in the tar file.
//
// If true is returned, the entry will be included in the final ACI; if false,
// the entry will not be included.
type TarHeaderWalkFunc func(hdr *tar.Header) bool

// BuildWalker creates a filepath.WalkFunc that walks over the given root
// (which should represent an ACI layout on disk) and adds the files in the
// rootfs/ subdirectory to the given ArchiveWriter
func BuildWalker(root string, aw ArchiveWriter, cb TarHeaderWalkFunc) filepath.WalkFunc {
	// cache of inode -> filepath, used to leverage hard links in the archive
	inos := map[uint64]string{}
	return func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		relpath, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		if relpath == "." {
			return nil
		}
		if relpath == ManifestFile {
			// ignore; this will be written by the archive writer
			// TODO(jonboulle): does this make sense? maybe just remove from archivewriter?
			return nil
		}

		link := ""
		var r io.Reader
		switch info.Mode() & os.ModeType {
		case os.ModeSocket:
			return nil
		case os.ModeNamedPipe:
		case os.ModeCharDevice:
		case os.ModeDevice:
		case os.ModeDir:
		case os.ModeSymlink:
			target, err := os.Readlink(path)
			if err != nil {
				return err
			}
			link = target
		default:
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			defer file.Close()
			r = file
		}

		hdr, err := tar.FileInfoHeader(info, link)
		if err != nil {
			panic(err)
		}
		// Because os.FileInfo's Name method returns only the base
		// name of the file it describes, it may be necessary to
		// modify the Name field of the returned header to provide the
		// full path name of the file.
		hdr.Name = relpath
		tarheader.Populate(hdr, info, inos)
		// If the file is a hard link to a file we've already seen, we
		// don't need the contents
		if hdr.Typeflag == tar.TypeLink {
			hdr.Size = 0
			r = nil
		}

		if cb != nil {
			if !cb(hdr) {
				return nil
			}
		}

		if err := aw.AddFile(hdr, r); err != nil {
			return err
		}

		return nil
	}
}
