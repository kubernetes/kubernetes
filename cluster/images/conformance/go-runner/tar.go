/*
Copyright 2019 The Kubernetes Authors.

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

package main

import (
	"archive/tar"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

// tarDir takes a source and variable writers and walks 'source' writing each file
// found to the tar writer.
func tarDir(dir, outpath string) error {
	// ensure the src actually exists before trying to tar it
	if _, err := os.Stat(dir); err != nil {
		return errors.Wrapf(err, "tar unable to stat directory %v", dir)
	}

	outfile, err := os.Create(outpath)
	if err != nil {
		return errors.Wrapf(err, "creating tarball %v", outpath)
	}
	defer outfile.Close()

	gzw := gzip.NewWriter(outfile)
	defer gzw.Close()

	tw := tar.NewWriter(gzw)
	defer tw.Close()

	return filepath.Walk(dir, func(file string, fi os.FileInfo, err error) error {
		// Return on any error.
		if err != nil {
			return err
		}

		// Only write regular files and don't include the archive itself.
		if !fi.Mode().IsRegular() || filepath.Join(dir, fi.Name()) == outpath {
			return nil
		}

		// Create a new dir/file header.
		header, err := tar.FileInfoHeader(fi, fi.Name())
		if err != nil {
			return errors.Wrapf(err, "creating file info header %v", fi.Name())
		}

		// Update the name to correctly reflect the desired destination when untaring.
		header.Name = strings.TrimPrefix(strings.Replace(file, dir, "", -1), string(filepath.Separator))
		if err := tw.WriteHeader(header); err != nil {
			return errors.Wrapf(err, "writing header for tarball %v", header.Name)
		}

		// Open files, copy into tarfile, and close.
		f, err := os.Open(file)
		if err != nil {
			return errors.Wrapf(err, "opening file %v for writing into tarball", file)
		}
		defer f.Close()

		_, err = io.Copy(tw, f)
		return errors.Wrapf(err, "creating file %v contents into tarball", file)
	})
}
