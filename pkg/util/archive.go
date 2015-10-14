/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path"
)

// ArchiveDirectory takes a directory and produces a tar ball of the contents.  If 'compress' is true, it is
// also gzip'd.
func ArchiveDirectoryToBytes(dir string, compress bool) ([]byte, error) {
	buff := &bytes.Buffer{}
	if err := ArchiveDirectory(dir, compress, buff); err != nil {
		return nil, err
	}
	return buff.Bytes(), nil
}

func ArchiveDirectory(dir string, compress bool, output io.Writer) error {
	var gz io.WriteCloser
	var writer *tar.Writer
	if compress {
		gz = gzip.NewWriter(output)
		writer = tar.NewWriter(gz)
	} else {
		writer = tar.NewWriter(output)
	}

	stat, err := os.Stat(dir)
	if err != nil {
		return err
	}
	if !stat.IsDir() {
		return fmt.Errorf("%s is not a directory", dir)
	}
	file, err := os.Open(dir)
	if err != nil {
		return err
	}
	files, err := file.Readdir(-1)
	if err != nil {
		return err
	}
	for ix := range files {
		hdr := &tar.Header{
			Name: files[ix].Name(),
			Size: files[ix].Size(),
		}
		if err := writer.WriteHeader(hdr); err != nil {
			return err
		}
		fs, err := os.Open(path.Join(file.Name(), files[ix].Name()))
		if err != nil {
			return err
		}
		if _, err := io.Copy(writer, fs); err != nil {
			return err
		}
	}
	if err := writer.Close(); err != nil {
		return err
	}
	if gz != nil {
		return gz.Close()
	}
	return nil
}
