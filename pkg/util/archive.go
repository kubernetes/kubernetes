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
func ArchiveDirectory(dir string, compress bool) ([]byte, error) {
	buff := &bytes.Buffer{}
	writer := tar.NewWriter(buff)

	stat, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !stat.IsDir() {
		return nil, fmt.Errorf("%s is not a directory", dir)
	}
	file, err := os.Open(dir)
	if err != nil {
		return nil, err
	}
	files, err := file.Readdir(-1)
	if err != nil {
		return nil, err
	}
	for ix := range files {
		hdr := &tar.Header{
			Name: files[ix].Name(),
			Size: files[ix].Size(),
		}
		if err := writer.WriteHeader(hdr); err != nil {
			return nil, err
		}
		fs, err := os.Open(path.Join(file.Name(), files[ix].Name()))
		if err != nil {
			return nil, err
		}
		if _, err := io.Copy(writer, fs); err != nil {
			return nil, err
		}
	}
	if err := writer.Close(); err != nil {
		return nil, err
	}
	data := buff.Bytes()
	if !compress {
		return data, nil
	}
	buff = &bytes.Buffer{}
	gz := gzip.NewWriter(buff)
	if _, err := gz.Write(data); err != nil {
		return nil, err
	}
	if err := gz.Close(); err != nil {
		return nil, err
	}
	return buff.Bytes(), nil
}
