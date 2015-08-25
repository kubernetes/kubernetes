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

package archive

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
)

// ZipWalker returns a filepath.WalkFunc that adds every filesystem node
// to the given *zip.Writer.
func ZipWalker(zw *zip.Writer) filepath.WalkFunc {
	var base string
	return func(path string, info os.FileInfo, err error) error {
		if base == "" {
			base = path
		}

		header, err := zip.FileInfoHeader(info)
		if err != nil {
			return err
		}

		if header.Name, err = filepath.Rel(base, path); err != nil {
			return err
		} else if info.IsDir() {
			header.Name = header.Name + string(filepath.Separator)
		} else {
			header.Method = zip.Deflate
		}

		w, err := zw.CreateHeader(header)
		if err != nil {
			return err
		}

		if info.IsDir() {
			return nil
		}

		f, err := os.Open(path)
		if err != nil {
			return err
		}

		_, err = io.Copy(w, f)
		f.Close()
		return err
	}
}

// Create a zip of all files in a directory recursively, return a byte array and
// the number of files archived.
func ZipDir(path string) ([]byte, []string, error) {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	zipWalker := ZipWalker(zw)
	paths := []string{}
	err := filepath.Walk(path, filepath.WalkFunc(func(path string, info os.FileInfo, err error) error {
		if !info.IsDir() {
			paths = append(paths, path)
		}
		return zipWalker(path, info, err)
	}))

	if err != nil {
		return nil, nil, err
	} else if err = zw.Close(); err != nil {
		return nil, nil, err
	}
	return buf.Bytes(), paths, nil
}

// UnzipDir unzips all files from a given zip byte array into a given directory.
// The directory is created if it does not exist yet.
func UnzipDir(data []byte, destPath string) error {
	// open zip
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return fmt.Errorf("Unzip archive read error: %v", err)
	}

	for _, file := range zr.File {
		// skip directories
		if file.FileInfo().IsDir() {
			continue
		}

		// open file
		rc, err := file.Open()
		defer rc.Close()
		if err != nil {
			return fmt.Errorf("Unzip file read error: %v", err)
		}

		// make sure the directory of the file exists, otherwise create
		destPath := filepath.Clean(filepath.Join(destPath, file.Name))
		destBasedir := path.Dir(destPath)
		err = os.MkdirAll(destBasedir, 0755)
		if err != nil {
			return fmt.Errorf("Unzip mkdir error: %v", err)
		}

		// create file
		f, err := os.Create(destPath)
		if err != nil {
			return fmt.Errorf("Unzip file creation error: %v", err)
		}
		defer f.Close()

		// write file
		if _, err := io.Copy(f, rc); err != nil {
			return fmt.Errorf("Unzip file write error: %v", err)
		}
	}

	return nil
}
