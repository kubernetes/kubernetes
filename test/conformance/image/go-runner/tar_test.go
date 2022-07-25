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
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestTar(t *testing.T) {
	tmp, err := os.MkdirTemp("", "testtar")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	if err := os.Mkdir(filepath.Join(tmp, "subdir"), os.FileMode(0755)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "file1"), []byte(`file1 data`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "file2"), []byte(`file2 data`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "subdir", "file4"), []byte(`file4 data`), os.FileMode(0644)); err != nil {
		t.Fatal(err)
	}

	testCases := []struct {
		desc      string
		dir       string
		outpath   string
		expectErr string
		expect    map[string]string
	}{
		{
			desc:    "Contents preserved and no self-reference",
			dir:     tmp,
			outpath: filepath.Join(tmp, "out.tar.gz"),
			expect: map[string]string{
				"file1":        "file1 data",
				"file2":        "file2 data",
				"subdir/file4": "file4 data",
			},
		}, {
			desc:      "Errors if directory does not exist",
			dir:       filepath.Join(tmp, "does-not-exist"),
			outpath:   filepath.Join(tmp, "out.tar.gz"),
			expectErr: "tar unable to stat directory",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			err := tarDir(tc.dir, tc.outpath)
			if err == nil {
				defer os.Remove(tc.outpath)
			}

			switch {
			case err != nil && len(tc.expectErr) == 0:
				t.Fatalf("Expected nil error but got %q", err)
			case err != nil && len(tc.expectErr) > 0:
				if !strings.Contains(fmt.Sprint(err), tc.expectErr) {
					t.Errorf("Expected error \n\t%q\nbut got\n\t%q", tc.expectErr, err)
				}
				return
			case err == nil && len(tc.expectErr) > 0:
				t.Fatalf("Expected error %q but got nil", tc.expectErr)
			default:
				// No error
			}

			data, err := readAllTar(tc.outpath)
			if err != nil {
				t.Fatalf("Failed to read tarball: %v", err)
			}

			if !reflect.DeepEqual(data, tc.expect) {
				t.Errorf("Expected data %v but got %v", tc.expect, data)
			}
		})
	}
}

// readAllTar walks all of the files in the archive. It returns a map
// of filenames and their contents and any error encountered.
func readAllTar(tarPath string) (map[string]string, error) {
	tarPath, err := filepath.Abs(tarPath)
	if err != nil {
		return nil, err
	}

	fileReader, err := os.Open(tarPath)
	if err != nil {
		return nil, err
	}
	defer fileReader.Close()

	gzStream, err := gzip.NewReader(fileReader)
	if err != nil {
		return nil, fmt.Errorf("couldn't uncompress reader: %w", err)
	}
	defer gzStream.Close()

	// Open and iterate through the files in the archive.
	tr := tar.NewReader(gzStream)
	fileData := map[string]string{}
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {

			return nil, err
		}

		b, err := io.ReadAll(tr)
		if err != nil {
			return nil, err
		}
		fileData[filepath.ToSlash(hdr.Name)] = string(b)
	}
	return fileData, nil
}
