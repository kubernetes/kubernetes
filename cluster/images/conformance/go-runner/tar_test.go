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
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/pkg/errors"
)

func TestTar(t *testing.T) {
	testCases := []struct {
		desc      string
		dir       string
		outpath   string
		expectErr string
		expect    map[string]string
	}{
		{
			desc:    "Contents preserved and no self-reference",
			dir:     "testdata/tartest",
			outpath: "testdata/tartest/out.tar.gz",
			expect: map[string]string{
				"file1":        "file1 data",
				"file2":        "file2 data",
				"subdir/file4": "file4 data",
			},
		}, {
			desc:      "Errors if directory does not exist",
			dir:       "testdata/does-not-exist",
			outpath:   "testdata/tartest/out.tar.gz",
			expectErr: "tar unable to stat directory",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			err := tarDir(tc.dir, tc.outpath)
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
		return nil, errors.Wrap(err, "couldn't uncompress reader")
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

		b, err := ioutil.ReadAll(tr)
		if err != nil {
			return nil, err
		}
		fileData[filepath.ToSlash(hdr.Name)] = string(b)
	}
	return fileData, nil
}
