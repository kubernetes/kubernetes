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
	"io"
	"io/ioutil"
	"os"
	"path"
	"testing"
)

func writeFileOrDie(path string, data []byte, t *testing.T) {
	if err := ioutil.WriteFile(path, data, os.FileMode(0644)); err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
		return
	}

}

func TestArchive(t *testing.T) {
	expected := map[string]string{
		"file1": "foobar",
		"file2": "baz",
		"file3": "blah",
	}

	tmp, err := ioutil.TempDir(os.TempDir(), "archive")
	defer os.RemoveAll(tmp)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
		return
	}
	if err := os.MkdirAll(tmp, os.FileMode(0755)); err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
		return
	}
	for file, data := range expected {
		writeFileOrDie(path.Join(tmp, file), []byte(data), t)
	}

	for _, compress := range []bool{true, false} {
		for _, stream := range []bool{true, false} {
			var input io.Reader
			if !stream {
				data, err := ArchiveDirectoryToBytes(tmp, compress)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
					continue
				}
				input = bytes.NewBuffer(data)
			} else {
				buff := &bytes.Buffer{}
				if err := ArchiveDirectory(tmp, compress, buff); err != nil {
					t.Errorf("unexpected error: %v", err)
					continue
				}
				input = buff
			}
			if compress {
				if input, err = gzip.NewReader(input); err != nil {
					t.Errorf("unexpected error: %v", err)
					continue
				}
			}

			reader := tar.NewReader(input)

			files := map[string][]byte{}
			for {
				hdr, err := reader.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					t.Errorf("unexpected error: %v", err)
					t.FailNow()
					return
				}
				output := &bytes.Buffer{}
				if _, err := io.Copy(output, reader); err != nil {
					t.Errorf("unexpected error: %v", err)
					t.FailNow()
					return
				}
				files[hdr.Name] = output.Bytes()
			}
			if len(files) != 3 {
				t.Errorf("expected 3 files, found: %d %v %v", len(files), compress, stream)
			}

			for file, str := range expected {
				data, found := files[file]
				if !found {
					t.Errorf("failed to find expected file: %s", file)
					continue
				}
				if string(data) != str {
					t.Errorf("expected: %s, saw: %s", str, string(data))
				}
			}
		}
	}
}
