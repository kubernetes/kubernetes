/*
Copyright 2014 The Kubernetes Authors.

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

package cmd

import (
	"bytes"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"testing"
)

func TestExtractFileSpec(t *testing.T) {
	tests := []struct {
		spec              string
		expectedPod       string
		expectedNamespace string
		expectedFile      string
		expectErr         bool
	}{
		{
			spec:              "namespace/pod:/some/file",
			expectedPod:       "pod",
			expectedNamespace: "namespace",
			expectedFile:      "/some/file",
		},
		{
			spec:         "pod:/some/file",
			expectedPod:  "pod",
			expectedFile: "/some/file",
		},
		{
			spec:         "/some/file",
			expectedFile: "/some/file",
		},
		{
			spec:      "some:bad:spec",
			expectErr: true,
		},
	}
	for _, test := range tests {
		spec, err := extractFileSpec(test.spec)
		if test.expectErr && err == nil {
			t.Errorf("unexpected non-error")
			continue
		}
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if spec.PodName != test.expectedPod {
			t.Errorf("expected: %s, saw: %s", test.expectedPod, spec.PodName)
		}
		if spec.PodNamespace != test.expectedNamespace {
			t.Errorf("expected: %s, saw: %s", test.expectedNamespace, spec.PodNamespace)
		}
		if spec.File != test.expectedFile {
			t.Errorf("expected: %s, saw: %s", test.expectedFile, spec.File)
		}
	}
}

func TestGetPrefix(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "/foo/bar",
			expected: "foo/bar",
		},
		{
			input:    "foo/bar",
			expected: "foo/bar",
		},
	}
	for _, test := range tests {
		out := getPrefix(test.input)
		if out != test.expected {
			t.Errorf("expected: %s, saw: %s", test.expected, out)
		}
	}
}

func TestTarUntar(t *testing.T) {
	dir, err := ioutil.TempDir(os.TempDir(), "input")
	dir2, err2 := ioutil.TempDir(os.TempDir(), "output")
	if err != nil || err2 != nil {
		t.Errorf("unexpected error: %v | %v", err, err2)
		t.FailNow()
	}
	dir = dir + "/"
	defer func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("Unexpected error cleaning up: %v", err)
		}
		if err := os.RemoveAll(dir2); err != nil {
			t.Errorf("Unexpected error cleaning up: %v", err)
		}
	}()

	files := []struct {
		name string
		data string
	}{
		{
			name: "foo",
			data: "foobarbaz",
		},
		{
			name: "dir/blah",
			data: "bazblahfoo",
		},
		{
			name: "some/other/directory",
			data: "with more data here",
		},
		{
			name: "blah",
			data: "same file name different data",
		},
	}

	for _, file := range files {
		filepath := path.Join(dir, file.name)
		if err := os.MkdirAll(path.Dir(filepath), 0755); err != nil {
			t.Errorf("unexpected error: %v", err)
			t.FailNow()
		}
		f, err := os.Create(filepath)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			t.FailNow()
		}
		defer f.Close()
		if _, err := io.Copy(f, bytes.NewBuffer([]byte(file.data))); err != nil {
			t.Errorf("unexpected error: %v", err)
			t.FailNow()
		}
	}

	writer := &bytes.Buffer{}
	if err := makeTar(dir, writer); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	reader := bytes.NewBuffer(writer.Bytes())
	if err := untarAll(reader, dir2, ""); err != nil {
		t.Errorf("unexpected error: %v", err)
		t.FailNow()
	}

	for _, file := range files {
		filepath := path.Join(dir, file.name)
		f, err := os.Open(filepath)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		defer f.Close()
		buff := &bytes.Buffer{}
		io.Copy(buff, f)
		if file.data != string(buff.Bytes()) {
			t.Errorf("expected: %s, saw: %s", file.data, string(buff.Bytes()))
		}
	}
}

// TestCopyToLocalFileOrDir tests untarAll in two cases :
// 1: copy pod file to local file
// 2: copy pod file into local directory
func TestCopyToLocalFileOrDir(t *testing.T) {
	dir, err := ioutil.TempDir(os.TempDir(), "input")
	dir2, err2 := ioutil.TempDir(os.TempDir(), "output")
	if err != nil || err2 != nil {
		t.Errorf("unexpected error: %v | %v", err, err2)
		t.FailNow()
	}
	defer func() {
		if err := os.RemoveAll(dir); err != nil {
			t.Errorf("Unexpected error cleaning up: %v", err)
		}
		if err := os.RemoveAll(dir2); err != nil {
			t.Errorf("Unexpected error cleaning up: %v", err)
		}
	}()

	files := []struct {
		name          string
		data          string
		dest          string
		destDirExists bool
	}{
		{
			name:          "foo",
			data:          "foobarbaz",
			dest:          "path/to/dest",
			destDirExists: false,
		},
		{
			name:          "dir/blah",
			data:          "bazblahfoo",
			dest:          "dest/file/path",
			destDirExists: true,
		},
	}

	for _, file := range files {
		func() {
			// setup
			srcFilePath := filepath.Join(dir, file.name)
			destPath := filepath.Join(dir2, file.dest)
			if err := os.MkdirAll(filepath.Dir(srcFilePath), 0755); err != nil {
				t.Errorf("unexpected error: %v", err)
				t.FailNow()
			}
			srcFile, err := os.Create(srcFilePath)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				t.FailNow()
			}
			defer srcFile.Close()

			if _, err := io.Copy(srcFile, bytes.NewBuffer([]byte(file.data))); err != nil {
				t.Errorf("unexpected error: %v", err)
				t.FailNow()
			}
			if file.destDirExists {
				if err := os.MkdirAll(destPath, 0755); err != nil {
					t.Errorf("unexpected error: %v", err)
					t.FailNow()
				}
			}

			// start tests
			srcTarFilePath := filepath.Join(dir, file.name+".tar")
			// here use tar command to create tar file instead of calling makeTar func
			// because makeTar func can not generate correct header name
			err = exec.Command("tar", "cf", srcTarFilePath, srcFilePath).Run()
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				t.FailNow()
			}
			srcTarFile, err := os.Open(srcTarFilePath)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				t.FailNow()
			}
			defer srcTarFile.Close()

			if err := untarAll(srcTarFile, destPath, getPrefix(srcFilePath)); err != nil {
				t.Errorf("unexpected error: %v", err)
				t.FailNow()
			}

			actualDestFilePath := destPath
			if file.destDirExists {
				actualDestFilePath = filepath.Join(destPath, filepath.Base(srcFilePath))
			}
			_, err = os.Stat(actualDestFilePath)
			if err != nil && os.IsNotExist(err) {
				t.Errorf("expecting %s exists, but actually it's missing", actualDestFilePath)
			}
			destFile, err := os.Open(actualDestFilePath)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				t.FailNow()
			}
			defer destFile.Close()
			buff := &bytes.Buffer{}
			io.Copy(buff, destFile)
			if file.data != string(buff.Bytes()) {
				t.Errorf("expected: %s, actual: %s", file.data, string(buff.Bytes()))
			}
		}()
	}

}
