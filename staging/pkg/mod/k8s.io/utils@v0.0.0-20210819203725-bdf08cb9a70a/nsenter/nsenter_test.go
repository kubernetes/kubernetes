// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package nsenter

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/utils/exec"
)

func TestExec(t *testing.T) {
	tests := []struct {
		name           string
		command        string
		args           []string
		expectedOutput string
		expectError    bool
	}{
		{
			name:           "simple command",
			command:        "echo",
			args:           []string{"hello", "world"},
			expectedOutput: "hello world\n",
		},
		{
			name:        "nozero exit code",
			command:     "false",
			expectError: true,
		},
	}

	executor := fakeExec{
		rootfsPath: "/rootfs",
	}
	for _, test := range tests {
		ns := NSEnter{
			hostRootFsPath: "/rootfs",
			executor:       executor,
		}
		cmd := ns.Exec(test.command, test.args)
		outBytes, err := cmd.CombinedOutput()
		out := string(outBytes)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error: %s", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error, got none", test.name)
		}
		if test.expectedOutput != out {
			t.Errorf("test %q: expected output %q, got %q", test.name, test.expectedOutput, out)
		}
	}
}

func TestKubeletPath(t *testing.T) {
	tests := []struct {
		rootfs              string
		hostpath            string
		expectedKubeletPath string
	}{
		{
			// simple join
			"/rootfs",
			"/some/path",
			"/rootfs/some/path",
		},
		{
			// squash slashes
			"/rootfs/",
			"//some/path",
			"/rootfs/some/path",
		},
	}

	for _, test := range tests {
		ns := NSEnter{
			hostRootFsPath: test.rootfs,
		}
		out := ns.KubeletPath(test.hostpath)
		if out != test.expectedKubeletPath {
			t.Errorf("Expected path %q, got %q", test.expectedKubeletPath, out)
		}

	}
}

func TestEvalSymlinks(t *testing.T) {
	tests := []struct {
		name        string
		mustExist   bool
		prepare     func(tmpdir string) (src string, expectedDst string, err error)
		expectError bool
	}{
		{
			name:      "simple file /src",
			mustExist: true,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				src = filepath.Join(tmpdir, "src")
				err = ioutil.WriteFile(src, []byte{}, 0644)
				return src, src, err
			},
		},
		{
			name:      "non-existing file /src",
			mustExist: true,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				src = filepath.Join(tmpdir, "src")
				return src, "", nil
			},
			expectError: true,
		},
		{
			name:      "non-existing file /src/ with mustExist=false",
			mustExist: false,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				src = filepath.Join(tmpdir, "src")
				return src, src, nil
			},
		},
		{
			name:      "non-existing file /existing/path/src with mustExist=false with existing directories",
			mustExist: false,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				src = filepath.Join(tmpdir, "existing/path")
				if err := os.MkdirAll(src, 0755); err != nil {
					return "", "", err
				}
				src = filepath.Join(src, "src")
				return src, src, nil
			},
		},
		{
			name:      "simple symlink /src -> /dst",
			mustExist: false,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				dst := filepath.Join(tmpdir, "dst")
				if err = ioutil.WriteFile(dst, []byte{}, 0644); err != nil {
					return "", "", err
				}
				src = filepath.Join(tmpdir, "src")
				err = os.Symlink(dst, src)
				return src, dst, err
			},
		},
		{
			name:      "dangling symlink /src -> /non-existing-path",
			mustExist: true,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				dst := filepath.Join(tmpdir, "non-existing-path")
				src = filepath.Join(tmpdir, "src")
				err = os.Symlink(dst, src)
				return src, "", err
			},
			expectError: true,
		},
		{
			name:      "dangling symlink /src -> /non-existing-path with mustExist=false",
			mustExist: false,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				dst := filepath.Join(tmpdir, "non-existing-path")
				src = filepath.Join(tmpdir, "src")
				err = os.Symlink(dst, src)
				return src, dst, err
			},
		},
		{
			name:      "symlink to directory /src/file, where /src is link to /dst",
			mustExist: true,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				dst := filepath.Join(tmpdir, "dst")
				if err = os.Mkdir(dst, 0755); err != nil {
					return "", "", err
				}
				dstFile := filepath.Join(dst, "file")
				if err = ioutil.WriteFile(dstFile, []byte{}, 0644); err != nil {
					return "", "", err
				}

				src = filepath.Join(tmpdir, "src")
				if err = os.Symlink(dst, src); err != nil {
					return "", "", err
				}
				srcFile := filepath.Join(src, "file")
				return srcFile, dstFile, nil
			},
		},
		{
			name:      "symlink to non-existing directory: /src/file, where /src is link to /dst and dst does not exist",
			mustExist: true,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				dst := filepath.Join(tmpdir, "dst")

				src = filepath.Join(tmpdir, "src")
				if err = os.Symlink(dst, src); err != nil {
					return "", "", err
				}
				srcFile := filepath.Join(src, "file")
				return srcFile, "", nil
			},
			expectError: true,
		},
		{
			name:      "symlink to non-existing directory: /src/file, where /src is link to /dst and dst does not exist with mustExist=false",
			mustExist: false,
			prepare: func(tmpdir string) (src string, expectedDst string, err error) {
				dst := filepath.Join(tmpdir, "dst")
				dstFile := filepath.Join(dst, "file")

				src = filepath.Join(tmpdir, "src")
				if err = os.Symlink(dst, src); err != nil {
					return "", "", err
				}
				srcFile := filepath.Join(src, "file")
				return srcFile, dstFile, nil
			},
		},
	}

	for _, test := range tests {
		ns := NSEnter{
			hostRootFsPath: "/rootfs",
			executor: fakeExec{
				rootfsPath: "/rootfs",
			},
		}

		tmpdir, err := ioutil.TempDir("", "nsenter-hostpath-")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(tmpdir)

		src, expectedDst, err := test.prepare(tmpdir)
		if err != nil {
			t.Error(err)
			continue
		}

		dst, err := ns.EvalSymlinks(src, test.mustExist)
		if err != nil && !test.expectError {
			t.Errorf("Test %q: unexpected error: %s", test.name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("Test %q: expected error, got none", test.name)
		}
		if dst != expectedDst {
			t.Errorf("Test %q: expected destination %q, got %q", test.name, expectedDst, dst)
		}
	}
}

func TestNewNsenter(t *testing.T) {
	// Create a symlink /tmp/xyz/rootfs -> / and use it as rootfs path
	// It should resolve all binaries correctly, the test runs on Linux

	tmpdir, err := ioutil.TempDir("", "nsenter-hostpath-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	rootfs := filepath.Join(tmpdir, "rootfs")
	if err = os.Symlink("/", rootfs); err != nil {
		t.Fatal(err)
	}

	_, err = NewNsenter(rootfs, exec.New())
	if err != nil {
		t.Errorf("Error: %s", err)
	}
}

func TestNewNsenterError(t *testing.T) {
	// Create empty dir /tmp/xyz/rootfs and use it as rootfs path
	// It should resolve all binaries correctly, the test runs on Linux

	tmpdir, err := ioutil.TempDir("", "nsenter-hostpath-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpdir)

	rootfs := filepath.Join(tmpdir, "rootfs")
	if err = os.MkdirAll(rootfs, 0755); err != nil {
		t.Fatal(err)
	}

	_, err = NewNsenter(rootfs, exec.New())
	if err == nil {
		t.Errorf("Expected error, got none")
	}
}
