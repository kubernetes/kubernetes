/*
Copyright 2016 The Kubernetes Authors.

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

package clusterinfo

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestSetupOutputWriterNoOp(t *testing.T) {
	tests := []struct {
		name         string
		outputWriter string
	}{
		{
			name:         "empty",
			outputWriter: "",
		},
		{
			name:         "stdout",
			outputWriter: "-",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, buf, _ := genericiooptions.NewTestIOStreams()
			f := cmdtesting.NewTestFactory()
			defer f.Cleanup()

			writer, closeWriter := setupOutputWriter(tt.outputWriter, buf, "/some/file/that/should/be/ignored", "")
			if writer != buf {
				t.Errorf("expected: %v, saw: %v", buf, writer)
			}
			// The default writer is owned by the caller, so releasing must not close it.
			closeWriter()
			if _, err := writer.Write([]byte("still open")); err != nil {
				t.Errorf("expected default writer to stay usable, saw: %v", err)
			}
		})
	}
}

func TestSetupOutputWriterFile(t *testing.T) {
	file := "output"
	extension := ".json"
	dir, err := os.MkdirTemp(os.TempDir(), "out")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fullPath := filepath.Join(dir, file) + extension
	defer os.RemoveAll(dir)

	_, _, buf, _ := genericiooptions.NewTestIOStreams()
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	writer, closeWriter := setupOutputWriter(dir, buf, file, extension)
	if writer == buf {
		t.Errorf("expected: %v, saw: %v", buf, writer)
	}
	output := "some data here"
	writer.Write([]byte(output))

	data, err := os.ReadFile(fullPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(data) != output {
		t.Errorf("expected: %v, saw: %v", output, data)
	}

	closeWriter()
	if _, err := writer.Write([]byte("more")); !errors.Is(err, os.ErrClosed) {
		t.Errorf("expected the file to be closed, saw: %v", err)
	}
}

func TestSetupOutputWriterPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("file modes are not enforced on windows")
	}
	file := filepath.Join("some-namespace", "some-pod", "logs")
	extension := ".txt"
	dir, err := os.MkdirTemp(os.TempDir(), "out")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fullPath := filepath.Join(dir, file) + extension
	defer os.RemoveAll(dir)

	_, _, buf, _ := genericiooptions.NewTestIOStreams()
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	writer, closeWriter := setupOutputWriter(dir, buf, file, extension)
	defer closeWriter()
	if writer == buf {
		t.Errorf("expected: %v, saw: %v", buf, writer)
	}

	fileInfo, err := os.Stat(fullPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fileInfo.Mode().Perm() != 0600 {
		t.Errorf("expected file mode: %v, saw: %v", os.FileMode(0600), fileInfo.Mode().Perm())
	}

	dirInfo, err := os.Stat(filepath.Dir(fullPath))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if dirInfo.Mode().Perm() != 0700 {
		t.Errorf("expected directory mode: %v, saw: %v", os.FileMode(0700), dirInfo.Mode().Perm())
	}
}
