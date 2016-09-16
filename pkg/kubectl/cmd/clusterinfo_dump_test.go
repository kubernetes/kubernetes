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

package cmd

import (
	"bytes"
	"io/ioutil"
	"os"
	"path"
	"testing"
)

func TestSetupOutputWriterNoOp(t *testing.T) {
	tests := []string{"", "-"}
	for _, test := range tests {
		out := &bytes.Buffer{}
		f, _, _, _ := NewAPIFactory()
		cmd := NewCmdClusterInfoDump(f, os.Stdout)
		cmd.Flag("output-directory").Value.Set(test)
		writer := setupOutputWriter(cmd, out, "/some/file/that/should/be/ignored")
		if writer != out {
			t.Errorf("expected: %v, saw: %v", out, writer)
		}
	}
}

func TestSetupOutputWriterFile(t *testing.T) {
	file := "output.json"
	dir, err := ioutil.TempDir(os.TempDir(), "out")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fullPath := path.Join(dir, file)
	defer os.RemoveAll(dir)

	out := &bytes.Buffer{}
	f, _, _, _ := NewAPIFactory()
	cmd := NewCmdClusterInfoDump(f, os.Stdout)
	cmd.Flag("output-directory").Value.Set(dir)
	writer := setupOutputWriter(cmd, out, file)
	if writer == out {
		t.Errorf("expected: %v, saw: %v", out, writer)
	}
	output := "some data here"
	writer.Write([]byte(output))

	data, err := ioutil.ReadFile(fullPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(data) != output {
		t.Errorf("expected: %v, saw: %v", output, data)
	}
}
