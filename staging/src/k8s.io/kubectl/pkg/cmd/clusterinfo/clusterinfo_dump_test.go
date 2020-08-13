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
	"io/ioutil"
	"os"
	"path"
	"testing"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestSetupOutputWriterNoOp(t *testing.T) {
	tests := []string{"", "-"}
	for _, test := range tests {
		_, _, buf, _ := genericclioptions.NewTestIOStreams()
		f := cmdtesting.NewTestFactory()
		defer f.Cleanup()

		writer := setupOutputWriter(test, buf, "/some/file/that/should/be/ignored", "")
		if writer != buf {
			t.Errorf("expected: %v, saw: %v", buf, writer)
		}
	}
}

func TestSetupOutputWriterFile(t *testing.T) {
	file := "output"
	extension := ".json"
	dir, err := ioutil.TempDir(os.TempDir(), "out")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fullPath := path.Join(dir, file) + extension
	defer os.RemoveAll(dir)

	_, _, buf, _ := genericclioptions.NewTestIOStreams()
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	writer := setupOutputWriter(dir, buf, file, extension)
	if writer == buf {
		t.Errorf("expected: %v, saw: %v", buf, writer)
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
