/*
Copyright 2017 The Kubernetes Authors.

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

package add

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

const (
	resourceFileName    = "myWonderfulResource.yaml"
	resourceFileContent = `
Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
`
)

func TestAddResourceHappyPath(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteFile(resourceFileName, []byte(resourceFileContent))
	fakeFS.WriteFile(resourceFileName+"another", []byte(resourceFileContent))
	fakeFS.WriteTestKustomization()

	cmd := newCmdAddResource(fakeFS)
	args := []string{resourceFileName + "*"}
	err := cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
	content, err := fakeFS.ReadTestKustomization()
	if err != nil {
		t.Errorf("unexpected read error: %v", err)
	}
	if !strings.Contains(string(content), resourceFileName) {
		t.Errorf("expected resource name in kustomization")
	}
	if !strings.Contains(string(content), resourceFileName+"another") {
		t.Errorf("expected resource name in kustomization")
	}
}

func TestAddResourceAlreadyThere(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteFile(resourceFileName, []byte(resourceFileContent))
	fakeFS.WriteTestKustomization()

	cmd := newCmdAddResource(fakeFS)
	args := []string{resourceFileName}
	err := cmd.RunE(cmd, args)
	if err != nil {
		t.Fatalf("unexpected cmd error: %v", err)
	}

	// adding an existing resource doesn't return an error
	err = cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error :%v", err)
	}
}

func TestAddResourceNoArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()

	cmd := newCmdAddResource(fakeFS)
	err := cmd.Execute()
	if err == nil {
		t.Errorf("expected error: %v", err)
	}
	if err.Error() != "must specify a resource file" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}
