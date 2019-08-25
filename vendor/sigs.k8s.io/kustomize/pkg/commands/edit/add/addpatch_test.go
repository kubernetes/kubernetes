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
	"testing"

	"strings"

	"sigs.k8s.io/kustomize/pkg/fs"
)

const (
	patchFileName    = "myWonderfulPatch.yaml"
	patchFileContent = `
Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
`
)

func TestAddPatchHappyPath(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteFile(patchFileName, []byte(patchFileContent))
	fakeFS.WriteFile(patchFileName+"another", []byte(patchFileContent))
	fakeFS.WriteTestKustomization()

	cmd := newCmdAddPatch(fakeFS)
	args := []string{patchFileName + "*"}
	err := cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
	content, err := fakeFS.ReadTestKustomization()
	if err != nil {
		t.Errorf("unexpected read error: %v", err)
	}
	if !strings.Contains(string(content), patchFileName) {
		t.Errorf("expected patch name in kustomization")
	}
	if !strings.Contains(string(content), patchFileName+"another") {
		t.Errorf("expected patch name in kustomization")
	}
}

func TestAddPatchAlreadyThere(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteFile(patchFileName, []byte(patchFileContent))
	fakeFS.WriteTestKustomization()

	cmd := newCmdAddPatch(fakeFS)
	args := []string{patchFileName}
	err := cmd.RunE(cmd, args)
	if err != nil {
		t.Fatalf("unexpected cmd error: %v", err)
	}

	// adding an existing patch shouldn't return an error
	err = cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
}

func TestAddPatchNoArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()

	cmd := newCmdAddPatch(fakeFS)
	err := cmd.Execute()
	if err == nil {
		t.Errorf("expected error: %v", err)
	}
	if err.Error() != "must specify a patch file" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}
