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

package set

import (
	"fmt"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/validators"
)

const (
	goodNamespaceValue = "staging"
)

func TestSetNamespaceHappyPath(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()

	cmd := newCmdSetNamespace(fakeFS, validators.MakeFakeValidator())
	args := []string{goodNamespaceValue}
	err := cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
	content, err := fakeFS.ReadTestKustomization()
	if err != nil {
		t.Errorf("unexpected read error: %v", err)
	}
	expected := []byte(fmt.Sprintf("namespace: %s", goodNamespaceValue))
	if !strings.Contains(string(content), string(expected)) {
		t.Errorf("expected namespace in kustomization file")
	}
}

func TestSetNamespaceOverride(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()

	cmd := newCmdSetNamespace(fakeFS, validators.MakeFakeValidator())
	args := []string{goodNamespaceValue}
	err := cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
	args = []string{"newnamespace"}
	err = cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
	content, err := fakeFS.ReadTestKustomization()
	if err != nil {
		t.Errorf("unexpected read error: %v", err)
	}
	expected := []byte("namespace: newnamespace")
	if !strings.Contains(string(content), string(expected)) {
		t.Errorf("expected namespace in kustomization file %s", string(content))
	}
}

func TestSetNamespaceNoArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()

	cmd := newCmdSetNamespace(fakeFS, validators.MakeFakeValidator())
	err := cmd.Execute()
	if err == nil {
		t.Errorf("expected error: %v", err)
	}
	if err.Error() != "must specify exactly one namespace value" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}

func TestSetNamespaceInvalid(t *testing.T) {
	fakeFS := fs.MakeFakeFS()

	cmd := newCmdSetNamespace(fakeFS, validators.MakeFakeValidator())
	args := []string{"/badnamespace/"}
	err := cmd.RunE(cmd, args)
	if err == nil {
		t.Errorf("expected error: %v", err)
	}
	if !strings.Contains(err.Error(), "is not a valid namespace name") {
		t.Errorf("unexpected error: %v", err.Error())
	}
}
