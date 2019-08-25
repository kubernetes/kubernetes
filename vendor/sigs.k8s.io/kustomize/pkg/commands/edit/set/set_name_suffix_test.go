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
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

const (
	goodSuffixValue = "-acme"
)

func TestSetNameSuffixHappyPath(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomization()

	cmd := newCmdSetNameSuffix(fakeFS)
	args := []string{goodSuffixValue}
	err := cmd.RunE(cmd, args)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
	content, err := fakeFS.ReadTestKustomization()
	if err != nil {
		t.Errorf("unexpected read error: %v", err)
	}
	if !strings.Contains(string(content), goodSuffixValue) {
		t.Errorf("expected suffix value in kustomization file")
	}
}

func TestSetNameSuffixNoArgs(t *testing.T) {
	fakeFS := fs.MakeFakeFS()

	cmd := newCmdSetNameSuffix(fakeFS)
	err := cmd.Execute()
	if err == nil {
		t.Errorf("expected error: %v", err)
	}
	if err.Error() != "must specify exactly one suffix value" {
		t.Errorf("incorrect error: %v", err.Error())
	}
}
