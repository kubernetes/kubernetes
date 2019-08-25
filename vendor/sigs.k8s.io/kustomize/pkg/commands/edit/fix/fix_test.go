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

package fix

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

func TestFix(t *testing.T) {
	fakeFS := fs.MakeFakeFS()
	fakeFS.WriteTestKustomizationWith([]byte(`nameprefix: some-prefix-`))

	cmd := NewCmdFix(fakeFS)
	err := cmd.RunE(cmd, nil)
	if err != nil {
		t.Errorf("unexpected cmd error: %v", err)
	}
	content, err := fakeFS.ReadTestKustomization()
	if err != nil {
		t.Errorf("unexpected read error: %v", err)
	}
	if !strings.Contains(string(content), "apiVersion: ") {
		t.Errorf("expected apiVersion in kustomization")
	}
	if !strings.Contains(string(content), "kind: Kustomization") {
		t.Errorf("expected kind in kustomization")
	}
}
