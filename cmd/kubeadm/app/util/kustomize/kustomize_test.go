/*
Copyright 2019 The Kubernetes Authors.

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

package kustomize

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lithammer/dedent"
)

func TestKustomize(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	resourceString := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
    `)

	patch1String := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
        annotations:
            kustomize: patch for kube-apiserver
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-1.yaml"), []byte(patch1String), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	patch2String := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-scheduler
        annotations:
            kustomize: patch for kube-scheduler
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-2.yaml"), []byte(patch2String), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	km, err := GetManager(tmpdir)
	if err != nil {
		t.Errorf("GetManager returned unexpected error: %v", err)
	}

	kustomized, err := km.Kustomize([]byte(resourceString))
	if err != nil {
		t.Errorf("Kustomize returned unexpected error: %v", err)
	}

	if !strings.Contains(string(kustomized), "kustomize: patch for kube-apiserver") {
		t.Error("Kustomize did not apply patches corresponding to the resource")
	}

	if strings.Contains(string(kustomized), "kustomize: patch for kube-scheduler") {
		t.Error("Kustomize did apply patches not corresponding to the resource")
	}
}
