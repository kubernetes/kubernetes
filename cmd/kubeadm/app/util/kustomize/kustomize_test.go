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

func TestKustomizeWithoutKustomizationFile(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	strategicMergePatch1 := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
        annotations:
            kustomize: patch for kube-apiserver
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-1.yaml"), []byte(strategicMergePatch1), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	strategicMergePatch2 := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-scheduler
        annotations:
            kustomize: patch for kube-scheduler
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-2.yaml"), []byte(strategicMergePatch2), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	km, err := GetManager(tmpdir)
	if err != nil {
		t.Fatalf("GetManager returned unexpected error: %v", err)
	}

	resource := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
    `)

	kustomized, err := km.Kustomize([]byte(resource))
	if err != nil {
		t.Fatalf("Kustomize returned unexpected error: %v", err)
	}

	if !strings.Contains(string(kustomized), "kustomize: patch for kube-apiserver") {
		t.Error("Kustomize did not apply strategicMergePatch")
	}

	if strings.Contains(string(kustomized), "kustomize: patch for kube-scheduler") {
		t.Error("Kustomize did apply patches not corresponding to the resource")
	}
}

func TestKustomizeWithKustomizationFile(t *testing.T) {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	kustomizationFile := dedent.Dedent(`
    patchesJson6902:
    - target:
        version: v1
        kind: Pod
        name: kube-apiserver
      path: patch-1.yaml
    - target:
        version: v1
        kind: Pod
        name: kube-scheduler
      path: patch-2.yaml
    patchesStrategicMerge:
    - patch-3.yaml
    - patch-4.yaml
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "kustomization.yaml"), []byte(kustomizationFile), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	jsonPatch1 := dedent.Dedent(`
    - op: add
      path: "/metadata/labels"
      value: 
          kustomize1: patch for kube-apiserver
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-1.yaml"), []byte(jsonPatch1), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	jsonPatch2 := dedent.Dedent(`
    - op: add
      path: "/metadata/labels"
      value:
      kustomize1: patch for kube-scheduler	
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-2.yaml"), []byte(jsonPatch2), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	strategicMergePatch1 := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
        annotations:
            kustomize2: patch for kube-apiserver
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-3.yaml"), []byte(strategicMergePatch1), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	strategicMergePatch2 := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-scheduler
        annotations:
            kustomize2: patch for kube-scheduler
    `)

	err = ioutil.WriteFile(filepath.Join(tmpdir, "patch-4.yaml"), []byte(strategicMergePatch2), 0644)
	if err != nil {
		t.Fatalf("WriteFile returned unexpected error: %v", err)
	}

	km, err := GetManager(tmpdir)
	if err != nil {
		t.Fatalf("GetManager returned unexpected error: %v", err)
	}

	resource := dedent.Dedent(`
    apiVersion: v1
    kind: Pod
    metadata:
        name: kube-apiserver
    `)

	kustomized, err := km.Kustomize([]byte(resource))
	if err != nil {
		t.Fatalf("Kustomize returned unexpected error: %v", err)
	}

	if !strings.Contains(string(kustomized), "kustomize1: patch for kube-apiserver") {
		t.Error("Kustomize did not apply json patches corresponding to the resource")
	}

	if strings.Contains(string(kustomized), "kustomize1: patch for kube-scheduler") {
		t.Error("Kustomize did apply json patches not corresponding to the resource")
	}

	if !strings.Contains(string(kustomized), "kustomize2: patch for kube-apiserver") {
		t.Error("Kustomize did not apply strategic merge patches corresponding to the resource")
	}

	if strings.Contains(string(kustomized), "kustomize2: patch for kube-scheduler") {
		t.Error("Kustomize did apply strategic merge patches not corresponding to the resource")
	}
}
