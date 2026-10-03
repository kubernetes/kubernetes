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

package config

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestExtractFromBadDataFile(t *testing.T) {
	dirName, err := mkTempDir("file-test")
	if err != nil {
		t.Fatalf("unable to create temp dir: %v", err)
	}
	defer removeAll(dirName, t)

	logger, _ := ktesting.NewTestContext(t)
	fileName := filepath.Join(dirName, "test_pod_config")
	err = os.WriteFile(fileName, []byte{1, 2, 3}, 0555)
	if err != nil {
		t.Fatalf("unable to write test file %#v", err)
	}

	ch := make(chan sourceUpdate, 1)
	lw := newSourceFile(fileName, "localhost", time.Millisecond, ch)
	err = lw.listConfig(logger)
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractFromEmptyDir(t *testing.T) {
	dirName, err := mkTempDir("file-test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer removeAll(dirName, t)

	logger, _ := ktesting.NewTestContext(t)
	ch := make(chan sourceUpdate, 1)
	lw := newSourceFile(dirName, "localhost", time.Millisecond, ch)
	err = lw.listConfig(logger)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	update, ok := <-ch
	if !ok {
		t.Fatalf("unexpected type: %#v", update)
	}
	expected := createSourceUpdate() // Expect empty update.
	if !apiequality.Semantic.DeepEqual(expected, update) {
		t.Fatalf("expected %#v, got %#v", expected, update)
	}
}

func mkTempDir(prefix string) (string, error) {
	return os.MkdirTemp(os.TempDir(), prefix)
}

func removeAll(dir string, t *testing.T) {
	if err := os.RemoveAll(dir); err != nil {
		t.Fatalf("unable to remove dir %s: %v", dir, err)
	}
}

func TestExtractFromDirDuplicatePod(t *testing.T) {
	dirName, err := mkTempDir("file-test")
	if err != nil {
		t.Fatalf("unable to create temp dir: %v", err)
	}
	defer removeAll(dirName, t)

	logger, _ := ktesting.NewTestContext(t)

	podManifest := `apiVersion: v1
kind: Pod
metadata:
  name: test-pod
  namespace: default
spec:
  containers:
  - name: test
    image: nginx:1.25
`

	backupManifest := `apiVersion: v1
kind: Pod
metadata:
  name: test-pod
  namespace: default
spec:
  containers:
  - name: test
    image: nginx:1.26
`

	manifestPath := filepath.Join(dirName, "test-pod.yaml")
	backupPath := filepath.Join(dirName, "test-pod.yaml.backup")

	if err := os.WriteFile(manifestPath, []byte(podManifest), 0644); err != nil {
		t.Fatalf("unable to write manifest: %v", err)
	}

	if err := os.WriteFile(backupPath, []byte(backupManifest), 0644); err != nil {
		t.Fatalf("unable to write backup manifest: %v", err)
	}

	ch := make(chan sourceUpdate, 1)
	lw := newSourceFile(dirName, "localhost", time.Millisecond, ch)

	pods, err := lw.extractFromDir(logger, dirName)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(pods) != 1 {
		t.Fatalf("expected 1 pod, got %d", len(pods))
	}

	if pods[0].Spec.Containers[0].Image != "nginx:1.25" {
		t.Fatalf(
			"expected first manifest to win, got image %q",
			pods[0].Spec.Containers[0].Image,
		)
	}
}
