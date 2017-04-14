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
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	utiltesting "k8s.io/client-go/util/testing"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func TestExtractFromBadDataFile(t *testing.T) {
	dirName, err := utiltesting.MkTmpdir("file-test")
	if err != nil {
		t.Fatalf("unable to create temp dir: %v", err)
	}
	defer os.RemoveAll(dirName)

	fileName := filepath.Join(dirName, "test_pod_config")
	err = ioutil.WriteFile(fileName, []byte{1, 2, 3}, 0555)
	if err != nil {
		t.Fatalf("unable to write test file %#v", err)
	}

	ch := make(chan interface{}, 1)
	c := new(fileName, "localhost", time.Millisecond, ch)
	err = c.fullScan()
	if err == nil {
		t.Fatalf("expected error, got nil")
	}
	expectEmptyChannel(t, ch)
}

func TestExtractFromEmptyDir(t *testing.T) {
	dirName, err := utiltesting.MkTmpdir("file-test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer os.RemoveAll(dirName)

	ch := make(chan interface{}, 1)
	c := new(dirName, "localhost", time.Millisecond, ch)
	err = c.fullScan()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	update := (<-ch).(kubetypes.PodUpdate)
	expected := CreatePodUpdate(kubetypes.SET, kubetypes.FileSource)
	if !apiequality.Semantic.DeepEqual(expected, update) {
		t.Fatalf("expected %#v, Got %#v", expected, update)
	}
}
