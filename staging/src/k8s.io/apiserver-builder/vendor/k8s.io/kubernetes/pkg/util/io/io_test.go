/*
Copyright 2015 The Kubernetes Authors.

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

package io_test

import (
	"fmt"
	"os"
	"testing"

	"github.com/pborman/uuid"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/volume"
)

func TestSavePodToFile(t *testing.T) {
	pod := volume.NewPersistentVolumeRecyclerPodTemplate()

	// sets all default values on a pod for equality comparison after decoding from file
	codec := api.Codecs.LegacyCodec(api.Registry.GroupOrDie(api.GroupName).GroupVersion)
	encoded, err := runtime.Encode(codec, pod)
	runtime.DecodeInto(codec, encoded, pod)

	tmpDir := utiltesting.MkTmpdirOrDie("kube-io-test")
	defer os.RemoveAll(tmpDir)
	path := fmt.Sprintf("/%s/kube-io-test-%s", tmpDir, uuid.New())

	if err := io.SavePodToFile(pod, path, 777); err != nil {
		t.Fatalf("failed to save pod to file: %v", err)
	}

	podFromFile, err := io.LoadPodFromFile(path)
	if err != nil {
		t.Fatalf("failed to load pod from file: %v", err)
	}
	if !apiequality.Semantic.DeepEqual(pod, podFromFile) {
		t.Errorf("\nexpected %#v\ngot	%#v\n", pod, podFromFile)
	}
}
