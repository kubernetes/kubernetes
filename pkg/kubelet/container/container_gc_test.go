/*
Copyright 2023 The Kubernetes Authors.

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

package container_test

import (
	"context"
	"reflect"
	"testing"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

func TestIsContainerFsSeparateFromImageFs(t *testing.T) {
	runtime := &ctest.FakeRuntime{}
	fakeSources := ctest.NewFakeReadyProvider()

	gcContainer, err := NewContainerGC(runtime, GCPolicy{}, fakeSources)
	if err != nil {
		t.Errorf("unexpected error")
	}

	cases := []struct {
		name                          string
		containerFs                   []*runtimeapi.FilesystemUsage
		imageFs                       []*runtimeapi.FilesystemUsage
		writeableSeparateFromReadOnly bool
	}{
		{
			name:                          "Only images",
			imageFs:                       []*runtimeapi.FilesystemUsage{{FsId: &runtimeapi.FilesystemIdentifier{Mountpoint: "image"}}},
			writeableSeparateFromReadOnly: false,
		},
		{
			name:                          "images and containers",
			imageFs:                       []*runtimeapi.FilesystemUsage{{FsId: &runtimeapi.FilesystemIdentifier{Mountpoint: "image"}}},
			containerFs:                   []*runtimeapi.FilesystemUsage{{FsId: &runtimeapi.FilesystemIdentifier{Mountpoint: "container"}}},
			writeableSeparateFromReadOnly: true,
		},
		{
			name:                          "same filesystem",
			imageFs:                       []*runtimeapi.FilesystemUsage{{FsId: &runtimeapi.FilesystemIdentifier{Mountpoint: "image"}}},
			containerFs:                   []*runtimeapi.FilesystemUsage{{FsId: &runtimeapi.FilesystemIdentifier{Mountpoint: "image"}}},
			writeableSeparateFromReadOnly: false,
		},

		{
			name:                          "Only containers",
			containerFs:                   []*runtimeapi.FilesystemUsage{{FsId: &runtimeapi.FilesystemIdentifier{Mountpoint: "image"}}},
			writeableSeparateFromReadOnly: false,
		},
		{
			name:                          "neither are specified",
			writeableSeparateFromReadOnly: false,
		},
		{
			name:                          "both are empty arrays",
			writeableSeparateFromReadOnly: false,
			containerFs:                   []*runtimeapi.FilesystemUsage{},
			imageFs:                       []*runtimeapi.FilesystemUsage{},
		},
		{
			name:                          "FsId does not exist",
			writeableSeparateFromReadOnly: false,
			containerFs:                   []*runtimeapi.FilesystemUsage{{UsedBytes: &runtimeapi.UInt64Value{Value: 10}}},
			imageFs:                       []*runtimeapi.FilesystemUsage{{UsedBytes: &runtimeapi.UInt64Value{Value: 10}}},
		},
	}

	for _, tc := range cases {
		runtime.SetContainerFsStats(tc.containerFs)
		runtime.SetImageFsStats(tc.imageFs)
		actualCommand := gcContainer.IsContainerFsSeparateFromImageFs(context.TODO())

		if e, a := tc.writeableSeparateFromReadOnly, actualCommand; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: unexpected value; expected %v, got %v", tc.name, e, a)
		}
		runtime.SetContainerFsStats(nil)
		runtime.SetImageFsStats(nil)
	}
}
