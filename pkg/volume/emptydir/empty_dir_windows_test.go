//go:build windows

/*
Copyright The Kubernetes Authors.

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

package emptydir

import (
	"os"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/mount-utils"
)

type fakeMountDetector struct {
	medium  v1.StorageMedium
	isMount bool
}

func (fake *fakeMountDetector) GetMountMedium(path string, requestedMedium v1.StorageMedium) (v1.StorageMedium, bool, *resource.Quantity, error) {
	return fake.medium, fake.isMount, nil, nil
}

func TestEmptyDirVolumeModeWindows(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EmptyDirVolumeMode, true)

	basePath, err := utiltesting.MkTmpdir("emptydir_mode_windows_test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(basePath)

	plugMgr := volume.VolumePluginMgr{}
	plugMgr.InitPlugins(ProbeVolumePlugins(), nil, volumetest.NewFakeVolumeHost(t, basePath, nil, nil))
	plug, err := plugMgr.FindPluginByName("kubernetes.io/empty-dir")
	if err != nil {
		t.Fatalf("Can't find the plugin by name: %v", err)
	}

	mode := int32(0o750)
	spec := &v1.Volume{
		Name: "test-volume",
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{Mode: &mode},
		},
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID("poduid"),
		},
	}

	mounter, err := plug.(*emptyDirPlugin).newMounterInternal(
		volume.NewSpecFromVolume(spec),
		pod,
		mount.NewFakeMounter(nil),
		&fakeMountDetector{},
	)
	if err != nil {
		t.Fatalf("Failed to make a new Mounter: %v", err)
	}

	if err := mounter.SetUp(volume.MounterArgs{}); err != nil {
		t.Fatalf("SetUp should not fail on Windows with mode set, got: %v", err)
	}

	volPath := mounter.GetPath()
	if _, err := os.Stat(volPath); err != nil {
		t.Fatalf("directory should exist after SetUp: %v", err)
	}
}
