/*
Copyright 2022 The Kubernetes Authors.

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

package reconciler

import (
	"os"
	"path/filepath"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/mount-utils"
)

func TestReconstructVolumes(t *testing.T) {
	// Arrange
	defer featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)()

	rootDir, err := os.MkdirTemp("", "volume-reconstruct-test")
	if err != nil {
		t.Fatalf("Unable to create temporary directory: %s", err)
	}
	defer os.RemoveAll(rootDir)

	volumePluginMgr, fakePlugin := volumetesting.GetTestKubeletVolumePluginMgrWithRootDir(t, rootDir)
	dsw := cache.NewDesiredStateOfWorld(volumePluginMgr)
	asw := cache.NewActualStateOfWorld(nodeName, volumePluginMgr)
	kubeClient := createTestClient()
	fakeRecorder := &record.FakeRecorder{}
	fakeHandler := volumetesting.NewBlockVolumePathHandler()
	oex := operationexecutor.NewOperationExecutor(operationexecutor.NewOperationGenerator(
		kubeClient,
		volumePluginMgr,
		fakeRecorder,
		fakeHandler))

	// Prepare pods/<uuid>/volumes/<plugin>/<name> directory with a fake mounted volume
	podUUID := "65b2b621-3f3a-422d-87e0-a1708850e85d"
	volumeSpecName := "foo"
	podsDir := filepath.Join(rootDir, config.DefaultKubeletPodsDirName)
	volumeMountPath := filepath.Join(podsDir, podUUID, config.DefaultKubeletVolumesDirName, fakePlugin.GetPluginName(), volumeSpecName)
	err = os.MkdirAll(volumeMountPath, 0755)
	if err != nil {
		t.Fatalf("Unable to create volume directory: %s", err)
	}

	// "Mount" the volume using the fake mounter, no root access is required
	mounts := []mount.MountPoint{
		{
			Device: "/dev/foo",
			Path:   volumeMountPath,
		},
	}
	mounter := mount.NewFakeMounter(mounts)

	rc := NewReconciler(
		kubeClient,
		true, /* controllerAttachDetachEnabled */
		reconcilerLoopSleepDuration,
		waitForAttachTimeout,
		nodeName,
		dsw,
		asw,
		hasAddedPods,
		oex,
		mounter,
		hostutil.NewFakeHostUtil(nil),
		volumePluginMgr,
		podsDir)

	// Act
	r := rc.(*reconciler)
	r.reconstructVolumes()

	// Assert
	volumeName, err := util.GetUniqueVolumeNameFromSpec(fakePlugin, &volume.Spec{Volume: &v1.Volume{Name: volumeSpecName}})
	if err != nil {
		t.Errorf("Failed to get volume name: %s", err)
	}
	mounted, devicePath, err := asw.PodExistsInVolume(volumetypes.UniquePodName(podUUID), volumeName)
	if err != nil {
		t.Errorf("Failed to check PodExistsInVolume: %s", err)
	}
	// PodExistsInVolume returns false on uncertain mounts (and error when the volume is not mounted at all)
	if mounted {
		t.Errorf("ASW reports the volume mounted, where uncertain was expected")
	}
	// Reconstruction does not reconstruct devicePaths
	if devicePath != "" {
		t.Errorf("Expected devicePath %q, where empty device was expected", devicePath)
	}
}
