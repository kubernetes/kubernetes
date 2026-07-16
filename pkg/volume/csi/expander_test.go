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

package csi

import (
	"context"
	"os"
	"reflect"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

func TestNodeExpand(t *testing.T) {
	tests := []struct {
		name                      string
		nodeExpansion             bool
		nodeStageSet              bool
		success                   bool
		fsVolume                  bool
		grpcError                 error
		hasVolumeInUseError       bool
		deviceStagePath           string
		enableCSINodeExpandSecret bool
		secret                    *api.Secret
	}{
		{
			name:    "when node expansion is not set",
			success: false,
		},
		{
			name:            "when nodeExpansion=on, nodeStage=on, volumePhase=staged",
			nodeExpansion:   true,
			nodeStageSet:    true,
			success:         true,
			fsVolume:        true,
			deviceStagePath: "/foo/bar",
		},
		{
			name:          "when nodeExpansion=on, nodeStage=on, volumePhase=published",
			nodeExpansion: true,
			nodeStageSet:  true,
			success:       true,
			fsVolume:      true,
		},
		{
			name:          "when nodeExpansion=on, nodeStage=off, volumePhase=published",
			nodeExpansion: true,
			success:       true,
			fsVolume:      true,
		},
		{
			name:          "when nodeExpansion=on, nodeStage=off, volumePhase=published, fsVolume=false",
			nodeExpansion: true,
			success:       true,
			fsVolume:      false,
		},
		{
			name:                "when nodeExpansion=on, nodeStage=on, volumePhase=published has grpc volume-in-use error",
			nodeExpansion:       true,
			nodeStageSet:        true,
			success:             false,
			fsVolume:            true,
			grpcError:           status.Error(codes.FailedPrecondition, "volume-in-use"),
			hasVolumeInUseError: true,
		},
		{
			name:                "when nodeExpansion=on, nodeStage=on, volumePhase=published has other grpc error",
			nodeExpansion:       true,
			nodeStageSet:        true,
			success:             false,
			fsVolume:            true,
			grpcError:           status.Error(codes.InvalidArgument, "invalid-argument"),
			hasVolumeInUseError: false,
		},
		{
			name:                      "when nodeExpansion=on, nodeStage=on, volumePhase=staged",
			nodeExpansion:             true,
			nodeStageSet:              true,
			success:                   true,
			fsVolume:                  true,
			deviceStagePath:           "/foo/bar",
			enableCSINodeExpandSecret: true,
			secret: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "expand-secret",
					Namespace: "default",
				},
				Data: map[string][]byte{
					"apiUsername": []byte("csiusername"),
					"apiPassword": []byte("csipassword"),
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plug, tmpDir := newTestPlugin(t, nil)
			defer os.RemoveAll(tmpDir)

			spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "expandable", "test-vol"), false)
			if tc.enableCSINodeExpandSecret {
				spec.PersistentVolume.Spec.CSI.NodeExpandSecretRef = &api.SecretReference{
					Name:      tc.secret.Name,
					Namespace: tc.secret.Namespace,
				}
			}

			newSize, _ := resource.ParseQuantity("20Gi")

			resizeOptions := volume.NodeResizeOptions{
				VolumeSpec:      spec,
				NewSize:         newSize,
				DeviceMountPath: "/foo/bar",
				DeviceStagePath: "/foo/bar",
				DevicePath:      "/mnt/foobar",
			}
			csiSource, _ := getCSISourceFromSpec(resizeOptions.VolumeSpec)
			csClient := setupClientWithExpansion(t, tc.nodeStageSet, tc.nodeExpansion)

			fakeCSIClient, _ := csClient.(*fakeCsiDriverClient)
			fakeNodeClient := fakeCSIClient.nodeClient

			if tc.enableCSINodeExpandSecret {
				_, err := plug.host.GetKubeClient().CoreV1().Secrets(tc.secret.Namespace).Create(context.TODO(), tc.secret, metav1.CreateOptions{})
				if err != nil {
					t.Fatal(err)
				}
			}

			if tc.grpcError != nil {
				fakeNodeClient.SetNextError(tc.grpcError)
			}

			ok, err := plug.nodeExpandWithClient(resizeOptions, csiSource, csClient, tc.fsVolume)

			if tc.hasVolumeInUseError {
				if !volumetypes.IsFailedPreconditionError(err) {
					t.Errorf("expected failed precondition error got: %v", err)
				}
			}

			// verify device staging target path
			stagingTargetPath := fakeNodeClient.FakeNodeExpansionRequest.GetStagingTargetPath()
			if tc.deviceStagePath != "" && tc.deviceStagePath != stagingTargetPath {
				t.Errorf("For %s: expected staging path %s got %s", tc.name, tc.deviceStagePath, stagingTargetPath)
			}

			if ok != tc.success {
				if err != nil {
					t.Errorf("For %s : expected %v got %v with %v", tc.name, tc.success, ok, err)
				} else {
					t.Errorf("For %s : expected %v got %v", tc.name, tc.success, ok)
				}
			}
			// verify volume capability received by node expansion request
			if tc.success {
				capability := fakeNodeClient.FakeNodeExpansionRequest.GetVolumeCapability()
				if tc.fsVolume {
					if capability.GetMount() == nil {
						t.Errorf("For %s: expected mount accesstype got: %v", tc.name, capability)
					}
				} else {
					if capability.GetBlock() == nil {
						t.Errorf("For %s: expected block accesstype got: %v", tc.name, capability)
					}
				}
			}
		})
	}
}

func TestNodeExpandNoClientError(t *testing.T) {
	transientError := volumetypes.NewTransientOperationFailure("")
	plug, tmpDir := newTestPlugin(t, nil)
	defer os.RemoveAll(tmpDir)
	spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "expandable", "test-vol"), false)

	newSize, _ := resource.ParseQuantity("20Gi")

	resizeOptions := volume.NodeResizeOptions{
		VolumeSpec:      spec,
		NewSize:         newSize,
		DeviceMountPath: "/foo/bar",
		DeviceStagePath: "/foo/bar",
		DevicePath:      "/mnt/foobar",
	}

	_, err := plug.NodeExpand(resizeOptions)

	if err == nil {
		t.Errorf("test should fail, but no error occurred")
	} else if reflect.TypeOf(transientError) != reflect.TypeOf(err) {
		t.Fatalf("expected exitError type: %v got: %v (%v)", reflect.TypeOf(transientError), reflect.TypeOf(err), err)
	}
}
