/*
Copyright 2017 The Kubernetes Authors.

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
	"errors"
	"io"
	"reflect"
	"testing"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/csi/fake"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

type fakeCsiDriverClient struct {
	t          *testing.T
	nodeClient *fake.NodeClient
}

func newFakeCsiDriverClient(t *testing.T, stagingCapable bool) *fakeCsiDriverClient {
	return &fakeCsiDriverClient{
		t:          t,
		nodeClient: fake.NewNodeClient(stagingCapable),
	}
}

func newFakeCsiDriverClientWithExpansion(t *testing.T, stagingCapable bool, expansionSet bool) *fakeCsiDriverClient {
	return &fakeCsiDriverClient{
		t:          t,
		nodeClient: fake.NewNodeClientWithExpansion(stagingCapable, expansionSet),
	}
}

func newFakeCsiDriverClientWithVolumeStats(t *testing.T, volumeStatsSet bool) *fakeCsiDriverClient {
	return &fakeCsiDriverClient{
		t:          t,
		nodeClient: fake.NewNodeClientWithVolumeStats(volumeStatsSet),
	}
}

func (c *fakeCsiDriverClient) NodeGetInfo(ctx context.Context) (
	nodeID string,
	maxVolumePerNode int64,
	accessibleTopology map[string]string,
	err error) {
	resp, err := c.nodeClient.NodeGetInfo(ctx, &csipbv1.NodeGetInfoRequest{})
	topology := resp.GetAccessibleTopology()
	if topology != nil {
		accessibleTopology = topology.Segments
	}
	return resp.GetNodeId(), resp.GetMaxVolumesPerNode(), accessibleTopology, err
}

func (c *fakeCsiDriverClient) NodeGetVolumeStats(ctx context.Context, volID string, targetPath string) (
	usageCountMap *volume.Metrics, err error) {
	c.t.Log("calling fake.NodeGetVolumeStats...")
	req := &csipbv1.NodeGetVolumeStatsRequest{
		VolumeId:   volID,
		VolumePath: targetPath,
	}
	resp, err := c.nodeClient.NodeGetVolumeStats(ctx, req)
	usages := resp.GetUsage()
	metrics := &volume.Metrics{}
	if usages == nil {
		return nil, nil
	}
	for _, usage := range usages {
		if usage == nil {
			continue
		}
		unit := usage.GetUnit()
		switch unit {
		case csipbv1.VolumeUsage_BYTES:
			metrics.Available = resource.NewQuantity(usage.GetAvailable(), resource.BinarySI)
			metrics.Capacity = resource.NewQuantity(usage.GetTotal(), resource.BinarySI)
			metrics.Used = resource.NewQuantity(usage.GetUsed(), resource.BinarySI)
		case csipbv1.VolumeUsage_INODES:
			metrics.InodesFree = resource.NewQuantity(usage.GetAvailable(), resource.BinarySI)
			metrics.Inodes = resource.NewQuantity(usage.GetTotal(), resource.BinarySI)
			metrics.InodesUsed = resource.NewQuantity(usage.GetUsed(), resource.BinarySI)
		}
	}
	return metrics, nil
}

func (c *fakeCsiDriverClient) NodeSupportsVolumeStats(ctx context.Context) (bool, error) {
	c.t.Log("calling fake.NodeSupportsVolumeStats...")
	req := &csipbv1.NodeGetCapabilitiesRequest{}
	resp, err := c.nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return false, err
	}
	capabilities := resp.GetCapabilities()
	if capabilities == nil {
		return false, nil
	}
	for _, capability := range capabilities {
		if capability.GetRpc().GetType() == csipbv1.NodeServiceCapability_RPC_GET_VOLUME_STATS {
			return true, nil
		}
	}
	return false, nil
}

func (c *fakeCsiDriverClient) NodePublishVolume(
	ctx context.Context,
	volID string,
	readOnly bool,
	stagingTargetPath string,
	targetPath string,
	accessMode api.PersistentVolumeAccessMode,
	publishContext map[string]string,
	volumeContext map[string]string,
	secrets map[string]string,
	fsType string,
	mountOptions []string,
) error {
	c.t.Log("calling fake.NodePublishVolume...")
	req := &csipbv1.NodePublishVolumeRequest{
		VolumeId:          volID,
		TargetPath:        targetPath,
		StagingTargetPath: stagingTargetPath,
		Readonly:          readOnly,
		PublishContext:    publishContext,
		VolumeContext:     volumeContext,
		Secrets:           secrets,
		VolumeCapability: &csipbv1.VolumeCapability{
			AccessMode: &csipbv1.VolumeCapability_AccessMode{
				Mode: asCSIAccessModeV1(accessMode),
			},
			AccessType: &csipbv1.VolumeCapability_Mount{
				Mount: &csipbv1.VolumeCapability_MountVolume{
					FsType:     fsType,
					MountFlags: mountOptions,
				},
			},
		},
	}

	_, err := c.nodeClient.NodePublishVolume(ctx, req)
	if err != nil && !isFinalError(err) {
		return volumetypes.NewUncertainProgressError(err.Error())
	}
	return err
}

func (c *fakeCsiDriverClient) NodeUnpublishVolume(ctx context.Context, volID string, targetPath string) error {
	c.t.Log("calling fake.NodeUnpublishVolume...")
	req := &csipbv1.NodeUnpublishVolumeRequest{
		VolumeId:   volID,
		TargetPath: targetPath,
	}

	_, err := c.nodeClient.NodeUnpublishVolume(ctx, req)
	return err
}

func (c *fakeCsiDriverClient) NodeStageVolume(ctx context.Context,
	volID string,
	publishContext map[string]string,
	stagingTargetPath string,
	fsType string,
	accessMode api.PersistentVolumeAccessMode,
	secrets map[string]string,
	volumeContext map[string]string,
	mountOptions []string,
) error {
	c.t.Log("calling fake.NodeStageVolume...")
	req := &csipbv1.NodeStageVolumeRequest{
		VolumeId:          volID,
		PublishContext:    publishContext,
		StagingTargetPath: stagingTargetPath,
		VolumeCapability: &csipbv1.VolumeCapability{
			AccessMode: &csipbv1.VolumeCapability_AccessMode{
				Mode: asCSIAccessModeV1(accessMode),
			},
			AccessType: &csipbv1.VolumeCapability_Mount{
				Mount: &csipbv1.VolumeCapability_MountVolume{
					FsType:     fsType,
					MountFlags: mountOptions,
				},
			},
		},
		Secrets:       secrets,
		VolumeContext: volumeContext,
	}

	_, err := c.nodeClient.NodeStageVolume(ctx, req)
	if err != nil && !isFinalError(err) {
		return volumetypes.NewUncertainProgressError(err.Error())
	}
	return err
}

func (c *fakeCsiDriverClient) NodeUnstageVolume(ctx context.Context, volID, stagingTargetPath string) error {
	c.t.Log("calling fake.NodeUnstageVolume...")
	req := &csipbv1.NodeUnstageVolumeRequest{
		VolumeId:          volID,
		StagingTargetPath: stagingTargetPath,
	}
	_, err := c.nodeClient.NodeUnstageVolume(ctx, req)
	return err
}

func (c *fakeCsiDriverClient) NodeSupportsNodeExpand(ctx context.Context) (bool, error) {
	c.t.Log("calling fake.NodeSupportsNodeExpand...")
	req := &csipbv1.NodeGetCapabilitiesRequest{}

	resp, err := c.nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return false, err
	}

	capabilities := resp.GetCapabilities()

	if capabilities == nil {
		return false, nil
	}
	for _, capability := range capabilities {
		if capability.GetRpc().GetType() == csipbv1.NodeServiceCapability_RPC_EXPAND_VOLUME {
			return true, nil
		}
	}
	return false, nil
}

func (c *fakeCsiDriverClient) NodeSupportsStageUnstage(ctx context.Context) (bool, error) {
	c.t.Log("calling fake.NodeGetCapabilities for NodeSupportsStageUnstage...")
	req := &csipbv1.NodeGetCapabilitiesRequest{}
	resp, err := c.nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return false, err
	}

	capabilities := resp.GetCapabilities()

	stageUnstageSet := false
	if capabilities == nil {
		return false, nil
	}
	for _, capability := range capabilities {
		if capability.GetRpc().GetType() == csipbv1.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME {
			stageUnstageSet = true
		}
	}
	return stageUnstageSet, nil
}

func (c *fakeCsiDriverClient) NodeExpandVolume(ctx context.Context, volumeid, volumePath string, newSize resource.Quantity) (resource.Quantity, error) {
	c.t.Log("calling fake.NodeExpandVolume")
	req := &csipbv1.NodeExpandVolumeRequest{
		VolumeId:      volumeid,
		VolumePath:    volumePath,
		CapacityRange: &csipbv1.CapacityRange{RequiredBytes: newSize.Value()},
	}
	resp, err := c.nodeClient.NodeExpandVolume(ctx, req)
	if err != nil {
		return newSize, err
	}
	updatedQuantity := resource.NewQuantity(resp.CapacityBytes, resource.BinarySI)
	return *updatedQuantity, nil
}

func setupClient(t *testing.T, stageUnstageSet bool) csiClient {
	return newFakeCsiDriverClient(t, stageUnstageSet)
}

func setupClientWithExpansion(t *testing.T, stageUnstageSet bool, expansionSet bool) csiClient {
	return newFakeCsiDriverClientWithExpansion(t, stageUnstageSet, expansionSet)
}

func setupClientWithVolumeStats(t *testing.T, volumeStatsSet bool) csiClient {
	return newFakeCsiDriverClientWithVolumeStats(t, volumeStatsSet)
}

func checkErr(t *testing.T, expectedAnError bool, actualError error) {
	t.Helper()

	errOccurred := actualError != nil

	if expectedAnError && !errOccurred {
		t.Error("expected an error")
	}

	if !expectedAnError && errOccurred {
		t.Errorf("expected no error, got: %v", actualError)
	}
}

func TestClientNodeGetInfo(t *testing.T) {
	testCases := []struct {
		name                       string
		expectedNodeID             string
		expectedMaxVolumePerNode   int64
		expectedAccessibleTopology map[string]string
		mustFail                   bool
		err                        error
	}{
		{
			name:                       "test ok",
			expectedNodeID:             "node1",
			expectedMaxVolumePerNode:   16,
			expectedAccessibleTopology: map[string]string{"com.example.csi-topology/zone": "zone1"},
		},
		{
			name:     "grpc error",
			mustFail: true,
			err:      errors.New("grpc error"),
		},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)

		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClient(false /* stagingCapable */)
				nodeClient.SetNextError(tc.err)
				nodeClient.SetNodeGetInfoResp(&csipbv1.NodeGetInfoResponse{
					NodeId:            tc.expectedNodeID,
					MaxVolumesPerNode: tc.expectedMaxVolumePerNode,
					AccessibleTopology: &csipbv1.Topology{
						Segments: tc.expectedAccessibleTopology,
					},
				})
				return nodeClient, fakeCloser, nil
			},
		}

		nodeID, maxVolumePerNode, accessibleTopology, err := client.NodeGetInfo(context.Background())
		checkErr(t, tc.mustFail, err)

		if nodeID != tc.expectedNodeID {
			t.Errorf("expected nodeID: %v; got: %v", tc.expectedNodeID, nodeID)
		}

		if maxVolumePerNode != tc.expectedMaxVolumePerNode {
			t.Errorf("expected maxVolumePerNode: %v; got: %v", tc.expectedMaxVolumePerNode, maxVolumePerNode)
		}

		if !reflect.DeepEqual(accessibleTopology, tc.expectedAccessibleTopology) {
			t.Errorf("expected accessibleTopology: %v; got: %v", tc.expectedAccessibleTopology, accessibleTopology)
		}

		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

func TestClientNodePublishVolume(t *testing.T) {
	testCases := []struct {
		name       string
		volID      string
		targetPath string
		fsType     string
		mustFail   bool
		err        error
	}{
		{name: "test ok", volID: "vol-test", targetPath: "/test/path"},
		{name: "missing volID", targetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "bad fs", volID: "vol-test", targetPath: "/test/path", fsType: "badfs", mustFail: true},
		{name: "grpc error", volID: "vol-test", targetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClient(false /* stagingCapable */)
				nodeClient.SetNextError(tc.err)
				return nodeClient, fakeCloser, nil
			},
		}

		err := client.NodePublishVolume(
			context.Background(),
			tc.volID,
			false,
			"",
			tc.targetPath,
			api.ReadWriteOnce,
			map[string]string{"device": "/dev/null"},
			map[string]string{"attr0": "val0"},
			map[string]string{},
			tc.fsType,
			[]string{},
		)
		checkErr(t, tc.mustFail, err)

		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

func TestClientNodeUnpublishVolume(t *testing.T) {
	testCases := []struct {
		name       string
		volID      string
		targetPath string
		mustFail   bool
		err        error
	}{
		{name: "test ok", volID: "vol-test", targetPath: "/test/path"},
		{name: "missing volID", targetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "grpc error", volID: "vol-test", targetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClient(false /* stagingCapable */)
				nodeClient.SetNextError(tc.err)
				return nodeClient, fakeCloser, nil
			},
		}

		err := client.NodeUnpublishVolume(context.Background(), tc.volID, tc.targetPath)
		checkErr(t, tc.mustFail, err)

		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

func TestClientNodeStageVolume(t *testing.T) {
	testCases := []struct {
		name              string
		volID             string
		stagingTargetPath string
		fsType            string
		secrets           map[string]string
		mountOptions      []string
		mustFail          bool
		err               error
	}{
		{name: "test ok", volID: "vol-test", stagingTargetPath: "/test/path", fsType: "ext4", mountOptions: []string{"unvalidated"}},
		{name: "missing volID", stagingTargetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "bad fs", volID: "vol-test", stagingTargetPath: "/test/path", fsType: "badfs", mustFail: true},
		{name: "grpc error", volID: "vol-test", stagingTargetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClient(false /* stagingCapable */)
				nodeClient.SetNextError(tc.err)
				return nodeClient, fakeCloser, nil
			},
		}

		err := client.NodeStageVolume(
			context.Background(),
			tc.volID,
			map[string]string{"device": "/dev/null"},
			tc.stagingTargetPath,
			tc.fsType,
			api.ReadWriteOnce,
			tc.secrets,
			map[string]string{"attr0": "val0"},
			tc.mountOptions,
		)
		checkErr(t, tc.mustFail, err)

		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

func TestClientNodeUnstageVolume(t *testing.T) {
	testCases := []struct {
		name              string
		volID             string
		stagingTargetPath string
		mustFail          bool
		err               error
	}{
		{name: "test ok", volID: "vol-test", stagingTargetPath: "/test/path"},
		{name: "missing volID", stagingTargetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "grpc error", volID: "vol-test", stagingTargetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClient(false /* stagingCapable */)
				nodeClient.SetNextError(tc.err)
				return nodeClient, fakeCloser, nil
			},
		}

		err := client.NodeUnstageVolume(
			context.Background(),
			tc.volID, tc.stagingTargetPath,
		)
		checkErr(t, tc.mustFail, err)

		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

func TestNodeExpandVolume(t *testing.T) {
	testCases := []struct {
		name       string
		volID      string
		volumePath string
		newSize    resource.Quantity
		mustFail   bool
		err        error
	}{
		{
			name:       "with all correct values",
			volID:      "vol-abcde",
			volumePath: "/foo/bar",
			newSize:    resource.MustParse("10Gi"),
			mustFail:   false,
		},
		{
			name:       "with missing volume-id",
			volumePath: "/foo/bar",
			newSize:    resource.MustParse("10Gi"),
			mustFail:   true,
		},
		{
			name:     "with missing volume path",
			volID:    "vol-1234",
			newSize:  resource.MustParse("10Gi"),
			mustFail: true,
		},
		{
			name:       "with invalid quantity",
			volID:      "vol-1234",
			volumePath: "/foo/bar",
			newSize:    *resource.NewQuantity(-10, resource.DecimalSI),
			mustFail:   true,
		},
	}

	for _, tc := range testCases {
		t.Logf("Running test cases : %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClient(false /* stagingCapable */)
				nodeClient.SetNextError(tc.err)
				return nodeClient, fakeCloser, nil
			},
		}
		_, err := client.NodeExpandVolume(context.Background(), tc.volID, tc.volumePath, tc.newSize)
		checkErr(t, tc.mustFail, err)
		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

type VolumeStatsOptions struct {
	VolumeSpec *volume.Spec

	// this just could be volumeID
	VolumeID string

	// DeviceMountPath location where device is mounted on the node. If volume type
	// is attachable - this would be global mount path otherwise
	// it would be location where volume was mounted for the pod
	DeviceMountPath string
}

func TestVolumeStats(t *testing.T) {
	spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "metrics", "test-vol"), false)
	tests := []struct {
		name           string
		volumeStatsSet bool
		volumeData     VolumeStatsOptions
		success        bool
	}{
		{
			name:           "when nodeVolumeStats=on, VolumeID=on, DeviceMountPath=on",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: true,
		},

		{
			name:           "when nodeVolumeStats=off, VolumeID=on, DeviceMountPath=on",
			volumeStatsSet: false,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: false,
		},

		{
			name:           "when nodeVolumeStats=on, VolumeID=off, DeviceMountPath=on",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "",
				DeviceMountPath: "/foo/bar",
			},
			success: false,
		},

		{
			name:           "when nodeVolumeStats=on, VolumeID=on, DeviceMountPath=off",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "",
			},
			success: false,
		},
		{
			name:           "when nodeVolumeStats=on, VolumeID=on, DeviceMountPath=off",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "",
				DeviceMountPath: "",
			},
			success: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
			defer cancel()
			csiSource, _ := getCSISourceFromSpec(tc.volumeData.VolumeSpec)
			csClient := setupClientWithVolumeStats(t, tc.volumeStatsSet)
			_, err := csClient.NodeGetVolumeStats(ctx, csiSource.VolumeHandle, tc.volumeData.DeviceMountPath)
			if err != nil && tc.success {
				t.Errorf("For %s : expected %v got %v", tc.name, tc.success, err)
			}
		})
	}

}
