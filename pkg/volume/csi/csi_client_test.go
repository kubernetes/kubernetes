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
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/stretchr/testify/assert"

	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltesting "k8s.io/client-go/util/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
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

func newFakeCsiDriverClientWithVolumeStatsAndCondition(t *testing.T, volumeStatsSet, volumeConditionSet, setVolumeStat, setVolumeCondition bool) *fakeCsiDriverClient {
	return &fakeCsiDriverClient{
		t:          t,
		nodeClient: fake.NewNodeClientWithVolumeStatsAndCondition(volumeStatsSet, volumeConditionSet, setVolumeStat, setVolumeCondition),
	}
}

func newFakeCsiDriverClientWithVolumeMountGroup(t *testing.T, stagingCapable, volumeMountGroupSet bool) *fakeCsiDriverClient {
	return &fakeCsiDriverClient{
		t:          t,
		nodeClient: fake.NewNodeClientWithVolumeMountGroup(stagingCapable, volumeMountGroupSet),
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
	fakeResp := &csipbv1.NodeGetVolumeStatsResponse{}
	if c.nodeClient.SetVolumeStats {
		fakeResp = getRawVolumeInfo()
	}
	if c.nodeClient.SetVolumecondition {
		fakeResp.VolumeCondition = &csipbv1.VolumeCondition{
			Abnormal: true,
			Message:  "Volume is abnormal",
		}
	}
	c.nodeClient.SetNodeVolumeStatsResp(fakeResp)

	resp, err := c.nodeClient.NodeGetVolumeStats(ctx, req)
	if err != nil {
		return nil, err
	}

	metrics := &volume.Metrics{}

	isSupportNodeVolumeCondition, err := c.nodeSupportsVolumeCondition(ctx)
	if err != nil {
		return nil, err
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIVolumeHealth) && isSupportNodeVolumeCondition {
		abnormal, message := resp.VolumeCondition.GetAbnormal(), resp.VolumeCondition.GetMessage()
		metrics.Abnormal, metrics.Message = &abnormal, &message
	}

	usages := resp.GetUsage()
	if !isSupportNodeVolumeCondition && usages == nil {
		return nil, errors.New("volume usage is nil")
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
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_GET_VOLUME_STATS)
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
	fsGroup *int64,
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
		},
	}

	if fsType == fsTypeBlockName {
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Block{
			Block: &csipbv1.VolumeCapability_BlockVolume{},
		}
	} else {
		mountVolume := &csipbv1.VolumeCapability_MountVolume{
			FsType:     fsType,
			MountFlags: mountOptions,
		}
		if fsGroup != nil {
			mountVolume.VolumeMountGroup = strconv.FormatInt(*fsGroup, 10 /* base */)
		}
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Mount{
			Mount: mountVolume,
		}
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
	fsGroup *int64,
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
		},
		Secrets:       secrets,
		VolumeContext: volumeContext,
	}
	if fsType == fsTypeBlockName {
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Block{
			Block: &csipbv1.VolumeCapability_BlockVolume{},
		}
	} else {
		mountVolume := &csipbv1.VolumeCapability_MountVolume{
			FsType:     fsType,
			MountFlags: mountOptions,
		}
		if fsGroup != nil {
			mountVolume.VolumeMountGroup = strconv.FormatInt(*fsGroup, 10 /* base */)
		}
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Mount{
			Mount: mountVolume,
		}
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
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_EXPAND_VOLUME)
}

func (c *fakeCsiDriverClient) NodeSupportsStageUnstage(ctx context.Context) (bool, error) {
	c.t.Log("calling fake.NodeGetCapabilities for NodeSupportsStageUnstage...")
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME)
}

func (c *fakeCsiDriverClient) NodeSupportsVolumeMountGroup(ctx context.Context) (bool, error) {
	c.t.Log("calling fake.NodeGetCapabilities for NodeSupportsVolumeMountGroup...")
	req := &csipbv1.NodeGetCapabilitiesRequest{}
	resp, err := c.nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return false, err
	}

	capabilities := resp.GetCapabilities()

	volumeMountGroupSet := false
	if capabilities == nil {
		return false, nil
	}
	for _, capability := range capabilities {
		if capability.GetRpc().GetType() == csipbv1.NodeServiceCapability_RPC_VOLUME_MOUNT_GROUP {
			volumeMountGroupSet = true
		}
	}
	return volumeMountGroupSet, nil
}

func (c *fakeCsiDriverClient) NodeExpandVolume(ctx context.Context, opts csiResizeOptions) (resource.Quantity, error) {
	c.t.Log("calling fake.NodeExpandVolume")
	req := &csipbv1.NodeExpandVolumeRequest{
		VolumeId:          opts.volumeID,
		VolumePath:        opts.volumePath,
		StagingTargetPath: opts.stagingTargetPath,
		CapacityRange:     &csipbv1.CapacityRange{RequiredBytes: opts.newSize.Value()},
		VolumeCapability: &csipbv1.VolumeCapability{
			AccessMode: &csipbv1.VolumeCapability_AccessMode{
				Mode: asCSIAccessModeV1(opts.accessMode),
			},
		},
	}
	if opts.fsType == fsTypeBlockName {
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Block{
			Block: &csipbv1.VolumeCapability_BlockVolume{},
		}
	} else {
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Mount{
			Mount: &csipbv1.VolumeCapability_MountVolume{
				FsType:     opts.fsType,
				MountFlags: opts.mountOptions,
			},
		}
	}
	resp, err := c.nodeClient.NodeExpandVolume(ctx, req)
	if err != nil {
		return opts.newSize, err
	}
	updatedQuantity := resource.NewQuantity(resp.CapacityBytes, resource.BinarySI)
	return *updatedQuantity, nil
}

func (c *fakeCsiDriverClient) nodeSupportsVolumeCondition(ctx context.Context) (bool, error) {
	c.t.Log("calling fake.nodeSupportsVolumeCondition...")
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_VOLUME_CONDITION)
}

func (c *fakeCsiDriverClient) NodeSupportsSingleNodeMultiWriterAccessMode(ctx context.Context) (bool, error) {
	c.t.Log("calling fake.NodeSupportsSingleNodeMultiWriterAccessMode...")
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_SINGLE_NODE_MULTI_WRITER)
}

func (c *fakeCsiDriverClient) nodeSupportsCapability(ctx context.Context, capabilityType csipbv1.NodeServiceCapability_RPC_Type) (bool, error) {
	capabilities, err := c.nodeGetCapabilities(ctx)
	if err != nil {
		return false, err
	}

	for _, capability := range capabilities {
		if capability.GetRpc().GetType() == capabilityType {
			return true, nil
		}
	}
	return false, nil
}

func (c *fakeCsiDriverClient) nodeGetCapabilities(ctx context.Context) ([]*csipbv1.NodeServiceCapability, error) {
	req := &csipbv1.NodeGetCapabilitiesRequest{}
	resp, err := c.nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return []*csipbv1.NodeServiceCapability{}, err
	}
	return resp.GetCapabilities(), nil
}

func setupClient(t *testing.T, stageUnstageSet bool) csiClient {
	return newFakeCsiDriverClient(t, stageUnstageSet)
}

func setupClientWithExpansion(t *testing.T, stageUnstageSet bool, expansionSet bool) csiClient {
	return newFakeCsiDriverClientWithExpansion(t, stageUnstageSet, expansionSet)
}

func setupClientWithVolumeStatsAndCondition(t *testing.T, volumeStatsSet, volumeConditionSet, setVolumeStat, setVolumecondition bool) csiClient {
	return newFakeCsiDriverClientWithVolumeStatsAndCondition(t, volumeStatsSet, volumeConditionSet, setVolumeStat, setVolumecondition)
}

func setupClientWithVolumeStats(t *testing.T, volumeStatsSet bool) csiClient {
	return newFakeCsiDriverClientWithVolumeStats(t, volumeStatsSet)
}

func setupClientWithVolumeMountGroup(t *testing.T, stageUnstageSet bool, volumeMountGroupSet bool) csiClient {
	return newFakeCsiDriverClientWithVolumeMountGroup(t, stageUnstageSet, volumeMountGroupSet)
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
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
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
	var testFSGroup int64 = 3000

	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	testPath := filepath.Join(tmpDir, "path")

	testCases := []struct {
		name                     string
		volID                    string
		targetPath               string
		fsType                   string
		fsGroup                  *int64
		expectedVolumeMountGroup string
		mustFail                 bool
		err                      error
	}{
		{name: "test ok", volID: "vol-test", targetPath: testPath},
		{name: "missing volID", targetPath: testPath, mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "bad fs", volID: "vol-test", targetPath: testPath, fsType: "badfs", mustFail: true},
		{name: "grpc error", volID: "vol-test", targetPath: testPath, mustFail: true, err: errors.New("grpc error")},
		{name: "fsgroup", volID: "vol-test", targetPath: testPath, fsGroup: &testFSGroup, expectedVolumeMountGroup: "3000"},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)

		nodeClient := fake.NewNodeClient(false /* stagingCapable */)
		nodeClient.SetNextError(tc.err)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
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
			tc.fsGroup,
		)
		checkErr(t, tc.mustFail, err)

		volumeMountGroup := nodeClient.GetNodePublishedVolumes()[tc.volID].VolumeMountGroup
		if volumeMountGroup != tc.expectedVolumeMountGroup {
			t.Errorf("Expected VolumeMountGroup in NodePublishVolumeRequest to be %q, got: %q", tc.expectedVolumeMountGroup, volumeMountGroup)
		}

		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

func TestClientNodeUnpublishVolume(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	testPath := filepath.Join(tmpDir, "path")

	testCases := []struct {
		name       string
		volID      string
		targetPath string
		mustFail   bool
		err        error
	}{
		{name: "test ok", volID: "vol-test", targetPath: testPath},
		{name: "missing volID", targetPath: testPath, mustFail: true},
		{name: "missing target path", volID: testPath, mustFail: true},
		{name: "grpc error", volID: "vol-test", targetPath: testPath, mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
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
	var testFSGroup int64 = 3000

	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	testPath := filepath.Join(tmpDir, "/test/path")

	testCases := []struct {
		name                     string
		volID                    string
		stagingTargetPath        string
		fsType                   string
		secrets                  map[string]string
		mountOptions             []string
		fsGroup                  *int64
		expectedVolumeMountGroup string
		mustFail                 bool
		err                      error
	}{
		{name: "test ok", volID: "vol-test", stagingTargetPath: testPath, fsType: "ext4", mountOptions: []string{"unvalidated"}},
		{name: "missing volID", stagingTargetPath: testPath, mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "bad fs", volID: "vol-test", stagingTargetPath: testPath, fsType: "badfs", mustFail: true},
		{name: "grpc error", volID: "vol-test", stagingTargetPath: testPath, mustFail: true, err: errors.New("grpc error")},
		{name: "fsgroup", volID: "vol-test", stagingTargetPath: testPath, fsGroup: &testFSGroup, expectedVolumeMountGroup: "3000"},
	}

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.name)

		nodeClient := fake.NewNodeClientWithVolumeMountGroup(true /* stagingCapable */, true /* volumeMountGroupCapable */)
		nodeClient.SetNextError(tc.err)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
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
			tc.fsGroup,
		)
		checkErr(t, tc.mustFail, err)

		volumeMountGroup := nodeClient.GetNodeStagedVolumes()[tc.volID].VolumeMountGroup
		if volumeMountGroup != tc.expectedVolumeMountGroup {
			t.Errorf("expected VolumeMountGroup parameter in NodePublishVolumeRequest to be %q, got: %q", tc.expectedVolumeMountGroup, volumeMountGroup)
		}

		if !tc.mustFail {
			fakeCloser.Check()
		}
	}
}

func TestClientNodeUnstageVolume(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("csi-test")
	if err != nil {
		t.Fatalf("can't create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)
	testPath := filepath.Join(tmpDir, "/test/path")

	testCases := []struct {
		name              string
		volID             string
		stagingTargetPath string
		mustFail          bool
		err               error
	}{
		{name: "test ok", volID: "vol-test", stagingTargetPath: testPath},
		{name: "missing volID", stagingTargetPath: testPath, mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "grpc error", volID: "vol-test", stagingTargetPath: testPath, mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
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

func TestClientNodeSupportsStageUnstage(t *testing.T) {
	testClientNodeSupportsCapabilities(t,
		func(client *csiDriverClient) (bool, error) {
			return client.NodeSupportsStageUnstage(context.Background())
		},
		func(stagingCapable bool) *fake.NodeClient {
			// Creates a staging-capable client
			return fake.NewNodeClient(stagingCapable)
		})
}

func TestClientNodeSupportsNodeExpand(t *testing.T) {
	testClientNodeSupportsCapabilities(t,
		func(client *csiDriverClient) (bool, error) {
			return client.NodeSupportsNodeExpand(context.Background())
		},
		func(expansionCapable bool) *fake.NodeClient {
			return fake.NewNodeClientWithExpansion(false /* stageCapable */, expansionCapable)
		})
}

func TestClientNodeSupportsVolumeStats(t *testing.T) {
	testClientNodeSupportsCapabilities(t,
		func(client *csiDriverClient) (bool, error) {
			return client.NodeSupportsVolumeStats(context.Background())
		},
		func(volumeStatsCapable bool) *fake.NodeClient {
			return fake.NewNodeClientWithVolumeStats(volumeStatsCapable)
		})
}

func TestClientNodeSupportsVolumeMountGroup(t *testing.T) {
	testClientNodeSupportsCapabilities(t,
		func(client *csiDriverClient) (bool, error) {
			return client.NodeSupportsVolumeMountGroup(context.Background())
		},
		func(volumeMountGroupCapable bool) *fake.NodeClient {
			return fake.NewNodeClientWithVolumeMountGroup(false /* stagingCapable */, volumeMountGroupCapable)
		})
}

func testClientNodeSupportsCapabilities(
	t *testing.T,
	capabilityMethodToTest func(*csiDriverClient) (bool, error),
	nodeClientGenerator func(bool) *fake.NodeClient) {

	testCases := []struct {
		name    string
		capable bool
	}{
		{name: "positive", capable: true},
		{name: "negative", capable: false},
	}

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.name)
		fakeCloser := fake.NewCloser(t)
		client := &csiDriverClient{
			driverName: "Fake Driver Name",
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := nodeClientGenerator(tc.capable)
				return nodeClient, fakeCloser, nil
			},
		}

		got, _ := capabilityMethodToTest(client)

		if got != tc.capable {
			t.Errorf("Expected capability support to be %v, got: %v", tc.capable, got)
		}

		fakeCloser.Check()
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
			nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
				nodeClient := fake.NewNodeClient(false /* stagingCapable */)
				nodeClient.SetNextError(tc.err)
				return nodeClient, fakeCloser, nil
			},
		}
		opts := csiResizeOptions{volumeID: tc.volID, volumePath: tc.volumePath, newSize: tc.newSize}
		_, err := client.NodeExpandVolume(context.Background(), opts)
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

func TestVolumeHealthEnable(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
	spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "metrics", "test-vol"), false)
	tests := []struct {
		name               string
		volumeStatsSet     bool
		setVolumeStat      bool
		setVolumecondition bool
		volumeConditionSet bool
		volumeData         VolumeStatsOptions
		success            bool
	}{
		{
			name:               "when nodeVolumeStats=on, volumeStatsSet=on, setVolumeStat=on, volumeCondition=on, setVolumecondition=on",
			volumeStatsSet:     true,
			setVolumeStat:      true,
			setVolumecondition: true,
			volumeConditionSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: true,
		},
		{
			name:               "when nodeVolumeStats=on, volumeStatsSet=on, setVolumeStat=on, volumeCondition=off, setVolumecondition=off",
			volumeStatsSet:     true,
			setVolumeStat:      true,
			setVolumecondition: false,
			volumeConditionSet: false,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: true,
		},
		{
			name:               "when nodeVolumeStats=on, volumeStatsSet=off, setVolumeStat=off, volumeCondition=on, setVolumecondition=on",
			volumeStatsSet:     false,
			setVolumeStat:      false,
			setVolumecondition: true,
			volumeConditionSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: true,
		},
		{
			name:               "when nodeVolumeStats=on, volumeStatsSet=off, setVolumeStat=off, volumeCondition=off, setVolumecondition=off",
			setVolumeStat:      false,
			volumeConditionSet: false,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
			defer cancel()
			csiSource, _ := getCSISourceFromSpec(tc.volumeData.VolumeSpec)
			csClient := setupClientWithVolumeStatsAndCondition(t, tc.volumeStatsSet, tc.volumeConditionSet, tc.setVolumeStat, tc.setVolumecondition)
			metrics, err := csClient.NodeGetVolumeStats(ctx, csiSource.VolumeHandle, tc.volumeData.DeviceMountPath)
			if err != nil && tc.success {
				t.Errorf("For %s : expected %v got %v", tc.name, tc.success, err)
			}
			if tc.success {
				if metrics == nil {
					t.Errorf("csi.NodeGetVolumeStats returned nil metrics for volume %s", tc.volumeData.VolumeID)
				} else {
					if tc.volumeConditionSet {
						assert.NotNil(t, metrics.Abnormal)
						assert.NotNil(t, metrics.Message)
					} else {
						assert.Nil(t, metrics.Abnormal)
						assert.Nil(t, metrics.Message)
					}
				}
			}
		})
	}
}

func TestVolumeHealthDisable(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, false)
	spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "metrics", "test-vol"), false)
	tests := []struct {
		name           string
		volumeStatsSet bool
		volumeData     VolumeStatsOptions
		success        bool
	}{
		{
			name:           "when nodeVolumeStats=on, VolumeID=on, DeviceMountPath=on, VolumeCondition=off",
			volumeStatsSet: true,
			volumeData: VolumeStatsOptions{
				VolumeSpec:      spec,
				VolumeID:        "volume1",
				DeviceMountPath: "/foo/bar",
			},
			success: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
			defer cancel()
			csiSource, _ := getCSISourceFromSpec(tc.volumeData.VolumeSpec)
			csClient := setupClientWithVolumeStatsAndCondition(t, tc.volumeStatsSet, false, true, false)
			metrics, err := csClient.NodeGetVolumeStats(ctx, csiSource.VolumeHandle, tc.volumeData.DeviceMountPath)
			if tc.success {
				assert.Nil(t, err)
			}

			if metrics == nil {
				t.Errorf("csi.NodeGetVolumeStats returned nil metrics for volume %s", tc.volumeData.VolumeID)
			} else {
				assert.Nil(t, metrics.Abnormal)
				assert.Nil(t, metrics.Message)
			}
		})
	}
}

func TestVolumeStats(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, true)
	spec := volume.NewSpecFromPersistentVolume(makeTestPV("test-pv", 10, "metrics", "test-vol"), false)
	tests := []struct {
		name               string
		volumeStatsSet     bool
		volumeConditionSet bool
		volumeData         VolumeStatsOptions
		success            bool
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

func TestAccessModeMapping(t *testing.T) {
	tests := []struct {
		name                     string
		singleNodeMultiWriterSet bool
		accessMode               api.PersistentVolumeAccessMode
		expectedMappedAccessMode csipbv1.VolumeCapability_AccessMode_Mode
	}{
		{
			name:                     "with ReadWriteOnce and incapable driver",
			singleNodeMultiWriterSet: false,
			accessMode:               api.ReadWriteOnce,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_WRITER,
		},
		{
			name:                     "with ReadOnlyMany and incapable driver",
			singleNodeMultiWriterSet: false,
			accessMode:               api.ReadOnlyMany,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY,
		},
		{
			name:                     "with ReadWriteMany and incapable driver",
			singleNodeMultiWriterSet: false,
			accessMode:               api.ReadWriteMany,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER,
		},
		{
			name:                     "with ReadWriteOncePod and incapable driver",
			singleNodeMultiWriterSet: false,
			accessMode:               api.ReadWriteOncePod,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_WRITER,
		},
		{
			name:                     "with ReadWriteOnce and capable driver",
			singleNodeMultiWriterSet: true,
			accessMode:               api.ReadWriteOnce,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_MULTI_WRITER,
		},
		{
			name:                     "with ReadOnlyMany and capable driver",
			singleNodeMultiWriterSet: true,
			accessMode:               api.ReadOnlyMany,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY,
		},
		{
			name:                     "with ReadWriteMany and capable driver",
			singleNodeMultiWriterSet: true,
			accessMode:               api.ReadWriteMany,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER,
		},
		{
			name:                     "with ReadWriteOncePod and capable driver",
			singleNodeMultiWriterSet: true,
			accessMode:               api.ReadWriteOncePod,
			expectedMappedAccessMode: csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_SINGLE_WRITER,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			fakeCloser := fake.NewCloser(t)
			client := &csiDriverClient{
				driverName: "Fake Driver Name",
				nodeV1ClientCreator: func(addr csiAddr, m *MetricsManager) (csipbv1.NodeClient, io.Closer, error) {
					nodeClient := fake.NewNodeClientWithSingleNodeMultiWriter(tc.singleNodeMultiWriterSet)
					return nodeClient, fakeCloser, nil
				},
			}

			accessModeMapper, err := client.getNodeV1AccessModeMapper(context.Background())
			if err != nil {
				t.Error(err)
			}

			mappedAccessMode := accessModeMapper(tc.accessMode)
			if mappedAccessMode != tc.expectedMappedAccessMode {
				t.Errorf("expected access mode: %v; got: %v", tc.expectedMappedAccessMode, mappedAccessMode)
			}
		})
	}
}
