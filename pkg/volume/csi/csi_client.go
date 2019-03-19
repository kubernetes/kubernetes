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
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	api "k8s.io/api/core/v1"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	csipbv0 "k8s.io/kubernetes/pkg/volume/csi/csiv0"
)

type csiClient interface {
	NodeGetInfo(ctx context.Context) (
		nodeID string,
		maxVolumePerNode int64,
		accessibleTopology map[string]string,
		err error)
	NodePublishVolume(
		ctx context.Context,
		volumeid string,
		readOnly bool,
		stagingTargetPath string,
		targetPath string,
		accessMode api.PersistentVolumeAccessMode,
		publishContext map[string]string,
		volumeContext map[string]string,
		secrets map[string]string,
		fsType string,
		mountOptions []string,
	) error
	NodeUnpublishVolume(
		ctx context.Context,
		volID string,
		targetPath string,
	) error
	NodeStageVolume(ctx context.Context,
		volID string,
		publishVolumeInfo map[string]string,
		stagingTargetPath string,
		fsType string,
		accessMode api.PersistentVolumeAccessMode,
		secrets map[string]string,
		volumeContext map[string]string,
	) error
	NodeUnstageVolume(ctx context.Context, volID, stagingTargetPath string) error
	NodeSupportsStageUnstage(ctx context.Context) (bool, error)
}

// Strongly typed address
type csiAddr string

// Strongly typed driver name
type csiDriverName string

// csiClient encapsulates all csi-plugin methods
type csiDriverClient struct {
	driverName          csiDriverName
	addr                csiAddr
	nodeV1ClientCreator nodeV1ClientCreator
	nodeV0ClientCreator nodeV0ClientCreator
}

var _ csiClient = &csiDriverClient{}

type nodeV1ClientCreator func(addr csiAddr) (
	nodeClient csipbv1.NodeClient,
	closer io.Closer,
	err error,
)

type nodeV0ClientCreator func(addr csiAddr) (
	nodeClient csipbv0.NodeClient,
	closer io.Closer,
	err error,
)

// newV1NodeClient creates a new NodeClient with the internally used gRPC
// connection set up. It also returns a closer which must to be called to close
// the gRPC connection when the NodeClient is not used anymore.
// This is the default implementation for the nodeV1ClientCreator, used in
// newCsiDriverClient.
func newV1NodeClient(addr csiAddr) (nodeClient csipbv1.NodeClient, closer io.Closer, err error) {
	var conn *grpc.ClientConn
	conn, err = newGrpcConn(addr)
	if err != nil {
		return nil, nil, err
	}

	nodeClient = csipbv1.NewNodeClient(conn)
	return nodeClient, conn, nil
}

// newV0NodeClient creates a new NodeClient with the internally used gRPC
// connection set up. It also returns a closer which must to be called to close
// the gRPC connection when the NodeClient is not used anymore.
// This is the default implementation for the nodeV1ClientCreator, used in
// newCsiDriverClient.
func newV0NodeClient(addr csiAddr) (nodeClient csipbv0.NodeClient, closer io.Closer, err error) {
	var conn *grpc.ClientConn
	conn, err = newGrpcConn(addr)
	if err != nil {
		return nil, nil, err
	}

	nodeClient = csipbv0.NewNodeClient(conn)
	return nodeClient, conn, nil
}

func newCsiDriverClient(driverName csiDriverName) (*csiDriverClient, error) {
	if driverName == "" {
		return nil, fmt.Errorf("driver name is empty")
	}

	addr := fmt.Sprintf(csiAddrTemplate, driverName)
	requiresV0Client := true
	if utilfeature.DefaultFeatureGate.Enabled(features.KubeletPluginsWatcher) {
		var existingDriver csiDriver
		driverExists := false
		func() {
			csiDrivers.RLock()
			defer csiDrivers.RUnlock()
			existingDriver, driverExists = csiDrivers.driversMap[string(driverName)]
		}()

		if !driverExists {
			return nil, fmt.Errorf("driver name %s not found in the list of registered CSI drivers", driverName)
		}

		addr = existingDriver.driverEndpoint
		requiresV0Client = versionRequiresV0Client(existingDriver.highestSupportedVersion)
	}

	nodeV1ClientCreator := newV1NodeClient
	nodeV0ClientCreator := newV0NodeClient
	if requiresV0Client {
		nodeV1ClientCreator = nil
	} else {
		nodeV0ClientCreator = nil
	}

	return &csiDriverClient{
		driverName:          driverName,
		addr:                csiAddr(addr),
		nodeV1ClientCreator: nodeV1ClientCreator,
		nodeV0ClientCreator: nodeV0ClientCreator,
	}, nil
}

func (c *csiDriverClient) NodeGetInfo(ctx context.Context) (
	nodeID string,
	maxVolumePerNode int64,
	accessibleTopology map[string]string,
	err error) {
	klog.V(4).Info(log("calling NodeGetInfo rpc"))
	if c.nodeV1ClientCreator != nil {
		return c.nodeGetInfoV1(ctx)
	} else if c.nodeV0ClientCreator != nil {
		return c.nodeGetInfoV0(ctx)
	}

	err = fmt.Errorf("failed to call NodeGetInfo. Both nodeV1ClientCreator and nodeV0ClientCreator are nil")

	return nodeID, maxVolumePerNode, accessibleTopology, err
}

func (c *csiDriverClient) nodeGetInfoV1(ctx context.Context) (
	nodeID string,
	maxVolumePerNode int64,
	accessibleTopology map[string]string,
	err error) {

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr)
	if err != nil {
		return "", 0, nil, err
	}
	defer closer.Close()

	res, err := nodeClient.NodeGetInfo(ctx, &csipbv1.NodeGetInfoRequest{})
	if err != nil {
		return "", 0, nil, err
	}

	topology := res.GetAccessibleTopology()
	if topology != nil {
		accessibleTopology = topology.Segments
	}
	return res.GetNodeId(), res.GetMaxVolumesPerNode(), accessibleTopology, nil
}

func (c *csiDriverClient) nodeGetInfoV0(ctx context.Context) (
	nodeID string,
	maxVolumePerNode int64,
	accessibleTopology map[string]string,
	err error) {

	nodeClient, closer, err := c.nodeV0ClientCreator(c.addr)
	if err != nil {
		return "", 0, nil, err
	}
	defer closer.Close()

	res, err := nodeClient.NodeGetInfo(ctx, &csipbv0.NodeGetInfoRequest{})
	if err != nil {
		return "", 0, nil, err
	}

	topology := res.GetAccessibleTopology()
	if topology != nil {
		accessibleTopology = topology.Segments
	}
	return res.GetNodeId(), res.GetMaxVolumesPerNode(), accessibleTopology, nil
}

func (c *csiDriverClient) NodePublishVolume(
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
	klog.V(4).Info(log("calling NodePublishVolume rpc [volid=%s,target_path=%s]", volID, targetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}
	if c.nodeV1ClientCreator != nil {
		return c.nodePublishVolumeV1(
			ctx,
			volID,
			readOnly,
			stagingTargetPath,
			targetPath,
			accessMode,
			publishContext,
			volumeContext,
			secrets,
			fsType,
			mountOptions,
		)
	} else if c.nodeV0ClientCreator != nil {
		return c.nodePublishVolumeV0(
			ctx,
			volID,
			readOnly,
			stagingTargetPath,
			targetPath,
			accessMode,
			publishContext,
			volumeContext,
			secrets,
			fsType,
			mountOptions,
		)
	}

	return fmt.Errorf("failed to call NodePublishVolume. Both nodeV1ClientCreator and nodeV0ClientCreator are nil")

}

func (c *csiDriverClient) nodePublishVolumeV1(
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
	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

	req := &csipbv1.NodePublishVolumeRequest{
		VolumeId:       volID,
		TargetPath:     targetPath,
		Readonly:       readOnly,
		PublishContext: publishContext,
		VolumeContext:  volumeContext,
		Secrets:        secrets,
		VolumeCapability: &csipbv1.VolumeCapability{
			AccessMode: &csipbv1.VolumeCapability_AccessMode{
				Mode: asCSIAccessModeV1(accessMode),
			},
		},
	}
	if stagingTargetPath != "" {
		req.StagingTargetPath = stagingTargetPath
	}

	if fsType == fsTypeBlockName {
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Block{
			Block: &csipbv1.VolumeCapability_BlockVolume{},
		}
	} else {
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Mount{
			Mount: &csipbv1.VolumeCapability_MountVolume{
				FsType:     fsType,
				MountFlags: mountOptions,
			},
		}
	}

	_, err = nodeClient.NodePublishVolume(ctx, req)
	return err
}

func (c *csiDriverClient) nodePublishVolumeV0(
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
	nodeClient, closer, err := c.nodeV0ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

	req := &csipbv0.NodePublishVolumeRequest{
		VolumeId:           volID,
		TargetPath:         targetPath,
		Readonly:           readOnly,
		PublishInfo:        publishContext,
		VolumeAttributes:   volumeContext,
		NodePublishSecrets: secrets,
		VolumeCapability: &csipbv0.VolumeCapability{
			AccessMode: &csipbv0.VolumeCapability_AccessMode{
				Mode: asCSIAccessModeV0(accessMode),
			},
		},
	}
	if stagingTargetPath != "" {
		req.StagingTargetPath = stagingTargetPath
	}

	if fsType == fsTypeBlockName {
		req.VolumeCapability.AccessType = &csipbv0.VolumeCapability_Block{
			Block: &csipbv0.VolumeCapability_BlockVolume{},
		}
	} else {
		req.VolumeCapability.AccessType = &csipbv0.VolumeCapability_Mount{
			Mount: &csipbv0.VolumeCapability_MountVolume{
				FsType:     fsType,
				MountFlags: mountOptions,
			},
		}
	}

	_, err = nodeClient.NodePublishVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeUnpublishVolume(ctx context.Context, volID string, targetPath string) error {
	klog.V(4).Info(log("calling NodeUnpublishVolume rpc: [volid=%s, target_path=%s", volID, targetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}

	if c.nodeV1ClientCreator != nil {
		return c.nodeUnpublishVolumeV1(ctx, volID, targetPath)
	} else if c.nodeV0ClientCreator != nil {
		return c.nodeUnpublishVolumeV0(ctx, volID, targetPath)
	}

	return fmt.Errorf("failed to call NodeUnpublishVolume. Both nodeV1ClientCreator and nodeV0ClientCreator are nil")
}

func (c *csiDriverClient) nodeUnpublishVolumeV1(ctx context.Context, volID string, targetPath string) error {
	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

	req := &csipbv1.NodeUnpublishVolumeRequest{
		VolumeId:   volID,
		TargetPath: targetPath,
	}

	_, err = nodeClient.NodeUnpublishVolume(ctx, req)
	return err
}

func (c *csiDriverClient) nodeUnpublishVolumeV0(ctx context.Context, volID string, targetPath string) error {
	nodeClient, closer, err := c.nodeV0ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

	req := &csipbv0.NodeUnpublishVolumeRequest{
		VolumeId:   volID,
		TargetPath: targetPath,
	}

	_, err = nodeClient.NodeUnpublishVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeStageVolume(ctx context.Context,
	volID string,
	publishContext map[string]string,
	stagingTargetPath string,
	fsType string,
	accessMode api.PersistentVolumeAccessMode,
	secrets map[string]string,
	volumeContext map[string]string,
) error {
	klog.V(4).Info(log("calling NodeStageVolume rpc [volid=%s,staging_target_path=%s]", volID, stagingTargetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if stagingTargetPath == "" {
		return errors.New("missing staging target path")
	}

	if c.nodeV1ClientCreator != nil {
		return c.nodeStageVolumeV1(ctx, volID, publishContext, stagingTargetPath, fsType, accessMode, secrets, volumeContext)
	} else if c.nodeV0ClientCreator != nil {
		return c.nodeStageVolumeV0(ctx, volID, publishContext, stagingTargetPath, fsType, accessMode, secrets, volumeContext)
	}

	return fmt.Errorf("failed to call NodeStageVolume. Both nodeV1ClientCreator and nodeV0ClientCreator are nil")
}

func (c *csiDriverClient) nodeStageVolumeV1(
	ctx context.Context,
	volID string,
	publishContext map[string]string,
	stagingTargetPath string,
	fsType string,
	accessMode api.PersistentVolumeAccessMode,
	secrets map[string]string,
	volumeContext map[string]string,
) error {
	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

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
		req.VolumeCapability.AccessType = &csipbv1.VolumeCapability_Mount{
			Mount: &csipbv1.VolumeCapability_MountVolume{
				FsType: fsType,
			},
		}
	}

	_, err = nodeClient.NodeStageVolume(ctx, req)
	return err
}

func (c *csiDriverClient) nodeStageVolumeV0(
	ctx context.Context,
	volID string,
	publishContext map[string]string,
	stagingTargetPath string,
	fsType string,
	accessMode api.PersistentVolumeAccessMode,
	secrets map[string]string,
	volumeContext map[string]string,
) error {
	nodeClient, closer, err := c.nodeV0ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

	req := &csipbv0.NodeStageVolumeRequest{
		VolumeId:          volID,
		PublishInfo:       publishContext,
		StagingTargetPath: stagingTargetPath,
		VolumeCapability: &csipbv0.VolumeCapability{
			AccessMode: &csipbv0.VolumeCapability_AccessMode{
				Mode: asCSIAccessModeV0(accessMode),
			},
		},
		NodeStageSecrets: secrets,
		VolumeAttributes: volumeContext,
	}

	if fsType == fsTypeBlockName {
		req.VolumeCapability.AccessType = &csipbv0.VolumeCapability_Block{
			Block: &csipbv0.VolumeCapability_BlockVolume{},
		}
	} else {
		req.VolumeCapability.AccessType = &csipbv0.VolumeCapability_Mount{
			Mount: &csipbv0.VolumeCapability_MountVolume{
				FsType: fsType,
			},
		}
	}

	_, err = nodeClient.NodeStageVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeUnstageVolume(ctx context.Context, volID, stagingTargetPath string) error {
	klog.V(4).Info(log("calling NodeUnstageVolume rpc [volid=%s,staging_target_path=%s]", volID, stagingTargetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if stagingTargetPath == "" {
		return errors.New("missing staging target path")
	}

	if c.nodeV1ClientCreator != nil {
		return c.nodeUnstageVolumeV1(ctx, volID, stagingTargetPath)
	} else if c.nodeV0ClientCreator != nil {
		return c.nodeUnstageVolumeV0(ctx, volID, stagingTargetPath)
	}

	return fmt.Errorf("failed to call NodeUnstageVolume. Both nodeV1ClientCreator and nodeV0ClientCreator are nil")
}

func (c *csiDriverClient) nodeUnstageVolumeV1(ctx context.Context, volID, stagingTargetPath string) error {
	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

	req := &csipbv1.NodeUnstageVolumeRequest{
		VolumeId:          volID,
		StagingTargetPath: stagingTargetPath,
	}
	_, err = nodeClient.NodeUnstageVolume(ctx, req)
	return err
}

func (c *csiDriverClient) nodeUnstageVolumeV0(ctx context.Context, volID, stagingTargetPath string) error {
	nodeClient, closer, err := c.nodeV0ClientCreator(c.addr)
	if err != nil {
		return err
	}
	defer closer.Close()

	req := &csipbv0.NodeUnstageVolumeRequest{
		VolumeId:          volID,
		StagingTargetPath: stagingTargetPath,
	}
	_, err = nodeClient.NodeUnstageVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeSupportsStageUnstage(ctx context.Context) (bool, error) {
	klog.V(4).Info(log("calling NodeGetCapabilities rpc to determine if NodeSupportsStageUnstage"))

	if c.nodeV1ClientCreator != nil {
		return c.nodeSupportsStageUnstageV1(ctx)
	} else if c.nodeV0ClientCreator != nil {
		return c.nodeSupportsStageUnstageV0(ctx)
	}

	return false, fmt.Errorf("failed to call NodeSupportsStageUnstage. Both nodeV1ClientCreator and nodeV0ClientCreator are nil")
}

func (c *csiDriverClient) nodeSupportsStageUnstageV1(ctx context.Context) (bool, error) {
	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr)
	if err != nil {
		return false, err
	}
	defer closer.Close()

	req := &csipbv1.NodeGetCapabilitiesRequest{}
	resp, err := nodeClient.NodeGetCapabilities(ctx, req)
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

func (c *csiDriverClient) nodeSupportsStageUnstageV0(ctx context.Context) (bool, error) {
	nodeClient, closer, err := c.nodeV0ClientCreator(c.addr)
	if err != nil {
		return false, err
	}
	defer closer.Close()

	req := &csipbv0.NodeGetCapabilitiesRequest{}
	resp, err := nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return false, err
	}

	capabilities := resp.GetCapabilities()

	stageUnstageSet := false
	if capabilities == nil {
		return false, nil
	}
	for _, capability := range capabilities {
		if capability.GetRpc().GetType() == csipbv0.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME {
			stageUnstageSet = true
		}
	}
	return stageUnstageSet, nil
}

func asCSIAccessModeV1(am api.PersistentVolumeAccessMode) csipbv1.VolumeCapability_AccessMode_Mode {
	switch am {
	case api.ReadWriteOnce:
		return csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_WRITER
	case api.ReadOnlyMany:
		return csipbv1.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY
	case api.ReadWriteMany:
		return csipbv1.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER
	}
	return csipbv1.VolumeCapability_AccessMode_UNKNOWN
}

func asCSIAccessModeV0(am api.PersistentVolumeAccessMode) csipbv0.VolumeCapability_AccessMode_Mode {
	switch am {
	case api.ReadWriteOnce:
		return csipbv0.VolumeCapability_AccessMode_SINGLE_NODE_WRITER
	case api.ReadOnlyMany:
		return csipbv0.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY
	case api.ReadWriteMany:
		return csipbv0.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER
	}
	return csipbv0.VolumeCapability_AccessMode_UNKNOWN
}

func newGrpcConn(addr csiAddr) (*grpc.ClientConn, error) {
	network := "unix"
	klog.V(4).Infof(log("creating new gRPC connection for [%s://%s]", network, addr))

	return grpc.Dial(
		string(addr),
		grpc.WithInsecure(),
		grpc.WithDialer(func(target string, timeout time.Duration) (net.Conn, error) {
			return net.Dial(network, target)
		}),
	)
}

func versionRequiresV0Client(version *utilversion.Version) bool {
	if version != nil && version.Major() == 0 {
		return true
	}

	return false
}

// CSI client getter with cache.
// This provides a method to initialize CSI client with driver name and caches
// it for later use. When CSI clients have not been discovered yet (e.g.
// on kubelet restart), client initialization will fail. Users of CSI client (e.g.
// mounter manager and block mapper) can use this to delay CSI client
// initialization until needed.
type csiClientGetter struct {
	sync.RWMutex
	csiClient  csiClient
	driverName csiDriverName
}

func (c *csiClientGetter) Get() (csiClient, error) {
	c.RLock()
	if c.csiClient != nil {
		c.RUnlock()
		return c.csiClient, nil
	}
	c.RUnlock()
	c.Lock()
	defer c.Unlock()
	// Double-checking locking criterion.
	if c.csiClient != nil {
		return c.csiClient, nil
	}
	csi, err := newCsiDriverClient(c.driverName)
	if err != nil {
		return nil, err
	}
	c.csiClient = csi
	return c.csiClient, nil
}
