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
	"strconv"
	"sync"

	csipbv1 "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	api "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

type csiClient interface {
	NodeGetInfo(ctx context.Context) (
		nodeID string,
		maxVolumePerNode int64,
		accessibleTopology map[string]string,
		err error)

	// The caller is responsible for checking whether the driver supports
	// applying FSGroup by calling NodeSupportsVolumeMountGroup().
	// If the driver does not, fsGroup must be set to nil.
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
		fsGroup *int64,
	) error

	NodeExpandVolume(ctx context.Context, rsOpts csiResizeOptions) (resource.Quantity, error)
	NodeUnpublishVolume(
		ctx context.Context,
		volID string,
		targetPath string,
	) error

	// The caller is responsible for checking whether the driver supports
	// applying FSGroup by calling NodeSupportsVolumeMountGroup().
	// If the driver does not, fsGroup must be set to nil.
	NodeStageVolume(ctx context.Context,
		volID string,
		publishVolumeInfo map[string]string,
		stagingTargetPath string,
		fsType string,
		accessMode api.PersistentVolumeAccessMode,
		secrets map[string]string,
		volumeContext map[string]string,
		mountOptions []string,
		fsGroup *int64,
	) error

	NodeGetVolumeStats(
		ctx context.Context,
		volID string,
		targetPath string,
	) (*volume.Metrics, error)
	NodeUnstageVolume(ctx context.Context, volID, stagingTargetPath string) error
	NodeSupportsStageUnstage(ctx context.Context) (bool, error)
	NodeSupportsNodeExpand(ctx context.Context) (bool, error)
	NodeSupportsVolumeStats(ctx context.Context) (bool, error)
	NodeSupportsSingleNodeMultiWriterAccessMode(ctx context.Context) (bool, error)
	NodeSupportsVolumeMountGroup(ctx context.Context) (bool, error)
}

// Strongly typed address
type csiAddr string

// Strongly typed driver name
type csiDriverName string

// csiClient encapsulates all csi-plugin methods
type csiDriverClient struct {
	driverName          csiDriverName
	addr                csiAddr
	metricsManager      *MetricsManager
	nodeV1ClientCreator nodeV1ClientCreator
}

type csiResizeOptions struct {
	volumeID          string
	volumePath        string
	stagingTargetPath string
	fsType            string
	accessMode        api.PersistentVolumeAccessMode
	newSize           resource.Quantity
	mountOptions      []string
	secrets           map[string]string
}

var _ csiClient = &csiDriverClient{}

type nodeV1ClientCreator func(addr csiAddr, metricsManager *MetricsManager) (
	nodeClient csipbv1.NodeClient,
	closer io.Closer,
	err error,
)

type nodeV1AccessModeMapper func(am api.PersistentVolumeAccessMode) csipbv1.VolumeCapability_AccessMode_Mode

// newV1NodeClient creates a new NodeClient with the internally used gRPC
// connection set up. It also returns a closer which must be called to close
// the gRPC connection when the NodeClient is not used anymore.
// This is the default implementation for the nodeV1ClientCreator, used in
// newCsiDriverClient.
func newV1NodeClient(addr csiAddr, metricsManager *MetricsManager) (nodeClient csipbv1.NodeClient, closer io.Closer, err error) {
	var conn *grpc.ClientConn
	conn, err = newGrpcConn(addr, metricsManager)
	if err != nil {
		return nil, nil, err
	}

	nodeClient = csipbv1.NewNodeClient(conn)
	return nodeClient, conn, nil
}

func newCsiDriverClient(driverName csiDriverName) (*csiDriverClient, error) {
	if driverName == "" {
		return nil, fmt.Errorf("driver name is empty")
	}

	existingDriver, driverExists := csiDrivers.Get(string(driverName))
	if !driverExists {
		return nil, fmt.Errorf("driver name %s not found in the list of registered CSI drivers", driverName)
	}

	nodeV1ClientCreator := newV1NodeClient
	return &csiDriverClient{
		driverName:          driverName,
		addr:                csiAddr(existingDriver.endpoint),
		nodeV1ClientCreator: nodeV1ClientCreator,
		metricsManager:      NewCSIMetricsManager(string(driverName)),
	}, nil
}

func (c *csiDriverClient) NodeGetInfo(ctx context.Context) (
	nodeID string,
	maxVolumePerNode int64,
	accessibleTopology map[string]string,
	err error) {
	klog.V(4).InfoS(log("calling NodeGetInfo rpc"))

	var getNodeInfoError error
	nodeID, maxVolumePerNode, accessibleTopology, getNodeInfoError = c.nodeGetInfoV1(ctx)
	if getNodeInfoError != nil {
		klog.InfoS("Error calling CSI NodeGetInfo()", "err", getNodeInfoError.Error())
	}
	return nodeID, maxVolumePerNode, accessibleTopology, getNodeInfoError
}

func (c *csiDriverClient) nodeGetInfoV1(ctx context.Context) (
	nodeID string,
	maxVolumePerNode int64,
	accessibleTopology map[string]string,
	err error) {

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
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
	fsGroup *int64,
) error {
	klog.V(4).InfoS(log("calling NodePublishVolume rpc"), "volID", volID, "targetPath", targetPath)
	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}

	if c.nodeV1ClientCreator == nil {
		return errors.New("failed to call NodePublishVolume. nodeV1ClientCreator is nil")
	}

	accessModeMapper, err := c.getNodeV1AccessModeMapper(ctx)
	if err != nil {
		return err
	}

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
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
				Mode: accessModeMapper(accessMode),
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

	_, err = nodeClient.NodePublishVolume(ctx, req)
	if err != nil && !isFinalError(err) {
		return volumetypes.NewUncertainProgressError(err.Error())
	}
	return err
}

func (c *csiDriverClient) NodeExpandVolume(ctx context.Context, opts csiResizeOptions) (resource.Quantity, error) {
	if c.nodeV1ClientCreator == nil {
		return opts.newSize, fmt.Errorf("version of CSI driver does not support volume expansion")
	}

	if opts.volumeID == "" {
		return opts.newSize, errors.New("missing volume id")
	}
	if opts.volumePath == "" {
		return opts.newSize, errors.New("missing volume path")
	}

	if opts.newSize.Value() < 0 {
		return opts.newSize, errors.New("size can not be less than 0")
	}

	accessModeMapper, err := c.getNodeV1AccessModeMapper(ctx)
	if err != nil {
		return opts.newSize, err
	}

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
	if err != nil {
		return opts.newSize, err
	}
	defer closer.Close()

	req := &csipbv1.NodeExpandVolumeRequest{
		VolumeId:      opts.volumeID,
		VolumePath:    opts.volumePath,
		CapacityRange: &csipbv1.CapacityRange{RequiredBytes: opts.newSize.Value()},
		VolumeCapability: &csipbv1.VolumeCapability{
			AccessMode: &csipbv1.VolumeCapability_AccessMode{
				Mode: accessModeMapper(opts.accessMode),
			},
		},
		Secrets: opts.secrets,
	}

	// not all CSI drivers support NodeStageUnstage and hence the StagingTargetPath
	// should only be set when available
	if opts.stagingTargetPath != "" {
		req.StagingTargetPath = opts.stagingTargetPath
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

	resp, err := nodeClient.NodeExpandVolume(ctx, req)
	if err != nil {
		if !isFinalError(err) {
			return opts.newSize, volumetypes.NewUncertainProgressError(err.Error())
		}
		return opts.newSize, err
	}

	updatedQuantity := resource.NewQuantity(resp.CapacityBytes, resource.BinarySI)
	return *updatedQuantity, nil
}

func (c *csiDriverClient) NodeUnpublishVolume(ctx context.Context, volID string, targetPath string) error {
	klog.V(4).InfoS(log("calling NodeUnpublishVolume rpc"), "volID", volID, "targetPath", targetPath)
	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}
	if c.nodeV1ClientCreator == nil {
		return errors.New("nodeV1ClientCreate is nil")
	}

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
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

func (c *csiDriverClient) NodeStageVolume(ctx context.Context,
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
	klog.V(4).InfoS(log("calling NodeStageVolume rpc"), "volID", volID, "stagingTargetPath", stagingTargetPath)
	if volID == "" {
		return errors.New("missing volume id")
	}
	if stagingTargetPath == "" {
		return errors.New("missing staging target path")
	}
	if c.nodeV1ClientCreator == nil {
		return errors.New("nodeV1ClientCreate is nil")
	}

	accessModeMapper, err := c.getNodeV1AccessModeMapper(ctx)
	if err != nil {
		return err
	}

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
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
				Mode: accessModeMapper(accessMode),
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

	_, err = nodeClient.NodeStageVolume(ctx, req)
	if err != nil && !isFinalError(err) {
		return volumetypes.NewUncertainProgressError(err.Error())
	}
	return err
}

func (c *csiDriverClient) NodeUnstageVolume(ctx context.Context, volID, stagingTargetPath string) error {
	klog.V(4).InfoS(log("calling NodeUnstageVolume rpc"), "volID", volID, "stagingTargetPath", stagingTargetPath)
	if volID == "" {
		return errors.New("missing volume id")
	}
	if stagingTargetPath == "" {
		return errors.New("missing staging target path")
	}
	if c.nodeV1ClientCreator == nil {
		return errors.New("nodeV1ClientCreate is nil")
	}

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
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

func (c *csiDriverClient) NodeSupportsNodeExpand(ctx context.Context) (bool, error) {
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_EXPAND_VOLUME)
}

func (c *csiDriverClient) NodeSupportsStageUnstage(ctx context.Context) (bool, error) {
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME)
}

func (c *csiDriverClient) getNodeV1AccessModeMapper(ctx context.Context) (nodeV1AccessModeMapper, error) {
	supported, err := c.NodeSupportsSingleNodeMultiWriterAccessMode(ctx)
	if err != nil {
		return nil, err
	}
	if supported {
		return asSingleNodeMultiWriterCapableCSIAccessModeV1, nil
	}
	return asCSIAccessModeV1, nil
}

func asCSIAccessModeV1(am api.PersistentVolumeAccessMode) csipbv1.VolumeCapability_AccessMode_Mode {
	switch am {
	case api.ReadWriteOnce:
		return csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_WRITER
	case api.ReadOnlyMany:
		return csipbv1.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY
	case api.ReadWriteMany:
		return csipbv1.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER
	// This mapping exists to enable CSI drivers that lack the
	// SINGLE_NODE_MULTI_WRITER capability to work with the
	// ReadWriteOncePod access mode.
	case api.ReadWriteOncePod:
		return csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_WRITER
	}
	return csipbv1.VolumeCapability_AccessMode_UNKNOWN
}

func asSingleNodeMultiWriterCapableCSIAccessModeV1(am api.PersistentVolumeAccessMode) csipbv1.VolumeCapability_AccessMode_Mode {
	switch am {
	case api.ReadWriteOnce:
		return csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_MULTI_WRITER
	case api.ReadOnlyMany:
		return csipbv1.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY
	case api.ReadWriteMany:
		return csipbv1.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER
	case api.ReadWriteOncePod:
		return csipbv1.VolumeCapability_AccessMode_SINGLE_NODE_SINGLE_WRITER
	}
	return csipbv1.VolumeCapability_AccessMode_UNKNOWN
}

func newGrpcConn(addr csiAddr, metricsManager *MetricsManager) (*grpc.ClientConn, error) {
	network := "unix"
	klog.V(4).InfoS(log("creating new gRPC connection"), "protocol", network, "endpoint", addr)

	return grpc.Dial(
		string(addr),
		grpc.WithAuthority("localhost"),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, target string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, target)
		}),
		grpc.WithChainUnaryInterceptor(metricsManager.RecordMetricsInterceptor),
	)
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

func (c *csiDriverClient) NodeSupportsVolumeStats(ctx context.Context) (bool, error) {
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_GET_VOLUME_STATS)
}

func (c *csiDriverClient) NodeSupportsSingleNodeMultiWriterAccessMode(ctx context.Context) (bool, error) {
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_SINGLE_NODE_MULTI_WRITER)
}

func (c *csiDriverClient) NodeGetVolumeStats(ctx context.Context, volID string, targetPath string) (*volume.Metrics, error) {
	klog.V(4).InfoS(log("calling NodeGetVolumeStats rpc"), "volID", volID, "targetPath", targetPath)
	if volID == "" {
		return nil, errors.New("missing volume id")
	}
	if targetPath == "" {
		return nil, errors.New("missing target path")
	}
	if c.nodeV1ClientCreator == nil {
		return nil, errors.New("nodeV1ClientCreate is nil")
	}

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
	if err != nil {
		return nil, err
	}
	defer closer.Close()

	req := &csipbv1.NodeGetVolumeStatsRequest{
		VolumeId:   volID,
		VolumePath: targetPath,
	}

	resp, err := nodeClient.NodeGetVolumeStats(ctx, req)
	if err != nil {
		return nil, err
	}
	usages := resp.GetUsage()
	if usages == nil {
		return nil, fmt.Errorf("failed to get usage from response. usage is nil")
	}
	metrics := &volume.Metrics{
		Used:       resource.NewQuantity(int64(0), resource.BinarySI),
		Capacity:   resource.NewQuantity(int64(0), resource.BinarySI),
		Available:  resource.NewQuantity(int64(0), resource.BinarySI),
		InodesUsed: resource.NewQuantity(int64(0), resource.BinarySI),
		Inodes:     resource.NewQuantity(int64(0), resource.BinarySI),
		InodesFree: resource.NewQuantity(int64(0), resource.BinarySI),
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CSIVolumeHealth) {
		isSupportNodeVolumeCondition, err := c.nodeSupportsVolumeCondition(ctx)
		if err != nil {
			return nil, err
		}

		if isSupportNodeVolumeCondition {
			abnormal, message := resp.VolumeCondition.GetAbnormal(), resp.VolumeCondition.GetMessage()
			metrics.Abnormal, metrics.Message = &abnormal, &message
		}
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
		default:
			klog.ErrorS(nil, "unknown unit in VolumeUsage", "unit", unit.String())
		}

	}
	return metrics, nil
}

func (c *csiDriverClient) nodeSupportsVolumeCondition(ctx context.Context) (bool, error) {
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_VOLUME_CONDITION)
}

func (c *csiDriverClient) NodeSupportsVolumeMountGroup(ctx context.Context) (bool, error) {
	return c.nodeSupportsCapability(ctx, csipbv1.NodeServiceCapability_RPC_VOLUME_MOUNT_GROUP)
}

func (c *csiDriverClient) nodeSupportsCapability(ctx context.Context, capabilityType csipbv1.NodeServiceCapability_RPC_Type) (bool, error) {
	klog.V(4).Info(log("calling NodeGetCapabilities rpc to determine if the node service has %s capability", capabilityType))
	capabilities, err := c.nodeGetCapabilities(ctx)
	if err != nil {
		return false, err
	}

	for _, capability := range capabilities {
		if capability == nil || capability.GetRpc() == nil {
			continue
		}
		if capability.GetRpc().GetType() == capabilityType {
			return true, nil
		}
	}
	return false, nil
}

func (c *csiDriverClient) nodeGetCapabilities(ctx context.Context) ([]*csipbv1.NodeServiceCapability, error) {
	if c.nodeV1ClientCreator == nil {
		return []*csipbv1.NodeServiceCapability{}, errors.New("nodeV1ClientCreate is nil")
	}

	nodeClient, closer, err := c.nodeV1ClientCreator(c.addr, c.metricsManager)
	if err != nil {
		return []*csipbv1.NodeServiceCapability{}, err
	}
	defer closer.Close()

	req := &csipbv1.NodeGetCapabilitiesRequest{}
	resp, err := nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return []*csipbv1.NodeServiceCapability{}, err
	}
	return resp.GetCapabilities(), nil
}

func isFinalError(err error) bool {
	// Sources:
	// https://github.com/grpc/grpc/blob/master/doc/statuscodes.md
	// https://github.com/container-storage-interface/spec/blob/master/spec.md
	st, ok := status.FromError(err)
	if !ok {
		// This is not gRPC error. The operation must have failed before gRPC
		// method was called, otherwise we would get gRPC error.
		// We don't know if any previous volume operation is in progress, be on the safe side.
		return false
	}
	switch st.Code() {
	case codes.Canceled, // gRPC: Client Application cancelled the request
		codes.DeadlineExceeded,  // gRPC: Timeout
		codes.Unavailable,       // gRPC: Server shutting down, TCP connection broken - previous volume operation may be still in progress.
		codes.ResourceExhausted, // gRPC: Server temporarily out of resources - previous volume operation may be still in progress.
		codes.Aborted:           // CSI: Operation pending for volume
		return false
	}
	// All other errors mean that operation either did not
	// even start or failed. It is for sure not in progress.
	return true
}
