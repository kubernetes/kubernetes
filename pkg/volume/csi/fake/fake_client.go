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

package fake

import (
	"context"
	"errors"
	"strings"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	// NodePublishTimeout_VolumeID is volume id that will result in NodePublish operation to timeout
	NodePublishTimeOut_VolumeID = "node-publish-timeout"
	// NodeStageTimeOut_VolumeID is a volume id that will result in NodeStage operation to timeout
	NodeStageTimeOut_VolumeID = "node-stage-timeout"
)

// IdentityClient is a CSI identity client used for testing
type IdentityClient struct {
	nextErr error
}

// NewIdentityClient returns a new IdentityClient
func NewIdentityClient() *IdentityClient {
	return &IdentityClient{}
}

// SetNextError injects expected error
func (f *IdentityClient) SetNextError(err error) {
	f.nextErr = err
}

// GetPluginInfo returns plugin info
func (f *IdentityClient) GetPluginInfo(ctx context.Context, in *csipb.GetPluginInfoRequest, opts ...grpc.CallOption) (*csipb.GetPluginInfoResponse, error) {
	return nil, nil
}

// GetPluginCapabilities implements csi method
func (f *IdentityClient) GetPluginCapabilities(ctx context.Context, in *csipb.GetPluginCapabilitiesRequest, opts ...grpc.CallOption) (*csipb.GetPluginCapabilitiesResponse, error) {
	return nil, nil
}

// Probe implements csi method
func (f *IdentityClient) Probe(ctx context.Context, in *csipb.ProbeRequest, opts ...grpc.CallOption) (*csipb.ProbeResponse, error) {
	return nil, nil
}

type CSIVolume struct {
	VolumeHandle    string
	VolumeContext   map[string]string
	Path            string
	DeviceMountPath string
	FSType          string
	MountFlags      []string
}

// NodeClient returns CSI node client
type NodeClient struct {
	nodePublishedVolumes map[string]CSIVolume
	nodeStagedVolumes    map[string]CSIVolume
	stageUnstageSet      bool
	expansionSet         bool
	volumeStatsSet       bool
	nodeGetInfoResp      *csipb.NodeGetInfoResponse
	nodeVolumeStatsResp  *csipb.NodeGetVolumeStatsResponse
	nextErr              error
}

// NewNodeClient returns fake node client
func NewNodeClient(stageUnstageSet bool) *NodeClient {
	return &NodeClient{
		nodePublishedVolumes: make(map[string]CSIVolume),
		nodeStagedVolumes:    make(map[string]CSIVolume),
		stageUnstageSet:      stageUnstageSet,
		volumeStatsSet:       true,
	}
}

func NewNodeClientWithExpansion(stageUnstageSet bool, expansionSet bool) *NodeClient {
	return &NodeClient{
		nodePublishedVolumes: make(map[string]CSIVolume),
		nodeStagedVolumes:    make(map[string]CSIVolume),
		stageUnstageSet:      stageUnstageSet,
		expansionSet:         expansionSet,
	}
}

func NewNodeClientWithVolumeStats(volumeStatsSet bool) *NodeClient {
	return &NodeClient{
		volumeStatsSet: volumeStatsSet,
	}
}

// SetNextError injects next expected error
func (f *NodeClient) SetNextError(err error) {
	f.nextErr = err
}

func (f *NodeClient) SetNodeGetInfoResp(resp *csipb.NodeGetInfoResponse) {
	f.nodeGetInfoResp = resp
}

func (f *NodeClient) SetNodeVolumeStatsResp(resp *csipb.NodeGetVolumeStatsResponse) {
	f.nodeVolumeStatsResp = resp
}

// GetNodePublishedVolumes returns node published volumes
func (f *NodeClient) GetNodePublishedVolumes() map[string]CSIVolume {
	return f.nodePublishedVolumes
}

// AddNodePublishedVolume adds specified volume to nodePublishedVolumes
func (f *NodeClient) AddNodePublishedVolume(volID, deviceMountPath string, volumeContext map[string]string) {
	f.nodePublishedVolumes[volID] = CSIVolume{
		Path:          deviceMountPath,
		VolumeContext: volumeContext,
	}
}

// GetNodeStagedVolumes returns node staged volumes
func (f *NodeClient) GetNodeStagedVolumes() map[string]CSIVolume {
	return f.nodeStagedVolumes
}

// AddNodeStagedVolume adds specified volume to nodeStagedVolumes
func (f *NodeClient) AddNodeStagedVolume(volID, deviceMountPath string, volumeContext map[string]string) {
	f.nodeStagedVolumes[volID] = CSIVolume{
		Path:          deviceMountPath,
		VolumeContext: volumeContext,
	}
}

// NodePublishVolume implements CSI NodePublishVolume
func (f *NodeClient) NodePublishVolume(ctx context.Context, req *csipb.NodePublishVolumeRequest, opts ...grpc.CallOption) (*csipb.NodePublishVolumeResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}

	if req.GetVolumeId() == "" {
		return nil, errors.New("missing volume id")
	}
	if req.GetTargetPath() == "" {
		return nil, errors.New("missing target path")
	}
	fsTypes := "block|ext4|xfs|zfs"
	fsType := req.GetVolumeCapability().GetMount().GetFsType()
	if !strings.Contains(fsTypes, fsType) {
		return nil, errors.New("invalid fstype")
	}

	if req.GetVolumeId() == NodePublishTimeOut_VolumeID {
		timeoutErr := status.Errorf(codes.DeadlineExceeded, "timeout exceeded")
		return nil, timeoutErr
	}

	f.nodePublishedVolumes[req.GetVolumeId()] = CSIVolume{
		VolumeHandle:    req.GetVolumeId(),
		Path:            req.GetTargetPath(),
		DeviceMountPath: req.GetStagingTargetPath(),
		VolumeContext:   req.GetVolumeContext(),
		FSType:          req.GetVolumeCapability().GetMount().GetFsType(),
		MountFlags:      req.GetVolumeCapability().GetMount().MountFlags,
	}
	return &csipb.NodePublishVolumeResponse{}, nil
}

// NodeUnpublishVolume implements csi method
func (f *NodeClient) NodeUnpublishVolume(ctx context.Context, req *csipb.NodeUnpublishVolumeRequest, opts ...grpc.CallOption) (*csipb.NodeUnpublishVolumeResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}

	if req.GetVolumeId() == "" {
		return nil, errors.New("missing volume id")
	}
	if req.GetTargetPath() == "" {
		return nil, errors.New("missing target path")
	}
	delete(f.nodePublishedVolumes, req.GetVolumeId())
	return &csipb.NodeUnpublishVolumeResponse{}, nil
}

// NodeStagevolume implements csi method
func (f *NodeClient) NodeStageVolume(ctx context.Context, req *csipb.NodeStageVolumeRequest, opts ...grpc.CallOption) (*csipb.NodeStageVolumeResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}

	if req.GetVolumeId() == "" {
		return nil, errors.New("missing volume id")
	}
	if req.GetStagingTargetPath() == "" {
		return nil, errors.New("missing staging target path")
	}

	csiVol := CSIVolume{
		Path:          req.GetStagingTargetPath(),
		VolumeContext: req.GetVolumeContext(),
	}

	fsType := ""
	fsTypes := "block|ext4|xfs|zfs"
	mounted := req.GetVolumeCapability().GetMount()
	if mounted != nil {
		fsType = mounted.GetFsType()
		csiVol.MountFlags = mounted.GetMountFlags()
	}
	if !strings.Contains(fsTypes, fsType) {
		return nil, errors.New("invalid fstype")
	}

	if req.GetVolumeId() == NodeStageTimeOut_VolumeID {
		timeoutErr := status.Errorf(codes.DeadlineExceeded, "timeout exceeded")
		return nil, timeoutErr
	}

	f.nodeStagedVolumes[req.GetVolumeId()] = csiVol
	return &csipb.NodeStageVolumeResponse{}, nil
}

// NodeUnstageVolume implements csi method
func (f *NodeClient) NodeUnstageVolume(ctx context.Context, req *csipb.NodeUnstageVolumeRequest, opts ...grpc.CallOption) (*csipb.NodeUnstageVolumeResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}

	if req.GetVolumeId() == "" {
		return nil, errors.New("missing volume id")
	}
	if req.GetStagingTargetPath() == "" {
		return nil, errors.New("missing staging target path")
	}

	delete(f.nodeStagedVolumes, req.GetVolumeId())
	return &csipb.NodeUnstageVolumeResponse{}, nil
}

// NodeExpandVolume implements csi method
func (f *NodeClient) NodeExpandVolume(ctx context.Context, req *csipb.NodeExpandVolumeRequest, opts ...grpc.CallOption) (*csipb.NodeExpandVolumeResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}

	if req.GetVolumeId() == "" {
		return nil, errors.New("missing volume id")
	}
	if req.GetVolumePath() == "" {
		return nil, errors.New("missing volume path")
	}

	if req.GetCapacityRange().RequiredBytes <= 0 {
		return nil, errors.New("required bytes should be greater than 0")
	}

	resp := &csipb.NodeExpandVolumeResponse{
		CapacityBytes: req.GetCapacityRange().RequiredBytes,
	}
	return resp, nil
}

// NodeGetId implements csi method
func (f *NodeClient) NodeGetInfo(ctx context.Context, in *csipb.NodeGetInfoRequest, opts ...grpc.CallOption) (*csipb.NodeGetInfoResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}
	return f.nodeGetInfoResp, nil
}

// NodeGetCapabilities implements csi method
func (f *NodeClient) NodeGetCapabilities(ctx context.Context, in *csipb.NodeGetCapabilitiesRequest, opts ...grpc.CallOption) (*csipb.NodeGetCapabilitiesResponse, error) {
	resp := &csipb.NodeGetCapabilitiesResponse{
		Capabilities: []*csipb.NodeServiceCapability{},
	}
	if f.stageUnstageSet {
		resp.Capabilities = append(resp.Capabilities, &csipb.NodeServiceCapability{
			Type: &csipb.NodeServiceCapability_Rpc{
				Rpc: &csipb.NodeServiceCapability_RPC{
					Type: csipb.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME,
				},
			},
		})
	}
	if f.expansionSet {
		resp.Capabilities = append(resp.Capabilities, &csipb.NodeServiceCapability{
			Type: &csipb.NodeServiceCapability_Rpc{
				Rpc: &csipb.NodeServiceCapability_RPC{
					Type: csipb.NodeServiceCapability_RPC_EXPAND_VOLUME,
				},
			},
		})
	}

	if f.volumeStatsSet {
		resp.Capabilities = append(resp.Capabilities, &csipb.NodeServiceCapability{
			Type: &csipb.NodeServiceCapability_Rpc{
				Rpc: &csipb.NodeServiceCapability_RPC{
					Type: csipb.NodeServiceCapability_RPC_GET_VOLUME_STATS,
				},
			},
		})
	}
	return resp, nil
}

/*
// NodeGetVolumeStats implements csi method
func (f *NodeClient) NodeGetVolumeStats(ctx context.Context, in *csipb.NodeGetVolumeStatsRequest, opts ...grpc.CallOption) (*csipb.NodeGetVolumeStatsResponse, error) {
	return nil, nil
}
*/

// NodeGetVolumeStats implements csi method
func (f *NodeClient) NodeGetVolumeStats(ctx context.Context, req *csipb.NodeGetVolumeStatsRequest, opts ...grpc.CallOption) (*csipb.NodeGetVolumeStatsResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}
	if req.GetVolumeId() == "" {
		return nil, errors.New("missing volume id")
	}
	if req.GetVolumePath() == "" {
		return nil, errors.New("missing Volume path")
	}
	if f.nodeVolumeStatsResp != nil {
		return f.nodeVolumeStatsResp, nil
	}
	return &csipb.NodeGetVolumeStatsResponse{}, nil
}

// ControllerClient represents a CSI Controller client
type ControllerClient struct {
	nextCapabilities []*csipb.ControllerServiceCapability
	nextErr          error
}

// NewControllerClient returns a ControllerClient
func NewControllerClient() *ControllerClient {
	return &ControllerClient{}
}

// SetNextError injects next expected error
func (f *ControllerClient) SetNextError(err error) {
	f.nextErr = err
}

// SetNextCapabilities injects next expected capabilities
func (f *ControllerClient) SetNextCapabilities(caps []*csipb.ControllerServiceCapability) {
	f.nextCapabilities = caps
}

// ControllerGetCapabilities implements csi method
func (f *ControllerClient) ControllerGetCapabilities(ctx context.Context, in *csipb.ControllerGetCapabilitiesRequest, opts ...grpc.CallOption) (*csipb.ControllerGetCapabilitiesResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}

	if f.nextCapabilities == nil {
		f.nextCapabilities = []*csipb.ControllerServiceCapability{
			{
				Type: &csipb.ControllerServiceCapability_Rpc{
					Rpc: &csipb.ControllerServiceCapability_RPC{
						Type: csipb.ControllerServiceCapability_RPC_PUBLISH_UNPUBLISH_VOLUME,
					},
				},
			},
		}
	}
	return &csipb.ControllerGetCapabilitiesResponse{
		Capabilities: f.nextCapabilities,
	}, nil
}

// CreateVolume implements csi method
func (f *ControllerClient) CreateVolume(ctx context.Context, in *csipb.CreateVolumeRequest, opts ...grpc.CallOption) (*csipb.CreateVolumeResponse, error) {
	return nil, nil
}

// DeleteVolume implements csi method
func (f *ControllerClient) DeleteVolume(ctx context.Context, in *csipb.DeleteVolumeRequest, opts ...grpc.CallOption) (*csipb.DeleteVolumeResponse, error) {
	return nil, nil
}

// ControllerPublishVolume implements csi method
func (f *ControllerClient) ControllerPublishVolume(ctx context.Context, in *csipb.ControllerPublishVolumeRequest, opts ...grpc.CallOption) (*csipb.ControllerPublishVolumeResponse, error) {
	return nil, nil
}

// ControllerUnpublishVolume implements csi method
func (f *ControllerClient) ControllerUnpublishVolume(ctx context.Context, in *csipb.ControllerUnpublishVolumeRequest, opts ...grpc.CallOption) (*csipb.ControllerUnpublishVolumeResponse, error) {
	return nil, nil
}

// ValidateVolumeCapabilities implements csi method
func (f *ControllerClient) ValidateVolumeCapabilities(ctx context.Context, in *csipb.ValidateVolumeCapabilitiesRequest, opts ...grpc.CallOption) (*csipb.ValidateVolumeCapabilitiesResponse, error) {
	return nil, nil
}

// ListVolumes implements csi method
func (f *ControllerClient) ListVolumes(ctx context.Context, in *csipb.ListVolumesRequest, opts ...grpc.CallOption) (*csipb.ListVolumesResponse, error) {
	return nil, nil
}

// GetCapacity implements csi method
func (f *ControllerClient) GetCapacity(ctx context.Context, in *csipb.GetCapacityRequest, opts ...grpc.CallOption) (*csipb.GetCapacityResponse, error) {
	return nil, nil
}
