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

	"google.golang.org/grpc"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
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
	VolumeContext map[string]string
	Path          string
	MountFlags    []string
}

// NodeClient returns CSI node client
type NodeClient struct {
	nodePublishedVolumes map[string]CSIVolume
	nodeStagedVolumes    map[string]CSIVolume
	stageUnstageSet      bool
	nodeGetInfoResp      *csipb.NodeGetInfoResponse
	nextErr              error
}

// NewNodeClient returns fake node client
func NewNodeClient(stageUnstageSet bool) *NodeClient {
	return &NodeClient{
		nodePublishedVolumes: make(map[string]CSIVolume),
		nodeStagedVolumes:    make(map[string]CSIVolume),
		stageUnstageSet:      stageUnstageSet,
	}
}

// SetNextError injects next expected error
func (f *NodeClient) SetNextError(err error) {
	f.nextErr = err
}

func (f *NodeClient) SetNodeGetInfoResp(resp *csipb.NodeGetInfoResponse) {
	f.nodeGetInfoResp = resp
}

// GetNodePublishedVolumes returns node published volumes
func (f *NodeClient) GetNodePublishedVolumes() map[string]CSIVolume {
	return f.nodePublishedVolumes
}

// GetNodeStagedVolumes returns node staged volumes
func (f *NodeClient) GetNodeStagedVolumes() map[string]CSIVolume {
	return f.nodeStagedVolumes
}

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
	f.nodePublishedVolumes[req.GetVolumeId()] = CSIVolume{
		Path:          req.GetTargetPath(),
		VolumeContext: req.GetVolumeContext(),
		MountFlags:    req.GetVolumeCapability().GetMount().MountFlags,
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

	fsType := ""
	fsTypes := "block|ext4|xfs|zfs"
	mounted := req.GetVolumeCapability().GetMount()
	if mounted != nil {
		fsType = mounted.GetFsType()
	}
	if !strings.Contains(fsTypes, fsType) {
		return nil, errors.New("invalid fstype")
	}

	f.nodeStagedVolumes[req.GetVolumeId()] = CSIVolume{
		Path:          req.GetStagingTargetPath(),
		VolumeContext: req.GetVolumeContext(),
	}
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
		Capabilities: []*csipb.NodeServiceCapability{
			{
				Type: &csipb.NodeServiceCapability_Rpc{
					Rpc: &csipb.NodeServiceCapability_RPC{
						Type: csipb.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME,
					},
				},
			},
		},
	}
	if f.stageUnstageSet {
		return resp, nil
	}
	return nil, nil
}

// NodeGetVolumeStats implements csi method
func (f *NodeClient) NodeGetVolumeStats(ctx context.Context, in *csipb.NodeGetVolumeStatsRequest, opts ...grpc.CallOption) (*csipb.NodeGetVolumeStatsResponse, error) {
	return nil, nil
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
