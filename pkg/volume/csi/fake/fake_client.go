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
	grpctx "golang.org/x/net/context"
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

// GetSupportedVersions returns supported version
func (f *IdentityClient) GetSupportedVersions(ctx grpctx.Context, req *csipb.GetSupportedVersionsRequest, opts ...grpc.CallOption) (*csipb.GetSupportedVersionsResponse, error) {
	// short circuit with an error
	if f.nextErr != nil {
		return nil, f.nextErr
	}

	rsp := &csipb.GetSupportedVersionsResponse{
		SupportedVersions: []*csipb.Version{
			{Major: 0, Minor: 0, Patch: 1},
			{Major: 0, Minor: 1, Patch: 0},
			{Major: 1, Minor: 0, Patch: 0},
			{Major: 1, Minor: 0, Patch: 1},
			{Major: 1, Minor: 1, Patch: 1},
		},
	}
	return rsp, nil
}

// GetPluginInfo returns plugin info
func (f *IdentityClient) GetPluginInfo(ctx context.Context, in *csipb.GetPluginInfoRequest, opts ...grpc.CallOption) (*csipb.GetPluginInfoResponse, error) {
	return nil, nil
}

// NodeClient returns CSI node client
type NodeClient struct {
	nodePublishedVolumes map[string]string
	nextErr              error
}

// NewNodeClient returns fake node client
func NewNodeClient() *NodeClient {
	return &NodeClient{nodePublishedVolumes: make(map[string]string)}
}

// SetNextError injects next expected error
func (f *NodeClient) SetNextError(err error) {
	f.nextErr = err
}

// GetNodePublishedVolumes returns node published volumes
func (f *NodeClient) GetNodePublishedVolumes() map[string]string {
	return f.nodePublishedVolumes
}

// NodePublishVolume implements CSI NodePublishVolume
func (f *NodeClient) NodePublishVolume(ctx grpctx.Context, req *csipb.NodePublishVolumeRequest, opts ...grpc.CallOption) (*csipb.NodePublishVolumeResponse, error) {

	if f.nextErr != nil {
		return nil, f.nextErr
	}

	if req.GetVolumeId() == "" {
		return nil, errors.New("missing volume id")
	}
	if req.GetTargetPath() == "" {
		return nil, errors.New("missing target path")
	}
	fsTypes := "ext4|xfs|zfs"
	fsType := req.GetVolumeCapability().GetMount().GetFsType()
	if !strings.Contains(fsTypes, fsType) {
		return nil, errors.New("invlid fstype")
	}
	f.nodePublishedVolumes[req.GetVolumeId()] = req.GetTargetPath()
	return &csipb.NodePublishVolumeResponse{}, nil
}

// NodeProbe implements csi NodeProbe
func (f *NodeClient) NodeProbe(ctx context.Context, req *csipb.NodeProbeRequest, opts ...grpc.CallOption) (*csipb.NodeProbeResponse, error) {
	if f.nextErr != nil {
		return nil, f.nextErr
	}
	if req.Version == nil {
		return nil, errors.New("missing version")
	}
	return &csipb.NodeProbeResponse{}, nil
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

// GetNodeID implements method
func (f *NodeClient) GetNodeID(ctx context.Context, in *csipb.GetNodeIDRequest, opts ...grpc.CallOption) (*csipb.GetNodeIDResponse, error) {
	return nil, nil
}

// NodeGetCapabilities implements csi method
func (f *NodeClient) NodeGetCapabilities(ctx context.Context, in *csipb.NodeGetCapabilitiesRequest, opts ...grpc.CallOption) (*csipb.NodeGetCapabilitiesResponse, error) {
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

// ControllerProbe implements csi method
func (f *ControllerClient) ControllerProbe(ctx context.Context, in *csipb.ControllerProbeRequest, opts ...grpc.CallOption) (*csipb.ControllerProbeResponse, error) {
	return nil, nil
}
