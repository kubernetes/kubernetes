/*
Copyright 2014 The Kubernetes Authors.

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

	grpctx "golang.org/x/net/context"
	csipb "k8s.io/kubernetes/pkg/volume/csi/proto/csi"
)

type FakeIdentityClient struct {
	nextErr error
}

func NewIdentityClient() *FakeIdentityClient {
	return &FakeIdentityClient{}
}

func (f *FakeIdentityClient) SetNextError(err error) {
	f.nextErr = err
}

func (f *FakeIdentityClient) GetSupportedVersions(ctx grpctx.Context, req *csipb.GetSupportedVersionsRequest, opts ...grpc.CallOption) (*csipb.GetSupportedVersionsResponse, error) {
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

func (f *FakeIdentityClient) GetPluginInfo(ctx context.Context, in *csipb.GetPluginInfoRequest, opts ...grpc.CallOption) (*csipb.GetPluginInfoResponse, error) {
	return nil, nil
}

type FakeNodeClient struct {
	nodePublishedVolumes map[string]string
	nextErr              error
}

func NewNodeClient() *FakeNodeClient {
	return &FakeNodeClient{nodePublishedVolumes: make(map[string]string)}
}

func (f *FakeNodeClient) SetNextError(err error) {
	f.nextErr = err
}

func (f *FakeNodeClient) GetNodePublishedVolumes() map[string]string {
	return f.nodePublishedVolumes
}

func (f *FakeNodeClient) NodePublishVolume(ctx grpctx.Context, req *csipb.NodePublishVolumeRequest, opts ...grpc.CallOption) (*csipb.NodePublishVolumeResponse, error) {

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

func (f *FakeNodeClient) NodeUnpublishVolume(ctx context.Context, req *csipb.NodeUnpublishVolumeRequest, opts ...grpc.CallOption) (*csipb.NodeUnpublishVolumeResponse, error) {
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

func (f *FakeNodeClient) GetNodeID(ctx context.Context, in *csipb.GetNodeIDRequest, opts ...grpc.CallOption) (*csipb.GetNodeIDResponse, error) {
	return nil, nil
}
func (f *FakeNodeClient) NodeProbe(ctx context.Context, in *csipb.NodeProbeRequest, opts ...grpc.CallOption) (*csipb.NodeProbeResponse, error) {
	return nil, nil
}
func (f *FakeNodeClient) NodeGetCapabilities(ctx context.Context, in *csipb.NodeGetCapabilitiesRequest, opts ...grpc.CallOption) (*csipb.NodeGetCapabilitiesResponse, error) {
	return nil, nil
}
