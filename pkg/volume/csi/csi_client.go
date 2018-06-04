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
	"errors"
	"fmt"
	"net"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi/v0"
	"github.com/golang/glog"
	grpctx "golang.org/x/net/context"
	"google.golang.org/grpc"
	api "k8s.io/api/core/v1"
)

type csiClient interface {
	NodePublishVolume(
		ctx grpctx.Context,
		volumeid string,
		readOnly bool,
		stagingTargetPath string,
		targetPath string,
		accessMode api.PersistentVolumeAccessMode,
		volumeInfo map[string]string,
		volumeAttribs map[string]string,
		nodePublishSecrets map[string]string,
		fsType string,
	) error
	NodeUnpublishVolume(
		ctx grpctx.Context,
		volID string,
		targetPath string,
	) error
	NodeStageVolume(ctx grpctx.Context,
		volID string,
		publishVolumeInfo map[string]string,
		stagingTargetPath string,
		fsType string,
		accessMode api.PersistentVolumeAccessMode,
		nodeStageSecrets map[string]string,
		volumeAttribs map[string]string,
	) error
	NodeUnstageVolume(ctx grpctx.Context, volID, stagingTargetPath string) error
	NodeGetCapabilities(ctx grpctx.Context) ([]*csipb.NodeServiceCapability, error)
}

// csiClient encapsulates all csi-plugin methods
type csiDriverClient struct {
	driverName string
	nodeClient csipb.NodeClient
}

var _ csiClient = &csiDriverClient{}

func newCsiDriverClient(driverName string) *csiDriverClient {
	c := &csiDriverClient{driverName: driverName}
	return c
}

func (c *csiDriverClient) NodePublishVolume(
	ctx grpctx.Context,
	volID string,
	readOnly bool,
	stagingTargetPath string,
	targetPath string,
	accessMode api.PersistentVolumeAccessMode,
	volumeInfo map[string]string,
	volumeAttribs map[string]string,
	nodePublishSecrets map[string]string,
	fsType string,
) error {
	glog.V(4).Info(log("calling NodePublishVolume rpc [volid=%s,target_path=%s]", volID, targetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}

	conn, err := newGrpcConn(c.driverName)
	if err != nil {
		return err
	}
	defer conn.Close()
	nodeClient := csipb.NewNodeClient(conn)

	req := &csipb.NodePublishVolumeRequest{
		VolumeId:           volID,
		TargetPath:         targetPath,
		Readonly:           readOnly,
		PublishInfo:        volumeInfo,
		VolumeAttributes:   volumeAttribs,
		NodePublishSecrets: nodePublishSecrets,
		VolumeCapability: &csipb.VolumeCapability{
			AccessMode: &csipb.VolumeCapability_AccessMode{
				Mode: asCSIAccessMode(accessMode),
			},
			AccessType: &csipb.VolumeCapability_Mount{
				Mount: &csipb.VolumeCapability_MountVolume{
					FsType: fsType,
				},
			},
		},
	}
	if stagingTargetPath != "" {
		req.StagingTargetPath = stagingTargetPath
	}

	_, err = nodeClient.NodePublishVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeUnpublishVolume(ctx grpctx.Context, volID string, targetPath string) error {
	glog.V(4).Info(log("calling NodeUnpublishVolume rpc: [volid=%s, target_path=%s", volID, targetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}

	conn, err := newGrpcConn(c.driverName)
	if err != nil {
		return err
	}
	defer conn.Close()
	nodeClient := csipb.NewNodeClient(conn)

	req := &csipb.NodeUnpublishVolumeRequest{
		VolumeId:   volID,
		TargetPath: targetPath,
	}

	_, err = nodeClient.NodeUnpublishVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeStageVolume(ctx grpctx.Context,
	volID string,
	publishInfo map[string]string,
	stagingTargetPath string,
	fsType string,
	accessMode api.PersistentVolumeAccessMode,
	nodeStageSecrets map[string]string,
	volumeAttribs map[string]string,
) error {
	glog.V(4).Info(log("calling NodeStageVolume rpc [volid=%s,staging_target_path=%s]", volID, stagingTargetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if stagingTargetPath == "" {
		return errors.New("missing staging target path")
	}

	conn, err := newGrpcConn(c.driverName)
	if err != nil {
		return err
	}
	defer conn.Close()
	nodeClient := csipb.NewNodeClient(conn)

	req := &csipb.NodeStageVolumeRequest{
		VolumeId:          volID,
		PublishInfo:       publishInfo,
		StagingTargetPath: stagingTargetPath,
		VolumeCapability: &csipb.VolumeCapability{
			AccessMode: &csipb.VolumeCapability_AccessMode{
				Mode: asCSIAccessMode(accessMode),
			},
			AccessType: &csipb.VolumeCapability_Mount{
				Mount: &csipb.VolumeCapability_MountVolume{
					FsType: fsType,
				},
			},
		},
		NodeStageSecrets: nodeStageSecrets,
		VolumeAttributes: volumeAttribs,
	}

	_, err = nodeClient.NodeStageVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeUnstageVolume(ctx grpctx.Context, volID, stagingTargetPath string) error {
	glog.V(4).Info(log("calling NodeUnstageVolume rpc [volid=%s,staging_target_path=%s]", volID, stagingTargetPath))
	if volID == "" {
		return errors.New("missing volume id")
	}
	if stagingTargetPath == "" {
		return errors.New("missing staging target path")
	}

	conn, err := newGrpcConn(c.driverName)
	if err != nil {
		return err
	}
	defer conn.Close()
	nodeClient := csipb.NewNodeClient(conn)

	req := &csipb.NodeUnstageVolumeRequest{
		VolumeId:          volID,
		StagingTargetPath: stagingTargetPath,
	}
	_, err = nodeClient.NodeUnstageVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeGetCapabilities(ctx grpctx.Context) ([]*csipb.NodeServiceCapability, error) {
	glog.V(4).Info(log("calling NodeGetCapabilities rpc"))

	conn, err := newGrpcConn(c.driverName)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	nodeClient := csipb.NewNodeClient(conn)

	req := &csipb.NodeGetCapabilitiesRequest{}
	resp, err := nodeClient.NodeGetCapabilities(ctx, req)
	if err != nil {
		return nil, err
	}
	return resp.GetCapabilities(), nil
}

func asCSIAccessMode(am api.PersistentVolumeAccessMode) csipb.VolumeCapability_AccessMode_Mode {
	switch am {
	case api.ReadWriteOnce:
		return csipb.VolumeCapability_AccessMode_SINGLE_NODE_WRITER
	case api.ReadOnlyMany:
		return csipb.VolumeCapability_AccessMode_MULTI_NODE_READER_ONLY
	case api.ReadWriteMany:
		return csipb.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER
	}
	return csipb.VolumeCapability_AccessMode_UNKNOWN
}

func newGrpcConn(driverName string) (*grpc.ClientConn, error) {
	if driverName == "" {
		return nil, fmt.Errorf("driver name is empty")
	}

	network := "unix"
	addr := fmt.Sprintf(csiAddrTemplate, driverName)
	glog.V(4).Infof(log("creating new gRPC connection for [%s://%s]", network, addr))

	return grpc.Dial(
		addr,
		grpc.WithInsecure(),
		grpc.WithDialer(func(target string, timeout time.Duration) (net.Conn, error) {
			return net.Dial(network, target)
		}),
	)
}
