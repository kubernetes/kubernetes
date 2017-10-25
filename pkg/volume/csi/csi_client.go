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
	"bytes"
	"errors"
	"fmt"
	"net"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"github.com/golang/glog"
	grpctx "golang.org/x/net/context"
	"google.golang.org/grpc"
	api "k8s.io/api/core/v1"
)

type csiClient interface {
	AssertSupportedVersion(ctx grpctx.Context, ver *csipb.Version) error
	NodePublishVolume(
		ctx grpctx.Context,
		volumeid string,
		readOnly bool,
		targetPath string,
		accessMode api.PersistentVolumeAccessMode,
		volumeInfo map[string]string,
		fsType string,
	) error
	NodeUnpublishVolume(ctx grpctx.Context, volID string, targetPath string) error
}

// csiClient encapsulates all csi-plugin methods
type csiDriverClient struct {
	network          string
	addr             string
	conn             *grpc.ClientConn
	idClient         csipb.IdentityClient
	nodeClient       csipb.NodeClient
	ctrlClient       csipb.ControllerClient
	versionAsserted  bool
	versionSupported bool
	publishAsserted  bool
	publishCapable   bool
}

func newCsiDriverClient(network, addr string) *csiDriverClient {
	return &csiDriverClient{network: network, addr: addr}
}

// assertConnection ensures a valid connection has been established
// if not, it creates a new connection and associated clients
func (c *csiDriverClient) assertConnection() error {
	if c.conn == nil {
		conn, err := grpc.Dial(
			c.addr,
			grpc.WithInsecure(),
			grpc.WithDialer(func(target string, timeout time.Duration) (net.Conn, error) {
				return net.Dial(c.network, target)
			}),
		)
		if err != nil {
			return err
		}
		c.conn = conn
		c.idClient = csipb.NewIdentityClient(conn)
		c.nodeClient = csipb.NewNodeClient(conn)
		c.ctrlClient = csipb.NewControllerClient(conn)

		// set supported version
	}

	return nil
}

// AssertSupportedVersion ensures driver supports specified spec version.
// If version is not supported, the assertion fails with an error.
// This test should be done early during the storage operation flow to avoid
// unnecessary calls later.
func (c *csiDriverClient) AssertSupportedVersion(ctx grpctx.Context, ver *csipb.Version) error {
	if c.versionAsserted {
		if !c.versionSupported {
			return fmt.Errorf("version %s not supported", verToStr(ver))
		}
		return nil
	}

	if err := c.assertConnection(); err != nil {
		c.versionAsserted = false
		return err
	}

	glog.V(4).Info(log("asserting version supported by driver"))
	rsp, err := c.idClient.GetSupportedVersions(ctx, &csipb.GetSupportedVersionsRequest{})
	if err != nil {
		c.versionAsserted = false
		return err
	}

	supported := false
	vers := rsp.GetSupportedVersions()
	glog.V(4).Info(log("driver reports %d versions supported: %s", len(vers), versToStr(vers)))

	for _, v := range vers {
		//TODO (vladimirvivien) use more lenient/heuristic for exact or match of ranges etc
		if verToStr(v) == verToStr(ver) {
			supported = true
			break
		}
	}

	c.versionAsserted = true
	c.versionSupported = supported

	if !supported {
		return fmt.Errorf("version %s not supported", verToStr(ver))
	}

	glog.V(4).Info(log("version %s supported", verToStr(ver)))
	return nil
}

func (c *csiDriverClient) NodePublishVolume(
	ctx grpctx.Context,
	volID string,
	readOnly bool,
	targetPath string,
	accessMode api.PersistentVolumeAccessMode,
	volumeInfo map[string]string,
	fsType string,
) error {

	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}
	if err := c.assertConnection(); err != nil {
		glog.Errorf("%v: failed to assert a connection: %v", csiPluginName, err)
		return err
	}

	req := &csipb.NodePublishVolumeRequest{
		Version:           csiVersion,
		VolumeId:          volID,
		TargetPath:        targetPath,
		Readonly:          readOnly,
		PublishVolumeInfo: volumeInfo,

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

	_, err := c.nodeClient.NodePublishVolume(ctx, req)
	return err
}

func (c *csiDriverClient) NodeUnpublishVolume(ctx grpctx.Context, volID string, targetPath string) error {

	if volID == "" {
		return errors.New("missing volume id")
	}
	if targetPath == "" {
		return errors.New("missing target path")
	}
	if err := c.assertConnection(); err != nil {
		glog.Error(log("failed to assert a connection: %v", err))
		return err
	}

	req := &csipb.NodeUnpublishVolumeRequest{
		Version:    csiVersion,
		VolumeId:   volID,
		TargetPath: targetPath,
	}

	_, err := c.nodeClient.NodeUnpublishVolume(ctx, req)
	return err
}

func asCSIAccessMode(am api.PersistentVolumeAccessMode) csipb.VolumeCapability_AccessMode_Mode {
	switch am {
	case api.ReadWriteOnce:
		return csipb.VolumeCapability_AccessMode_SINGLE_NODE_WRITER
	case api.ReadOnlyMany:
		return csipb.VolumeCapability_AccessMode_MULTI_NODE_SINGLE_WRITER
	case api.ReadWriteMany:
		return csipb.VolumeCapability_AccessMode_MULTI_NODE_MULTI_WRITER
	}
	return csipb.VolumeCapability_AccessMode_UNKNOWN
}

func verToStr(ver *csipb.Version) string {
	if ver == nil {
		return ""
	}
	return fmt.Sprintf("%d.%d.%d", ver.GetMajor(), ver.GetMinor(), ver.GetPatch())
}

func versToStr(vers []*csipb.Version) string {
	if vers == nil {
		return ""
	}
	str := bytes.NewBufferString("[")
	for _, v := range vers {
		str.WriteString(fmt.Sprintf("{%s};", verToStr(v)))
	}
	str.WriteString("]")
	return str.String()
}
