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
	"path"

	"github.com/golang/glog"
	grpctx "golang.org/x/net/context"
	api "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1alpha1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

type csiMountMgr struct {
	k8s        kubernetes.Interface
	csiClient  csiClient
	plugin     *csiPlugin
	driverName string
	volumeID   string
	readOnly   bool
	spec       *volume.Spec
	pod        *api.Pod
	podUID     types.UID
	options    volume.VolumeOptions
	volumeInfo map[string]string
	volume.MetricsNil
}

// volume.Volume methods
var _ volume.Volume = &csiMountMgr{}

func (c *csiMountMgr) GetPath() string {
	return getTargetPath(c.podUID, c.driverName, c.volumeID, c.plugin.host)
}

func getTargetPath(uid types.UID, driverName string, volID string, host volume.VolumeHost) string {
	// driverName validated at Mounter creation
	// sanitize (replace / with ~) in volumeID before it's appended to path:w
	driverPath := fmt.Sprintf("%s/%s", driverName, kstrings.EscapeQualifiedNameForDisk(volID))
	return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(csiPluginName), driverPath)
}

// volume.Mounter methods
var _ volume.Mounter = &csiMountMgr{}

func (c *csiMountMgr) CanMount() error {
	//TODO (vladimirvivien) use this method to probe controller using CSI.NodeProbe() call
	// to ensure Node service is ready in the CSI plugin
	return nil
}

func (c *csiMountMgr) SetUp(fsGroup *int64) error {
	return c.SetUpAt(c.GetPath(), fsGroup)
}

func (c *csiMountMgr) SetUpAt(dir string, fsGroup *int64) error {
	glog.V(4).Infof(log("Mounter.SetUpAt(%s)", dir))

	ctx, cancel := grpctx.WithTimeout(grpctx.Background(), csiTimeout)
	defer cancel()

	csi := c.csiClient
	pvName := c.spec.PersistentVolume.GetName()

	// ensure version is supported
	if err := csi.AssertSupportedVersion(ctx, csiVersion); err != nil {
		glog.Errorf(log("failed to assert version: %v", err))
		return err
	}

	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	if c.volumeInfo == nil {

		//TODO (vladimirvivien) consider using VolumesAttachments().Get() to retrieve
		//the object directly. This requires the ability to reconstruct the ID using volumeName+nodeName (nodename may not be avilable)
		attachList, err := c.k8s.StorageV1alpha1().VolumeAttachments().List(meta.ListOptions{})
		if err != nil {
			glog.Error(log("failed to get volume attachments: %v", err))
			return err
		}

		var attachment *storage.VolumeAttachment
		for _, attach := range attachList.Items {
			if attach.Spec.Source.PersistentVolumeName != nil &&
				*attach.Spec.Source.PersistentVolumeName == pvName {
				attachment = &attach
				break
			}
		}

		if attachment == nil {
			glog.Error(log("unable to find VolumeAttachment with PV.name = %s", pvName))
			return errors.New("no existing VolumeAttachment found")
		}
		c.volumeInfo = attachment.Status.AttachmentMetadata
	}

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := api.ReadWriteOnce
	if c.spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = c.spec.PersistentVolume.Spec.AccessModes[0]
	}

	err := csi.NodePublishVolume(
		ctx,
		c.volumeID,
		c.readOnly,
		dir,
		accessMode,
		c.volumeInfo,
		"ext4", //TODO needs to be sourced from PV or somewhere else
	)

	if err != nil {
		glog.Errorf(log("Mounter.Setup failed: %v", err))
		return err
	}
	glog.V(4).Infof(log("successfully mounted %s", dir))

	return nil
}

func (c *csiMountMgr) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        c.readOnly,
		Managed:         !c.readOnly,
		SupportsSELinux: false,
	}
}

// volume.Unmounter methods
var _ volume.Unmounter = &csiMountMgr{}

func (c *csiMountMgr) TearDown() error {
	return c.TearDownAt(c.GetPath())
}
func (c *csiMountMgr) TearDownAt(dir string) error {
	glog.V(4).Infof(log("Unmounter.TearDown(%s)", dir))

	// extract driverName and volID from path
	base, volID := path.Split(dir)
	volID = kstrings.UnescapeQualifiedNameForDisk(volID)
	driverName := path.Base(base)

	if c.csiClient == nil {
		addr := fmt.Sprintf(csiAddrTemplate, driverName)
		client := newCsiDriverClient("unix", addr)
		glog.V(4).Infof(log("unmounter csiClient setup [volume=%v,driver=%v]", volID, driverName))
		c.csiClient = client
	}

	ctx, cancel := grpctx.WithTimeout(grpctx.Background(), csiTimeout)
	defer cancel()

	csi := c.csiClient

	// TODO make all assertion calls private within the client itself
	if err := csi.AssertSupportedVersion(ctx, csiVersion); err != nil {
		glog.Errorf(log("failed to assert version: %v", err))
		return err
	}

	err := csi.NodeUnpublishVolume(ctx, volID, dir)

	if err != nil {
		glog.Errorf(log("Mounter.Setup failed: %v", err))
		return err
	}

	glog.V(4).Infof(log("successfully unmounted %s", dir))

	return nil
}
