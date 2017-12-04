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
	"encoding/json"
	"errors"
	"fmt"
	"path"

	"github.com/golang/glog"
	grpctx "golang.org/x/net/context"
	api "k8s.io/api/core/v1"
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

	csiSource, err := getCSISourceFromSpec(c.spec)
	if err != nil {
		glog.Error(log("mounter.SetupAt failed to get CSI persistent source: %v", err))
		return err
	}

	ctx, cancel := grpctx.WithTimeout(grpctx.Background(), csiTimeout)
	defer cancel()

	csi := c.csiClient
	nodeName := string(c.plugin.host.GetNodeName())
	attachID := getAttachmentName(csiSource.VolumeHandle, csiSource.Driver, nodeName)

	// ensure version is supported
	if err := csi.AssertSupportedVersion(ctx, csiVersion); err != nil {
		glog.Errorf(log("failed to assert version: %v", err))
		return err
	}

	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	if c.volumeInfo == nil {

		attachment, err := c.k8s.StorageV1alpha1().VolumeAttachments().Get(attachID, meta.GetOptions{})
		if err != nil {
			glog.Error(log("mounter.SetupAt failed while getting volume attachment [id=%v]: %v", attachID, err))
			return err
		}

		if attachment == nil {
			glog.Error(log("unable to find VolumeAttachment [id=%s]", attachID))
			return errors.New("no existing VolumeAttachment found")
		}
		c.volumeInfo = attachment.Status.AttachmentMetadata
	}

	// get volume attributes
	// TODO: for alpha vol atttributes are passed via PV.Annotations
	// Beta will fix that
	attribs, err := getVolAttribsFromSpec(c.spec)
	if err != nil {
		glog.Error(log("mounter.SetUpAt failed to extract volume attributes from PV annotations: %v", err))
		return err
	}

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := api.ReadWriteOnce
	if c.spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = c.spec.PersistentVolume.Spec.AccessModes[0]
	}

	err = csi.NodePublishVolume(
		ctx,
		c.volumeID,
		c.readOnly,
		dir,
		accessMode,
		c.volumeInfo,
		attribs,
		"ext4", //TODO needs to be sourced from PV or somewhere else
	)

	if err != nil {
		glog.Errorf(log("Mounter.SetupAt failed: %v", err))
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

// getVolAttribsFromSpec exracts CSI VolumeAttributes information from PV.Annotations
// using key csi.kubernetes.io/volume-attributes.  The annotation value is expected
// to be a JSON-encoded object of form {"key0":"val0",...,"keyN":"valN"}
func getVolAttribsFromSpec(spec *volume.Spec) (map[string]string, error) {
	if spec == nil {
		return nil, errors.New("missing volume spec")
	}
	annotations := spec.PersistentVolume.GetAnnotations()
	if annotations == nil {
		return nil, nil // no annotations found
	}
	jsonAttribs := annotations[csiVolAttribsAnnotationKey]
	if jsonAttribs == "" {
		return nil, nil // csi annotation not found
	}
	attribs := map[string]string{}
	if err := json.Unmarshal([]byte(jsonAttribs), &attribs); err != nil {
		glog.Error(log("error parsing csi PV.Annotation [%s]=%s: %v", csiVolAttribsAnnotationKey, jsonAttribs, err))
		return nil, err
	}
	return attribs, nil
}
