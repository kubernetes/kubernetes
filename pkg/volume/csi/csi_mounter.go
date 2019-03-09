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
	"os"
	"path"

	"k8s.io/klog"

	api "k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	kstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

//TODO (vladimirvivien) move this in a central loc later
var (
	volDataKey = struct {
		specVolID,
		volHandle,
		driverName,
		nodeName,
		attachmentID string
	}{
		"specVolID",
		"volumeHandle",
		"driverName",
		"nodeName",
		"attachmentID",
	}
	currentPodInfoMountVersion = "v1"
)

type csiMountMgr struct {
	csiClientGetter
	k8s            kubernetes.Interface
	plugin         *csiPlugin
	driverName     csiDriverName
	volumeID       string
	specVolumeID   string
	readOnly       bool
	spec           *volume.Spec
	pod            *api.Pod
	podUID         types.UID
	options        volume.VolumeOptions
	publishContext map[string]string
	volume.MetricsNil
}

// volume.Volume methods
var _ volume.Volume = &csiMountMgr{}

func (c *csiMountMgr) GetPath() string {
	dir := path.Join(getTargetPath(c.podUID, c.specVolumeID, c.plugin.host), "/mount")
	klog.V(4).Info(log("mounter.GetPath generated [%s]", dir))
	return dir
}

func getTargetPath(uid types.UID, specVolumeID string, host volume.VolumeHost) string {
	specVolID := kstrings.EscapeQualifiedNameForDisk(specVolumeID)
	return host.GetPodVolumeDir(uid, kstrings.EscapeQualifiedNameForDisk(csiPluginName), specVolID)
}

// volume.Mounter methods
var _ volume.Mounter = &csiMountMgr{}

func (c *csiMountMgr) CanMount() error {
	return nil
}

func (c *csiMountMgr) SetUp(fsGroup *int64) error {
	return c.SetUpAt(c.GetPath(), fsGroup)
}

func (c *csiMountMgr) SetUpAt(dir string, fsGroup *int64) error {
	klog.V(4).Infof(log("Mounter.SetUpAt(%s)", dir))

	mounted, err := isDirMounted(c.plugin, dir)
	if err != nil {
		klog.Error(log("mounter.SetUpAt failed while checking mount status for dir [%s]", dir))
		return err
	}

	if mounted {
		klog.V(4).Info(log("mounter.SetUpAt skipping mount, dir already mounted [%s]", dir))
		return nil
	}

	csiSource, err := getCSISourceFromSpec(c.spec)
	if err != nil {
		klog.Error(log("mounter.SetupAt failed to get CSI persistent source: %v", err))
		return err
	}

	csi, err := c.csiClientGetter.Get()
	if err != nil {
		klog.Error(log("mounter.SetUpAt failed to get CSI client: %v", err))
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	// Check for STAGE_UNSTAGE_VOLUME set and populate deviceMountPath if so
	deviceMountPath := ""
	stageUnstageSet, err := csi.NodeSupportsStageUnstage(ctx)
	if err != nil {
		klog.Error(log("mounter.SetUpAt failed to check for STAGE_UNSTAGE_VOLUME capabilty: %v", err))
		return err
	}

	if stageUnstageSet {
		deviceMountPath, err = makeDeviceMountPath(c.plugin, c.spec)
		if err != nil {
			klog.Error(log("mounter.SetUpAt failed to make device mount path: %v", err))
			return err
		}
	}
	// search for attachment by VolumeAttachment.Spec.Source.PersistentVolumeName
	if c.publishContext == nil {
		nodeName := string(c.plugin.host.GetNodeName())
		c.publishContext, err = c.plugin.getPublishContext(c.k8s, c.volumeID, string(c.driverName), nodeName)
		if err != nil {
			return err
		}
	}

	attribs := csiSource.VolumeAttributes

	nodePublishSecrets := map[string]string{}
	if csiSource.NodePublishSecretRef != nil {
		nodePublishSecrets, err = getCredentialsFromSecret(c.k8s, csiSource.NodePublishSecretRef)
		if err != nil {
			return fmt.Errorf("fetching NodePublishSecretRef %s/%s failed: %v",
				csiSource.NodePublishSecretRef.Namespace, csiSource.NodePublishSecretRef.Name, err)
		}
	}

	// create target_dir before call to NodePublish
	if err := os.MkdirAll(dir, 0750); err != nil {
		klog.Error(log("mouter.SetUpAt failed to create dir %#v:  %v", dir, err))
		return err
	}
	klog.V(4).Info(log("created target path successfully [%s]", dir))

	//TODO (vladimirvivien) implement better AccessModes mapping between k8s and CSI
	accessMode := api.ReadWriteOnce
	if c.spec.PersistentVolume.Spec.AccessModes != nil {
		accessMode = c.spec.PersistentVolume.Spec.AccessModes[0]
	}

	// Inject pod information into volume_attributes
	podAttrs, err := c.podAttributes()
	if err != nil {
		klog.Error(log("mouter.SetUpAt failed to assemble volume attributes: %v", err))
		return err
	}
	if podAttrs != nil {
		if attribs == nil {
			attribs = podAttrs
		} else {
			for k, v := range podAttrs {
				attribs[k] = v
			}
		}
	}

	fsType := csiSource.FSType
	err = csi.NodePublishVolume(
		ctx,
		c.volumeID,
		c.readOnly,
		deviceMountPath,
		dir,
		accessMode,
		c.publishContext,
		attribs,
		nodePublishSecrets,
		fsType,
		c.spec.PersistentVolume.Spec.MountOptions,
	)

	if err != nil {
		klog.Errorf(log("mounter.SetupAt failed: %v", err))
		if removeMountDirErr := removeMountDir(c.plugin, dir); removeMountDirErr != nil {
			klog.Error(log("mounter.SetupAt failed to remove mount dir after a NodePublish() error [%s]: %v", dir, removeMountDirErr))
		}
		return err
	}

	// apply volume ownership
	// The following logic is derived from https://github.com/kubernetes/kubernetes/issues/66323
	// if fstype is "", then skip fsgroup (could be indication of non-block filesystem)
	// if fstype is provided and pv.AccessMode == ReadWriteOnly, then apply fsgroup

	err = c.applyFSGroup(fsType, fsGroup)
	if err != nil {
		// attempt to rollback mount.
		fsGrpErr := fmt.Errorf("applyFSGroup failed for vol %s: %v", c.volumeID, err)
		if unpubErr := csi.NodeUnpublishVolume(ctx, c.volumeID, dir); unpubErr != nil {
			klog.Error(log("NodeUnpublishVolume failed for [%s]: %v", c.volumeID, unpubErr))
			return fsGrpErr
		}

		if unmountErr := removeMountDir(c.plugin, dir); unmountErr != nil {
			klog.Error(log("removeMountDir failed for [%s]: %v", dir, unmountErr))
			return fsGrpErr
		}
		return fsGrpErr
	}

	klog.V(4).Infof(log("mounter.SetUp successfully requested NodePublish [%s]", dir))
	return nil
}

func (c *csiMountMgr) podAttributes() (map[string]string, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIDriverRegistry) {
		return nil, nil
	}
	if c.plugin.csiDriverLister == nil {
		return nil, errors.New("CSIDriver lister does not exist")
	}

	csiDriver, err := c.plugin.csiDriverLister.Get(string(c.driverName))
	if err != nil {
		if apierrs.IsNotFound(err) {
			klog.V(4).Infof(log("CSIDriver %q not found, not adding pod information", c.driverName))
			return nil, nil
		}
		return nil, err
	}

	// if PodInfoOnMountVersion is not set or not v1 we do not set pod attributes
	if csiDriver.Spec.PodInfoOnMountVersion == nil || *csiDriver.Spec.PodInfoOnMountVersion != currentPodInfoMountVersion {
		klog.V(4).Infof(log("CSIDriver %q does not require pod information", c.driverName))
		return nil, nil
	}

	attrs := map[string]string{
		"csi.storage.k8s.io/pod.name":            c.pod.Name,
		"csi.storage.k8s.io/pod.namespace":       c.pod.Namespace,
		"csi.storage.k8s.io/pod.uid":             string(c.pod.UID),
		"csi.storage.k8s.io/serviceAccount.name": c.pod.Spec.ServiceAccountName,
	}
	klog.V(4).Infof(log("CSIDriver %q requires pod information", c.driverName))
	return attrs, nil
}

func (c *csiMountMgr) GetAttributes() volume.Attributes {
	mounter := c.plugin.host.GetMounter(c.plugin.GetPluginName())
	path := c.GetPath()
	supportSelinux, err := mounter.GetSELinuxSupport(path)
	if err != nil {
		klog.V(2).Info(log("error checking for SELinux support: %s", err))
		// Best guess
		supportSelinux = false
	}
	return volume.Attributes{
		ReadOnly:        c.readOnly,
		Managed:         !c.readOnly,
		SupportsSELinux: supportSelinux,
	}
}

// volume.Unmounter methods
var _ volume.Unmounter = &csiMountMgr{}

func (c *csiMountMgr) TearDown() error {
	return c.TearDownAt(c.GetPath())
}
func (c *csiMountMgr) TearDownAt(dir string) error {
	klog.V(4).Infof(log("Unmounter.TearDown(%s)", dir))

	// is dir even mounted ?
	// TODO (vladimirvivien) this check may not work for an emptyDir or local storage
	// see https://github.com/kubernetes/kubernetes/pull/56836#discussion_r155834524
	mounted, err := isDirMounted(c.plugin, dir)
	if err != nil {
		klog.Error(log("unmounter.Teardown failed while checking mount status for dir [%s]: %v", dir, err))
		return err
	}

	if !mounted {
		klog.V(4).Info(log("unmounter.Teardown skipping unmount, dir not mounted [%s]", dir))
		return nil
	}

	volID := c.volumeID
	csi, err := c.csiClientGetter.Get()
	if err != nil {
		klog.Error(log("mounter.SetUpAt failed to get CSI client: %v", err))
		return err
	}

	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()

	if err := csi.NodeUnpublishVolume(ctx, volID, dir); err != nil {
		klog.Errorf(log("mounter.TearDownAt failed: %v", err))
		return err
	}

	// clean mount point dir
	if err := removeMountDir(c.plugin, dir); err != nil {
		klog.Error(log("mounter.TearDownAt failed to clean mount dir [%s]: %v", dir, err))
		return err
	}
	klog.V(4).Infof(log("mounte.TearDownAt successfully unmounted dir [%s]", dir))

	return nil
}

// applyFSGroup applies the volume ownership it derives its logic
// from https://github.com/kubernetes/kubernetes/issues/66323
// 1) if fstype is "", then skip fsgroup (could be indication of non-block filesystem)
// 2) if fstype is provided and pv.AccessMode == ReadWriteOnly and !c.spec.ReadOnly then apply fsgroup
func (c *csiMountMgr) applyFSGroup(fsType string, fsGroup *int64) error {
	if fsGroup != nil {
		if fsType == "" {
			klog.V(4).Info(log("mounter.SetupAt WARNING: skipping fsGroup, fsType not provided"))
			return nil
		}

		accessModes := c.spec.PersistentVolume.Spec.AccessModes
		if c.spec.PersistentVolume.Spec.AccessModes == nil {
			klog.V(4).Info(log("mounter.SetupAt WARNING: skipping fsGroup, access modes not provided"))
			return nil
		}
		if !hasReadWriteOnce(accessModes) {
			klog.V(4).Info(log("mounter.SetupAt WARNING: skipping fsGroup, only support ReadWriteOnce access mode"))
			return nil
		}

		if c.readOnly {
			klog.V(4).Info(log("mounter.SetupAt WARNING: skipping fsGroup, volume is readOnly"))
			return nil
		}

		err := volume.SetVolumeOwnership(c, fsGroup)
		if err != nil {
			return err
		}

		klog.V(4).Info(log("mounter.SetupAt fsGroup [%d] applied successfully to %s", *fsGroup, c.volumeID))
	}

	return nil
}

// isDirMounted returns the !notMounted result from IsLikelyNotMountPoint check
func isDirMounted(plug *csiPlugin, dir string) (bool, error) {
	mounter := plug.host.GetMounter(plug.GetPluginName())
	notMnt, err := mounter.IsLikelyNotMountPoint(dir)
	if err != nil && !os.IsNotExist(err) {
		klog.Error(log("isDirMounted IsLikelyNotMountPoint test failed for dir [%v]", dir))
		return false, err
	}
	return !notMnt, nil
}

// removeMountDir cleans the mount dir when dir is not mounted and removed the volume data file in dir
func removeMountDir(plug *csiPlugin, mountPath string) error {
	klog.V(4).Info(log("removing mount path [%s]", mountPath))
	if pathExists, pathErr := util.PathExists(mountPath); pathErr != nil {
		klog.Error(log("failed while checking mount path stat [%s]", pathErr))
		return pathErr
	} else if !pathExists {
		klog.Warning(log("skipping mount dir removal, path does not exist [%v]", mountPath))
		return nil
	}

	mounter := plug.host.GetMounter(plug.GetPluginName())
	notMnt, err := mounter.IsLikelyNotMountPoint(mountPath)
	if err != nil {
		klog.Error(log("mount dir removal failed [%s]: %v", mountPath, err))
		return err
	}
	if notMnt {
		klog.V(4).Info(log("dir not mounted, deleting it [%s]", mountPath))
		if err := os.Remove(mountPath); err != nil && !os.IsNotExist(err) {
			klog.Error(log("failed to remove dir [%s]: %v", mountPath, err))
			return err
		}
		// remove volume data file as well
		volPath := path.Dir(mountPath)
		dataFile := path.Join(volPath, volDataFileName)
		klog.V(4).Info(log("also deleting volume info data file [%s]", dataFile))
		if err := os.Remove(dataFile); err != nil && !os.IsNotExist(err) {
			klog.Error(log("failed to delete volume data file [%s]: %v", dataFile, err))
			return err
		}
		// remove volume path
		klog.V(4).Info(log("deleting volume path [%s]", volPath))
		if err := os.Remove(volPath); err != nil && !os.IsNotExist(err) {
			klog.Error(log("failed to delete volume path [%s]: %v", volPath, err))
			return err
		}
	}
	return nil
}
