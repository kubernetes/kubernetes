/*
Copyright 2015 The Kubernetes Authors.

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

package iscsi

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	ioutil "k8s.io/kubernetes/pkg/volume/util"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&iscsiPlugin{nil}}
}

type iscsiPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &iscsiPlugin{}
var _ volume.PersistentVolumePlugin = &iscsiPlugin{}

const (
	iscsiPluginName = "kubernetes.io/iscsi"
)

func (plugin *iscsiPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *iscsiPlugin) GetPluginName() string {
	return iscsiPluginName
}

func (plugin *iscsiPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	tp, _, iqn, lun, err := getISCSITargetInfo(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%v:%v:%v", tp, iqn, lun), nil
}

func (plugin *iscsiPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.Volume != nil && spec.Volume.ISCSI == nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.ISCSI == nil) {
		return false
	}

	return true
}

func (plugin *iscsiPlugin) RequiresRemount() bool {
	return false
}

func (plugin *iscsiPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *iscsiPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *iscsiPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *iscsiPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	// Inject real implementations here, test through the internal function.
	var secret map[string]string
	if pod == nil {
		return nil, fmt.Errorf("nil pod")
	}
	chapDiscover, err := getISCSIDiscoveryCHAPInfo(spec)
	if err != nil {
		return nil, err
	}
	chapSession, err := getISCSISessionCHAPInfo(spec)
	if err != nil {
		return nil, err
	}
	if chapDiscover || chapSession {
		secretName, secretNamespace, err := getISCSISecretNameAndNamespace(spec, pod.Namespace)
		if err != nil {
			return nil, err
		}

		if len(secretName) > 0 && len(secretNamespace) > 0 {
			// if secret is provideded, retrieve it
			kubeClient := plugin.host.GetKubeClient()
			if kubeClient == nil {
				return nil, fmt.Errorf("Cannot get kube client")
			}
			secretObj, err := kubeClient.Core().Secrets(secretNamespace).Get(secretName, metav1.GetOptions{})
			if err != nil {
				err = fmt.Errorf("Couldn't get secret %v/%v error: %v", secretNamespace, secretName, err)
				return nil, err
			}
			secret = make(map[string]string)
			for name, data := range secretObj.Data {
				glog.V(4).Infof("retrieving CHAP secret name: %s", name)
				secret[name] = string(data)
			}
		}
	}
	return plugin.newMounterInternal(spec, pod.UID, &ISCSIUtil{}, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()), secret)
}

func (plugin *iscsiPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface, exec mount.Exec, secret map[string]string) (volume.Mounter, error) {
	// iscsi volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// iscsi volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	readOnly, fsType, err := getISCSIVolumeInfo(spec)
	if err != nil {
		return nil, err
	}
	tp, portals, iqn, lunStr, err := getISCSITargetInfo(spec)
	if err != nil {
		return nil, err
	}

	lun := strconv.Itoa(int(lunStr))
	portal := portalMounter(tp)
	var bkportal []string
	bkportal = append(bkportal, portal)
	for _, p := range portals {
		bkportal = append(bkportal, portalMounter(string(p)))
	}

	iface, initiatorNamePtr, err := getISCSIInitiatorInfo(spec)
	if err != nil {
		return nil, err
	}

	var initiatorName string
	if initiatorNamePtr != nil {
		initiatorName = *initiatorNamePtr
	}
	chapDiscovery, err := getISCSIDiscoveryCHAPInfo(spec)
	if err != nil {
		return nil, err
	}
	chapSession, err := getISCSISessionCHAPInfo(spec)
	if err != nil {
		return nil, err
	}

	return &iscsiDiskMounter{
		iscsiDisk: &iscsiDisk{
			podUID:         podUID,
			VolName:        spec.Name(),
			Portals:        bkportal,
			Iqn:            iqn,
			lun:            lun,
			Iface:          iface,
			chap_discovery: chapDiscovery,
			chap_session:   chapSession,
			secret:         secret,
			InitiatorName:  initiatorName,
			manager:        manager,
			plugin:         plugin},
		fsType:       fsType,
		readOnly:     readOnly,
		mounter:      &mount.SafeFormatAndMount{Interface: mounter, Exec: exec},
		exec:         exec,
		deviceUtil:   ioutil.NewDeviceHandler(ioutil.NewIOHandler()),
		mountOptions: volume.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *iscsiPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &ISCSIUtil{}, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()))
}

func (plugin *iscsiPlugin) newUnmounterInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface, exec mount.Exec) (volume.Unmounter, error) {
	return &iscsiDiskUnmounter{
		iscsiDisk: &iscsiDisk{
			podUID:  podUID,
			VolName: volName,
			manager: manager,
			plugin:  plugin,
		},
		mounter: mounter,
		exec:    exec,
	}, nil
}

func (plugin *iscsiPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	iscsiVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			ISCSI: &v1.ISCSIVolumeSource{
				TargetPortal: volumeName,
				IQN:          volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(iscsiVolume), nil
}

type iscsiDisk struct {
	VolName        string
	podUID         types.UID
	Portals        []string
	Iqn            string
	lun            string
	Iface          string
	chap_discovery bool
	chap_session   bool
	secret         map[string]string
	InitiatorName  string
	plugin         *iscsiPlugin
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
	volume.MetricsNil
}

func (iscsi *iscsiDisk) GetPath() string {
	name := iscsiPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return iscsi.plugin.host.GetPodVolumeDir(iscsi.podUID, utilstrings.EscapeQualifiedNameForDisk(name), iscsi.VolName)
}

type iscsiDiskMounter struct {
	*iscsiDisk
	readOnly     bool
	fsType       string
	mounter      *mount.SafeFormatAndMount
	exec         mount.Exec
	deviceUtil   ioutil.DeviceUtil
	mountOptions []string
}

var _ volume.Mounter = &iscsiDiskMounter{}

func (b *iscsiDiskMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         !b.readOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *iscsiDiskMounter) CanMount() error {
	return nil
}

func (b *iscsiDiskMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *iscsiDiskMounter) SetUpAt(dir string, fsGroup *int64) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(b.manager, *b, dir, b.mounter, fsGroup)
	if err != nil {
		glog.Errorf("iscsi: failed to setup")
	}
	return err
}

type iscsiDiskUnmounter struct {
	*iscsiDisk
	mounter mount.Interface
	exec    mount.Exec
}

var _ volume.Unmounter = &iscsiDiskUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (c *iscsiDiskUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *iscsiDiskUnmounter) TearDownAt(dir string) error {
	return ioutil.UnmountPath(dir, c.mounter)
}

func portalMounter(portal string) string {
	if !strings.Contains(portal, ":") {
		portal = portal + ":3260"
	}
	return portal
}

// get iSCSI volume info: readOnly and fstype
func getISCSIVolumeInfo(spec *volume.Spec) (bool, string, error) {
	// for volume source, readonly is in volume spec
	// for PV, readonly is in PV spec
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		return spec.Volume.ISCSI.ReadOnly, spec.Volume.ISCSI.FSType, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ISCSI != nil {
		return spec.ReadOnly, spec.PersistentVolume.Spec.ISCSI.FSType, nil
	}

	return false, "", fmt.Errorf("Spec does not reference an ISCSI volume type")
}

// get iSCSI target info: target portal, portals, iqn, and lun
func getISCSITargetInfo(spec *volume.Spec) (string, []string, string, int32, error) {
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		return spec.Volume.ISCSI.TargetPortal, spec.Volume.ISCSI.Portals, spec.Volume.ISCSI.IQN, spec.Volume.ISCSI.Lun, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ISCSI != nil {
		return spec.PersistentVolume.Spec.ISCSI.TargetPortal, spec.PersistentVolume.Spec.ISCSI.Portals, spec.PersistentVolume.Spec.ISCSI.IQN, spec.PersistentVolume.Spec.ISCSI.Lun, nil
	}

	return "", nil, "", 0, fmt.Errorf("Spec does not reference an ISCSI volume type")
}

// get iSCSI initiator info: iface and initiator name
func getISCSIInitiatorInfo(spec *volume.Spec) (string, *string, error) {
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		return spec.Volume.ISCSI.ISCSIInterface, spec.Volume.ISCSI.InitiatorName, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ISCSI != nil {
		return spec.PersistentVolume.Spec.ISCSI.ISCSIInterface, spec.PersistentVolume.Spec.ISCSI.InitiatorName, nil
	}

	return "", nil, fmt.Errorf("Spec does not reference an ISCSI volume type")
}

// get iSCSI Discovery CHAP boolean
func getISCSIDiscoveryCHAPInfo(spec *volume.Spec) (bool, error) {
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		return spec.Volume.ISCSI.DiscoveryCHAPAuth, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ISCSI != nil {
		return spec.PersistentVolume.Spec.ISCSI.DiscoveryCHAPAuth, nil
	}

	return false, fmt.Errorf("Spec does not reference an ISCSI volume type")
}

// get iSCSI Session CHAP boolean
func getISCSISessionCHAPInfo(spec *volume.Spec) (bool, error) {
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		return spec.Volume.ISCSI.SessionCHAPAuth, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ISCSI != nil {
		return spec.PersistentVolume.Spec.ISCSI.SessionCHAPAuth, nil
	}

	return false, fmt.Errorf("Spec does not reference an ISCSI volume type")
}

// get iSCSI CHAP Secret info: secret name and namespace
func getISCSISecretNameAndNamespace(spec *volume.Spec, defaultSecretNamespace string) (string, string, error) {
	if spec.Volume != nil && spec.Volume.ISCSI != nil {
		if spec.Volume.ISCSI.SecretRef != nil {
			return spec.Volume.ISCSI.SecretRef.Name, defaultSecretNamespace, nil
		}
		return "", "", nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ISCSI != nil {
		secretRef := spec.PersistentVolume.Spec.ISCSI.SecretRef
		secretNs := defaultSecretNamespace
		if secretRef != nil {
			if len(secretRef.Namespace) != 0 {
				secretNs = secretRef.Namespace
			}
			return secretRef.Name, secretNs, nil
		}
		return "", "", nil
	}

	return "", "", fmt.Errorf("Spec does not reference an ISCSI volume type")
}
