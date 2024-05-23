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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/mount-utils"
	utilexec "k8s.io/utils/exec"
	"k8s.io/utils/io"
	"k8s.io/utils/keymutex"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/volume"
	ioutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&iscsiPlugin{}}
}

type iscsiPlugin struct {
	host        volume.VolumeHost
	targetLocks keymutex.KeyMutex
}

var _ volume.VolumePlugin = &iscsiPlugin{}
var _ volume.PersistentVolumePlugin = &iscsiPlugin{}
var _ volume.BlockVolumePlugin = &iscsiPlugin{}

const (
	iscsiPluginName = "kubernetes.io/iscsi"
)

func (plugin *iscsiPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.targetLocks = keymutex.NewHashed(0)
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
	return (spec.Volume != nil && spec.Volume.ISCSI != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.ISCSI != nil)
}

func (plugin *iscsiPlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *iscsiPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *iscsiPlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *iscsiPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *iscsiPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod) (volume.Mounter, error) {
	if pod == nil {
		return nil, fmt.Errorf("nil pod")
	}
	secret, err := createSecretMap(spec, plugin, pod.Namespace)
	if err != nil {
		return nil, err
	}
	return plugin.newMounterInternal(spec, pod.UID, &ISCSIUtil{}, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()), secret)
}

func (plugin *iscsiPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface, exec utilexec.Interface, secret map[string]string) (volume.Mounter, error) {
	readOnly, fsType, err := getISCSIVolumeInfo(spec)
	if err != nil {
		return nil, err
	}
	iscsiDisk, err := createISCSIDisk(spec, podUID, plugin, manager, secret)
	if err != nil {
		return nil, err
	}

	if iscsiDisk != nil {

		//Add volume metrics
		iscsiDisk.MetricsProvider = volume.NewMetricsStatFS(iscsiDisk.GetPath())
	}
	return &iscsiDiskMounter{
		iscsiDisk:    iscsiDisk,
		fsType:       fsType,
		readOnly:     readOnly,
		mounter:      &mount.SafeFormatAndMount{Interface: mounter, Exec: exec},
		exec:         exec,
		deviceUtil:   ioutil.NewDeviceHandler(ioutil.NewIOHandler()),
		mountOptions: ioutil.MountOptionFromSpec(spec),
	}, nil
}

// NewBlockVolumeMapper creates a new volume.BlockVolumeMapper from an API specification.
func (plugin *iscsiPlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod) (volume.BlockVolumeMapper, error) {
	// If this is called via GenerateUnmapDeviceFunc(), pod is nil.
	// Pass empty string as dummy uid since uid isn't used in the case.
	var uid types.UID
	var secret map[string]string
	var err error
	if pod != nil {
		uid = pod.UID
		secret, err = createSecretMap(spec, plugin, pod.Namespace)
		if err != nil {
			return nil, err
		}
	}
	return plugin.newBlockVolumeMapperInternal(spec, uid, &ISCSIUtil{}, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()), secret)
}

func (plugin *iscsiPlugin) newBlockVolumeMapperInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface, exec utilexec.Interface, secret map[string]string) (volume.BlockVolumeMapper, error) {
	readOnly, _, err := getISCSIVolumeInfo(spec)
	if err != nil {
		return nil, err
	}
	iscsiDisk, err := createISCSIDisk(spec, podUID, plugin, manager, secret)
	if err != nil {
		return nil, err
	}
	mapper := &iscsiDiskMapper{
		iscsiDisk:  iscsiDisk,
		readOnly:   readOnly,
		exec:       exec,
		deviceUtil: ioutil.NewDeviceHandler(ioutil.NewIOHandler()),
	}

	blockPath, err := mapper.GetGlobalMapPath(spec)
	if err != nil {
		return nil, fmt.Errorf("failed to get device path: %v", err)
	}
	mapper.MetricsProvider = volume.NewMetricsBlock(filepath.Join(blockPath, string(podUID)))

	return mapper, nil
}

func (plugin *iscsiPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &ISCSIUtil{}, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()))
}

func (plugin *iscsiPlugin) newUnmounterInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface, exec utilexec.Interface) (volume.Unmounter, error) {
	return &iscsiDiskUnmounter{
		iscsiDisk: &iscsiDisk{
			podUID:          podUID,
			VolName:         volName,
			manager:         manager,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(plugin.host.GetPodVolumeDir(podUID, utilstrings.EscapeQualifiedName(iscsiPluginName), volName)),
		},
		mounter:    mounter,
		exec:       exec,
		deviceUtil: ioutil.NewDeviceHandler(ioutil.NewIOHandler()),
	}, nil
}

// NewBlockVolumeUnmapper creates a new volume.BlockVolumeUnmapper from recoverable state.
func (plugin *iscsiPlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return plugin.newUnmapperInternal(volName, podUID, &ISCSIUtil{}, plugin.host.GetExec(plugin.GetPluginName()))
}

func (plugin *iscsiPlugin) newUnmapperInternal(volName string, podUID types.UID, manager diskManager, exec utilexec.Interface) (volume.BlockVolumeUnmapper, error) {
	return &iscsiDiskUnmapper{
		iscsiDisk: &iscsiDisk{
			podUID:  podUID,
			VolName: volName,
			manager: manager,
			plugin:  plugin,
		},
		exec:       exec,
		deviceUtil: ioutil.NewDeviceHandler(ioutil.NewIOHandler()),
	}, nil
}

func (plugin *iscsiPlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	// Find globalPDPath from pod volume directory(mountPath)
	var globalPDPath string
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	// Try really hard to get the global mount of the volume, an error returned from here would
	// leave the global mount still mounted, while marking the volume as unused.
	// The volume can then be mounted on several nodes, resulting in volume
	// corruption.
	paths, err := ioutil.GetReliableMountRefs(mounter, mountPath)
	if io.IsInconsistentReadError(err) {
		klog.Errorf("Failed to read mount refs from /proc/mounts for %s: %s", mountPath, err)
		klog.Errorf("Kubelet cannot unmount volume at %s, please unmount it and all mounts of the same device manually.", mountPath)
		return volume.ReconstructedVolume{}, err
	}
	if err != nil {
		return volume.ReconstructedVolume{}, err
	}

	for _, path := range paths {
		if strings.Contains(path, plugin.host.GetPluginDir(iscsiPluginName)) {
			globalPDPath = path
			break
		}
	}
	// Couldn't fetch globalPDPath
	if len(globalPDPath) == 0 {
		return volume.ReconstructedVolume{}, fmt.Errorf("couldn't fetch globalPDPath. failed to obtain volume spec")
	}

	// Obtain iscsi disk configurations from globalPDPath
	device, _, err := extractDeviceAndPrefix(globalPDPath)
	if err != nil {
		return volume.ReconstructedVolume{}, err
	}
	bkpPortal, iqn, err := extractPortalAndIqn(device)
	if err != nil {
		return volume.ReconstructedVolume{}, err
	}
	arr := strings.Split(device, "-lun-")
	if len(arr) < 2 {
		return volume.ReconstructedVolume{}, fmt.Errorf("failed to retrieve lun from globalPDPath: %v", globalPDPath)
	}
	lun, err := strconv.Atoi(arr[1])
	if err != nil {
		return volume.ReconstructedVolume{}, err
	}
	iface, _ := extractIface(globalPDPath)
	iscsiVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			ISCSI: &v1.ISCSIVolumeSource{
				TargetPortal:   bkpPortal,
				IQN:            iqn,
				Lun:            int32(lun),
				ISCSIInterface: iface,
			},
		},
	}

	var mountContext string
	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
		kvh, ok := plugin.host.(volume.KubeletVolumeHost)
		if !ok {
			return volume.ReconstructedVolume{}, fmt.Errorf("plugin volume host does not implement KubeletVolumeHost interface")
		}
		hu := kvh.GetHostUtil()
		mountContext, err = hu.GetSELinuxMountContext(mountPath)
		if err != nil {
			return volume.ReconstructedVolume{}, err
		}
	}

	return volume.ReconstructedVolume{
		Spec:                volume.NewSpecFromVolume(iscsiVolume),
		SELinuxMountContext: mountContext,
	}, nil
}

func (plugin *iscsiPlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mapPath string) (*volume.Spec, error) {
	pluginDir := plugin.host.GetVolumeDevicePluginDir(iscsiPluginName)
	blkutil := volumepathhandler.NewBlockVolumePathHandler()
	globalMapPathUUID, err := blkutil.FindGlobalMapPathUUIDFromPod(pluginDir, mapPath, podUID)
	if err != nil {
		return nil, err
	}
	klog.V(5).Infof("globalMapPathUUID: %v, err: %v", globalMapPathUUID, err)
	// Retrieve volume information from globalMapPathUUID
	// globalMapPathUUID example:
	// plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}/{pod uuid}
	// plugins/kubernetes.io/iscsi/volumeDevices/iface-default/192.168.0.10:3260-iqn.2017-05.com.example:test-lun-0/{pod uuid}
	globalMapPath := filepath.Dir(globalMapPathUUID)
	return getVolumeSpecFromGlobalMapPath(volumeName, globalMapPath)
}

type iscsiDisk struct {
	VolName       string
	podUID        types.UID
	Portals       []string
	Iqn           string
	Lun           string
	InitIface     string
	Iface         string
	chapDiscovery bool
	chapSession   bool
	secret        map[string]string `datapolicy:"token"`
	InitiatorName string
	plugin        *iscsiPlugin
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
	volume.MetricsProvider
}

func (iscsi *iscsiDisk) GetPath() string {
	name := iscsiPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return iscsi.plugin.host.GetPodVolumeDir(iscsi.podUID, utilstrings.EscapeQualifiedName(name), iscsi.VolName)
}

func (iscsi *iscsiDisk) iscsiGlobalMapPath(spec *volume.Spec) (string, error) {
	mounter, err := volumeSpecToMounter(spec, iscsi.plugin.host, iscsi.plugin.targetLocks, nil /* pod */)
	if err != nil {
		klog.Warningf("failed to get iscsi mounter: %v", err)
		return "", err
	}
	return iscsi.manager.MakeGlobalVDPDName(*mounter.iscsiDisk), nil
}

func (iscsi *iscsiDisk) iscsiPodDeviceMapPath() (string, string) {
	name := iscsiPluginName
	return iscsi.plugin.host.GetPodVolumeDeviceDir(iscsi.podUID, utilstrings.EscapeQualifiedName(name)), iscsi.VolName
}

type iscsiDiskMounter struct {
	*iscsiDisk
	readOnly                  bool
	fsType                    string
	volumeMode                v1.PersistentVolumeMode
	mounter                   *mount.SafeFormatAndMount
	exec                      utilexec.Interface
	deviceUtil                ioutil.DeviceUtil
	mountOptions              []string
	mountedWithSELinuxContext bool
}

var _ volume.Mounter = &iscsiDiskMounter{}

func (b *iscsiDiskMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:       b.readOnly,
		Managed:        !b.readOnly,
		SELinuxRelabel: !b.mountedWithSELinuxContext,
	}
}

func (b *iscsiDiskMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

func (b *iscsiDiskMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(b.manager, *b, dir, b.mounter, mounterArgs.FsGroup, mounterArgs.FSGroupChangePolicy)
	if err != nil {
		klog.Errorf("iscsi: failed to setup")
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.SELinuxMountReadWriteOncePod) {
		// The volume must have been mounted in MountDevice with -o context.
		// TODO: extract from mount table in GetAttributes() to be sure?
		b.mountedWithSELinuxContext = mounterArgs.SELinuxLabel != ""
	}
	return err
}

type iscsiDiskUnmounter struct {
	*iscsiDisk
	mounter    mount.Interface
	exec       utilexec.Interface
	deviceUtil ioutil.DeviceUtil
}

var _ volume.Unmounter = &iscsiDiskUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (c *iscsiDiskUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *iscsiDiskUnmounter) TearDownAt(dir string) error {
	return mount.CleanupMountPoint(dir, c.mounter, false)
}

// Block Volumes Support
type iscsiDiskMapper struct {
	*iscsiDisk
	readOnly   bool
	exec       utilexec.Interface
	deviceUtil ioutil.DeviceUtil
}

var _ volume.BlockVolumeMapper = &iscsiDiskMapper{}

type iscsiDiskUnmapper struct {
	*iscsiDisk
	exec       utilexec.Interface
	deviceUtil ioutil.DeviceUtil
	volume.MetricsNil
}

// SupportsMetrics returns true for SupportsMetrics as it initializes the
// MetricsProvider.
func (idm *iscsiDiskMapper) SupportsMetrics() bool {
	return true
}

var _ volume.BlockVolumeUnmapper = &iscsiDiskUnmapper{}
var _ volume.CustomBlockVolumeUnmapper = &iscsiDiskUnmapper{}

// Even though iSCSI plugin has attacher/detacher implementation, iSCSI plugin
// needs volume detach operation during TearDownDevice(). This method is only
// chance that operations are done on kubelet node during volume teardown sequences.
func (c *iscsiDiskUnmapper) TearDownDevice(mapPath, _ string) error {
	err := c.manager.DetachBlockISCSIDisk(*c, mapPath)
	if err != nil {
		return fmt.Errorf("iscsi: failed to detach disk: %s\nError: %v", mapPath, err)
	}
	klog.V(4).Infof("iscsi: %q is unmounted, deleting the directory", mapPath)
	err = os.RemoveAll(mapPath)
	if err != nil {
		return fmt.Errorf("iscsi: failed to delete the directory: %s\nError: %v", mapPath, err)
	}
	klog.V(4).Infof("iscsi: successfully detached disk: %s", mapPath)
	return nil
}

func (c *iscsiDiskUnmapper) UnmapPodDevice() error {
	return nil
}

// GetGlobalMapPath returns global map path and error
// path: plugins/kubernetes.io/{PluginName}/volumeDevices/{ifaceName}/{portal-some_iqn-lun-lun_id}
func (iscsi *iscsiDisk) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	return iscsi.iscsiGlobalMapPath(spec)
}

// GetPodDeviceMapPath returns pod device map path and volume name
// path: pods/{podUid}/volumeDevices/kubernetes.io~iscsi
// volumeName: pv0001
func (iscsi *iscsiDisk) GetPodDeviceMapPath() (string, string) {
	return iscsi.iscsiPodDeviceMapPath()
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
	// for PV, readonly is in PV spec. PV gets the ReadOnly flag indirectly through the PVC source
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

func createISCSIDisk(spec *volume.Spec, podUID types.UID, plugin *iscsiPlugin, manager diskManager, secret map[string]string) (*iscsiDisk, error) {
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

	initIface := iface
	if initiatorName != "" {
		iface = bkportal[0] + ":" + spec.Name()
	}

	return &iscsiDisk{
		podUID:        podUID,
		VolName:       spec.Name(),
		Portals:       bkportal,
		Iqn:           iqn,
		Lun:           lun,
		InitIface:     initIface,
		Iface:         iface,
		chapDiscovery: chapDiscovery,
		chapSession:   chapSession,
		secret:        secret,
		InitiatorName: initiatorName,
		manager:       manager,
		plugin:        plugin}, nil
}

func createSecretMap(spec *volume.Spec, plugin *iscsiPlugin, namespace string) (map[string]string, error) {
	var secret map[string]string
	chapDiscover, err := getISCSIDiscoveryCHAPInfo(spec)
	if err != nil {
		return nil, err
	}
	chapSession, err := getISCSISessionCHAPInfo(spec)
	if err != nil {
		return nil, err
	}
	if chapDiscover || chapSession {
		secretName, secretNamespace, err := getISCSISecretNameAndNamespace(spec, namespace)
		if err != nil {
			return nil, err
		}

		if len(secretName) > 0 && len(secretNamespace) > 0 {
			// if secret is provideded, retrieve it
			kubeClient := plugin.host.GetKubeClient()
			if kubeClient == nil {
				return nil, fmt.Errorf("cannot get kube client")
			}
			secretObj, err := kubeClient.CoreV1().Secrets(secretNamespace).Get(context.TODO(), secretName, metav1.GetOptions{})
			if err != nil {
				err = fmt.Errorf("couldn't get secret %v/%v error: %w", secretNamespace, secretName, err)
				return nil, err
			}
			secret = make(map[string]string)
			for name, data := range secretObj.Data {
				klog.V(4).Infof("retrieving CHAP secret name: %s", name)
				secret[name] = string(data)
			}
		}
	}
	return secret, err
}

func createPersistentVolumeFromISCSIPVSource(volumeName string, iscsi v1.ISCSIPersistentVolumeSource) *v1.PersistentVolume {
	block := v1.PersistentVolumeBlock
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				ISCSI: &iscsi,
			},
			VolumeMode: &block,
		},
	}
}

func getVolumeSpecFromGlobalMapPath(volumeName, globalMapPath string) (*volume.Spec, error) {
	// Retrieve volume spec information from globalMapPath
	// globalMapPath example:
	// plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}
	// plugins/kubernetes.io/iscsi/volumeDevices/iface-default/192.168.0.10:3260-iqn.2017-05.com.example:test-lun-0

	// device: 192.168.0.10:3260-iqn.2017-05.com.example:test-lun-0
	device, _, err := extractDeviceAndPrefix(globalMapPath)
	if err != nil {
		return nil, err
	}
	bkpPortal, iqn, err := extractPortalAndIqn(device)
	if err != nil {
		return nil, err
	}
	arr := strings.Split(device, "-lun-")
	if len(arr) < 2 {
		return nil, fmt.Errorf("failed to retrieve lun from globalMapPath: %v", globalMapPath)
	}
	lun, err := strconv.Atoi(arr[1])
	if err != nil {
		return nil, err
	}
	iface, found := extractIface(globalMapPath)
	if !found {
		return nil, fmt.Errorf("failed to retrieve iface from globalMapPath: %v", globalMapPath)
	}
	iscsiPV := createPersistentVolumeFromISCSIPVSource(volumeName,
		v1.ISCSIPersistentVolumeSource{
			TargetPortal:   bkpPortal,
			IQN:            iqn,
			Lun:            int32(lun),
			ISCSIInterface: iface,
		},
	)
	klog.V(5).Infof("ConstructBlockVolumeSpec: TargetPortal: %v, IQN: %v, Lun: %v, ISCSIInterface: %v",
		iscsiPV.Spec.PersistentVolumeSource.ISCSI.TargetPortal,
		iscsiPV.Spec.PersistentVolumeSource.ISCSI.IQN,
		iscsiPV.Spec.PersistentVolumeSource.ISCSI.Lun,
		iscsiPV.Spec.PersistentVolumeSource.ISCSI.ISCSIInterface,
	)
	return volume.NewSpecFromPersistentVolume(iscsiPV, false), nil
}
