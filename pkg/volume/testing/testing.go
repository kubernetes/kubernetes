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

package testing

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	authenticationv1 "k8s.io/api/authentication/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	storagelisters "k8s.io/client-go/listers/storage/v1beta1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	utiltesting "k8s.io/client-go/util/testing"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/util/mount"
	. "k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/recyclerclient"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
	utilstrings "k8s.io/utils/strings"
)

const (
	// A hook specified in storage class to indicate it's provisioning
	// is expected to fail.
	ExpectProvisionFailureKey = "expect-provision-failure"
	// The node is marked as uncertain. The attach operation will fail and return timeout error
	// for the first attach call. The following call will return sucesssfully.
	UncertainAttachNode = "uncertain-attach-node"
	// The node is marked as timeout. The attach operation will always fail and return timeout error
	// but the operation is actually succeeded.
	TimeoutAttachNode = "timeout-attach-node"
	// The node is marked as multi-attach which means it is allowed to attach the volume to multiple nodes.
	MultiAttachNode = "multi-attach-node"
)

// fakeVolumeHost is useful for testing volume plugins.
type fakeVolumeHost struct {
	rootDir         string
	kubeClient      clientset.Interface
	pluginMgr       VolumePluginMgr
	cloud           cloudprovider.Interface
	mounter         mount.Interface
	hostUtil        mount.HostUtils
	exec            mount.Exec
	nodeLabels      map[string]string
	nodeName        string
	subpather       subpath.Interface
	csiDriverLister storagelisters.CSIDriverLister
	informerFactory informers.SharedInformerFactory
}

var _ VolumeHost = &fakeVolumeHost{}
var _ AttachDetachVolumeHost = &fakeVolumeHost{}

func NewFakeVolumeHost(rootDir string, kubeClient clientset.Interface, plugins []VolumePlugin) *fakeVolumeHost {
	return newFakeVolumeHost(rootDir, kubeClient, plugins, nil, nil)
}

func NewFakeVolumeHostWithCloudProvider(rootDir string, kubeClient clientset.Interface, plugins []VolumePlugin, cloud cloudprovider.Interface) *fakeVolumeHost {
	return newFakeVolumeHost(rootDir, kubeClient, plugins, cloud, nil)
}

func NewFakeVolumeHostWithNodeLabels(rootDir string, kubeClient clientset.Interface, plugins []VolumePlugin, labels map[string]string) *fakeVolumeHost {
	volHost := newFakeVolumeHost(rootDir, kubeClient, plugins, nil, nil)
	volHost.nodeLabels = labels
	return volHost
}

func NewFakeVolumeHostWithCSINodeName(rootDir string, kubeClient clientset.Interface, plugins []VolumePlugin, nodeName string, driverLister storagelisters.CSIDriverLister) *fakeVolumeHost {
	volHost := newFakeVolumeHost(rootDir, kubeClient, plugins, nil, nil)
	volHost.nodeName = nodeName
	if driverLister != nil {
		volHost.csiDriverLister = driverLister
	}
	return volHost
}

func newFakeVolumeHost(rootDir string, kubeClient clientset.Interface, plugins []VolumePlugin, cloud cloudprovider.Interface, pathToTypeMap map[string]mount.FileType) *fakeVolumeHost {
	host := &fakeVolumeHost{rootDir: rootDir, kubeClient: kubeClient, cloud: cloud}
	host.mounter = &mount.FakeMounter{}
	host.hostUtil = &mount.FakeHostUtil{
		Filesystem: pathToTypeMap,
	}
	host.exec = mount.NewFakeExec(nil)
	host.pluginMgr.InitPlugins(plugins, nil /* prober */, host)
	host.subpather = &subpath.FakeSubpath{}
	host.informerFactory = informers.NewSharedInformerFactory(kubeClient, time.Minute)
	return host
}

func NewFakeVolumeHostWithMounterFSType(rootDir string, kubeClient clientset.Interface, plugins []VolumePlugin, pathToTypeMap map[string]mount.FileType) *fakeVolumeHost {
	volHost := newFakeVolumeHost(rootDir, kubeClient, plugins, nil, pathToTypeMap)
	return volHost
}

func (f *fakeVolumeHost) GetPluginDir(podUID string) string {
	return filepath.Join(f.rootDir, "plugins", podUID)
}

func (f *fakeVolumeHost) GetVolumeDevicePluginDir(pluginName string) string {
	return filepath.Join(f.rootDir, "plugins", pluginName, "volumeDevices")
}

func (f *fakeVolumeHost) GetPodsDir() string {
	return filepath.Join(f.rootDir, "pods")
}

func (f *fakeVolumeHost) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return filepath.Join(f.rootDir, "pods", string(podUID), "volumes", pluginName, volumeName)
}

func (f *fakeVolumeHost) GetPodVolumeDeviceDir(podUID types.UID, pluginName string) string {
	return filepath.Join(f.rootDir, "pods", string(podUID), "volumeDevices", pluginName)
}

func (f *fakeVolumeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return filepath.Join(f.rootDir, "pods", string(podUID), "plugins", pluginName)
}

func (f *fakeVolumeHost) GetKubeClient() clientset.Interface {
	return f.kubeClient
}

func (f *fakeVolumeHost) GetCloudProvider() cloudprovider.Interface {
	return f.cloud
}

func (f *fakeVolumeHost) GetMounter(pluginName string) mount.Interface {
	return f.mounter
}

func (f *fakeVolumeHost) GetHostUtil() mount.HostUtils {
	return f.hostUtil
}

func (f *fakeVolumeHost) GetSubpather() subpath.Interface {
	return f.subpather
}

func (f *fakeVolumeHost) NewWrapperMounter(volName string, spec Spec, pod *v1.Pod, opts VolumeOptions) (Mounter, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}
	plug, err := f.pluginMgr.FindPluginBySpec(&spec)
	if err != nil {
		return nil, err
	}
	return plug.NewMounter(&spec, pod, opts)
}

func (f *fakeVolumeHost) NewWrapperUnmounter(volName string, spec Spec, podUID types.UID) (Unmounter, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}
	plug, err := f.pluginMgr.FindPluginBySpec(&spec)
	if err != nil {
		return nil, err
	}
	return plug.NewUnmounter(spec.Name(), podUID)
}

// Returns the hostname of the host kubelet is running on
func (f *fakeVolumeHost) GetHostName() string {
	return "fakeHostName"
}

// Returns host IP or nil in the case of error.
func (f *fakeVolumeHost) GetHostIP() (net.IP, error) {
	return nil, fmt.Errorf("GetHostIP() not implemented")
}

func (f *fakeVolumeHost) GetNodeAllocatable() (v1.ResourceList, error) {
	return v1.ResourceList{}, nil
}

func (f *fakeVolumeHost) GetSecretFunc() func(namespace, name string) (*v1.Secret, error) {
	return func(namespace, name string) (*v1.Secret, error) {
		return f.kubeClient.CoreV1().Secrets(namespace).Get(name, metav1.GetOptions{})
	}
}

func (f *fakeVolumeHost) GetExec(pluginName string) mount.Exec {
	return f.exec
}

func (f *fakeVolumeHost) GetConfigMapFunc() func(namespace, name string) (*v1.ConfigMap, error) {
	return func(namespace, name string) (*v1.ConfigMap, error) {
		return f.kubeClient.CoreV1().ConfigMaps(namespace).Get(name, metav1.GetOptions{})
	}
}

func (f *fakeVolumeHost) GetServiceAccountTokenFunc() func(string, string, *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
	return func(namespace, name string, tr *authenticationv1.TokenRequest) (*authenticationv1.TokenRequest, error) {
		return f.kubeClient.CoreV1().ServiceAccounts(namespace).CreateToken(name, tr)
	}
}

func (f *fakeVolumeHost) DeleteServiceAccountTokenFunc() func(types.UID) {
	return func(types.UID) {}
}

func (f *fakeVolumeHost) GetNodeLabels() (map[string]string, error) {
	if f.nodeLabels == nil {
		f.nodeLabels = map[string]string{"test-label": "test-value"}
	}
	return f.nodeLabels, nil
}

func (f *fakeVolumeHost) GetNodeName() types.NodeName {
	return types.NodeName(f.nodeName)
}

func (f *fakeVolumeHost) GetEventRecorder() record.EventRecorder {
	return nil
}

func ProbeVolumePlugins(config VolumeConfig) []VolumePlugin {
	if _, ok := config.OtherAttributes["fake-property"]; ok {
		return []VolumePlugin{
			&FakeVolumePlugin{
				PluginName: "fake-plugin",
				Host:       nil,
				// SomeFakeProperty: config.OtherAttributes["fake-property"] -- string, may require parsing by plugin
			},
		}
	}
	return []VolumePlugin{&FakeVolumePlugin{PluginName: "fake-plugin"}}
}

// FakeVolumePlugin is useful for testing.  It tries to be a fully compliant
// plugin, but all it does is make empty directories.
// Use as:
//   volume.RegisterPlugin(&FakePlugin{"fake-name"})
type FakeVolumePlugin struct {
	sync.RWMutex
	PluginName             string
	Host                   VolumeHost
	Config                 VolumeConfig
	LastProvisionerOptions VolumeOptions
	NewAttacherCallCount   int
	NewDetacherCallCount   int
	VolumeLimits           map[string]int64
	VolumeLimitsError      error
	LimitKey               string
	ProvisionDelaySeconds  int

	// Add callbacks as needed
	WaitForAttachHook func(spec *Spec, devicePath string, pod *v1.Pod, spectimeout time.Duration) (string, error)
	UnmountDeviceHook func(globalMountPath string) error

	Mounters             []*FakeVolume
	Unmounters           []*FakeVolume
	Attachers            []*FakeVolume
	Detachers            []*FakeVolume
	BlockVolumeMappers   []*FakeVolume
	BlockVolumeUnmappers []*FakeVolume
}

var _ VolumePlugin = &FakeVolumePlugin{}
var _ BlockVolumePlugin = &FakeVolumePlugin{}
var _ RecyclableVolumePlugin = &FakeVolumePlugin{}
var _ DeletableVolumePlugin = &FakeVolumePlugin{}
var _ ProvisionableVolumePlugin = &FakeVolumePlugin{}
var _ AttachableVolumePlugin = &FakeVolumePlugin{}
var _ VolumePluginWithAttachLimits = &FakeVolumePlugin{}
var _ DeviceMountableVolumePlugin = &FakeVolumePlugin{}
var _ NodeExpandableVolumePlugin = &FakeVolumePlugin{}

func (plugin *FakeVolumePlugin) getFakeVolume(list *[]*FakeVolume) *FakeVolume {
	volumeList := *list
	if list != nil && len(volumeList) > 0 {
		volume := volumeList[0]
		volume.Lock()
		defer volume.Unlock()
		volume.WaitForAttachHook = plugin.WaitForAttachHook
		volume.UnmountDeviceHook = plugin.UnmountDeviceHook
		return volume
	}
	volume := &FakeVolume{
		WaitForAttachHook: plugin.WaitForAttachHook,
		UnmountDeviceHook: plugin.UnmountDeviceHook,
	}
	volume.VolumesAttached = make(map[string]types.NodeName)
	*list = append(*list, volume)
	return volume
}

func (plugin *FakeVolumePlugin) Init(host VolumeHost) error {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.Host = host
	return nil
}

func (plugin *FakeVolumePlugin) GetPluginName() string {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.PluginName
}

func (plugin *FakeVolumePlugin) GetVolumeName(spec *Spec) (string, error) {
	var volumeName string
	if spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil {
		volumeName = spec.Volume.GCEPersistentDisk.PDName
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.GCEPersistentDisk != nil {
		volumeName = spec.PersistentVolume.Spec.GCEPersistentDisk.PDName
	}
	if volumeName == "" {
		volumeName = spec.Name()
	}
	return volumeName, nil
}

func (plugin *FakeVolumePlugin) CanSupport(spec *Spec) bool {
	// TODO: maybe pattern-match on spec.Name() to decide?
	return true
}

func (plugin *FakeVolumePlugin) IsMigratedToCSI() bool {
	return false
}

func (plugin *FakeVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *FakeVolumePlugin) SupportsMountOption() bool {
	return true
}

func (plugin *FakeVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *FakeVolumePlugin) NewMounter(spec *Spec, pod *v1.Pod, opts VolumeOptions) (Mounter, error) {
	plugin.Lock()
	defer plugin.Unlock()
	volume := plugin.getFakeVolume(&plugin.Mounters)
	volume.Lock()
	defer volume.Unlock()
	volume.PodUID = pod.UID
	volume.VolName = spec.Name()
	volume.Plugin = plugin
	volume.MetricsNil = MetricsNil{}
	return volume, nil
}

func (plugin *FakeVolumePlugin) GetMounters() (Mounters []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.Mounters
}

func (plugin *FakeVolumePlugin) NewUnmounter(volName string, podUID types.UID) (Unmounter, error) {
	plugin.Lock()
	defer plugin.Unlock()
	volume := plugin.getFakeVolume(&plugin.Unmounters)
	volume.Lock()
	defer volume.Unlock()
	volume.PodUID = podUID
	volume.VolName = volName
	volume.Plugin = plugin
	volume.MetricsNil = MetricsNil{}
	return volume, nil
}

func (plugin *FakeVolumePlugin) GetUnmounters() (Unmounters []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.Unmounters
}

// Block volume support
func (plugin *FakeVolumePlugin) NewBlockVolumeMapper(spec *Spec, pod *v1.Pod, opts VolumeOptions) (BlockVolumeMapper, error) {
	plugin.Lock()
	defer plugin.Unlock()
	volume := plugin.getFakeVolume(&plugin.BlockVolumeMappers)
	volume.Lock()
	defer volume.Unlock()
	if pod != nil {
		volume.PodUID = pod.UID
	}
	volume.VolName = spec.Name()
	volume.Plugin = plugin
	return volume, nil
}

// Block volume support
func (plugin *FakeVolumePlugin) GetBlockVolumeMapper() (BlockVolumeMappers []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.BlockVolumeMappers
}

// Block volume support
func (plugin *FakeVolumePlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (BlockVolumeUnmapper, error) {
	plugin.Lock()
	defer plugin.Unlock()
	volume := plugin.getFakeVolume(&plugin.BlockVolumeUnmappers)
	volume.Lock()
	defer volume.Unlock()
	volume.PodUID = podUID
	volume.VolName = volName
	volume.Plugin = plugin
	return volume, nil
}

// Block volume support
func (plugin *FakeVolumePlugin) GetBlockVolumeUnmapper() (BlockVolumeUnmappers []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.BlockVolumeUnmappers
}

func (plugin *FakeVolumePlugin) NewAttacher() (Attacher, error) {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.NewAttacherCallCount = plugin.NewAttacherCallCount + 1
	return plugin.getFakeVolume(&plugin.Attachers), nil
}

func (plugin *FakeVolumePlugin) NewDeviceMounter() (DeviceMounter, error) {
	return plugin.NewAttacher()
}

func (plugin *FakeVolumePlugin) GetAttachers() (Attachers []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.Attachers
}

func (plugin *FakeVolumePlugin) GetNewAttacherCallCount() int {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.NewAttacherCallCount
}

func (plugin *FakeVolumePlugin) NewDetacher() (Detacher, error) {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.NewDetacherCallCount = plugin.NewDetacherCallCount + 1
	detacher := plugin.getFakeVolume(&plugin.Detachers)
	attacherList := plugin.Attachers
	if attacherList != nil && len(attacherList) > 0 {
		detacherList := plugin.Detachers
		if detacherList != nil && len(detacherList) > 0 {
			detacherList[0].VolumesAttached = attacherList[0].VolumesAttached
		}

	}
	return detacher, nil
}

func (plugin *FakeVolumePlugin) NewDeviceUnmounter() (DeviceUnmounter, error) {
	return plugin.NewDetacher()
}

func (plugin *FakeVolumePlugin) GetDetachers() (Detachers []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.Detachers
}

func (plugin *FakeVolumePlugin) GetNewDetacherCallCount() int {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.NewDetacherCallCount
}

func (plugin *FakeVolumePlugin) CanAttach(spec *Spec) (bool, error) {
	return true, nil
}

func (plugin *FakeVolumePlugin) CanDeviceMount(spec *Spec) (bool, error) {
	return true, nil
}

func (plugin *FakeVolumePlugin) Recycle(pvName string, spec *Spec, eventRecorder recyclerclient.RecycleEventRecorder) error {
	return nil
}

func (plugin *FakeVolumePlugin) NewDeleter(spec *Spec) (Deleter, error) {
	return &FakeDeleter{"/attributesTransferredFromSpec", MetricsNil{}}, nil
}

func (plugin *FakeVolumePlugin) NewProvisioner(options VolumeOptions) (Provisioner, error) {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.LastProvisionerOptions = options
	return &FakeProvisioner{options, plugin.Host, plugin.ProvisionDelaySeconds}, nil
}

func (plugin *FakeVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{}
}

func (plugin *FakeVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*Spec, error) {
	return &Spec{
		Volume: &v1.Volume{
			Name: volumeName,
		},
	}, nil
}

// Block volume support
func (plugin *FakeVolumePlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mountPath string) (*Spec, error) {
	return &Spec{
		Volume: &v1.Volume{
			Name: volumeName,
		},
	}, nil
}

func (plugin *FakeVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	return []string{}, nil
}

// Expandable volume support
func (plugin *FakeVolumePlugin) ExpandVolumeDevice(spec *Spec, newSize resource.Quantity, oldSize resource.Quantity) (resource.Quantity, error) {
	return resource.Quantity{}, nil
}

func (plugin *FakeVolumePlugin) RequiresFSResize() bool {
	return true
}

func (plugin *FakeVolumePlugin) NodeExpand(resizeOptions NodeResizeOptions) (bool, error) {
	return true, nil
}

func (plugin *FakeVolumePlugin) GetVolumeLimits() (map[string]int64, error) {
	return plugin.VolumeLimits, plugin.VolumeLimitsError
}

func (plugin *FakeVolumePlugin) VolumeLimitKey(spec *Spec) string {
	return plugin.LimitKey
}

// FakeBasicVolumePlugin implements a basic volume plugin. It wrappers on
// FakeVolumePlugin but implements VolumePlugin interface only.
// It is useful to test logic involving plugin interfaces.
type FakeBasicVolumePlugin struct {
	Plugin FakeVolumePlugin
}

func (f *FakeBasicVolumePlugin) GetPluginName() string {
	return f.Plugin.GetPluginName()
}

func (f *FakeBasicVolumePlugin) GetVolumeName(spec *Spec) (string, error) {
	return f.Plugin.GetVolumeName(spec)
}

// CanSupport tests whether the plugin supports a given volume specification by
// testing volume spec name begins with plugin name or not.
// This is useful to choose plugin by volume in testing.
func (f *FakeBasicVolumePlugin) CanSupport(spec *Spec) bool {
	return strings.HasPrefix(spec.Name(), f.GetPluginName())
}

func (plugin *FakeBasicVolumePlugin) IsMigratedToCSI() bool {
	return false
}

func (f *FakeBasicVolumePlugin) ConstructVolumeSpec(ame, mountPath string) (*Spec, error) {
	return f.Plugin.ConstructVolumeSpec(ame, mountPath)
}

func (f *FakeBasicVolumePlugin) Init(ost VolumeHost) error {
	return f.Plugin.Init(ost)
}

func (f *FakeBasicVolumePlugin) NewMounter(spec *Spec, pod *v1.Pod, opts VolumeOptions) (Mounter, error) {
	return f.Plugin.NewMounter(spec, pod, opts)
}

func (f *FakeBasicVolumePlugin) NewUnmounter(volName string, podUID types.UID) (Unmounter, error) {
	return f.Plugin.NewUnmounter(volName, podUID)
}

func (f *FakeBasicVolumePlugin) RequiresRemount() bool {
	return f.Plugin.RequiresRemount()
}

func (f *FakeBasicVolumePlugin) SupportsBulkVolumeVerification() bool {
	return f.Plugin.SupportsBulkVolumeVerification()
}

func (f *FakeBasicVolumePlugin) SupportsMountOption() bool {
	return f.Plugin.SupportsMountOption()
}

var _ VolumePlugin = &FakeBasicVolumePlugin{}

// FakeDeviceMountableVolumePlugin implements an device mountable plugin based on FakeBasicVolumePlugin.
type FakeDeviceMountableVolumePlugin struct {
	FakeBasicVolumePlugin
}

func (f *FakeDeviceMountableVolumePlugin) CanDeviceMount(spec *Spec) (bool, error) {
	return true, nil
}

func (f *FakeDeviceMountableVolumePlugin) NewDeviceMounter() (DeviceMounter, error) {
	return f.Plugin.NewDeviceMounter()
}

func (f *FakeDeviceMountableVolumePlugin) NewDeviceUnmounter() (DeviceUnmounter, error) {
	return f.Plugin.NewDeviceUnmounter()
}

func (f *FakeDeviceMountableVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	return f.Plugin.GetDeviceMountRefs(deviceMountPath)
}

var _ VolumePlugin = &FakeDeviceMountableVolumePlugin{}
var _ DeviceMountableVolumePlugin = &FakeDeviceMountableVolumePlugin{}

// FakeAttachableVolumePlugin implements an attachable plugin based on FakeDeviceMountableVolumePlugin.
type FakeAttachableVolumePlugin struct {
	FakeDeviceMountableVolumePlugin
}

func (f *FakeAttachableVolumePlugin) NewAttacher() (Attacher, error) {
	return f.Plugin.NewAttacher()
}

func (f *FakeAttachableVolumePlugin) NewDetacher() (Detacher, error) {
	return f.Plugin.NewDetacher()
}

func (f *FakeAttachableVolumePlugin) CanAttach(spec *Spec) (bool, error) {
	return true, nil
}

var _ VolumePlugin = &FakeAttachableVolumePlugin{}
var _ AttachableVolumePlugin = &FakeAttachableVolumePlugin{}

type FakeFileVolumePlugin struct {
}

func (plugin *FakeFileVolumePlugin) Init(host VolumeHost) error {
	return nil
}

func (plugin *FakeFileVolumePlugin) GetPluginName() string {
	return "fake-file-plugin"
}

func (plugin *FakeFileVolumePlugin) GetVolumeName(spec *Spec) (string, error) {
	return "", nil
}

func (plugin *FakeFileVolumePlugin) CanSupport(spec *Spec) bool {
	return true
}

func (plugin *FakeFileVolumePlugin) IsMigratedToCSI() bool {
	return false
}

func (plugin *FakeFileVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *FakeFileVolumePlugin) SupportsMountOption() bool {
	return false
}

func (plugin *FakeFileVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *FakeFileVolumePlugin) NewMounter(spec *Spec, podRef *v1.Pod, opts VolumeOptions) (Mounter, error) {
	return nil, nil
}

func (plugin *FakeFileVolumePlugin) NewUnmounter(name string, podUID types.UID) (Unmounter, error) {
	return nil, nil
}

func (plugin *FakeFileVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*Spec, error) {
	return nil, nil
}

func NewFakeFileVolumePlugin() []VolumePlugin {
	return []VolumePlugin{&FakeFileVolumePlugin{}}
}

type FakeVolume struct {
	sync.RWMutex
	PodUID  types.UID
	VolName string
	Plugin  *FakeVolumePlugin
	MetricsNil
	VolumesAttached map[string]types.NodeName

	// Add callbacks as needed
	WaitForAttachHook func(spec *Spec, devicePath string, pod *v1.Pod, spectimeout time.Duration) (string, error)
	UnmountDeviceHook func(globalMountPath string) error

	SetUpCallCount              int
	TearDownCallCount           int
	AttachCallCount             int
	DetachCallCount             int
	WaitForAttachCallCount      int
	MountDeviceCallCount        int
	UnmountDeviceCallCount      int
	GetDeviceMountPathCallCount int
	SetUpDeviceCallCount        int
	TearDownDeviceCallCount     int
	MapDeviceCallCount          int
	GlobalMapPathCallCount      int
	PodDeviceMapPathCallCount   int
}

func getUniqueVolumeName(spec *Spec) (string, error) {
	var volumeName string
	if spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil {
		volumeName = spec.Volume.GCEPersistentDisk.PDName
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.GCEPersistentDisk != nil {
		volumeName = spec.PersistentVolume.Spec.GCEPersistentDisk.PDName
	}
	if volumeName == "" {
		volumeName = spec.Name()
	}
	return volumeName, nil
}

func (_ *FakeVolume) GetAttributes() Attributes {
	return Attributes{
		ReadOnly:        false,
		Managed:         true,
		SupportsSELinux: true,
	}
}

func (fv *FakeVolume) CanMount() error {
	return nil
}

func (fv *FakeVolume) SetUp(mounterArgs MounterArgs) error {
	fv.Lock()
	defer fv.Unlock()
	fv.SetUpCallCount++
	return fv.SetUpAt(fv.getPath(), mounterArgs)
}

func (fv *FakeVolume) GetSetUpCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.SetUpCallCount
}

func (fv *FakeVolume) SetUpAt(dir string, mounterArgs MounterArgs) error {
	return os.MkdirAll(dir, 0750)
}

func (fv *FakeVolume) GetPath() string {
	fv.RLock()
	defer fv.RUnlock()
	return fv.getPath()
}

func (fv *FakeVolume) getPath() string {
	return filepath.Join(fv.Plugin.Host.GetPodVolumeDir(fv.PodUID, utilstrings.EscapeQualifiedName(fv.Plugin.PluginName), fv.VolName))
}

func (fv *FakeVolume) TearDown() error {
	fv.Lock()
	defer fv.Unlock()
	fv.TearDownCallCount++
	return fv.TearDownAt(fv.getPath())
}

func (fv *FakeVolume) GetTearDownCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.TearDownCallCount
}

func (fv *FakeVolume) TearDownAt(dir string) error {
	return os.RemoveAll(dir)
}

// Block volume support
func (fv *FakeVolume) SetUpDevice() (string, error) {
	fv.Lock()
	defer fv.Unlock()
	fv.SetUpDeviceCallCount++
	return "", nil
}

// Block volume support
func (fv *FakeVolume) GetSetUpDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.SetUpDeviceCallCount
}

// Block volume support
func (fv *FakeVolume) GetGlobalMapPath(spec *Spec) (string, error) {
	fv.RLock()
	defer fv.RUnlock()
	fv.GlobalMapPathCallCount++
	return fv.getGlobalMapPath()
}

// Block volume support
func (fv *FakeVolume) getGlobalMapPath() (string, error) {
	return filepath.Join(fv.Plugin.Host.GetVolumeDevicePluginDir(utilstrings.EscapeQualifiedName(fv.Plugin.PluginName)), "pluginDependentPath"), nil
}

// Block volume support
func (fv *FakeVolume) GetGlobalMapPathCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.GlobalMapPathCallCount
}

// Block volume support
func (fv *FakeVolume) GetPodDeviceMapPath() (string, string) {
	fv.RLock()
	defer fv.RUnlock()
	fv.PodDeviceMapPathCallCount++
	return fv.getPodDeviceMapPath()
}

// Block volume support
func (fv *FakeVolume) getPodDeviceMapPath() (string, string) {
	return filepath.Join(fv.Plugin.Host.GetPodVolumeDeviceDir(fv.PodUID, utilstrings.EscapeQualifiedName(fv.Plugin.PluginName))), fv.VolName
}

// Block volume support
func (fv *FakeVolume) GetPodDeviceMapPathCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.PodDeviceMapPathCallCount
}

// Block volume support
func (fv *FakeVolume) TearDownDevice(mapPath string, devicePath string) error {
	fv.Lock()
	defer fv.Unlock()
	fv.TearDownDeviceCallCount++
	return nil
}

// Block volume support
func (fv *FakeVolume) GetTearDownDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.TearDownDeviceCallCount
}

// Block volume support
func (fv *FakeVolume) MapDevice(devicePath, globalMapPath, volumeMapPath, volumeMapName string, pod types.UID) error {
	fv.Lock()
	defer fv.Unlock()
	fv.MapDeviceCallCount++
	return nil
}

// Block volume support
func (fv *FakeVolume) GetMapDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.MapDeviceCallCount
}

func (fv *FakeVolume) Attach(spec *Spec, nodeName types.NodeName) (string, error) {
	fv.Lock()
	defer fv.Unlock()
	fv.AttachCallCount++
	volumeName, err := getUniqueVolumeName(spec)
	if err != nil {
		return "", err
	}
	volumeNode, exist := fv.VolumesAttached[volumeName]
	if exist {
		if nodeName == UncertainAttachNode {
			return "/dev/vdb-test", nil
		}
		// even if volume was previously attached to time out, we need to keep returning error
		// so as reconciler can not confirm this volume as attached.
		if nodeName == TimeoutAttachNode {
			return "", fmt.Errorf("Timed out to attach volume %q to node %q", volumeName, nodeName)
		}
		if volumeNode == nodeName || volumeNode == MultiAttachNode || nodeName == MultiAttachNode {
			return "/dev/vdb-test", nil
		}
		return "", fmt.Errorf("volume %q trying to attach to node %q is already attached to node %q", volumeName, nodeName, volumeNode)
	}

	fv.VolumesAttached[volumeName] = nodeName
	if nodeName == UncertainAttachNode || nodeName == TimeoutAttachNode {
		return "", fmt.Errorf("Timed out to attach volume %q to node %q", volumeName, nodeName)
	}
	return "/dev/vdb-test", nil
}

func (fv *FakeVolume) GetAttachCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.AttachCallCount
}

func (fv *FakeVolume) WaitForAttach(spec *Spec, devicePath string, pod *v1.Pod, spectimeout time.Duration) (string, error) {
	fv.Lock()
	defer fv.Unlock()
	fv.WaitForAttachCallCount++
	if fv.WaitForAttachHook != nil {
		return fv.WaitForAttachHook(spec, devicePath, pod, spectimeout)
	}
	return "/dev/sdb", nil
}

func (fv *FakeVolume) GetWaitForAttachCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.WaitForAttachCallCount
}

func (fv *FakeVolume) GetDeviceMountPath(spec *Spec) (string, error) {
	fv.Lock()
	defer fv.Unlock()
	fv.GetDeviceMountPathCallCount++
	return "", nil
}

func (fv *FakeVolume) MountDevice(spec *Spec, devicePath string, deviceMountPath string) error {
	fv.Lock()
	defer fv.Unlock()
	fv.MountDeviceCallCount++
	return nil
}

func (fv *FakeVolume) GetMountDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.MountDeviceCallCount
}

func (fv *FakeVolume) Detach(volumeName string, nodeName types.NodeName) error {
	fv.Lock()
	defer fv.Unlock()
	fv.DetachCallCount++
	if _, exist := fv.VolumesAttached[volumeName]; !exist {
		return fmt.Errorf("Trying to detach volume %q that is not attached to the node %q", volumeName, nodeName)
	}
	delete(fv.VolumesAttached, volumeName)
	return nil
}

func (fv *FakeVolume) VolumesAreAttached(spec []*Spec, nodeName types.NodeName) (map[*Spec]bool, error) {
	fv.Lock()
	defer fv.Unlock()
	return nil, nil
}

func (fv *FakeVolume) GetDetachCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.DetachCallCount
}

func (fv *FakeVolume) UnmountDevice(globalMountPath string) error {
	fv.Lock()
	defer fv.Unlock()
	fv.UnmountDeviceCallCount++
	if fv.UnmountDeviceHook != nil {
		return fv.UnmountDeviceHook(globalMountPath)
	}
	return nil
}

type FakeDeleter struct {
	path string
	MetricsNil
}

func (fd *FakeDeleter) Delete() error {
	// nil is success, else error
	return nil
}

func (fd *FakeDeleter) GetPath() string {
	return fd.path
}

type FakeProvisioner struct {
	Options               VolumeOptions
	Host                  VolumeHost
	ProvisionDelaySeconds int
}

func (fc *FakeProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	// Add provision failure hook
	if fc.Options.Parameters != nil {
		if _, ok := fc.Options.Parameters[ExpectProvisionFailureKey]; ok {
			return nil, fmt.Errorf("expected error")
		}
	}
	fullpath := fmt.Sprintf("/tmp/hostpath_pv/%s", uuid.NewUUID())

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fc.Options.PVName,
			Annotations: map[string]string{
				util.VolumeDynamicallyCreatedByKey: "fakeplugin-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: fc.Options.PersistentVolumeReclaimPolicy,
			AccessModes:                   fc.Options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): fc.Options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)],
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: fullpath,
				},
			},
		},
	}

	if fc.ProvisionDelaySeconds > 0 {
		time.Sleep(time.Duration(fc.ProvisionDelaySeconds) * time.Second)
	}

	return pv, nil
}

var _ volumepathhandler.BlockVolumePathHandler = &FakeVolumePathHandler{}

//NewDeviceHandler Create a new IoHandler implementation
func NewBlockVolumePathHandler() volumepathhandler.BlockVolumePathHandler {
	return &FakeVolumePathHandler{}
}

type FakeVolumePathHandler struct {
	sync.RWMutex
}

func (fv *FakeVolumePathHandler) MapDevice(devicePath string, mapDir string, linkName string) error {
	// nil is success, else error
	return nil
}

func (fv *FakeVolumePathHandler) UnmapDevice(mapDir string, linkName string) error {
	// nil is success, else error
	return nil
}

func (fv *FakeVolumePathHandler) RemoveMapPath(mapPath string) error {
	// nil is success, else error
	return nil
}

func (fv *FakeVolumePathHandler) IsSymlinkExist(mapPath string) (bool, error) {
	// nil is success, else error
	return true, nil
}

func (fv *FakeVolumePathHandler) GetDeviceSymlinkRefs(devPath string, mapPath string) ([]string, error) {
	// nil is success, else error
	return []string{}, nil
}

func (fv *FakeVolumePathHandler) FindGlobalMapPathUUIDFromPod(pluginDir, mapPath string, podUID types.UID) (string, error) {
	// nil is success, else error
	return "", nil
}

func (fv *FakeVolumePathHandler) AttachFileDevice(path string) (string, error) {
	// nil is success, else error
	return "", nil
}

func (fv *FakeVolumePathHandler) GetLoopDevice(path string) (string, error) {
	// nil is success, else error
	return "/dev/loop1", nil
}

func (fv *FakeVolumePathHandler) RemoveLoopDevice(device string) error {
	// nil is success, else error
	return nil
}

// FindEmptyDirectoryUsageOnTmpfs finds the expected usage of an empty directory existing on
// a tmpfs filesystem on this system.
func FindEmptyDirectoryUsageOnTmpfs() (*resource.Quantity, error) {
	tmpDir, err := utiltesting.MkTmpdir("metrics_du_test")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	out, err := exec.Command("nice", "-n", "19", "du", "-s", "-B", "1", tmpDir).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed command 'du' on %s with error %v", tmpDir, err)
	}
	used, err := resource.ParseQuantity(strings.Fields(string(out))[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse 'du' output %s due to error %v", out, err)
	}
	used.Format = resource.BinarySI
	return &used, nil
}

// VerifyAttachCallCount ensures that at least one of the Attachers for this
// plugin has the expectedAttachCallCount number of calls. Otherwise it returns
// an error.
func VerifyAttachCallCount(
	expectedAttachCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, attacher := range fakeVolumePlugin.GetAttachers() {
		actualCallCount := attacher.GetAttachCallCount()
		if actualCallCount >= expectedAttachCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No attachers have expected AttachCallCount. Expected: <%v>.",
		expectedAttachCallCount)
}

// VerifyZeroAttachCalls ensures that all of the Attachers for this plugin have
// a zero AttachCallCount. Otherwise it returns an error.
func VerifyZeroAttachCalls(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, attacher := range fakeVolumePlugin.GetAttachers() {
		actualCallCount := attacher.GetAttachCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one attacher has non-zero AttachCallCount: <%v>.",
				actualCallCount)
		}
	}

	return nil
}

// VerifyWaitForAttachCallCount ensures that at least one of the Mounters for
// this plugin has the expectedWaitForAttachCallCount number of calls. Otherwise
// it returns an error.
func VerifyWaitForAttachCallCount(
	expectedWaitForAttachCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, attacher := range fakeVolumePlugin.GetAttachers() {
		actualCallCount := attacher.GetWaitForAttachCallCount()
		if actualCallCount >= expectedWaitForAttachCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Attachers have expected WaitForAttachCallCount. Expected: <%v>.",
		expectedWaitForAttachCallCount)
}

// VerifyZeroWaitForAttachCallCount ensures that all Attachers for this plugin
// have a zero WaitForAttachCallCount. Otherwise it returns an error.
func VerifyZeroWaitForAttachCallCount(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, attacher := range fakeVolumePlugin.GetAttachers() {
		actualCallCount := attacher.GetWaitForAttachCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one attacher has non-zero WaitForAttachCallCount: <%v>.",
				actualCallCount)
		}
	}

	return nil
}

// VerifyMountDeviceCallCount ensures that at least one of the Mounters for
// this plugin has the expectedMountDeviceCallCount number of calls. Otherwise
// it returns an error.
func VerifyMountDeviceCallCount(
	expectedMountDeviceCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, attacher := range fakeVolumePlugin.GetAttachers() {
		actualCallCount := attacher.GetMountDeviceCallCount()
		if actualCallCount >= expectedMountDeviceCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Attachers have expected MountDeviceCallCount. Expected: <%v>.",
		expectedMountDeviceCallCount)
}

// VerifyZeroMountDeviceCallCount ensures that all Attachers for this plugin
// have a zero MountDeviceCallCount. Otherwise it returns an error.
func VerifyZeroMountDeviceCallCount(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, attacher := range fakeVolumePlugin.GetAttachers() {
		actualCallCount := attacher.GetMountDeviceCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one attacher has non-zero MountDeviceCallCount: <%v>.",
				actualCallCount)
		}
	}

	return nil
}

// VerifySetUpCallCount ensures that at least one of the Mounters for this
// plugin has the expectedSetUpCallCount number of calls. Otherwise it returns
// an error.
func VerifySetUpCallCount(
	expectedSetUpCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mounter := range fakeVolumePlugin.GetMounters() {
		actualCallCount := mounter.GetSetUpCallCount()
		if actualCallCount >= expectedSetUpCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Mounters have expected SetUpCallCount. Expected: <%v>.",
		expectedSetUpCallCount)
}

// VerifyZeroSetUpCallCount ensures that all Mounters for this plugin have a
// zero SetUpCallCount. Otherwise it returns an error.
func VerifyZeroSetUpCallCount(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mounter := range fakeVolumePlugin.GetMounters() {
		actualCallCount := mounter.GetSetUpCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one mounter has non-zero SetUpCallCount: <%v>.",
				actualCallCount)
		}
	}

	return nil
}

// VerifyTearDownCallCount ensures that at least one of the Unounters for this
// plugin has the expectedTearDownCallCount number of calls. Otherwise it
// returns an error.
func VerifyTearDownCallCount(
	expectedTearDownCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, unmounter := range fakeVolumePlugin.GetUnmounters() {
		actualCallCount := unmounter.GetTearDownCallCount()
		if actualCallCount >= expectedTearDownCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Unmounters have expected SetUpCallCount. Expected: <%v>.",
		expectedTearDownCallCount)
}

// VerifyZeroTearDownCallCount ensures that all Mounters for this plugin have a
// zero TearDownCallCount. Otherwise it returns an error.
func VerifyZeroTearDownCallCount(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mounter := range fakeVolumePlugin.GetMounters() {
		actualCallCount := mounter.GetTearDownCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one mounter has non-zero TearDownCallCount: <%v>.",
				actualCallCount)
		}
	}

	return nil
}

// VerifyDetachCallCount ensures that at least one of the Attachers for this
// plugin has the expectedDetachCallCount number of calls. Otherwise it returns
// an error.
func VerifyDetachCallCount(
	expectedDetachCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, detacher := range fakeVolumePlugin.GetDetachers() {
		actualCallCount := detacher.GetDetachCallCount()
		if actualCallCount == expectedDetachCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Detachers have expected DetachCallCount. Expected: <%v>.",
		expectedDetachCallCount)
}

// VerifyZeroDetachCallCount ensures that all Detachers for this plugin have a
// zero DetachCallCount. Otherwise it returns an error.
func VerifyZeroDetachCallCount(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, detacher := range fakeVolumePlugin.GetDetachers() {
		actualCallCount := detacher.GetDetachCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one detacher has non-zero DetachCallCount: <%v>.",
				actualCallCount)
		}
	}

	return nil
}

// VerifySetUpDeviceCallCount ensures that at least one of the Mappers for this
// plugin has the expectedSetUpDeviceCallCount number of calls. Otherwise it
// returns an error.
func VerifySetUpDeviceCallCount(
	expectedSetUpDeviceCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mapper := range fakeVolumePlugin.GetBlockVolumeMapper() {
		actualCallCount := mapper.GetSetUpDeviceCallCount()
		if actualCallCount >= expectedSetUpDeviceCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Mapper have expected SetUpDeviceCallCount. Expected: <%v>.",
		expectedSetUpDeviceCallCount)
}

// VerifyTearDownDeviceCallCount ensures that at least one of the Unmappers for this
// plugin has the expectedTearDownDeviceCallCount number of calls. Otherwise it
// returns an error.
func VerifyTearDownDeviceCallCount(
	expectedTearDownDeviceCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, unmapper := range fakeVolumePlugin.GetBlockVolumeUnmapper() {
		actualCallCount := unmapper.GetTearDownDeviceCallCount()
		if actualCallCount >= expectedTearDownDeviceCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Unmapper have expected TearDownDeviceCallCount. Expected: <%v>.",
		expectedTearDownDeviceCallCount)
}

// VerifyZeroTearDownDeviceCallCount ensures that all Mappers for this plugin have a
// zero TearDownDeviceCallCount. Otherwise it returns an error.
func VerifyZeroTearDownDeviceCallCount(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, unmapper := range fakeVolumePlugin.GetBlockVolumeUnmapper() {
		actualCallCount := unmapper.GetTearDownDeviceCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one unmapper has non-zero TearDownDeviceCallCount: <%v>.",
				actualCallCount)
		}
	}

	return nil
}

// VerifyGetGlobalMapPathCallCount ensures that at least one of the Mappers for this
// plugin has the expectedGlobalMapPathCallCount number of calls. Otherwise it returns
// an error.
func VerifyGetGlobalMapPathCallCount(
	expectedGlobalMapPathCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mapper := range fakeVolumePlugin.GetBlockVolumeMapper() {
		actualCallCount := mapper.GetGlobalMapPathCallCount()
		if actualCallCount == expectedGlobalMapPathCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Mappers have expected GetGlobalMapPathCallCount. Expected: <%v>.",
		expectedGlobalMapPathCallCount)
}

// VerifyGetPodDeviceMapPathCallCount ensures that at least one of the Mappers for this
// plugin has the expectedPodDeviceMapPathCallCount number of calls. Otherwise it returns
// an error.
func VerifyGetPodDeviceMapPathCallCount(
	expectedPodDeviceMapPathCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mapper := range fakeVolumePlugin.GetBlockVolumeMapper() {
		actualCallCount := mapper.GetPodDeviceMapPathCallCount()
		if actualCallCount == expectedPodDeviceMapPathCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Mappers have expected GetPodDeviceMapPathCallCount. Expected: <%v>.",
		expectedPodDeviceMapPathCallCount)
}

// VerifyGetMapDeviceCallCount ensures that at least one of the Mappers for this
// plugin has the expectedMapDeviceCallCount number of calls. Otherwise it
// returns an error.
func VerifyGetMapDeviceCallCount(
	expectedMapDeviceCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mapper := range fakeVolumePlugin.GetBlockVolumeMapper() {
		actualCallCount := mapper.GetMapDeviceCallCount()
		if actualCallCount >= expectedMapDeviceCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Mapper have expected MapdDeviceCallCount. Expected: <%v>.",
		expectedMapDeviceCallCount)
}

// GetTestVolumePluginMgr creates, initializes, and returns a test volume plugin
// manager and fake volume plugin using a fake volume host.
func GetTestVolumePluginMgr(
	t *testing.T) (*VolumePluginMgr, *FakeVolumePlugin) {
	v := NewFakeVolumeHost(
		"",  /* rootDir */
		nil, /* kubeClient */
		nil, /* plugins */
	)
	plugins := ProbeVolumePlugins(VolumeConfig{})
	if err := v.pluginMgr.InitPlugins(plugins, nil /* prober */, v); err != nil {
		t.Fatal(err)
	}

	return &v.pluginMgr, plugins[0].(*FakeVolumePlugin)
}

// CreateTestPVC returns a provisionable PVC for tests
func CreateTestPVC(capacity string, accessModes []v1.PersistentVolumeAccessMode) *v1.PersistentVolumeClaim {
	claim := v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dummy",
			Namespace: "default",
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: accessModes,
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(capacity),
				},
			},
		},
	}
	return &claim
}

func MetricsEqualIgnoreTimestamp(a *Metrics, b *Metrics) bool {
	available := a.Available == b.Available
	capacity := a.Capacity == b.Capacity
	used := a.Used == b.Used
	inodes := a.Inodes == b.Inodes
	inodesFree := a.InodesFree == b.InodesFree
	inodesUsed := a.InodesUsed == b.InodesUsed
	return available && capacity && used && inodes && inodesFree && inodesUsed
}

func ContainsAccessMode(modes []v1.PersistentVolumeAccessMode, mode v1.PersistentVolumeAccessMode) bool {
	for _, m := range modes {
		if m == mode {
			return true
		}
	}
	return false
}

func (f *fakeVolumeHost) CSIDriverLister() storagelisters.CSIDriverLister {
	return f.csiDriverLister
}

func (f *fakeVolumeHost) CSIDriversSynced() cache.InformerSynced {
	// not needed for testing
	return nil
}

func (f *fakeVolumeHost) CSINodeLister() storagelisters.CSINodeLister {
	// not needed for testing
	return nil
}

func (f *fakeVolumeHost) GetInformerFactory() informers.SharedInformerFactory {
	return f.informerFactory
}

func (f *fakeVolumeHost) IsAttachDetachController() bool {
	return true
}

func (f *fakeVolumeHost) SetKubeletError(err error) {
	return
}

func (f *fakeVolumeHost) WaitForCacheSync() error {
	return nil
}
