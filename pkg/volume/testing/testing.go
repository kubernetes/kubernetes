/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io/ioutil"
	"os"
	"os/exec"
	"path"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	. "k8s.io/kubernetes/pkg/volume"
)

// fakeVolumeHost is useful for testing volume plugins.
type fakeVolumeHost struct {
	rootDir    string
	kubeClient clientset.Interface
	pluginMgr  VolumePluginMgr
	cloud      cloudprovider.Interface
	mounter    mount.Interface
	writer     io.Writer
}

func NewFakeVolumeHost(rootDir string, kubeClient clientset.Interface, plugins []VolumePlugin) *fakeVolumeHost {
	host := &fakeVolumeHost{rootDir: rootDir, kubeClient: kubeClient, cloud: nil}
	host.mounter = &mount.FakeMounter{}
	host.writer = &io.StdWriter{}
	host.pluginMgr.InitPlugins(plugins, host)
	return host
}

func (f *fakeVolumeHost) GetPluginDir(podUID string) string {
	return path.Join(f.rootDir, "plugins", podUID)
}

func (f *fakeVolumeHost) GetPodVolumeDir(podUID types.UID, pluginName, volumeName string) string {
	return path.Join(f.rootDir, "pods", string(podUID), "volumes", pluginName, volumeName)
}

func (f *fakeVolumeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return path.Join(f.rootDir, "pods", string(podUID), "plugins", pluginName)
}

func (f *fakeVolumeHost) GetKubeClient() clientset.Interface {
	return f.kubeClient
}

func (f *fakeVolumeHost) GetCloudProvider() cloudprovider.Interface {
	return f.cloud
}

func (f *fakeVolumeHost) GetMounter() mount.Interface {
	return f.mounter
}

func (f *fakeVolumeHost) GetWriter() io.Writer {
	return f.writer
}

func (f *fakeVolumeHost) NewWrapperMounter(volName string, spec Spec, pod *api.Pod, opts VolumeOptions) (Mounter, error) {
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
	PluginName             string
	Host                   VolumeHost
	Config                 VolumeConfig
	LastProvisionerOptions VolumeOptions
	NewAttacherCallCount   int
	NewDetacherCallCount   int

	Mounters   []*FakeVolume
	Unmounters []*FakeVolume
	Attachers  []*FakeVolume
	Detachers  []*FakeVolume
}

var _ VolumePlugin = &FakeVolumePlugin{}
var _ RecyclableVolumePlugin = &FakeVolumePlugin{}
var _ DeletableVolumePlugin = &FakeVolumePlugin{}
var _ ProvisionableVolumePlugin = &FakeVolumePlugin{}
var _ AttachableVolumePlugin = &FakeVolumePlugin{}

func (plugin *FakeVolumePlugin) getFakeVolume(list *[]*FakeVolume) *FakeVolume {
	volume := &FakeVolume{}
	*list = append(*list, volume)
	return volume
}

func (plugin *FakeVolumePlugin) Init(host VolumeHost) error {
	plugin.Host = host
	return nil
}

func (plugin *FakeVolumePlugin) Name() string {
	return plugin.PluginName
}

func (plugin *FakeVolumePlugin) CanSupport(spec *Spec) bool {
	// TODO: maybe pattern-match on spec.Name() to decide?
	return true
}

func (plugin *FakeVolumePlugin) NewMounter(spec *Spec, pod *api.Pod, opts VolumeOptions) (Mounter, error) {
	volume := plugin.getFakeVolume(&plugin.Mounters)
	volume.PodUID = pod.UID
	volume.VolName = spec.Name()
	volume.Plugin = plugin
	volume.MetricsNil = MetricsNil{}
	return volume, nil
}

func (plugin *FakeVolumePlugin) NewUnmounter(volName string, podUID types.UID) (Unmounter, error) {
	volume := plugin.getFakeVolume(&plugin.Unmounters)
	volume.PodUID = podUID
	volume.VolName = volName
	volume.Plugin = plugin
	volume.MetricsNil = MetricsNil{}
	return volume, nil
}

func (plugin *FakeVolumePlugin) NewAttacher(spec *Spec) (Attacher, error) {
	plugin.NewAttacherCallCount = plugin.NewAttacherCallCount + 1
	return plugin.getFakeVolume(&plugin.Attachers), nil
}

func (plugin *FakeVolumePlugin) NewDetacher(name string, podUID types.UID) (Detacher, error) {
	plugin.NewDetacherCallCount = plugin.NewDetacherCallCount + 1
	return plugin.getFakeVolume(&plugin.Detachers), nil
}

func (plugin *FakeVolumePlugin) NewRecycler(spec *Spec) (Recycler, error) {
	return &fakeRecycler{"/attributesTransferredFromSpec", MetricsNil{}}, nil
}

func (plugin *FakeVolumePlugin) NewDeleter(spec *Spec) (Deleter, error) {
	return &FakeDeleter{"/attributesTransferredFromSpec", MetricsNil{}}, nil
}

func (plugin *FakeVolumePlugin) NewProvisioner(options VolumeOptions) (Provisioner, error) {
	plugin.LastProvisionerOptions = options
	return &FakeProvisioner{options, plugin.Host}, nil
}

func (plugin *FakeVolumePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{}
}

type FakeVolume struct {
	PodUID  types.UID
	VolName string
	Plugin  *FakeVolumePlugin
	MetricsNil

	SetUpCallCount              int
	TearDownCallCount           int
	AttachCallCount             int
	DetachCallCount             int
	WaitForAttachCallCount      int
	WaitForDetachCallCount      int
	MountDeviceCallCount        int
	UnmountDeviceCallCount      int
	GetDeviceMountPathCallCount int
}

func (_ *FakeVolume) GetAttributes() Attributes {
	return Attributes{
		ReadOnly:        false,
		Managed:         true,
		SupportsSELinux: true,
	}
}

func (fv *FakeVolume) SetUp(fsGroup *int64) error {
	fv.SetUpCallCount++
	return fv.SetUpAt(fv.GetPath(), fsGroup)
}

func (fv *FakeVolume) SetUpAt(dir string, fsGroup *int64) error {
	return os.MkdirAll(dir, 0750)
}

func (fv *FakeVolume) GetPath() string {
	return path.Join(fv.Plugin.Host.GetPodVolumeDir(fv.PodUID, utilstrings.EscapeQualifiedNameForDisk(fv.Plugin.PluginName), fv.VolName))
}

func (fv *FakeVolume) TearDown() error {
	fv.TearDownCallCount++
	return fv.TearDownAt(fv.GetPath())
}

func (fv *FakeVolume) TearDownAt(dir string) error {
	return os.RemoveAll(dir)
}

func (fv *FakeVolume) Attach(spec *Spec, hostName string) error {
	fv.AttachCallCount++
	return nil
}

func (fv *FakeVolume) WaitForAttach(spec *Spec, spectimeout time.Duration) (string, error) {
	fv.WaitForAttachCallCount++
	return "", nil
}

func (fv *FakeVolume) GetDeviceMountPath(spec *Spec) string {
	fv.GetDeviceMountPathCallCount++
	return ""
}

func (fv *FakeVolume) MountDevice(devicePath string, deviceMountPath string, mounter mount.Interface) error {
	fv.MountDeviceCallCount++
	return nil
}

func (fv *FakeVolume) Detach(deviceMountPath string, hostName string) error {
	fv.DetachCallCount++
	return nil
}

func (fv *FakeVolume) WaitForDetach(devicePath string, timeout time.Duration) error {
	fv.WaitForDetachCallCount++
	return nil
}

func (fv *FakeVolume) UnmountDevice(globalMountPath string, mounter mount.Interface) error {
	fv.UnmountDeviceCallCount++
	return nil
}

type fakeRecycler struct {
	path string
	MetricsNil
}

func (fr *fakeRecycler) Recycle() error {
	// nil is success, else error
	return nil
}

func (fr *fakeRecycler) GetPath() string {
	return fr.path
}

func NewFakeRecycler(spec *Spec, host VolumeHost, config VolumeConfig) (Recycler, error) {
	if spec.PersistentVolume == nil || spec.PersistentVolume.Spec.HostPath == nil {
		return nil, fmt.Errorf("fakeRecycler only supports spec.PersistentVolume.Spec.HostPath")
	}
	return &fakeRecycler{
		path: spec.PersistentVolume.Spec.HostPath.Path,
	}, nil
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
	Options VolumeOptions
	Host    VolumeHost
}

func (fc *FakeProvisioner) NewPersistentVolumeTemplate() (*api.PersistentVolume, error) {
	fullpath := fmt.Sprintf("/tmp/hostpath_pv/%s", util.NewUUID())
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-fakeplugin-",
			Annotations: map[string]string{
				"kubernetes.io/createdby": "fakeplugin-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: fc.Options.PersistentVolumeReclaimPolicy,
			AccessModes:                   fc.Options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): fc.Options.Capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				HostPath: &api.HostPathVolumeSource{
					Path: fullpath,
				},
			},
		},
	}, nil
}

func (fc *FakeProvisioner) Provision(pv *api.PersistentVolume) error {
	return nil
}

// FindEmptyDirectoryUsageOnTmpfs finds the expected usage of an empty directory existing on
// a tmpfs filesystem on this system.
func FindEmptyDirectoryUsageOnTmpfs() (*resource.Quantity, error) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), "metrics_du_test")
	if err != nil {
		return nil, err
	}
	out, err := exec.Command("nice", "-n", "19", "du", "-s", "-B", "1", tmpDir).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed command 'du' on %s with error %v", tmpDir, err)
	}
	used, err := resource.ParseQuantity(strings.Fields(string(out))[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse 'du' output %s due to error %v", out, err)
	}
	used.Format = resource.BinarySI
	return used, nil
}
