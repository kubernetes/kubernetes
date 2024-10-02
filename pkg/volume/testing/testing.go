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
	"os"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/utils/exec"
	testingexec "k8s.io/utils/exec/testing"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/recyclerclient"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

const (
	// A hook specified in storage class to indicate it's provisioning
	// is expected to fail.
	ExpectProvisionFailureKey = "expect-provision-failure"
	// The node is marked as uncertain. The attach operation will fail and return timeout error
	// for the first attach call. The following call will return successfully.
	UncertainAttachNode = "uncertain-attach-node"
	// The detach operation will keep failing on the node.
	FailDetachNode = "fail-detach-node"
	// The node is marked as timeout. The attach operation will always fail and return timeout error
	// but the operation is actually succeeded.
	TimeoutAttachNode = "timeout-attach-node"
	// The node is marked as multi-attach which means it is allowed to attach the volume to multiple nodes.
	MultiAttachNode = "multi-attach-node"
	// TimeoutOnSetupVolumeName will cause Setup call to timeout but volume will finish mounting.
	TimeoutOnSetupVolumeName = "timeout-setup-volume"
	// FailOnSetupVolumeName will cause setup call to fail
	FailOnSetupVolumeName = "fail-setup-volume"
	//TimeoutAndFailOnSetupVolumeName will first timeout and then fail the setup
	TimeoutAndFailOnSetupVolumeName = "timeout-and-fail-setup-volume"
	// SuccessAndTimeoutSetupVolumeName will cause first mount operation to succeed but subsequent attempts to timeout
	SuccessAndTimeoutSetupVolumeName = "success-and-timeout-setup-volume-name"
	// SuccessAndFailOnSetupVolumeName will cause first mount operation to succeed but subsequent attempts to fail
	SuccessAndFailOnSetupVolumeName = "success-and-failed-setup-device-name"

	// TimeoutOnMountDeviceVolumeName will cause MountDevice call to timeout but Setup will finish.
	TimeoutOnMountDeviceVolumeName = "timeout-mount-device-volume"
	// TimeoutAndFailOnMountDeviceVolumeName will cause first MountDevice call to timeout but second call will fail
	TimeoutAndFailOnMountDeviceVolumeName = "timeout-and-fail-mount-device-name"
	// FailMountDeviceVolumeName will cause MountDevice operation on volume to fail
	FailMountDeviceVolumeName = "fail-mount-device-volume-name"
	// SuccessAndTimeoutDeviceName will cause first mount operation to succeed but subsequent attempts to timeout
	SuccessAndTimeoutDeviceName = "success-and-timeout-device-name"
	// SuccessAndFailOnMountDeviceName will cause first mount operation to succeed but subsequent attempts to fail
	SuccessAndFailOnMountDeviceName = "success-and-failed-mount-device-name"

	// FailWithInUseVolumeName will cause NodeExpandVolume to result in FailedPrecondition error
	FailWithInUseVolumeName       = "fail-expansion-in-use"
	FailWithUnSupportedVolumeName = "fail-expansion-unsupported"

	FailVolumeExpansion = "fail-expansion-test"

	InfeasibleNodeExpansion      = "infeasible-fail-node-expansion"
	OtherFinalNodeExpansionError = "other-final-node-expansion-error"

	deviceNotMounted     = "deviceNotMounted"
	deviceMountUncertain = "deviceMountUncertain"
	deviceMounted        = "deviceMounted"

	volumeNotMounted     = "volumeNotMounted"
	volumeMountUncertain = "volumeMountUncertain"
	volumeMounted        = "volumeMounted"

	FailNewMounter = "fail-new-mounter"
)

// CommandScript is used to pre-configure a command that will be executed and
// optionally set it's output (stdout and stderr combined) and return code.
type CommandScript struct {
	// Cmd is the command to execute, e.g. "ls"
	Cmd string
	// Args is a slice of arguments to pass to the command, e.g. "-a"
	Args []string
	// Output is the combined stdout and stderr of the command to return
	Output string
	// ReturnCode is the exit code for the command. Setting this to non-zero will
	// cause the command to return an error with this exit code set.
	ReturnCode int
}

// ScriptCommands configures fe, the FakeExec, to have a pre-configured list of
// commands to expect. Calling more commands using fe than those scripted will
// result in a panic. By default, the fe does not enforce command argument checking
// or order -- if you have given an Output to the command, the first command scripted
// will return its output on the first command call, even if the command called is
// different than the one scripted. This is mostly useful to make sure that the
// right number of commands were called. If you want to check the exact commands
// and arguments were called, set fe.ExectOrder to true.
func ScriptCommands(fe *testingexec.FakeExec, scripts []CommandScript) {
	fe.DisableScripts = false
	for _, script := range scripts {
		fakeCmd := &testingexec.FakeCmd{}
		cmdAction := makeFakeCmd(fakeCmd, script.Cmd, script.Args...)
		outputAction := makeFakeOutput(script.Output, script.ReturnCode)
		fakeCmd.CombinedOutputScript = append(fakeCmd.CombinedOutputScript, outputAction)
		fe.CommandScript = append(fe.CommandScript, cmdAction)
	}
}

func makeFakeCmd(fakeCmd *testingexec.FakeCmd, cmd string, args ...string) testingexec.FakeCommandAction {
	fc := fakeCmd
	c := cmd
	a := args
	return func(cmd string, args ...string) exec.Cmd {
		command := testingexec.InitFakeCmd(fc, c, a...)
		return command
	}
}

func makeFakeOutput(output string, rc int) testingexec.FakeAction {
	o := output
	var e error
	if rc != 0 {
		e = testingexec.FakeExitError{Status: rc}
	}
	return func() ([]byte, []byte, error) {
		return []byte(o), nil, e
	}
}

func ProbeVolumePlugins(config volume.VolumeConfig) []volume.VolumePlugin {
	if _, ok := config.OtherAttributes["fake-property"]; ok {
		return []volume.VolumePlugin{
			&FakeVolumePlugin{
				PluginName: "fake-plugin",
				Host:       nil,
				// SomeFakeProperty: config.OtherAttributes["fake-property"] -- string, may require parsing by plugin
			},
		}
	}
	return []volume.VolumePlugin{&FakeVolumePlugin{PluginName: "fake-plugin"}}
}

// FakeVolumePlugin is useful for testing.  It tries to be a fully compliant
// plugin, but all it does is make empty directories.
// Use as:
//
//	volume.RegisterPlugin(&FakePlugin{"fake-name"})
type FakeVolumePlugin struct {
	sync.RWMutex
	PluginName             string
	Host                   volume.VolumeHost
	Config                 volume.VolumeConfig
	LastProvisionerOptions volume.VolumeOptions
	LastResizeOptions      volume.NodeResizeOptions
	NewAttacherCallCount   int
	NewDetacherCallCount   int
	NodeExpandCallCount    int
	VolumeLimits           map[string]int64
	VolumeLimitsError      error
	LimitKey               string
	ProvisionDelaySeconds  int
	SupportsRemount        bool
	SupportsSELinux        bool
	DisableNodeExpansion   bool
	CanSupportFn           func(*volume.Spec) bool

	// default to false which means it is attachable by default
	NonAttachable bool

	// Add callbacks as needed
	WaitForAttachHook func(spec *volume.Spec, devicePath string, pod *v1.Pod, spectimeout time.Duration) (string, error)
	UnmountDeviceHook func(globalMountPath string) error

	Mounters             []*FakeVolume
	Unmounters           []*FakeVolume
	Attachers            []*FakeVolume
	Detachers            []*FakeVolume
	BlockVolumeMappers   []*FakeVolume
	BlockVolumeUnmappers []*FakeVolume
}

var _ volume.VolumePlugin = &FakeVolumePlugin{}
var _ volume.BlockVolumePlugin = &FakeVolumePlugin{}
var _ volume.RecyclableVolumePlugin = &FakeVolumePlugin{}
var _ volume.DeletableVolumePlugin = &FakeVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &FakeVolumePlugin{}
var _ volume.AttachableVolumePlugin = &FakeVolumePlugin{}
var _ volume.DeviceMountableVolumePlugin = &FakeVolumePlugin{}
var _ volume.NodeExpandableVolumePlugin = &FakeVolumePlugin{}

func (plugin *FakeVolumePlugin) getFakeVolume(list *[]*FakeVolume) *FakeVolume {
	if list != nil {
		volumeList := *list
		if len(volumeList) > 0 {
			volume := volumeList[0]
			volume.Lock()
			defer volume.Unlock()
			volume.WaitForAttachHook = plugin.WaitForAttachHook
			volume.UnmountDeviceHook = plugin.UnmountDeviceHook
			return volume
		}
	}
	volume := &FakeVolume{
		WaitForAttachHook: plugin.WaitForAttachHook,
		UnmountDeviceHook: plugin.UnmountDeviceHook,
	}
	volume.VolumesAttached = make(map[string]sets.Set[string])
	volume.DeviceMountState = make(map[string]string)
	volume.VolumeMountState = make(map[string]string)
	if list != nil {
		*list = append(*list, volume)
	}
	return volume
}

func (plugin *FakeVolumePlugin) Init(host volume.VolumeHost) error {
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

func (plugin *FakeVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	var volumeName string
	if spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil {
		volumeName = spec.Volume.GCEPersistentDisk.PDName
	} else if spec.Volume != nil && spec.Volume.RBD != nil {
		volumeName = spec.Volume.RBD.RBDImage
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.GCEPersistentDisk != nil {
		volumeName = spec.PersistentVolume.Spec.GCEPersistentDisk.PDName
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		volumeName = spec.PersistentVolume.Spec.RBD.RBDImage
	} else if spec.Volume != nil && spec.Volume.CSI != nil {
		volumeName = spec.Volume.CSI.Driver
	}
	if volumeName == "" {
		volumeName = spec.Name()
	}
	return volumeName, nil
}

func (plugin *FakeVolumePlugin) CanSupport(spec *volume.Spec) bool {
	if plugin.CanSupportFn != nil {
		return plugin.CanSupportFn(spec)
	}

	return true
}

func (plugin *FakeVolumePlugin) RequiresRemount(spec *volume.Spec) bool {
	return plugin.SupportsRemount
}

func (plugin *FakeVolumePlugin) SupportsMountOption() bool {
	return true
}

func (plugin *FakeVolumePlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return plugin.SupportsSELinux, nil
}

func (plugin *FakeVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod) (volume.Mounter, error) {
	plugin.Lock()
	defer plugin.Unlock()
	if spec.Name() == FailNewMounter {
		return nil, fmt.Errorf("AlwaysFailNewMounter")
	}
	fakeVolume := plugin.getFakeVolume(&plugin.Mounters)
	fakeVolume.Lock()
	defer fakeVolume.Unlock()
	fakeVolume.PodUID = pod.UID
	fakeVolume.VolName = spec.Name()
	fakeVolume.Plugin = plugin
	fakeVolume.MetricsNil = volume.MetricsNil{}
	return fakeVolume, nil
}

func (plugin *FakeVolumePlugin) GetMounters() (Mounters []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.Mounters
}

func (plugin *FakeVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	plugin.Lock()
	defer plugin.Unlock()
	fakeVolume := plugin.getFakeVolume(&plugin.Unmounters)
	fakeVolume.Lock()
	defer fakeVolume.Unlock()
	fakeVolume.PodUID = podUID
	fakeVolume.VolName = volName
	fakeVolume.Plugin = plugin
	fakeVolume.MetricsNil = volume.MetricsNil{}
	return fakeVolume, nil
}

func (plugin *FakeVolumePlugin) GetUnmounters() (Unmounters []*FakeVolume) {
	plugin.RLock()
	defer plugin.RUnlock()
	return plugin.Unmounters
}

// Block volume support
func (plugin *FakeVolumePlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod) (volume.BlockVolumeMapper, error) {
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
func (plugin *FakeVolumePlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
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

func (plugin *FakeVolumePlugin) NewAttacher() (volume.Attacher, error) {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.NewAttacherCallCount = plugin.NewAttacherCallCount + 1
	return plugin.getFakeVolume(&plugin.Attachers), nil
}

func (plugin *FakeVolumePlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
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

func (plugin *FakeVolumePlugin) NewDetacher() (volume.Detacher, error) {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.NewDetacherCallCount = plugin.NewDetacherCallCount + 1
	detacher := plugin.getFakeVolume(&plugin.Detachers)
	attacherList := plugin.Attachers
	if len(attacherList) > 0 {
		detacherList := plugin.Detachers
		if len(detacherList) > 0 {
			detacherList[0].VolumesAttached = attacherList[0].VolumesAttached
		}

	}
	return detacher, nil
}

func (plugin *FakeVolumePlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
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

func (plugin *FakeVolumePlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return !plugin.NonAttachable, nil
}

func (plugin *FakeVolumePlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (plugin *FakeVolumePlugin) Recycle(pvName string, spec *volume.Spec, eventRecorder recyclerclient.RecycleEventRecorder) error {
	return nil
}

func (plugin *FakeVolumePlugin) NewDeleter(logger klog.Logger, spec *volume.Spec) (volume.Deleter, error) {
	return &FakeDeleter{"/attributesTransferredFromSpec", volume.MetricsNil{}}, nil
}

func (plugin *FakeVolumePlugin) NewProvisioner(logger klog.Logger, options volume.VolumeOptions) (volume.Provisioner, error) {
	plugin.Lock()
	defer plugin.Unlock()
	plugin.LastProvisionerOptions = options
	return &FakeProvisioner{options, plugin.Host, plugin.ProvisionDelaySeconds}, nil
}

func (plugin *FakeVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{}
}

func (plugin *FakeVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	return volume.ReconstructedVolume{
		Spec: &volume.Spec{
			Volume: &v1.Volume{
				Name: volumeName,
			},
		},
	}, nil
}

// Block volume support
func (plugin *FakeVolumePlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mountPath string) (*volume.Spec, error) {
	return &volume.Spec{
		Volume: &v1.Volume{
			Name: volumeName,
		},
	}, nil
}

func (plugin *FakeVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	return []string{}, nil
}

// Expandable volume support
func (plugin *FakeVolumePlugin) ExpandVolumeDevice(spec *volume.Spec, newSize resource.Quantity, oldSize resource.Quantity) (resource.Quantity, error) {
	return resource.Quantity{}, nil
}

func (plugin *FakeVolumePlugin) RequiresFSResize() bool {
	return !plugin.DisableNodeExpansion
}

func (plugin *FakeVolumePlugin) NodeExpand(resizeOptions volume.NodeResizeOptions) (bool, error) {
	plugin.NodeExpandCallCount++
	plugin.LastResizeOptions = resizeOptions
	if resizeOptions.VolumeSpec.Name() == FailWithInUseVolumeName {
		return false, volumetypes.NewFailedPreconditionError("volume-in-use")
	}
	if resizeOptions.VolumeSpec.Name() == FailWithUnSupportedVolumeName {
		return false, volumetypes.NewOperationNotSupportedError("volume-unsupported")
	}

	if resizeOptions.VolumeSpec.Name() == InfeasibleNodeExpansion {
		return false, volumetypes.NewInfeasibleError("infeasible-expansion")
	}

	if resizeOptions.VolumeSpec.Name() == OtherFinalNodeExpansionError {
		return false, fmt.Errorf("other-final-node-expansion-error")
	}

	if resizeOptions.VolumeSpec.Name() == FailVolumeExpansion {
		return false, fmt.Errorf("fail volume expansion for volume: %s", FailVolumeExpansion)
	}
	return true, nil
}

func (plugin *FakeVolumePlugin) GetVolumeLimits() (map[string]int64, error) {
	return plugin.VolumeLimits, plugin.VolumeLimitsError
}

func (plugin *FakeVolumePlugin) VolumeLimitKey(spec *volume.Spec) string {
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

func (f *FakeBasicVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	return f.Plugin.GetVolumeName(spec)
}

// CanSupport tests whether the plugin supports a given volume specification by
// testing volume spec name begins with plugin name or not.
// This is useful to choose plugin by volume in testing.
func (f *FakeBasicVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return strings.HasPrefix(spec.Name(), f.GetPluginName())
}

func (f *FakeBasicVolumePlugin) ConstructVolumeSpec(ame, mountPath string) (volume.ReconstructedVolume, error) {
	return f.Plugin.ConstructVolumeSpec(ame, mountPath)
}

func (f *FakeBasicVolumePlugin) Init(ost volume.VolumeHost) error {
	return f.Plugin.Init(ost)
}

func (f *FakeBasicVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod) (volume.Mounter, error) {
	return f.Plugin.NewMounter(spec, pod)
}

func (f *FakeBasicVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return f.Plugin.NewUnmounter(volName, podUID)
}

func (f *FakeBasicVolumePlugin) RequiresRemount(spec *volume.Spec) bool {
	return f.Plugin.RequiresRemount(spec)
}

func (f *FakeBasicVolumePlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return f.Plugin.SupportsSELinuxContextMount(spec)
}

func (f *FakeBasicVolumePlugin) SupportsMountOption() bool {
	return f.Plugin.SupportsMountOption()
}

var _ volume.VolumePlugin = &FakeBasicVolumePlugin{}

// FakeDeviceMountableVolumePlugin implements an device mountable plugin based on FakeBasicVolumePlugin.
type FakeDeviceMountableVolumePlugin struct {
	FakeBasicVolumePlugin
}

func (f *FakeDeviceMountableVolumePlugin) CanDeviceMount(spec *volume.Spec) (bool, error) {
	return true, nil
}

func (f *FakeDeviceMountableVolumePlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	return f.Plugin.NewDeviceMounter()
}

func (f *FakeDeviceMountableVolumePlugin) NewDeviceUnmounter() (volume.DeviceUnmounter, error) {
	return f.Plugin.NewDeviceUnmounter()
}

func (f *FakeDeviceMountableVolumePlugin) GetDeviceMountRefs(deviceMountPath string) ([]string, error) {
	return f.Plugin.GetDeviceMountRefs(deviceMountPath)
}

var _ volume.VolumePlugin = &FakeDeviceMountableVolumePlugin{}
var _ volume.DeviceMountableVolumePlugin = &FakeDeviceMountableVolumePlugin{}

// FakeAttachableVolumePlugin implements an attachable plugin based on FakeDeviceMountableVolumePlugin.
type FakeAttachableVolumePlugin struct {
	FakeDeviceMountableVolumePlugin
}

func (f *FakeAttachableVolumePlugin) NewAttacher() (volume.Attacher, error) {
	return f.Plugin.NewAttacher()
}

func (f *FakeAttachableVolumePlugin) NewDetacher() (volume.Detacher, error) {
	return f.Plugin.NewDetacher()
}

func (f *FakeAttachableVolumePlugin) CanAttach(spec *volume.Spec) (bool, error) {
	return true, nil
}

var _ volume.VolumePlugin = &FakeAttachableVolumePlugin{}
var _ volume.AttachableVolumePlugin = &FakeAttachableVolumePlugin{}

type FakeFileVolumePlugin struct {
}

func (plugin *FakeFileVolumePlugin) Init(host volume.VolumeHost) error {
	return nil
}

func (plugin *FakeFileVolumePlugin) GetPluginName() string {
	return "fake-file-plugin"
}

func (plugin *FakeFileVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	return "", nil
}

func (plugin *FakeFileVolumePlugin) CanSupport(spec *volume.Spec) bool {
	return true
}

func (plugin *FakeFileVolumePlugin) RequiresRemount(spec *volume.Spec) bool {
	return false
}

func (plugin *FakeFileVolumePlugin) SupportsMountOption() bool {
	return false
}

func (plugin *FakeFileVolumePlugin) SupportsSELinuxContextMount(spec *volume.Spec) (bool, error) {
	return false, nil
}

func (plugin *FakeFileVolumePlugin) NewMounter(spec *volume.Spec, podRef *v1.Pod) (volume.Mounter, error) {
	return nil, nil
}

func (plugin *FakeFileVolumePlugin) NewUnmounter(name string, podUID types.UID) (volume.Unmounter, error) {
	return nil, nil
}

func (plugin *FakeFileVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (volume.ReconstructedVolume, error) {
	return volume.ReconstructedVolume{}, nil
}

func NewFakeFileVolumePlugin() []volume.VolumePlugin {
	return []volume.VolumePlugin{&FakeFileVolumePlugin{}}
}

type FakeVolume struct {
	sync.RWMutex
	PodUID  types.UID
	VolName string
	Plugin  *FakeVolumePlugin
	volume.MetricsNil
	VolumesAttached  map[string]sets.Set[string]
	DeviceMountState map[string]string
	VolumeMountState map[string]string

	// Add callbacks as needed
	WaitForAttachHook func(spec *volume.Spec, devicePath string, pod *v1.Pod, spectimeout time.Duration) (string, error)
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
	MapPodDeviceCallCount       int
	UnmapPodDeviceCallCount     int
	GlobalMapPathCallCount      int
	PodDeviceMapPathCallCount   int
}

func getUniqueVolumeName(spec *volume.Spec) (string, error) {
	var volumeName string
	if spec.Volume != nil && spec.Volume.GCEPersistentDisk != nil {
		volumeName = spec.Volume.GCEPersistentDisk.PDName
	} else if spec.Volume != nil && spec.Volume.RBD != nil {
		volumeName = spec.Volume.RBD.RBDImage
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.GCEPersistentDisk != nil {
		volumeName = spec.PersistentVolume.Spec.GCEPersistentDisk.PDName
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		volumeName = spec.PersistentVolume.Spec.RBD.RBDImage
	}
	if volumeName == "" {
		volumeName = spec.Name()
	}
	return volumeName, nil
}

func (_ *FakeVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:       false,
		Managed:        true,
		SELinuxRelabel: true,
	}
}

func (fv *FakeVolume) SetUp(mounterArgs volume.MounterArgs) error {
	fv.Lock()
	defer fv.Unlock()
	err := fv.setupInternal(mounterArgs)
	fv.SetUpCallCount++
	return err
}

func (fv *FakeVolume) setupInternal(mounterArgs volume.MounterArgs) error {
	if fv.VolName == TimeoutOnSetupVolumeName {
		fv.VolumeMountState[fv.VolName] = volumeMountUncertain
		return volumetypes.NewUncertainProgressError("time out on setup")
	}

	if fv.VolName == FailOnSetupVolumeName {
		fv.VolumeMountState[fv.VolName] = volumeNotMounted
		return fmt.Errorf("mounting volume failed")
	}

	if fv.VolName == TimeoutAndFailOnSetupVolumeName {
		_, ok := fv.VolumeMountState[fv.VolName]
		if !ok {
			fv.VolumeMountState[fv.VolName] = volumeMountUncertain
			return volumetypes.NewUncertainProgressError("time out on setup")
		}
		fv.VolumeMountState[fv.VolName] = volumeNotMounted
		return fmt.Errorf("mounting volume failed")

	}

	if fv.VolName == SuccessAndFailOnSetupVolumeName {
		_, ok := fv.VolumeMountState[fv.VolName]
		if ok {
			fv.VolumeMountState[fv.VolName] = volumeNotMounted
			return fmt.Errorf("mounting volume failed")
		}
	}

	if fv.VolName == SuccessAndTimeoutSetupVolumeName {
		_, ok := fv.VolumeMountState[fv.VolName]
		if ok {
			fv.VolumeMountState[fv.VolName] = volumeMountUncertain
			return volumetypes.NewUncertainProgressError("time out on setup")
		}
	}

	fv.VolumeMountState[fv.VolName] = volumeNotMounted
	return fv.SetUpAt(fv.getPath(), mounterArgs)
}

func (fv *FakeVolume) GetSetUpCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.SetUpCallCount
}

func (fv *FakeVolume) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
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
	if fv.VolName == TimeoutOnMountDeviceVolumeName {
		fv.DeviceMountState[fv.VolName] = deviceMountUncertain
		return "", volumetypes.NewUncertainProgressError("mount failed")
	}
	if fv.VolName == FailMountDeviceVolumeName {
		fv.DeviceMountState[fv.VolName] = deviceNotMounted
		return "", fmt.Errorf("error mapping disk: %s", fv.VolName)
	}

	if fv.VolName == TimeoutAndFailOnMountDeviceVolumeName {
		_, ok := fv.DeviceMountState[fv.VolName]
		if !ok {
			fv.DeviceMountState[fv.VolName] = deviceMountUncertain
			return "", volumetypes.NewUncertainProgressError("timed out mounting error")
		}
		fv.DeviceMountState[fv.VolName] = deviceNotMounted
		return "", fmt.Errorf("error mapping disk: %s", fv.VolName)
	}

	if fv.VolName == SuccessAndTimeoutDeviceName {
		_, ok := fv.DeviceMountState[fv.VolName]
		if ok {
			fv.DeviceMountState[fv.VolName] = deviceMountUncertain
			return "", volumetypes.NewUncertainProgressError("error mounting state")
		}
	}
	if fv.VolName == SuccessAndFailOnMountDeviceName {
		_, ok := fv.DeviceMountState[fv.VolName]
		if ok {
			return "", fmt.Errorf("error mapping disk: %s", fv.VolName)
		}
	}

	fv.DeviceMountState[fv.VolName] = deviceMounted
	fv.SetUpDeviceCallCount++

	return "", nil
}

func (fv *FakeVolume) GetStagingPath() string {
	return filepath.Join(fv.Plugin.Host.GetVolumeDevicePluginDir(utilstrings.EscapeQualifiedName(fv.Plugin.PluginName)), "staging", fv.VolName)
}

// Block volume support
func (fv *FakeVolume) GetSetUpDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.SetUpDeviceCallCount
}

// Block volume support
func (fv *FakeVolume) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	fv.Lock()
	defer fv.Unlock()
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
func (fv *FakeVolume) UnmapPodDevice() error {
	fv.Lock()
	defer fv.Unlock()
	fv.UnmapPodDeviceCallCount++
	return nil
}

// Block volume support
func (fv *FakeVolume) GetUnmapPodDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.UnmapPodDeviceCallCount
}

// Block volume support
func (fv *FakeVolume) MapPodDevice() (string, error) {
	fv.Lock()
	defer fv.Unlock()

	if fv.VolName == TimeoutOnSetupVolumeName {
		fv.VolumeMountState[fv.VolName] = volumeMountUncertain
		return "", volumetypes.NewUncertainProgressError("time out on setup")
	}

	if fv.VolName == FailOnSetupVolumeName {
		fv.VolumeMountState[fv.VolName] = volumeNotMounted
		return "", fmt.Errorf("mounting volume failed")
	}

	if fv.VolName == TimeoutAndFailOnSetupVolumeName {
		_, ok := fv.VolumeMountState[fv.VolName]
		if !ok {
			fv.VolumeMountState[fv.VolName] = volumeMountUncertain
			return "", volumetypes.NewUncertainProgressError("time out on setup")
		}
		fv.VolumeMountState[fv.VolName] = volumeNotMounted
		return "", fmt.Errorf("mounting volume failed")

	}

	if fv.VolName == SuccessAndFailOnSetupVolumeName {
		_, ok := fv.VolumeMountState[fv.VolName]
		if ok {
			fv.VolumeMountState[fv.VolName] = volumeNotMounted
			return "", fmt.Errorf("mounting volume failed")
		}
	}

	if fv.VolName == SuccessAndTimeoutSetupVolumeName {
		_, ok := fv.VolumeMountState[fv.VolName]
		if ok {
			fv.VolumeMountState[fv.VolName] = volumeMountUncertain
			return "", volumetypes.NewUncertainProgressError("time out on setup")
		}
	}

	fv.VolumeMountState[fv.VolName] = volumeMounted
	fv.MapPodDeviceCallCount++
	return "", nil
}

// Block volume support
func (fv *FakeVolume) GetMapPodDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.MapPodDeviceCallCount
}

func (fv *FakeVolume) Attach(spec *volume.Spec, nodeName types.NodeName) (string, error) {
	fv.Lock()
	defer fv.Unlock()
	fv.AttachCallCount++

	volumeName, err := getUniqueVolumeName(spec)
	if err != nil {
		return "", err
	}
	volumeNodes, exist := fv.VolumesAttached[volumeName]
	if exist {
		if nodeName == UncertainAttachNode {
			return "/dev/vdb-test", nil
		}
		// even if volume was previously attached to time out, we need to keep returning error
		// so as reconciler can not confirm this volume as attached.
		if nodeName == TimeoutAttachNode {
			return "", fmt.Errorf("timed out to attach volume %q to node %q", volumeName, nodeName)
		}
		if volumeNodes.Has(string(nodeName)) || volumeNodes.Has(MultiAttachNode) || nodeName == MultiAttachNode {
			volumeNodes.Insert(string(nodeName))
			return "/dev/vdb-test", nil
		}
		return "", fmt.Errorf("volume %q trying to attach to node %q is already attached to node %q", volumeName, nodeName, volumeNodes)
	}

	fv.VolumesAttached[volumeName] = sets.New[string](string(nodeName))
	if nodeName == UncertainAttachNode || nodeName == TimeoutAttachNode {
		return "", fmt.Errorf("timed out to attach volume %q to node %q", volumeName, nodeName)
	}
	return "/dev/vdb-test", nil
}

func (fv *FakeVolume) GetAttachCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.AttachCallCount
}

func (fv *FakeVolume) WaitForAttach(spec *volume.Spec, devicePath string, pod *v1.Pod, spectimeout time.Duration) (string, error) {
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

func (fv *FakeVolume) GetDeviceMountPath(spec *volume.Spec) (string, error) {
	fv.Lock()
	defer fv.Unlock()
	fv.GetDeviceMountPathCallCount++
	return "", nil
}

func (fv *FakeVolume) mountDeviceInternal(spec *volume.Spec, devicePath string, deviceMountPath string) error {
	fv.Lock()
	defer fv.Unlock()
	if spec.Name() == TimeoutOnMountDeviceVolumeName {
		fv.DeviceMountState[spec.Name()] = deviceMountUncertain
		return volumetypes.NewUncertainProgressError("mount failed")
	}

	if spec.Name() == FailMountDeviceVolumeName {
		fv.DeviceMountState[spec.Name()] = deviceNotMounted
		return fmt.Errorf("error mounting disk: %s", devicePath)
	}

	if spec.Name() == TimeoutAndFailOnMountDeviceVolumeName {
		_, ok := fv.DeviceMountState[spec.Name()]
		if !ok {
			fv.DeviceMountState[spec.Name()] = deviceMountUncertain
			return volumetypes.NewUncertainProgressError("timed out mounting error")
		}
		fv.DeviceMountState[spec.Name()] = deviceNotMounted
		return fmt.Errorf("error mounting disk: %s", devicePath)
	}

	if spec.Name() == SuccessAndTimeoutDeviceName {
		_, ok := fv.DeviceMountState[spec.Name()]
		if ok {
			fv.DeviceMountState[spec.Name()] = deviceMountUncertain
			return volumetypes.NewUncertainProgressError("error mounting state")
		}
	}

	if spec.Name() == SuccessAndFailOnMountDeviceName {
		_, ok := fv.DeviceMountState[spec.Name()]
		if ok {
			return fmt.Errorf("error mounting disk: %s", devicePath)
		}
	}
	fv.DeviceMountState[spec.Name()] = deviceMounted
	fv.MountDeviceCallCount++
	return nil
}

func (fv *FakeVolume) MountDevice(spec *volume.Spec, devicePath string, deviceMountPath string, _ volume.DeviceMounterArgs) error {
	return fv.mountDeviceInternal(spec, devicePath, deviceMountPath)
}

func (fv *FakeVolume) GetMountDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.MountDeviceCallCount
}

func (fv *FakeVolume) GetUnmountDeviceCallCount() int {
	fv.RLock()
	defer fv.RUnlock()
	return fv.UnmountDeviceCallCount
}

func (fv *FakeVolume) Detach(volumeName string, nodeName types.NodeName) error {
	fv.Lock()
	defer fv.Unlock()

	node := string(nodeName)
	volumeNodes, exist := fv.VolumesAttached[volumeName]
	if !exist || !volumeNodes.Has(node) {
		return fmt.Errorf("trying to detach volume %q that is not attached to the node %q", volumeName, node)
	}

	fv.DetachCallCount++
	if nodeName == FailDetachNode {
		return fmt.Errorf("fail to detach volume %q to node %q", volumeName, nodeName)
	}

	volumeNodes.Delete(node)
	if volumeNodes.Len() == 0 {
		delete(fv.VolumesAttached, volumeName)
	}

	return nil
}

func (fv *FakeVolume) VolumesAreAttached(spec []*volume.Spec, nodeName types.NodeName) (map[*volume.Spec]bool, error) {
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
	volume.MetricsNil
}

func (fd *FakeDeleter) Delete() error {
	// nil is success, else error
	return nil
}

func (fd *FakeDeleter) GetPath() string {
	return fd.path
}

type FakeProvisioner struct {
	Options               volume.VolumeOptions
	Host                  volume.VolumeHost
	ProvisionDelaySeconds int
}

func (fc *FakeProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	// Add provision failure hook
	if fc.Options.Parameters != nil {
		if _, ok := fc.Options.Parameters[ExpectProvisionFailureKey]; ok {
			return nil, fmt.Errorf("expected error")
		}
	}
	fullpath := fmt.Sprintf("/%s/hostpath_pv/%s", os.TempDir(), uuid.NewUUID())

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

// NewDeviceHandler Create a new IoHandler implementation
func NewBlockVolumePathHandler() volumepathhandler.BlockVolumePathHandler {
	return &FakeVolumePathHandler{}
}

type FakeVolumePathHandler struct {
	sync.RWMutex
}

func (fv *FakeVolumePathHandler) MapDevice(devicePath string, mapDir string, linkName string, bindMount bool) error {
	// nil is success, else error
	return nil
}

func (fv *FakeVolumePathHandler) UnmapDevice(mapDir string, linkName string, bindMount bool) error {
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

func (fv *FakeVolumePathHandler) IsDeviceBindMountExist(mapPath string) (bool, error) {
	// nil is success, else error
	return true, nil
}

func (fv *FakeVolumePathHandler) GetDeviceBindMountRefs(devPath string, mapPath string) ([]string, error) {
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

func (fv *FakeVolumePathHandler) DetachFileDevice(path string) error {
	// nil is success, else error
	return nil
}

func (fv *FakeVolumePathHandler) GetLoopDevice(path string) (string, error) {
	// nil is success, else error
	return "/dev/loop1", nil
}

// FindEmptyDirectoryUsageOnTmpfs finds the expected usage of an empty directory existing on
// a tmpfs filesystem on this system.
func FindEmptyDirectoryUsageOnTmpfs() (*resource.Quantity, error) {
	// The command below does not exist on Windows. Additionally, empty folders have size 0 on Windows.
	if goruntime.GOOS == "windows" {
		used, err := resource.ParseQuantity("0")
		return &used, err
	}
	tmpDir, err := utiltesting.MkTmpdir("metrics_du_test")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	out, err := exec.New().Command("nice", "-n", "19", "du", "-x", "-s", "-B", "1", tmpDir).CombinedOutput()
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

func VerifyUnmountDeviceCallCount(expectedCallCount int, fakeVolumePlugin *FakeVolumePlugin) error {
	detachers := fakeVolumePlugin.GetDetachers()
	if len(detachers) == 0 && (expectedCallCount == 0) {
		return nil
	}
	actualCallCount := 0
	for _, detacher := range detachers {
		actualCallCount = detacher.GetUnmountDeviceCallCount()
		if expectedCallCount == 0 && actualCallCount == expectedCallCount {
			return nil
		}

		if (expectedCallCount > 0) && (actualCallCount >= expectedCallCount) {
			return nil
		}
	}

	return fmt.Errorf(
		"Expected DeviceUnmount Call %d, got %d",
		expectedCallCount, actualCallCount)
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
	unmounters := fakeVolumePlugin.GetUnmounters()
	if len(unmounters) == 0 && (expectedTearDownCallCount == 0) {
		return nil
	}

	for _, unmounter := range unmounters {
		actualCallCount := unmounter.GetTearDownCallCount()
		if expectedTearDownCallCount == 0 && actualCallCount == expectedTearDownCallCount {
			return nil
		}

		if (expectedTearDownCallCount > 0) && (actualCallCount >= expectedTearDownCallCount) {
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

// VerifyUnmapPodDeviceCallCount ensures that at least one of the Unmappers for this
// plugin has the expected number of UnmapPodDevice calls. Otherwise it
// returns an error.
func VerifyUnmapPodDeviceCallCount(
	expectedUnmapPodDeviceCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, unmapper := range fakeVolumePlugin.GetBlockVolumeUnmapper() {
		actualCallCount := unmapper.GetUnmapPodDeviceCallCount()
		if actualCallCount >= expectedUnmapPodDeviceCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Unmapper have expected UnmapPodDeviceCallCount. Expected: <%v>.",
		expectedUnmapPodDeviceCallCount)
}

// VerifyZeroUnmapPodDeviceCallCount ensures that all Mappers for this plugin have a
// zero UnmapPodDevice calls. Otherwise it returns an error.
func VerifyZeroUnmapPodDeviceCallCount(fakeVolumePlugin *FakeVolumePlugin) error {
	for _, unmapper := range fakeVolumePlugin.GetBlockVolumeUnmapper() {
		actualCallCount := unmapper.GetUnmapPodDeviceCallCount()
		if actualCallCount != 0 {
			return fmt.Errorf(
				"At least one unmapper has non-zero UnmapPodDeviceCallCount: <%v>.",
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

// VerifyGetMapPodDeviceCallCount ensures that at least one of the Mappers for this
// plugin has the expectedMapPodDeviceCallCount number of calls. Otherwise it
// returns an error.
func VerifyGetMapPodDeviceCallCount(
	expectedMapPodDeviceCallCount int,
	fakeVolumePlugin *FakeVolumePlugin) error {
	for _, mapper := range fakeVolumePlugin.GetBlockVolumeMapper() {
		actualCallCount := mapper.GetMapPodDeviceCallCount()
		if actualCallCount >= expectedMapPodDeviceCallCount {
			return nil
		}
	}

	return fmt.Errorf(
		"No Mapper have expected MapPodDeviceCallCount. Expected: <%v>.",
		expectedMapPodDeviceCallCount)
}

// GetTestVolumePluginMgr creates, initializes, and returns a test volume plugin
// manager and fake volume plugin using a fake volume host.
func GetTestVolumePluginMgr(t *testing.T) (*volume.VolumePluginMgr, *FakeVolumePlugin) {
	plugins := ProbeVolumePlugins(volume.VolumeConfig{})
	v := NewFakeVolumeHost(
		t,
		"",      /* rootDir */
		nil,     /* kubeClient */
		plugins, /* plugins */
	)
	return v.GetPluginMgr(), plugins[0].(*FakeVolumePlugin)
}

func GetTestKubeletVolumePluginMgr(t *testing.T) (*volume.VolumePluginMgr, *FakeVolumePlugin) {
	plugins := ProbeVolumePlugins(volume.VolumeConfig{})
	v := NewFakeKubeletVolumeHost(
		t,
		"",      /* rootDir */
		nil,     /* kubeClient */
		plugins, /* plugins */
	)
	return v.GetPluginMgr(), plugins[0].(*FakeVolumePlugin)
}

func GetTestKubeletVolumePluginMgrWithNode(t *testing.T, node *v1.Node) (*volume.VolumePluginMgr, *FakeVolumePlugin) {
	plugins := ProbeVolumePlugins(volume.VolumeConfig{})
	v := NewFakeKubeletVolumeHost(
		t,
		"",      /* rootDir */
		nil,     /* kubeClient */
		plugins, /* plugins */
	)
	v.WithNode(node)

	return v.GetPluginMgr(), plugins[0].(*FakeVolumePlugin)
}

func GetTestKubeletVolumePluginMgrWithNodeAndRoot(t *testing.T, node *v1.Node, rootDir string) (*volume.VolumePluginMgr, *FakeVolumePlugin) {
	plugins := ProbeVolumePlugins(volume.VolumeConfig{})
	v := NewFakeKubeletVolumeHost(
		t,
		rootDir, /* rootDir */
		nil,     /* kubeClient */
		plugins, /* plugins */
	)
	v.WithNode(node)

	return v.GetPluginMgr(), plugins[0].(*FakeVolumePlugin)
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
			Resources: v1.VolumeResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(capacity),
				},
			},
		},
	}
	return &claim
}

func MetricsEqualIgnoreTimestamp(a *volume.Metrics, b *volume.Metrics) bool {
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
