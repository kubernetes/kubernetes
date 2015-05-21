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

package empty_dir

import (
	"fmt"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&emptyDirPlugin{nil, false},
		&emptyDirPlugin{nil, true},
	}
}

type emptyDirPlugin struct {
	host       volume.VolumeHost
	legacyMode bool // if set, plugin answers to the legacy name
}

var _ volume.VolumePlugin = &emptyDirPlugin{}

const (
	emptyDirPluginName       = "kubernetes.io/empty-dir"
	emptyDirPluginLegacyName = "empty"
)

func (plugin *emptyDirPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *emptyDirPlugin) Name() string {
	if plugin.legacyMode {
		return emptyDirPluginLegacyName
	}
	return emptyDirPluginName
}

func (plugin *emptyDirPlugin) CanSupport(spec *volume.Spec) bool {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return false
	}

	if spec.VolumeSource.EmptyDir != nil {
		return true
	}
	return false
}

func (plugin *emptyDirPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, pod, mounter, &realMountDetector{mounter}, opts)
}

func (plugin *emptyDirPlugin) newBuilderInternal(spec *volume.Spec, pod *api.Pod, mounter mount.Interface, mountDetector mountDetector, opts volume.VolumeOptions) (volume.Builder, error) {
	if plugin.legacyMode {
		// Legacy mode instances can be cleaned up but not created anew.
		return nil, fmt.Errorf("legacy mode: can not create new instances")
	}
	medium := api.StorageMediumDefault
	if spec.VolumeSource.EmptyDir != nil { // Support a non-specified source as EmptyDir.
		medium = spec.VolumeSource.EmptyDir.Medium
	}
	return &emptyDir{
		podUID:        pod.UID,
		volName:       spec.Name,
		medium:        medium,
		mounter:       mounter,
		mountDetector: mountDetector,
		plugin:        plugin,
		legacyMode:    false,
		rootContext:   opts.RootContext,
	}, nil
}

func (plugin *emptyDirPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, mounter, &realMountDetector{mounter})
}

func (plugin *emptyDirPlugin) newCleanerInternal(volName string, podUID types.UID, mounter mount.Interface, mountDetector mountDetector) (volume.Cleaner, error) {
	legacy := false
	if plugin.legacyMode {
		legacy = true
	}
	ed := &emptyDir{
		podUID:        podUID,
		volName:       volName,
		medium:        api.StorageMediumDefault, // might be changed later
		mounter:       mounter,
		mountDetector: mountDetector,
		plugin:        plugin,
		legacyMode:    legacy,
	}
	return ed, nil
}

// mountDetector abstracts how to find what kind of mount a path is backed by.
type mountDetector interface {
	// GetMountMedium determines what type of medium a given path is backed
	// by and whether that path is a mount point.  For example, if this
	// returns (mediumMemory, false, nil), the caller knows that the path is
	// on a memory FS (tmpfs on Linux) but is not the root mountpoint of
	// that tmpfs.
	GetMountMedium(path string) (storageMedium, bool, error)
}

type storageMedium int

const (
	mediumUnknown storageMedium = 0 // assume anything we don't explicitly handle is this
	mediumMemory  storageMedium = 1 // memory (e.g. tmpfs on linux)
)

// EmptyDir volumes are temporary directories exposed to the pod.
// These do not persist beyond the lifetime of a pod.
type emptyDir struct {
	podUID        types.UID
	volName       string
	medium        api.StorageMedium
	mounter       mount.Interface
	mountDetector mountDetector
	plugin        *emptyDirPlugin
	legacyMode    bool
	rootContext   string
}

// SetUp creates new directory.
func (ed *emptyDir) SetUp() error {
	return ed.SetUpAt(ed.GetPath())
}

// SetUpAt creates new directory.
func (ed *emptyDir) SetUpAt(dir string) error {
	if ed.legacyMode {
		return fmt.Errorf("legacy mode: can not create new instances")
	}
	switch ed.medium {
	case api.StorageMediumDefault:
		return ed.setupDefault(dir)
	case api.StorageMediumMemory:
		return ed.setupTmpfs(dir)
	default:
		return fmt.Errorf("unknown storage medium %q", ed.medium)
	}
}

func (ed *emptyDir) setupDefault(dir string) error {
	return os.MkdirAll(dir, 0750)
}

func (ed *emptyDir) setupTmpfs(dir string) error {
	if ed.mounter == nil {
		return fmt.Errorf("memory storage requested, but mounter is nil")
	}
	if err := os.MkdirAll(dir, 0750); err != nil {
		return err
	}
	// Make SetUp idempotent.
	medium, isMnt, err := ed.mountDetector.GetMountMedium(dir)
	if err != nil {
		return err
	}
	// If the directory is a mountpoint with medium memory, there is no
	// work to do since we are already in the desired state.
	if isMnt && medium == mediumMemory {
		return nil
	}

	// By default a tmpfs mount will receive a different SELinux context
	// from that of the Kubelet root directory which is not readable from
	// the SELinux context of a docker container.
	//
	// getTmpfsMountOptions gets the mount option to set the context of
	// the tmpfs mount so that it can be read from the SELinux context of
	// the container.
	opts := ed.getTmpfsMountOptions()
	glog.V(3).Infof("pod %v: mounting tmpfs for volume %v with opts %v", ed.podUID, ed.volName, opts)
	return ed.mounter.Mount("tmpfs", dir, "tmpfs", opts)
}

func (ed *emptyDir) getTmpfsMountOptions() []string {
	if ed.rootContext == "" {
		return []string{""}
	}

	return []string{fmt.Sprintf("rootcontext=\"%v\"", ed.rootContext)}
}

func (ed *emptyDir) GetPath() string {
	name := emptyDirPluginName
	if ed.legacyMode {
		name = emptyDirPluginLegacyName
	}
	return ed.plugin.host.GetPodVolumeDir(ed.podUID, util.EscapeQualifiedNameForDisk(name), ed.volName)
}

// TearDown simply discards everything in the directory.
func (ed *emptyDir) TearDown() error {
	return ed.TearDownAt(ed.GetPath())
}

// TearDownAt simply discards everything in the directory.
func (ed *emptyDir) TearDownAt(dir string) error {
	// Figure out the medium.
	medium, isMnt, err := ed.mountDetector.GetMountMedium(dir)
	if err != nil {
		return err
	}
	if isMnt && medium == mediumMemory {
		ed.medium = api.StorageMediumMemory
		return ed.teardownTmpfs(dir)
	}
	// assume StorageMediumDefault
	return ed.teardownDefault(dir)
}

func (ed *emptyDir) teardownDefault(dir string) error {
	tmpDir, err := volume.RenameDirectory(dir, ed.volName+".deleting~")
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}

func (ed *emptyDir) teardownTmpfs(dir string) error {
	if ed.mounter == nil {
		return fmt.Errorf("memory storage requested, but mounter is nil")
	}
	if err := ed.mounter.Unmount(dir); err != nil {
		return err
	}
	if err := os.RemoveAll(dir); err != nil {
		return err
	}
	return nil
}
