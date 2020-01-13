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

package emptydir

import (
	"fmt"
	"os"
	"path/filepath"

	"k8s.io/klog"
	"k8s.io/utils/mount"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/fsquota"
)

// TODO: in the near future, this will be changed to be more restrictive
// and the group will be set to allow containers to use emptyDir volumes
// from the group attribute.
//
// http://issue.k8s.io/2630
const perm os.FileMode = 0777

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{
		&emptyDirPlugin{nil},
	}
}

type emptyDirPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &emptyDirPlugin{}

const (
	emptyDirPluginName           = "kubernetes.io/empty-dir"
	hugePagesPageSizeMountOption = "pagesize"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(emptyDirPluginName), volName)
}

func (plugin *emptyDirPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host

	return nil
}

func (plugin *emptyDirPlugin) GetPluginName() string {
	return emptyDirPluginName
}

func (plugin *emptyDirPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _ := getVolumeSource(spec)
	if volumeSource == nil {
		return "", fmt.Errorf("Spec does not reference an EmptyDir volume type")
	}

	// Return user defined volume name, since this is an ephemeral volume type
	return spec.Name(), nil
}

func (plugin *emptyDirPlugin) CanSupport(spec *volume.Spec) bool {
	if spec.Volume != nil && spec.Volume.EmptyDir != nil {
		return true
	}
	return false
}

func (plugin *emptyDirPlugin) RequiresRemount() bool {
	return false
}

func (plugin *emptyDirPlugin) SupportsMountOption() bool {
	return false
}

func (plugin *emptyDirPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *emptyDirPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod, plugin.host.GetMounter(plugin.GetPluginName()), &realMountDetector{plugin.host.GetMounter(plugin.GetPluginName())}, opts)
}

func (plugin *emptyDirPlugin) newMounterInternal(spec *volume.Spec, pod *v1.Pod, mounter mount.Interface, mountDetector mountDetector, opts volume.VolumeOptions) (volume.Mounter, error) {
	medium := v1.StorageMediumDefault

	if spec.Volume.EmptyDir != nil { // Support a non-specified source as EmptyDir.
		medium = spec.Volume.EmptyDir.Medium
	}

	return &emptyDir{
		pod:             pod,
		volName:         spec.Name(),
		medium:          medium,
		mounter:         mounter,
		mountDetector:   mountDetector,
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsDu(getPath(pod.UID, spec.Name(), plugin.host)),
	}, nil
}

func (plugin *emptyDirPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter(plugin.GetPluginName()), &realMountDetector{plugin.host.GetMounter(plugin.GetPluginName())})
}

func (plugin *emptyDirPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface, mountDetector mountDetector) (volume.Unmounter, error) {
	ed := &emptyDir{
		pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}},
		volName:         volName,
		medium:          v1.StorageMediumDefault, // might be changed later
		mounter:         mounter,
		mountDetector:   mountDetector,
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsDu(getPath(podUID, volName, plugin.host)),
	}
	return ed, nil
}

func (plugin *emptyDirPlugin) ConstructVolumeSpec(volName, mountPath string) (*volume.Spec, error) {
	emptyDirVolume := &v1.Volume{
		Name: volName,
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	}
	return volume.NewSpecFromVolume(emptyDirVolume), nil
}

// mountDetector abstracts how to find what kind of mount a path is backed by.
type mountDetector interface {
	// GetMountMedium determines what type of medium a given path is backed
	// by and whether that path is a mount point.  For example, if this
	// returns (v1.StorageMediumMemory, false, nil), the caller knows that the path is
	// on a memory FS (tmpfs on Linux) but is not the root mountpoint of
	// that tmpfs.
	GetMountMedium(path string) (v1.StorageMedium, bool, error)
}

// EmptyDir volumes are temporary directories exposed to the pod.
// These do not persist beyond the lifetime of a pod.
type emptyDir struct {
	pod           *v1.Pod
	volName       string
	medium        v1.StorageMedium
	mounter       mount.Interface
	mountDetector mountDetector
	plugin        *emptyDirPlugin
	volume.MetricsProvider
}

func (ed *emptyDir) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        false,
		Managed:         true,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (ed *emptyDir) CanMount() error {
	return nil
}

// SetUp creates new directory.
func (ed *emptyDir) SetUp(mounterArgs volume.MounterArgs) error {
	return ed.SetUpAt(ed.GetPath(), mounterArgs)
}

// SetUpAt creates new directory.
func (ed *emptyDir) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	notMnt, err := ed.mounter.IsLikelyNotMountPoint(dir)
	// Getting an os.IsNotExist err from is a contingency; the directory
	// may not exist yet, in which case, setup should run.
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	// If the plugin readiness file is present for this volume, and the
	// storage medium is the default, then the volume is ready.  If the
	// medium is memory, and a mountpoint is present, then the volume is
	// ready.
	if volumeutil.IsReady(ed.getMetaDir()) {
		if ed.medium == v1.StorageMediumMemory && !notMnt {
			return nil
		} else if ed.medium == v1.StorageMediumDefault {
			return nil
		}
	}

	switch ed.medium {
	case v1.StorageMediumDefault:
		err = ed.setupDir(dir)
	case v1.StorageMediumMemory:
		err = ed.setupTmpfs(dir)
	case v1.StorageMediumHugePages:
		err = ed.setupHugepages(dir)
	default:
		err = fmt.Errorf("unknown storage medium %q", ed.medium)
	}

	volume.SetVolumeOwnership(ed, mounterArgs.FsGroup)

	// If setting up the quota fails, just log a message but don't actually error out.
	// We'll use the old du mechanism in this case, at least until we support
	// enforcement.
	if err == nil {
		volumeutil.SetReady(ed.getMetaDir())
		if mounterArgs.DesiredSize != nil {
			// Deliberately shadow the outer use of err as noted
			// above.
			hasQuotas, err := fsquota.SupportsQuotas(ed.mounter, dir)
			if err != nil {
				klog.V(3).Infof("Unable to check for quota support on %s: %s", dir, err.Error())
			} else if hasQuotas {
				klog.V(4).Infof("emptydir trying to assign quota %v on %s", mounterArgs.DesiredSize, dir)
				err := fsquota.AssignQuota(ed.mounter, dir, ed.pod.UID, mounterArgs.DesiredSize)
				if err != nil {
					klog.V(3).Infof("Set quota on %s failed %s", dir, err.Error())
				}
			}
		}
	}
	return err
}

// setupTmpfs creates a tmpfs mount at the specified directory.
func (ed *emptyDir) setupTmpfs(dir string) error {
	if ed.mounter == nil {
		return fmt.Errorf("memory storage requested, but mounter is nil")
	}
	if err := ed.setupDir(dir); err != nil {
		return err
	}
	// Make SetUp idempotent.
	medium, isMnt, err := ed.mountDetector.GetMountMedium(dir)
	if err != nil {
		return err
	}
	// If the directory is a mountpoint with medium memory, there is no
	// work to do since we are already in the desired state.
	if isMnt && medium == v1.StorageMediumMemory {
		return nil
	}

	klog.V(3).Infof("pod %v: mounting tmpfs for volume %v", ed.pod.UID, ed.volName)
	return ed.mounter.Mount("tmpfs", dir, "tmpfs", nil /* options */)
}

// setupHugepages creates a hugepage mount at the specified directory.
func (ed *emptyDir) setupHugepages(dir string) error {
	if ed.mounter == nil {
		return fmt.Errorf("memory storage requested, but mounter is nil")
	}
	if err := ed.setupDir(dir); err != nil {
		return err
	}
	// Make SetUp idempotent.
	medium, isMnt, err := ed.mountDetector.GetMountMedium(dir)
	if err != nil {
		return err
	}
	// If the directory is a mountpoint with medium hugepages, there is no
	// work to do since we are already in the desired state.
	if isMnt && medium == v1.StorageMediumHugePages {
		return nil
	}

	pageSizeMountOption, err := getPageSizeMountOptionFromPod(ed.pod)
	if err != nil {
		return err
	}

	klog.V(3).Infof("pod %v: mounting hugepages for volume %v", ed.pod.UID, ed.volName)
	return ed.mounter.Mount("nodev", dir, "hugetlbfs", []string{pageSizeMountOption})
}

// getPageSizeMountOptionFromPod retrieves pageSize mount option from Pod's resources
// and validates pageSize options in all containers of given Pod.
func getPageSizeMountOptionFromPod(pod *v1.Pod) (string, error) {
	pageSizeFound := false
	pageSize := resource.Quantity{}
	// In some rare cases init containers can also consume Huge pages.
	containers := append(pod.Spec.Containers, pod.Spec.InitContainers...)
	for _, container := range containers {
		// We can take request because limit and requests must match.
		for requestName := range container.Resources.Requests {
			if v1helper.IsHugePageResourceName(requestName) {
				currentPageSize, err := v1helper.HugePageSizeFromResourceName(requestName)
				if err != nil {
					return "", err
				}
				// PageSize for all volumes in a POD are equal, except for the first one discovered.
				if pageSizeFound && pageSize.Cmp(currentPageSize) != 0 {
					return "", fmt.Errorf("multiple pageSizes for huge pages in a single PodSpec")
				}
				pageSize = currentPageSize
				pageSizeFound = true
			}
		}
	}

	if !pageSizeFound {
		return "", fmt.Errorf("hugePages storage requested, but there is no resource request for huge pages")
	}

	return fmt.Sprintf("%s=%s", hugePagesPageSizeMountOption, pageSize.String()), nil

}

// setupDir creates the directory with the default permissions specified by the perm constant.
func (ed *emptyDir) setupDir(dir string) error {
	// Create the directory if it doesn't already exist.
	if err := os.MkdirAll(dir, perm); err != nil {
		return err
	}

	// stat the directory to read permission bits
	fileinfo, err := os.Lstat(dir)
	if err != nil {
		return err
	}

	if fileinfo.Mode().Perm() != perm.Perm() {
		// If the permissions on the created directory are wrong, the
		// kubelet is probably running with a umask set.  In order to
		// avoid clearing the umask for the entire process or locking
		// the thread, clearing the umask, creating the dir, restoring
		// the umask, and unlocking the thread, we do a chmod to set
		// the specific bits we need.
		err := os.Chmod(dir, perm)
		if err != nil {
			return err
		}

		fileinfo, err = os.Lstat(dir)
		if err != nil {
			return err
		}

		if fileinfo.Mode().Perm() != perm.Perm() {
			klog.Errorf("Expected directory %q permissions to be: %s; got: %s", dir, perm.Perm(), fileinfo.Mode().Perm())
		}
	}

	return nil
}

func (ed *emptyDir) GetPath() string {
	return getPath(ed.pod.UID, ed.volName, ed.plugin.host)
}

// TearDown simply discards everything in the directory.
func (ed *emptyDir) TearDown() error {
	return ed.TearDownAt(ed.GetPath())
}

// TearDownAt simply discards everything in the directory.
func (ed *emptyDir) TearDownAt(dir string) error {
	if pathExists, pathErr := mount.PathExists(dir); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", dir)
		return nil
	}

	// Figure out the medium.
	medium, isMnt, err := ed.mountDetector.GetMountMedium(dir)
	if err != nil {
		return err
	}
	if isMnt {
		if medium == v1.StorageMediumMemory {
			ed.medium = v1.StorageMediumMemory
			return ed.teardownTmpfsOrHugetlbfs(dir)
		} else if medium == v1.StorageMediumHugePages {
			ed.medium = v1.StorageMediumHugePages
			return ed.teardownTmpfsOrHugetlbfs(dir)
		}
	}
	// assume StorageMediumDefault
	return ed.teardownDefault(dir)
}

func (ed *emptyDir) teardownDefault(dir string) error {
	// Remove any quota
	err := fsquota.ClearQuota(ed.mounter, dir)
	if err != nil {
		klog.Warningf("Warning: Failed to clear quota on %s: %v", dir, err)
	}
	// Renaming the directory is not required anymore because the operation executor
	// now handles duplicate operations on the same volume
	err = os.RemoveAll(dir)
	if err != nil {
		return err
	}
	return nil
}

func (ed *emptyDir) teardownTmpfsOrHugetlbfs(dir string) error {
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

func (ed *emptyDir) getMetaDir() string {
	return filepath.Join(ed.plugin.host.GetPodPluginDir(ed.pod.UID, utilstrings.EscapeQualifiedName(emptyDirPluginName)), ed.volName)
}

func getVolumeSource(spec *volume.Spec) (*v1.EmptyDirVolumeSource, bool) {
	var readOnly bool
	var volumeSource *v1.EmptyDirVolumeSource

	if spec.Volume != nil && spec.Volume.EmptyDir != nil {
		volumeSource = spec.Volume.EmptyDir
		readOnly = spec.ReadOnly
	}

	return volumeSource, readOnly
}
