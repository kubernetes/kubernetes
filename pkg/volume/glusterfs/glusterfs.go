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

package glusterfs

import (
	"fmt"
	"math"
	"os"
	"path"
	"runtime"
	"strconv"
	dstrings "strings"
	"sync"

	"github.com/golang/glog"
	gcli "github.com/heketi/heketi/client/api/go-client"
	gapi "github.com/heketi/heketi/pkg/glusterfs/api"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&glusterfsPlugin{host: nil, gidTable: make(map[string]*MinMaxAllocator)}}
}

type glusterfsPlugin struct {
	host         volume.VolumeHost
	gidTable     map[string]*MinMaxAllocator
	gidTableLock sync.Mutex
}

var _ volume.VolumePlugin = &glusterfsPlugin{}
var _ volume.PersistentVolumePlugin = &glusterfsPlugin{}
var _ volume.DeletableVolumePlugin = &glusterfsPlugin{}
var _ volume.ProvisionableVolumePlugin = &glusterfsPlugin{}
var _ volume.ExpandableVolumePlugin = &glusterfsPlugin{}
var _ volume.Provisioner = &glusterfsVolumeProvisioner{}
var _ volume.Deleter = &glusterfsVolumeDeleter{}

const (
	glusterfsPluginName            = "kubernetes.io/glusterfs"
	volPrefix                      = "vol_"
	dynamicEpSvcPrefix             = "glusterfs-dynamic-"
	replicaCount                   = 3
	durabilityType                 = "replicate"
	secretKeyName                  = "key" // key name used in secret
	gciLinuxGlusterMountBinaryPath = "/sbin/mount.glusterfs"
	defaultGidMin                  = 2000
	defaultGidMax                  = math.MaxInt32

	// absoluteGidMin/Max are currently the same as the
	// default values, but they play a different role and
	// could take a different value. Only thing we need is:
	// absGidMin <= defGidMin <= defGidMax <= absGidMax
	absoluteGidMin          = 2000
	absoluteGidMax          = math.MaxInt32
	linuxGlusterMountBinary = "mount.glusterfs"
	heketiAnn               = "heketi-dynamic-provisioner"
	glusterTypeAnn          = "gluster.org/type"
	glusterDescAnn          = "Gluster-Internal: Dynamically provisioned PV"
	heketiVolIDAnn          = "gluster.kubernetes.io/heketi-volume-id"
)

func (plugin *glusterfsPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *glusterfsPlugin) GetPluginName() string {
	return glusterfsPluginName
}

func (plugin *glusterfsPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v:%v",
		volumeSource.EndpointsName,
		volumeSource.Path), nil
}

func (plugin *glusterfsPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Glusterfs != nil) ||
		(spec.Volume != nil && spec.Volume.Glusterfs != nil)
}

func (plugin *glusterfsPlugin) RequiresRemount() bool {
	return false
}

func (plugin *glusterfsPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *glusterfsPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *glusterfsPlugin) RequiresFSResize() bool {
	return false
}

func (plugin *glusterfsPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
	}
}

func (plugin *glusterfsPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	source, _, err := getVolumeSource(spec)
	if err != nil {
		glog.Errorf("failed to get gluster volumesource: %v", err)
		return nil, err
	}
	epName := source.EndpointsName
	// PVC/POD is in same namespace.
	podNs := pod.Namespace
	kubeClient := plugin.host.GetKubeClient()
	if kubeClient == nil {
		return nil, fmt.Errorf("failed to get kube client to initialize mounter")
	}
	ep, err := kubeClient.CoreV1().Endpoints(podNs).Get(epName, metav1.GetOptions{})
	if err != nil {
		glog.Errorf("failed to get endpoint %s: %v", epName, err)
		return nil, err
	}
	glog.V(4).Infof("glusterfs pv endpoint %v", ep)
	return plugin.newMounterInternal(spec, ep, pod, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *glusterfsPlugin) newMounterInternal(spec *volume.Spec, ep *v1.Endpoints, pod *v1.Pod, mounter mount.Interface) (volume.Mounter, error) {
	source, readOnly, _ := getVolumeSource(spec)
	return &glusterfsMounter{
		glusterfs: &glusterfs{
			volName:         spec.Name(),
			mounter:         mounter,
			pod:             pod,
			plugin:          plugin,
			MetricsProvider: volume.NewMetricsStatFS(plugin.host.GetPodVolumeDir(pod.UID, strings.EscapeQualifiedNameForDisk(glusterfsPluginName), spec.Name())),
		},
		hosts:        ep,
		path:         source.Path,
		readOnly:     readOnly,
		mountOptions: volutil.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *glusterfsPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *glusterfsPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &glusterfsUnmounter{&glusterfs{
		volName:         volName,
		mounter:         mounter,
		pod:             &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}},
		plugin:          plugin,
		MetricsProvider: volume.NewMetricsStatFS(plugin.host.GetPodVolumeDir(podUID, strings.EscapeQualifiedNameForDisk(glusterfsPluginName), volName)),
	}}, nil
}

func (plugin *glusterfsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {

	// To reconstruct volume spec we need endpoint where fetching endpoint from mount
	// string looks to be impossible, so returning error.

	return nil, fmt.Errorf("impossible to reconstruct glusterfs volume spec from volume mountpath")
}

// Glusterfs volumes represent a bare host file or directory mount of an Glusterfs export.
type glusterfs struct {
	volName string
	pod     *v1.Pod
	mounter mount.Interface
	plugin  *glusterfsPlugin
	volume.MetricsProvider
}

type glusterfsMounter struct {
	*glusterfs
	hosts        *v1.Endpoints
	path         string
	readOnly     bool
	mountOptions []string
}

var _ volume.Mounter = &glusterfsMounter{}

func (b *glusterfsMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *glusterfsMounter) CanMount() error {
	exe := b.plugin.host.GetExec(b.plugin.GetPluginName())
	switch runtime.GOOS {
	case "linux":
		if _, err := exe.Run("test", "-x", gciLinuxGlusterMountBinaryPath); err != nil {
			return fmt.Errorf("Required binary %s is missing", gciLinuxGlusterMountBinaryPath)
		}
	}
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *glusterfsMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *glusterfsMounter) SetUpAt(dir string, fsGroup *int64) error {
	notMnt, err := b.mounter.IsLikelyNotMountPoint(dir)
	glog.V(4).Infof("mount setup: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}
	if err := os.MkdirAll(dir, 0750); err != nil {
		return err
	}
	err = b.setUpAtInternal(dir)
	if err == nil {
		return nil
	}

	// Cleanup upon failure.
	volutil.UnmountPath(dir, b.mounter)
	return err
}

func (glusterfsVolume *glusterfs) GetPath() string {
	name := glusterfsPluginName
	return glusterfsVolume.plugin.host.GetPodVolumeDir(glusterfsVolume.pod.UID, strings.EscapeQualifiedNameForDisk(name), glusterfsVolume.volName)
}

type glusterfsUnmounter struct {
	*glusterfs
}

var _ volume.Unmounter = &glusterfsUnmounter{}

func (c *glusterfsUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *glusterfsUnmounter) TearDownAt(dir string) error {
	return volutil.UnmountPath(dir, c.mounter)
}

func (b *glusterfsMounter) setUpAtInternal(dir string) error {
	var errs error
	options := []string{}
	hasLogFile := false
	log := ""

	if b.readOnly {
		options = append(options, "ro")

	}

	// Check logfile has been provided by user, if provided, use that as the log file.
	for _, userOpt := range b.mountOptions {
		if dstrings.HasPrefix(userOpt, "log-file") {
			glog.V(4).Infof("log-file mount option has provided")
			hasLogFile = true
			break
		}
	}

	// If logfile has not been provided, create driver specific log file.
	if !hasLogFile {
		log = ""
		p := path.Join(b.glusterfs.plugin.host.GetPluginDir(glusterfsPluginName), b.glusterfs.volName)
		if err := os.MkdirAll(p, 0750); err != nil {
			return fmt.Errorf("failed to create directory %v: %v", p, err)
		}

		// adding log-level ERROR to remove noise
		// and more specific log path so each pod has
		// its own log based on PV + Pod
		log = path.Join(p, b.pod.Name+"-glusterfs.log")

		// Use derived log file in gluster fuse mount
		options = append(options, "log-file="+log)

	}

	options = append(options, "log-level=ERROR")

	var addrlist []string
	if b.hosts == nil {
		return fmt.Errorf("glusterfs endpoint is nil in mounter")
	}
	addr := sets.String{}
	if b.hosts.Subsets != nil {
		for _, s := range b.hosts.Subsets {
			for _, a := range s.Addresses {
				if !addr.Has(a.IP) {
					addr.Insert(a.IP)
					addrlist = append(addrlist, a.IP)
				}
			}
		}

	}

	//Add backup-volfile-servers and auto_unmount options.
	options = append(options, "backup-volfile-servers="+dstrings.Join(addrlist[:], ":"))
	options = append(options, "auto_unmount")

	mountOptions := volutil.JoinMountOptions(b.mountOptions, options)

	// with `backup-volfile-servers` mount option in place, it is not required to
	// iterate over all the servers in the addrlist. A mount attempt with this option
	// will fetch all the servers mentioned in the backup-volfile-servers list.
	// Refer to backup-volfile-servers @ http://docs.gluster.org/en/latest/Administrator%20Guide/Setting%20Up%20Clients/

	if (len(addrlist) > 0) && (addrlist[0] != "") {
		ip := addrlist[0]
		errs = b.mounter.Mount(ip+":"+b.path, dir, "glusterfs", mountOptions)
		if errs == nil {
			glog.Infof("successfully mounted directory %s", dir)
			return nil
		}

		if dstrings.Contains(errs.Error(), "Invalid option auto_unmount") ||
			dstrings.Contains(errs.Error(), "Invalid argument") {
			// Give a try without `auto_unmount` mount option, because
			// it could be that gluster fuse client is older version and
			// mount.glusterfs is unaware of `auto_unmount`.
			noAutoMountOptions := make([]string, 0, len(mountOptions))
			for _, opt := range mountOptions {
				if opt != "auto_unmount" {
					noAutoMountOptions = append(noAutoMountOptions, opt)
				}
			}
			errs = b.mounter.Mount(ip+":"+b.path, dir, "glusterfs", noAutoMountOptions)
			if errs == nil {
				glog.Infof("successfully mounted %s", dir)
				return nil
			}
		}
	} else {
		return fmt.Errorf("failed to execute mount command:[no valid ipaddress found in endpoint address list]")
	}

	// Failed mount scenario.
	// Since glusterfs does not return error text
	// it all goes in a log file, we will read the log file
	logErr := readGlusterLog(log, b.pod.Name)
	if logErr != nil {
		return fmt.Errorf("mount failed: %v the following error information was pulled from the glusterfs log to help diagnose this issue: %v", errs, logErr)
	}
	return fmt.Errorf("mount failed: %v", errs)

}

func getVolumeSource(spec *volume.Spec) (*v1.GlusterfsVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Glusterfs != nil {
		return spec.Volume.Glusterfs, spec.Volume.Glusterfs.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Glusterfs != nil {
		return spec.PersistentVolume.Spec.Glusterfs, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Glusterfs volume type")
}

func (plugin *glusterfsPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options)
}

func (plugin *glusterfsPlugin) newProvisionerInternal(options volume.VolumeOptions) (volume.Provisioner, error) {
	return &glusterfsVolumeProvisioner{
		glusterfsMounter: &glusterfsMounter{
			glusterfs: &glusterfs{
				plugin: plugin,
			},
		},
		options: options,
	}, nil
}

type provisionerConfig struct {
	url                string
	user               string
	userKey            string
	secretNamespace    string
	secretName         string
	secretValue        string
	clusterID          string
	gidMin             int
	gidMax             int
	volumeType         gapi.VolumeDurabilityInfo
	volumeOptions      []string
	volumeNamePrefix   string
	thinPoolSnapFactor float32
}

type glusterfsVolumeProvisioner struct {
	*glusterfsMounter
	provisionerConfig
	options volume.VolumeOptions
}

func convertGid(gidString string) (int, error) {
	gid64, err := strconv.ParseInt(gidString, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("failed to parse gid %v: %v", gidString, err)
	}

	if gid64 < 0 {
		return 0, fmt.Errorf("negative GIDs %v are not allowed", gidString)
	}

	// ParseInt returns a int64, but since we parsed only
	// for 32 bit, we can cast to int without loss:
	gid := int(gid64)
	return gid, nil
}

func convertVolumeParam(volumeString string) (int, error) {

	count, err := strconv.Atoi(volumeString)
	if err != nil {
		return 0, fmt.Errorf("failed to parse volumestring %q: %v", volumeString, err)
	}

	if count < 0 {
		return 0, fmt.Errorf("negative values are not allowed")
	}
	return count, nil
}

func (plugin *glusterfsPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec)
}

func (plugin *glusterfsPlugin) newDeleterInternal(spec *volume.Spec) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Glusterfs == nil {
		return nil, fmt.Errorf("spec.PersistentVolume.Spec.Glusterfs is nil")
	}
	return &glusterfsVolumeDeleter{
		glusterfsMounter: &glusterfsMounter{
			glusterfs: &glusterfs{
				volName: spec.Name(),
				plugin:  plugin,
			},
			path: spec.PersistentVolume.Spec.Glusterfs.Path,
		},
		spec: spec.PersistentVolume,
	}, nil
}

type glusterfsVolumeDeleter struct {
	*glusterfsMounter
	provisionerConfig
	spec *v1.PersistentVolume
}

func (d *glusterfsVolumeDeleter) GetPath() string {
	name := glusterfsPluginName
	return d.plugin.host.GetPodVolumeDir(d.glusterfsMounter.glusterfs.pod.UID, strings.EscapeQualifiedNameForDisk(name), d.glusterfsMounter.glusterfs.volName)
}

// Traverse the PVs, fetching all the GIDs from those
// in a given storage class, and mark them in the table.
func (plugin *glusterfsPlugin) collectGids(className string, gidTable *MinMaxAllocator) error {
	kubeClient := plugin.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("failed to get kube client when collecting gids")
	}
	pvList, err := kubeClient.CoreV1().PersistentVolumes().List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		glog.Error("failed to get existing persistent volumes")
		return err
	}

	for _, pv := range pvList.Items {
		if v1helper.GetPersistentVolumeClass(&pv) != className {
			continue
		}

		pvName := pv.ObjectMeta.Name

		gidStr, ok := pv.Annotations[volutil.VolumeGidAnnotationKey]

		if !ok {
			glog.Warningf("no GID found in pv %v", pvName)
			continue
		}

		gid, err := convertGid(gidStr)
		if err != nil {
			glog.Errorf("failed to parse gid %s: %v", gidStr, err)
			continue
		}

		_, err = gidTable.Allocate(gid)
		if err == ErrConflict {
			glog.Warningf("GID %v found in pv %v was already allocated", gid, pvName)
		} else if err != nil {
			glog.Errorf("failed to store gid %v found in pv %v: %v", gid, pvName, err)
			return err
		}
	}

	return nil
}

// Return the gid table for a storage class.
// - If this is the first time, fill it with all the gids
//   used in PVs of this storage class by traversing the PVs.
// - Adapt the range of the table to the current range of the SC.
func (plugin *glusterfsPlugin) getGidTable(className string, min int, max int) (*MinMaxAllocator, error) {
	plugin.gidTableLock.Lock()
	gidTable, ok := plugin.gidTable[className]
	plugin.gidTableLock.Unlock()

	if ok {
		err := gidTable.SetRange(min, max)
		if err != nil {
			return nil, err
		}

		return gidTable, nil
	}

	// create a new table and fill it
	newGidTable, err := NewMinMaxAllocator(0, absoluteGidMax)
	if err != nil {
		return nil, err
	}

	// collect gids with the full range
	err = plugin.collectGids(className, newGidTable)
	if err != nil {
		return nil, err
	}

	// and only reduce the range afterwards
	err = newGidTable.SetRange(min, max)
	if err != nil {
		return nil, err
	}

	// if in the meantime a table appeared, use it
	plugin.gidTableLock.Lock()
	defer plugin.gidTableLock.Unlock()

	gidTable, ok = plugin.gidTable[className]
	if ok {
		err = gidTable.SetRange(min, max)
		if err != nil {
			return nil, err
		}

		return gidTable, nil
	}

	plugin.gidTable[className] = newGidTable

	return newGidTable, nil
}

func (d *glusterfsVolumeDeleter) getGid() (int, bool, error) {
	gidStr, ok := d.spec.Annotations[volutil.VolumeGidAnnotationKey]

	if !ok {
		return 0, false, nil
	}

	gid, err := convertGid(gidStr)

	return gid, true, err
}

func (d *glusterfsVolumeDeleter) Delete() error {
	glog.V(2).Infof("delete volume %s", d.glusterfsMounter.path)

	volumeName := d.glusterfsMounter.path
	volumeID, err := getVolumeID(d.spec, volumeName)
	if err != nil {
		return fmt.Errorf("failed to get volumeID: %v", err)
	}

	class, err := volutil.GetClassForVolume(d.plugin.host.GetKubeClient(), d.spec)
	if err != nil {
		return err
	}

	cfg, err := parseClassParameters(class.Parameters, d.plugin.host.GetKubeClient())
	if err != nil {
		return err
	}
	d.provisionerConfig = *cfg

	glog.V(4).Infof("deleting volume %q", volumeID)

	gid, exists, err := d.getGid()
	if err != nil {
		glog.Error(err)
	} else if exists {
		gidTable, err := d.plugin.getGidTable(class.Name, cfg.gidMin, cfg.gidMax)
		if err != nil {
			return fmt.Errorf("failed to get gidTable: %v", err)
		}

		err = gidTable.Release(gid)
		if err != nil {
			return fmt.Errorf("failed to release gid %v: %v", gid, err)
		}
	}

	cli := gcli.NewClient(d.url, d.user, d.secretValue)
	if cli == nil {
		glog.Errorf("failed to create glusterfs REST client")
		return fmt.Errorf("failed to create glusterfs REST client, REST server authentication failed")
	}
	err = cli.VolumeDelete(volumeID)
	if err != nil {
		glog.Errorf("failed to delete volume %s: %v", volumeName, err)
		return err
	}
	glog.V(2).Infof("volume %s deleted successfully", volumeName)

	//Deleter takes endpoint and namespace from pv spec.
	pvSpec := d.spec.Spec
	var dynamicEndpoint, dynamicNamespace string
	if pvSpec.ClaimRef == nil {
		glog.Errorf("ClaimRef is nil")
		return fmt.Errorf("ClaimRef is nil")
	}
	if pvSpec.ClaimRef.Namespace == "" {
		glog.Errorf("namespace is nil")
		return fmt.Errorf("namespace is nil")
	}
	dynamicNamespace = pvSpec.ClaimRef.Namespace
	if pvSpec.Glusterfs.EndpointsName != "" {
		dynamicEndpoint = pvSpec.Glusterfs.EndpointsName
	}
	glog.V(3).Infof("dynamic namespace and endpoint %v/%v", dynamicNamespace, dynamicEndpoint)
	err = d.deleteEndpointService(dynamicNamespace, dynamicEndpoint)
	if err != nil {
		glog.Errorf("failed to delete endpoint/service %v/%v: %v", dynamicNamespace, dynamicEndpoint, err)
	} else {
		glog.V(1).Infof("endpoint %v/%v is deleted successfully ", dynamicNamespace, dynamicEndpoint)
	}
	return nil
}

func (p *glusterfsVolumeProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if !volutil.AccessModesContainedInAll(p.plugin.GetAccessModes(), p.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", p.options.PVC.Spec.AccessModes, p.plugin.GetAccessModes())
	}

	if p.options.PVC.Spec.Selector != nil {
		glog.V(4).Infof("not able to parse your claim Selector")
		return nil, fmt.Errorf("not able to parse your claim Selector")
	}

	if volutil.CheckPersistentVolumeClaimModeBlock(p.options.PVC) {
		return nil, fmt.Errorf("%s does not support block volume provisioning", p.plugin.GetPluginName())
	}

	glog.V(4).Infof("Provision VolumeOptions %v", p.options)
	scName := v1helper.GetPersistentVolumeClaimClass(p.options.PVC)
	cfg, err := parseClassParameters(p.options.Parameters, p.plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}
	p.provisionerConfig = *cfg

	gidTable, err := p.plugin.getGidTable(scName, cfg.gidMin, cfg.gidMax)
	if err != nil {
		return nil, fmt.Errorf("failed to get gidTable: %v", err)
	}

	gid, _, err := gidTable.AllocateNext()
	if err != nil {
		glog.Errorf("failed to reserve GID from table: %v", err)
		return nil, fmt.Errorf("failed to reserve GID from table: %v", err)
	}

	glog.V(2).Infof("Allocated GID %d for PVC %s", gid, p.options.PVC.Name)

	glusterfs, sizeGiB, volID, err := p.CreateVolume(gid)
	if err != nil {
		if releaseErr := gidTable.Release(gid); releaseErr != nil {
			glog.Errorf("error when releasing GID in storageclass %s: %v", scName, releaseErr)
		}

		glog.Errorf("failed to create volume: %v", err)
		return nil, fmt.Errorf("failed to create volume: %v", err)
	}
	mode := v1.PersistentVolumeFilesystem
	pv := new(v1.PersistentVolume)
	pv.Spec.PersistentVolumeSource.Glusterfs = glusterfs
	pv.Spec.PersistentVolumeReclaimPolicy = p.options.PersistentVolumeReclaimPolicy
	pv.Spec.AccessModes = p.options.PVC.Spec.AccessModes
	pv.Spec.VolumeMode = &mode
	if len(pv.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = p.plugin.GetAccessModes()
	}

	pv.Spec.MountOptions = p.options.MountOptions

	gidStr := strconv.FormatInt(int64(gid), 10)

	pv.Annotations = map[string]string{
		volutil.VolumeGidAnnotationKey:        gidStr,
		volutil.VolumeDynamicallyCreatedByKey: heketiAnn,
		glusterTypeAnn:                        "file",
		"Description":                         glusterDescAnn,
		heketiVolIDAnn:                        volID,
	}

	pv.Spec.Capacity = v1.ResourceList{
		v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGiB)),
	}
	return pv, nil
}

func (p *glusterfsVolumeProvisioner) CreateVolume(gid int) (r *v1.GlusterfsVolumeSource, size int, volID string, err error) {
	var clusterIDs []string
	customVolumeName := ""
	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]

	// GlusterFS/heketi creates volumes in units of GiB.
	sz, err := volutil.RoundUpToGiBInt(capacity)
	if err != nil {
		return nil, 0, "", err
	}
	glog.V(2).Infof("create volume of size %dGiB", sz)

	if p.url == "" {
		glog.Errorf("REST server endpoint is empty")
		return nil, 0, "", fmt.Errorf("failed to create glusterfs REST client, REST URL is empty")
	}
	cli := gcli.NewClient(p.url, p.user, p.secretValue)
	if cli == nil {
		glog.Errorf("failed to create glusterfs REST client")
		return nil, 0, "", fmt.Errorf("failed to create glusterfs REST client, REST server authentication failed")
	}
	if p.provisionerConfig.clusterID != "" {
		clusterIDs = dstrings.Split(p.clusterID, ",")
		glog.V(4).Infof("provided clusterIDs %v", clusterIDs)
	}

	if p.provisionerConfig.volumeNamePrefix != "" {
		customVolumeName = fmt.Sprintf("%s_%s_%s_%s", p.provisionerConfig.volumeNamePrefix, p.options.PVC.Namespace, p.options.PVC.Name, uuid.NewUUID())
	}

	gid64 := int64(gid)
	snaps := struct {
		Enable bool    `json:"enable"`
		Factor float32 `json:"factor"`
	}{
		true,
		p.provisionerConfig.thinPoolSnapFactor,
	}

	volumeReq := &gapi.VolumeCreateRequest{Size: sz, Name: customVolumeName, Clusters: clusterIDs, Gid: gid64, Durability: p.volumeType, GlusterVolumeOptions: p.volumeOptions, Snapshot: snaps}
	volume, err := cli.VolumeCreate(volumeReq)
	if err != nil {
		glog.Errorf("failed to create volume: %v", err)
		return nil, 0, "", fmt.Errorf("failed to create volume: %v", err)
	}
	glog.V(1).Infof("volume with size %d and name %s created", volume.Size, volume.Name)
	volID = volume.Id
	dynamicHostIps, err := getClusterNodes(cli, volume.Cluster)
	if err != nil {
		glog.Errorf("failed to get cluster nodes for volume %s: %v", volume, err)
		return nil, 0, "", fmt.Errorf("failed to get cluster nodes for volume %s: %v", volume, err)
	}

	// The 'endpointname' is created in form of 'glusterfs-dynamic-<claimname>'.
	// createEndpointService() checks for this 'endpoint' existence in PVC's namespace and
	// If not found, it create an endpoint and service using the IPs we dynamically picked at time
	// of volume creation.
	epServiceName := dynamicEpSvcPrefix + p.options.PVC.Name
	epNamespace := p.options.PVC.Namespace
	endpoint, service, err := p.createEndpointService(epNamespace, epServiceName, dynamicHostIps, p.options.PVC.Name)
	if err != nil {
		glog.Errorf("failed to create endpoint/service %v/%v: %v", epNamespace, epServiceName, err)
		deleteErr := cli.VolumeDelete(volume.Id)
		if deleteErr != nil {
			glog.Errorf("failed to delete volume: %v, manual deletion of the volume required", deleteErr)
		}
		return nil, 0, "", fmt.Errorf("failed to create endpoint/service %v/%v: %v", epNamespace, epServiceName, err)
	}
	glog.V(3).Infof("dynamic endpoint %v and service %v ", endpoint, service)
	return &v1.GlusterfsVolumeSource{
		EndpointsName: endpoint.Name,
		Path:          volume.Name,
		ReadOnly:      false,
	}, sz, volID, nil
}

func (p *glusterfsVolumeProvisioner) createEndpointService(namespace string, epServiceName string, hostips []string, pvcname string) (endpoint *v1.Endpoints, service *v1.Service, err error) {

	addrlist := make([]v1.EndpointAddress, len(hostips))
	for i, v := range hostips {
		addrlist[i].IP = v
	}
	endpoint = &v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      epServiceName,
			Labels: map[string]string{
				"gluster.kubernetes.io/provisioned-for-pvc": pvcname,
			},
		},
		Subsets: []v1.EndpointSubset{{
			Addresses: addrlist,
			Ports:     []v1.EndpointPort{{Port: 1, Protocol: "TCP"}},
		}},
	}
	kubeClient := p.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return nil, nil, fmt.Errorf("failed to get kube client when creating endpoint service")
	}
	_, err = kubeClient.CoreV1().Endpoints(namespace).Create(endpoint)
	if err != nil && errors.IsAlreadyExists(err) {
		glog.V(1).Infof("endpoint %s already exist in namespace %s", endpoint, namespace)
		err = nil
	}
	if err != nil {
		glog.Errorf("failed to create endpoint: %v", err)
		return nil, nil, fmt.Errorf("failed to create endpoint: %v", err)
	}
	service = &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      epServiceName,
			Namespace: namespace,
			Labels: map[string]string{
				"gluster.kubernetes.io/provisioned-for-pvc": pvcname,
			},
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				{Protocol: "TCP", Port: 1}}}}
	_, err = kubeClient.CoreV1().Services(namespace).Create(service)
	if err != nil && errors.IsAlreadyExists(err) {
		glog.V(1).Infof("service %s already exist in namespace %s", service, namespace)
		err = nil
	}
	if err != nil {
		glog.Errorf("failed to create service: %v", err)
		return nil, nil, fmt.Errorf("error creating service: %v", err)
	}
	return endpoint, service, nil
}

func (d *glusterfsVolumeDeleter) deleteEndpointService(namespace string, epServiceName string) (err error) {
	kubeClient := d.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("failed to get kube client when deleting endpoint service")
	}
	err = kubeClient.CoreV1().Services(namespace).Delete(epServiceName, nil)
	if err != nil {
		glog.Errorf("failed to delete service %s/%s: %v", namespace, epServiceName, err)
		return fmt.Errorf("failed to delete service %s/%s: %v", namespace, epServiceName, err)
	}
	glog.V(1).Infof("service/endpoint: %s/%s deleted successfully", namespace, epServiceName)
	return nil
}

// parseSecret finds a given Secret instance and reads user password from it.
func parseSecret(namespace, secretName string, kubeClient clientset.Interface) (string, error) {
	secretMap, err := volutil.GetSecretForPV(namespace, secretName, glusterfsPluginName, kubeClient)
	if err != nil {
		glog.Errorf("failed to get secret: %s/%s: %v", namespace, secretName, err)
		return "", fmt.Errorf("failed to get secret %s/%s: %v", namespace, secretName, err)
	}
	if len(secretMap) == 0 {
		return "", fmt.Errorf("empty secret map")
	}
	secret := ""
	for k, v := range secretMap {
		if k == secretKeyName {
			return v, nil
		}
		secret = v
	}

	// If not found, the last secret in the map wins as done before
	return secret, nil
}

// getClusterNodes() returns the cluster nodes of a given cluster
func getClusterNodes(cli *gcli.Client, cluster string) (dynamicHostIps []string, err error) {
	clusterinfo, err := cli.ClusterInfo(cluster)
	if err != nil {
		glog.Errorf("failed to get cluster details: %v", err)
		return nil, fmt.Errorf("failed to get cluster details: %v", err)
	}

	// For the dynamically provisioned volume, we gather the list of node IPs
	// of the cluster on which provisioned volume belongs to, as there can be multiple
	// clusters.
	for _, node := range clusterinfo.Nodes {
		nodeInfo, err := cli.NodeInfo(string(node))
		if err != nil {
			glog.Errorf("failed to get host ipaddress: %v", err)
			return nil, fmt.Errorf("failed to get host ipaddress: %v", err)
		}
		ipaddr := dstrings.Join(nodeInfo.NodeAddRequest.Hostnames.Storage, "")
		dynamicHostIps = append(dynamicHostIps, ipaddr)
	}
	glog.V(3).Infof("host list :%v", dynamicHostIps)
	if len(dynamicHostIps) == 0 {
		glog.Errorf("no hosts found: %v", err)
		return nil, fmt.Errorf("no hosts found: %v", err)
	}
	return dynamicHostIps, nil
}

// parseClassParameters parses StorageClass parameters.
func parseClassParameters(params map[string]string, kubeClient clientset.Interface) (*provisionerConfig, error) {
	var cfg provisionerConfig
	var err error
	cfg.gidMin = defaultGidMin
	cfg.gidMax = defaultGidMax

	authEnabled := true
	parseVolumeType := ""
	parseVolumeOptions := ""
	parseVolumeNamePrefix := ""
	parseThinPoolSnapFactor := ""

	//thin pool snap factor default to 1.0
	cfg.thinPoolSnapFactor = float32(1.0)

	for k, v := range params {
		switch dstrings.ToLower(k) {
		case "resturl":
			cfg.url = v
		case "restuser":
			cfg.user = v
		case "restuserkey":
			cfg.userKey = v
		case "secretname":
			cfg.secretName = v
		case "secretnamespace":
			cfg.secretNamespace = v
		case "clusterid":
			if len(v) != 0 {
				cfg.clusterID = v
			}
		case "restauthenabled":
			authEnabled = dstrings.ToLower(v) == "true"
		case "gidmin":
			parseGidMin, err := convertGid(v)
			if err != nil {
				return nil, fmt.Errorf("invalid gidMin value %q for volume plugin %s", k, glusterfsPluginName)
			}
			if parseGidMin < absoluteGidMin {
				return nil, fmt.Errorf("gidMin must be >= %v", absoluteGidMin)
			}
			if parseGidMin > absoluteGidMax {
				return nil, fmt.Errorf("gidMin must be <= %v", absoluteGidMax)
			}
			cfg.gidMin = parseGidMin
		case "gidmax":
			parseGidMax, err := convertGid(v)
			if err != nil {
				return nil, fmt.Errorf("invalid gidMax value %q for volume plugin %s", k, glusterfsPluginName)
			}
			if parseGidMax < absoluteGidMin {
				return nil, fmt.Errorf("gidMax must be >= %v", absoluteGidMin)
			}
			if parseGidMax > absoluteGidMax {
				return nil, fmt.Errorf("gidMax must be <= %v", absoluteGidMax)
			}
			cfg.gidMax = parseGidMax
		case "volumetype":
			parseVolumeType = v

		case "volumeoptions":
			if len(v) != 0 {
				parseVolumeOptions = v
			}
		case "volumenameprefix":
			if len(v) != 0 {
				parseVolumeNamePrefix = v
			}
		case "snapfactor":
			if len(v) != 0 {
				parseThinPoolSnapFactor = v
			}

		default:
			return nil, fmt.Errorf("invalid option %q for volume plugin %s", k, glusterfsPluginName)
		}
	}

	if len(cfg.url) == 0 {
		return nil, fmt.Errorf("StorageClass for provisioner %s must contain 'resturl' parameter", glusterfsPluginName)
	}

	if len(parseVolumeType) == 0 {
		cfg.volumeType = gapi.VolumeDurabilityInfo{Type: gapi.DurabilityReplicate, Replicate: gapi.ReplicaDurability{Replica: replicaCount}}
	} else {
		parseVolumeTypeInfo := dstrings.Split(parseVolumeType, ":")

		switch parseVolumeTypeInfo[0] {
		case "replicate":
			if len(parseVolumeTypeInfo) >= 2 {
				newReplicaCount, err := convertVolumeParam(parseVolumeTypeInfo[1])
				if err != nil {
					return nil, fmt.Errorf("error parsing volumeType %q: %s", parseVolumeTypeInfo[1], err)
				}
				cfg.volumeType = gapi.VolumeDurabilityInfo{Type: gapi.DurabilityReplicate, Replicate: gapi.ReplicaDurability{Replica: newReplicaCount}}
			} else {
				cfg.volumeType = gapi.VolumeDurabilityInfo{Type: gapi.DurabilityReplicate, Replicate: gapi.ReplicaDurability{Replica: replicaCount}}
			}
		case "disperse":
			if len(parseVolumeTypeInfo) >= 3 {
				newDisperseData, err := convertVolumeParam(parseVolumeTypeInfo[1])
				if err != nil {
					return nil, fmt.Errorf("error parsing volumeType %q: %s", parseVolumeTypeInfo[1], err)
				}
				newDisperseRedundancy, err := convertVolumeParam(parseVolumeTypeInfo[2])
				if err != nil {
					return nil, fmt.Errorf("error parsing volumeType %q: %s", parseVolumeTypeInfo[2], err)
				}
				cfg.volumeType = gapi.VolumeDurabilityInfo{Type: gapi.DurabilityEC, Disperse: gapi.DisperseDurability{Data: newDisperseData, Redundancy: newDisperseRedundancy}}
			} else {
				return nil, fmt.Errorf("StorageClass for provisioner %q must have data:redundancy count set for disperse volumes in storage class option '%s'", glusterfsPluginName, "volumetype")
			}
		case "none":
			cfg.volumeType = gapi.VolumeDurabilityInfo{Type: gapi.DurabilityDistributeOnly}
		default:
			return nil, fmt.Errorf("error parsing value for option 'volumetype' for volume plugin %s", glusterfsPluginName)
		}
	}
	if !authEnabled {
		cfg.user = ""
		cfg.secretName = ""
		cfg.secretNamespace = ""
		cfg.userKey = ""
		cfg.secretValue = ""
	}

	if len(cfg.secretName) != 0 || len(cfg.secretNamespace) != 0 {

		// secretName + Namespace has precedence over userKey
		if len(cfg.secretName) != 0 && len(cfg.secretNamespace) != 0 {
			cfg.secretValue, err = parseSecret(cfg.secretNamespace, cfg.secretName, kubeClient)
			if err != nil {
				return nil, err
			}
		} else {
			return nil, fmt.Errorf("StorageClass for provisioner %q must have secretNamespace and secretName either both set or both empty", glusterfsPluginName)
		}
	} else {
		cfg.secretValue = cfg.userKey
	}

	if cfg.gidMin > cfg.gidMax {
		return nil, fmt.Errorf("StorageClass for provisioner %q must have gidMax value >= gidMin", glusterfsPluginName)
	}

	if len(parseVolumeOptions) != 0 {
		volOptions := dstrings.Split(parseVolumeOptions, ",")
		if len(volOptions) == 0 {
			return nil, fmt.Errorf("StorageClass for provisioner %q must have valid (for e.g., 'client.ssl on') volume option", glusterfsPluginName)
		}
		cfg.volumeOptions = volOptions

	}

	if len(parseVolumeNamePrefix) != 0 {
		if dstrings.Contains(parseVolumeNamePrefix, "_") {
			return nil, fmt.Errorf("Storageclass parameter 'volumenameprefix' should not contain '_' in its value")
		}
		cfg.volumeNamePrefix = parseVolumeNamePrefix
	}

	if len(parseThinPoolSnapFactor) != 0 {
		thinPoolSnapFactor, err := strconv.ParseFloat(parseThinPoolSnapFactor, 32)
		if err != nil {
			return nil, fmt.Errorf("failed to convert snapfactor %v to float: %v", parseThinPoolSnapFactor, err)
		}
		if thinPoolSnapFactor < 1.0 || thinPoolSnapFactor > 100.0 {
			return nil, fmt.Errorf("invalid snapshot factor %v, the value must be between 1 to 100", thinPoolSnapFactor)
		}
		cfg.thinPoolSnapFactor = float32(thinPoolSnapFactor)
	}
	return &cfg, nil
}

// getVolumeID returns volumeID from the PV or volumename.
func getVolumeID(pv *v1.PersistentVolume, volumeName string) (string, error) {
	volumeID := ""

	// Get volID from pvspec if available, else fill it from volumename.
	if pv != nil {
		if pv.Annotations[heketiVolIDAnn] != "" {
			volumeID = pv.Annotations[heketiVolIDAnn]
		} else {
			volumeID = dstrings.TrimPrefix(volumeName, volPrefix)
		}
	} else {
		return volumeID, fmt.Errorf("provided PV spec is nil")
	}
	if volumeID == "" {
		return volumeID, fmt.Errorf("volume ID is empty")
	}
	return volumeID, nil
}

func (plugin *glusterfsPlugin) ExpandVolumeDevice(spec *volume.Spec, newSize resource.Quantity, oldSize resource.Quantity) (resource.Quantity, error) {
	pvSpec := spec.PersistentVolume.Spec
	volumeName := pvSpec.Glusterfs.Path
	glog.V(2).Infof("Received request to expand volume %s", volumeName)
	volumeID, err := getVolumeID(spec.PersistentVolume, volumeName)

	if err != nil {
		return oldSize, fmt.Errorf("failed to get volumeID for volume %s: %v", volumeName, err)
	}

	//Get details of StorageClass.
	class, err := volutil.GetClassForVolume(plugin.host.GetKubeClient(), spec.PersistentVolume)
	if err != nil {
		return oldSize, err
	}
	cfg, err := parseClassParameters(class.Parameters, plugin.host.GetKubeClient())
	if err != nil {
		return oldSize, err
	}

	glog.V(4).Infof("expanding volume: %q", volumeID)

	//Create REST server connection
	cli := gcli.NewClient(cfg.url, cfg.user, cfg.secretValue)
	if cli == nil {
		glog.Errorf("failed to create glusterfs REST client")
		return oldSize, fmt.Errorf("failed to create glusterfs REST client, REST server authentication failed")
	}

	// Find out delta size
	expansionSize := (newSize.Value() - oldSize.Value())
	expansionSizeGiB := int(volutil.RoundUpSize(expansionSize, volutil.GIB))

	// Find out requested Size
	requestGiB := volutil.RoundUpToGiB(newSize)

	//Check the existing volume size
	currentVolumeInfo, err := cli.VolumeInfo(volumeID)
	if err != nil {
		glog.Errorf("error when fetching details of volume %s: %v", volumeName, err)
		return oldSize, err
	}

	if int64(currentVolumeInfo.Size) >= requestGiB {
		return newSize, nil
	}

	// Make volume expansion request
	volumeExpandReq := &gapi.VolumeExpandRequest{Size: expansionSizeGiB}

	// Expand the volume
	volumeInfoRes, err := cli.VolumeExpand(volumeID, volumeExpandReq)
	if err != nil {
		glog.Errorf("failed to expand volume %s: %v", volumeName, err)
		return oldSize, err
	}

	glog.V(2).Infof("volume %s expanded to new size %d successfully", volumeName, volumeInfoRes.Size)
	newVolumeSize := resource.MustParse(fmt.Sprintf("%dGi", volumeInfoRes.Size))
	return newVolumeSize, nil
}
