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
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&glusterfsPlugin{host: nil, exe: exec.New(), gidTable: make(map[string]*MinMaxAllocator)}}
}

type glusterfsPlugin struct {
	host         volume.VolumeHost
	exe          exec.Interface
	gidTable     map[string]*MinMaxAllocator
	gidTableLock sync.Mutex
}

var _ volume.VolumePlugin = &glusterfsPlugin{}
var _ volume.PersistentVolumePlugin = &glusterfsPlugin{}
var _ volume.DeletableVolumePlugin = &glusterfsPlugin{}
var _ volume.ProvisionableVolumePlugin = &glusterfsPlugin{}
var _ volume.Provisioner = &glusterfsVolumeProvisioner{}
var _ volume.Deleter = &glusterfsVolumeDeleter{}

const (
	glusterfsPluginName         = "kubernetes.io/glusterfs"
	volPrefix                   = "vol_"
	dynamicEpSvcPrefix          = "glusterfs-dynamic-"
	replicaCount                = 3
	durabilityType              = "replicate"
	secretKeyName               = "key" // key name used in secret
	gciGlusterMountBinariesPath = "/sbin/mount.glusterfs"
	defaultGidMin               = 2000
	defaultGidMax               = math.MaxInt32
	// absoluteGidMin/Max are currently the same as the
	// default values, but they play a different role and
	// could take a different value. Only thing we need is:
	// absGidMin <= defGidMin <= defGidMax <= absGidMax
	absoluteGidMin = 2000
	absoluteGidMax = math.MaxInt32
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
	if (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Glusterfs == nil) ||
		(spec.Volume != nil && spec.Volume.Glusterfs == nil) {
		return false
	}

	return true
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

func (plugin *glusterfsPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
	}
}

func (plugin *glusterfsPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	source, _ := plugin.getGlusterVolumeSource(spec)
	ep_name := source.EndpointsName
	// PVC/POD is in same ns.
	ns := pod.Namespace
	kubeClient := plugin.host.GetKubeClient()
	if kubeClient == nil {
		return nil, fmt.Errorf("glusterfs: failed to get kube client to initialize mounter")
	}
	ep, err := kubeClient.Core().Endpoints(ns).Get(ep_name, metav1.GetOptions{})
	if err != nil {
		glog.Errorf("glusterfs: failed to get endpoints %s[%v]", ep_name, err)
		return nil, err
	}
	glog.V(1).Infof("glusterfs: endpoints %v", ep)
	return plugin.newMounterInternal(spec, ep, pod, plugin.host.GetMounter(), exec.New())
}

func (plugin *glusterfsPlugin) getGlusterVolumeSource(spec *volume.Spec) (*v1.GlusterfsVolumeSource, bool) {
	// Glusterfs volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// Glusterfs volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	if spec.Volume != nil && spec.Volume.Glusterfs != nil {
		return spec.Volume.Glusterfs, spec.Volume.Glusterfs.ReadOnly
	} else {
		return spec.PersistentVolume.Spec.Glusterfs, spec.ReadOnly
	}
}

func (plugin *glusterfsPlugin) newMounterInternal(spec *volume.Spec, ep *v1.Endpoints, pod *v1.Pod, mounter mount.Interface, exe exec.Interface) (volume.Mounter, error) {
	source, readOnly := plugin.getGlusterVolumeSource(spec)
	return &glusterfsMounter{
		glusterfs: &glusterfs{
			volName: spec.Name(),
			mounter: mounter,
			pod:     pod,
			plugin:  plugin,
		},
		hosts:        ep,
		path:         source.Path,
		readOnly:     readOnly,
		exe:          exe,
		mountOptions: volume.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *glusterfsPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *glusterfsPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &glusterfsUnmounter{&glusterfs{
		volName: volName,
		mounter: mounter,
		pod:     &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: podUID}},
		plugin:  plugin,
	}}, nil
}

func (plugin *glusterfsPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

func (plugin *glusterfsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glusterfsVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			Glusterfs: &v1.GlusterfsVolumeSource{
				EndpointsName: volumeName,
				Path:          volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(glusterfsVolume), nil
}

// Glusterfs volumes represent a bare host file or directory mount of an Glusterfs export.
type glusterfs struct {
	volName string
	pod     *v1.Pod
	mounter mount.Interface
	plugin  *glusterfsPlugin
	volume.MetricsNil
}

type glusterfsMounter struct {
	*glusterfs
	hosts        *v1.Endpoints
	path         string
	readOnly     bool
	exe          exec.Interface
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
	exe := exec.New()
	switch runtime.GOOS {
	case "linux":
		if _, err := exe.Command("/bin/ls", gciGlusterMountBinariesPath).CombinedOutput(); err != nil {
			return fmt.Errorf("Required binary %s is missing", gciGlusterMountBinariesPath)
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
	glog.V(4).Infof("glusterfs: mount set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}

	os.MkdirAll(dir, 0750)
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
	if b.readOnly {
		options = append(options, "ro")
	}

	p := path.Join(b.glusterfs.plugin.host.GetPluginDir(glusterfsPluginName), b.glusterfs.volName)
	if err := os.MkdirAll(p, 0750); err != nil {
		return fmt.Errorf("glusterfs: mkdir failed: %v", err)
	}

	// adding log-level ERROR to remove noise
	// and more specific log path so each pod has
	// its own log based on PV + Pod
	log := path.Join(p, b.pod.Name+"-glusterfs.log")
	options = append(options, "log-level=ERROR")
	options = append(options, "log-file="+log)

	var addrlist []string
	if b.hosts == nil {
		return fmt.Errorf("glusterfs: endpoint is nil")
	} else {
		addr := make(map[string]struct{})
		if b.hosts.Subsets != nil {
			for _, s := range b.hosts.Subsets {
				for _, a := range s.Addresses {
					addr[a.IP] = struct{}{}
					addrlist = append(addrlist, a.IP)
				}
			}

		}

		// Avoid mount storm, pick a host randomly.
		// Iterate all hosts until mount succeeds.
		for _, ip := range addrlist {
			mountOptions := volume.JoinMountOptions(b.mountOptions, options)
			errs = b.mounter.Mount(ip+":"+b.path, dir, "glusterfs", mountOptions)
			if errs == nil {
				glog.Infof("glusterfs: successfully mounted %s", dir)
				return nil
			}
		}
	}

	// Failed mount scenario.
	// Since glusterfs does not return error text
	// it all goes in a log file, we will read the log file
	logerror := readGlusterLog(log, b.pod.Name)
	if logerror != nil {
		// return fmt.Errorf("glusterfs: mount failed: %v", logerror)
		return fmt.Errorf("glusterfs: mount failed: %v the following error information was pulled from the glusterfs log to help diagnose this issue: %v", errs, logerror)
	}
	return fmt.Errorf("glusterfs: mount failed: %v", errs)
}

func getVolumeSource(
	spec *volume.Spec) (*v1.GlusterfsVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Glusterfs != nil {
		return spec.Volume.Glusterfs, spec.Volume.Glusterfs.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Glusterfs != nil {
		return spec.PersistentVolume.Spec.Glusterfs, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a GlusterFS volume type")
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
	url             string
	user            string
	userKey         string
	secretNamespace string
	secretName      string
	secretValue     string
	clusterId       string
	gidMin          int
	gidMax          int
	volumeType      gapi.VolumeDurabilityInfo
}

type glusterfsVolumeProvisioner struct {
	*glusterfsMounter
	provisionerConfig
	options volume.VolumeOptions
}

func convertGid(gidString string) (int, error) {
	gid64, err := strconv.ParseInt(gidString, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("glusterfs: failed to parse gid %v ", gidString)
	}

	if gid64 < 0 {
		return 0, fmt.Errorf("glusterfs: negative GIDs are not allowed: %v", gidString)
	}

	// ParseInt returns a int64, but since we parsed only
	// for 32 bit, we can cast to int without loss:
	gid := int(gid64)
	return gid, nil
}

func convertVolumeParam(volumeString string) (int, error) {

	count, err := strconv.Atoi(volumeString)
	if err != nil {
		return 0, fmt.Errorf("failed to parse %q", volumeString)
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
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Spec.Glusterfs is nil")
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

//
// Traverse the PVs, fetching all the GIDs from those
// in a given storage class, and mark them in the table.
//
func (p *glusterfsPlugin) collectGids(className string, gidTable *MinMaxAllocator) error {
	kubeClient := p.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("glusterfs: failed to get kube client when collecting gids")
	}
	pvList, err := kubeClient.Core().PersistentVolumes().List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	if err != nil {
		glog.Errorf("glusterfs: failed to get existing persistent volumes")
		return err
	}

	for _, pv := range pvList.Items {
		if v1.GetPersistentVolumeClass(&pv) != className {
			continue
		}

		pvName := pv.ObjectMeta.Name

		gidStr, ok := pv.Annotations[volumehelper.VolumeGidAnnotationKey]

		if !ok {
			glog.Warningf("glusterfs: no gid found in pv '%v'", pvName)
			continue
		}

		gid, err := convertGid(gidStr)
		if err != nil {
			glog.Error(err)
			continue
		}

		_, err = gidTable.Allocate(gid)
		if err == ErrConflict {
			glog.Warningf("glusterfs: gid %v found in pv %v was already allocated", gid)
		} else if err != nil {
			glog.Errorf("glusterfs: failed to store gid %v found in pv '%v': %v", gid, pvName, err)
			return err
		}
	}

	return nil
}

//
// Return the gid table for a storage class.
// - If this is the first time, fill it with all the gids
//   used in PVs of this storage class by traversing the PVs.
// - Adapt the range of the table to the current range of the SC.
//
func (p *glusterfsPlugin) getGidTable(className string, min int, max int) (*MinMaxAllocator, error) {
	var err error
	p.gidTableLock.Lock()
	gidTable, ok := p.gidTable[className]
	p.gidTableLock.Unlock()

	if ok {
		err = gidTable.SetRange(min, max)
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
	err = p.collectGids(className, newGidTable)
	if err != nil {
		return nil, err
	}

	// and only reduce the range afterwards
	err = newGidTable.SetRange(min, max)
	if err != nil {
		return nil, err
	}

	// if in the meantime a table appeared, use it

	p.gidTableLock.Lock()
	defer p.gidTableLock.Unlock()

	gidTable, ok = p.gidTable[className]
	if ok {
		err = gidTable.SetRange(min, max)
		if err != nil {
			return nil, err
		}

		return gidTable, nil
	}

	p.gidTable[className] = newGidTable

	return newGidTable, nil
}

func (d *glusterfsVolumeDeleter) getGid() (int, bool, error) {
	gidStr, ok := d.spec.Annotations[volumehelper.VolumeGidAnnotationKey]

	if !ok {
		return 0, false, nil
	}

	gid, err := convertGid(gidStr)

	return gid, true, err
}

func (d *glusterfsVolumeDeleter) Delete() error {
	var err error
	glog.V(2).Infof("glusterfs: delete volume: %s ", d.glusterfsMounter.path)
	volumeName := d.glusterfsMounter.path
	volumeId := dstrings.TrimPrefix(volumeName, volPrefix)
	class, err := volutil.GetClassForVolume(d.plugin.host.GetKubeClient(), d.spec)
	if err != nil {
		return err
	}

	cfg, err := parseClassParameters(class.Parameters, d.plugin.host.GetKubeClient())
	if err != nil {
		return err
	}
	d.provisionerConfig = *cfg

	glog.V(4).Infof("glusterfs: deleting volume %q with configuration %+v", volumeId, d.provisionerConfig)

	gid, exists, err := d.getGid()
	if err != nil {
		glog.Error(err)
	} else if exists {
		gidTable, err := d.plugin.getGidTable(class.Name, cfg.gidMin, cfg.gidMax)
		if err != nil {
			return fmt.Errorf("glusterfs: failed to get gidTable: %v", err)
		}

		err = gidTable.Release(gid)
		if err != nil {
			return fmt.Errorf("glusterfs: failed to release gid %v: %v", gid, err)
		}
	}

	cli := gcli.NewClient(d.url, d.user, d.secretValue)
	if cli == nil {
		glog.Errorf("glusterfs: failed to create glusterfs rest client")
		return fmt.Errorf("glusterfs: failed to create glusterfs rest client, REST server authentication failed")
	}
	err = cli.VolumeDelete(volumeId)
	if err != nil {
		glog.Errorf("glusterfs: error when deleting the volume :%v", err)
		return err
	}
	glog.V(2).Infof("glusterfs: volume %s deleted successfully", volumeName)

	//Deleter takes endpoint and endpointnamespace from pv spec.
	pvSpec := d.spec.Spec
	var dynamicEndpoint, dynamicNamespace string
	if pvSpec.ClaimRef == nil {
		glog.Errorf("glusterfs: ClaimRef is nil")
		return fmt.Errorf("glusterfs: ClaimRef is nil")
	}
	if pvSpec.ClaimRef.Namespace == "" {
		glog.Errorf("glusterfs: namespace is nil")
		return fmt.Errorf("glusterfs: namespace is nil")
	}
	dynamicNamespace = pvSpec.ClaimRef.Namespace
	if pvSpec.Glusterfs.EndpointsName != "" {
		dynamicEndpoint = pvSpec.Glusterfs.EndpointsName
	}
	glog.V(3).Infof("glusterfs: dynamic namespace and endpoint : [%v/%v]", dynamicNamespace, dynamicEndpoint)
	err = d.deleteEndpointService(dynamicNamespace, dynamicEndpoint)
	if err != nil {
		glog.Errorf("glusterfs: error when deleting endpoint/service :%v", err)
	} else {
		glog.V(1).Infof("glusterfs: [%v/%v] deleted successfully ", dynamicNamespace, dynamicEndpoint)
	}
	return nil
}

func (r *glusterfsVolumeProvisioner) Provision() (*v1.PersistentVolume, error) {
	var err error
	if r.options.PVC.Spec.Selector != nil {
		glog.V(4).Infof("glusterfs: not able to parse your claim Selector")
		return nil, fmt.Errorf("glusterfs: not able to parse your claim Selector")
	}
	glog.V(4).Infof("glusterfs: Provison VolumeOptions %v", r.options)
	scName := v1.GetPersistentVolumeClaimClass(r.options.PVC)
	cfg, err := parseClassParameters(r.options.Parameters, r.plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}
	r.provisionerConfig = *cfg

	glog.V(4).Infof("glusterfs: creating volume with configuration %+v", r.provisionerConfig)

	gidTable, err := r.plugin.getGidTable(scName, cfg.gidMin, cfg.gidMax)
	if err != nil {
		return nil, fmt.Errorf("glusterfs: failed to get gidTable: %v", err)
	}

	gid, _, err := gidTable.AllocateNext()
	if err != nil {
		glog.Errorf("glusterfs: failed to reserve gid from table: %v", err)
		return nil, fmt.Errorf("glusterfs: failed to reserve gid from table: %v", err)
	}

	glog.V(2).Infof("glusterfs: got gid [%d] for PVC %s", gid, r.options.PVC.Name)

	glusterfs, sizeGB, err := r.CreateVolume(gid)
	if err != nil {
		if release_err := gidTable.Release(gid); release_err != nil {
			glog.Errorf("glusterfs:  error when releasing gid in storageclass: %s", scName)
		}

		glog.Errorf("glusterfs: create volume err: %v.", err)
		return nil, fmt.Errorf("glusterfs: create volume err: %v.", err)
	}
	pv := new(v1.PersistentVolume)
	pv.Spec.PersistentVolumeSource.Glusterfs = glusterfs
	pv.Spec.PersistentVolumeReclaimPolicy = r.options.PersistentVolumeReclaimPolicy
	pv.Spec.AccessModes = r.options.PVC.Spec.AccessModes
	if len(pv.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = r.plugin.GetAccessModes()
	}

	gidStr := strconv.FormatInt(int64(gid), 10)
	pv.Annotations = map[string]string{volumehelper.VolumeGidAnnotationKey: gidStr}

	pv.Spec.Capacity = v1.ResourceList{
		v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return pv, nil
}

func (p *glusterfsVolumeProvisioner) GetClusterNodes(cli *gcli.Client, cluster string) (dynamicHostIps []string, err error) {
	clusterinfo, err := cli.ClusterInfo(cluster)
	if err != nil {
		glog.Errorf("glusterfs: failed to get cluster details: %v", err)
		return nil, fmt.Errorf("failed to get cluster details: %v", err)
	}

	// For the dynamically provisioned volume, we gather the list of node IPs
	// of the cluster on which provisioned volume belongs to, as there can be multiple
	// clusters.
	for _, node := range clusterinfo.Nodes {
		nodei, err := cli.NodeInfo(string(node))
		if err != nil {
			glog.Errorf("glusterfs: failed to get hostip: %v", err)
			return nil, fmt.Errorf("failed to get hostip: %v", err)
		}
		ipaddr := dstrings.Join(nodei.NodeAddRequest.Hostnames.Storage, "")
		dynamicHostIps = append(dynamicHostIps, ipaddr)
	}
	glog.V(3).Infof("glusterfs: hostlist :%v", dynamicHostIps)
	if len(dynamicHostIps) == 0 {
		glog.Errorf("glusterfs: no hosts found: %v", err)
		return nil, fmt.Errorf("no hosts found: %v", err)
	}
	return dynamicHostIps, nil
}

func (p *glusterfsVolumeProvisioner) CreateVolume(gid int) (r *v1.GlusterfsVolumeSource, size int, err error) {
	var clusterIds []string
	capacity := p.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	volSizeBytes := capacity.Value()
	sz := int(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))
	glog.V(2).Infof("glusterfs: create volume of size: %d bytes and configuration %+v", volSizeBytes, p.provisionerConfig)
	if p.url == "" {
		glog.Errorf("glusterfs : rest server endpoint is empty")
		return nil, 0, fmt.Errorf("failed to create glusterfs REST client, REST URL is empty")
	}
	cli := gcli.NewClient(p.url, p.user, p.secretValue)
	if cli == nil {
		glog.Errorf("glusterfs: failed to create glusterfs rest client")
		return nil, 0, fmt.Errorf("failed to create glusterfs REST client, REST server authentication failed")
	}
	if p.provisionerConfig.clusterId != "" {
		clusterIds = dstrings.Split(p.clusterId, ",")
		glog.V(4).Infof("glusterfs: provided clusterids: %v", clusterIds)
	}
	gid64 := int64(gid)
	volumeReq := &gapi.VolumeCreateRequest{Size: sz, Clusters: clusterIds, Gid: gid64, Durability: p.volumeType}
	volume, err := cli.VolumeCreate(volumeReq)
	if err != nil {
		glog.Errorf("glusterfs: error creating volume %v ", err)
		return nil, 0, fmt.Errorf("error creating volume %v", err)
	}
	glog.V(1).Infof("glusterfs: volume with size: %d and name: %s created", volume.Size, volume.Name)
	dynamicHostIps, err := p.GetClusterNodes(cli, volume.Cluster)
	if err != nil {
		glog.Errorf("glusterfs: error [%v] when getting cluster nodes for volume %s", err, volume)
		return nil, 0, fmt.Errorf("error [%v] when getting cluster nodes for volume %s", err, volume)
	}

	// The 'endpointname' is created in form of 'gluster-dynamic-<claimname>'.
	// createEndpointService() checks for this 'endpoint' existence in PVC's namespace and
	// If not found, it create an endpoint and svc using the IPs we dynamically picked at time
	// of volume creation.
	epServiceName := dynamicEpSvcPrefix + p.options.PVC.Name
	epNamespace := p.options.PVC.Namespace
	endpoint, service, err := p.createEndpointService(epNamespace, epServiceName, dynamicHostIps, p.options.PVC.Name)
	if err != nil {
		glog.Errorf("glusterfs: failed to create endpoint/service: %v", err)
		err = cli.VolumeDelete(volume.Id)
		if err != nil {
			glog.Errorf("glusterfs: error when deleting the volume :%v , manual deletion required", err)
		}
		return nil, 0, fmt.Errorf("failed to create endpoint/service %v", err)
	}
	glog.V(3).Infof("glusterfs: dynamic ep %v and svc : %v ", endpoint, service)
	return &v1.GlusterfsVolumeSource{
		EndpointsName: endpoint.Name,
		Path:          volume.Name,
		ReadOnly:      false,
	}, sz, nil
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
		return nil, nil, fmt.Errorf("glusterfs: failed to get kube client when creating endpoint service")
	}
	_, err = kubeClient.Core().Endpoints(namespace).Create(endpoint)
	if err != nil && errors.IsAlreadyExists(err) {
		glog.V(1).Infof("glusterfs: endpoint [%s] already exist in namespace [%s]", endpoint, namespace)
		err = nil
	}
	if err != nil {
		glog.Errorf("glusterfs: failed to create endpoint: %v", err)
		return nil, nil, fmt.Errorf("error creating endpoint: %v", err)
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
	_, err = kubeClient.Core().Services(namespace).Create(service)
	if err != nil && errors.IsAlreadyExists(err) {
		glog.V(1).Infof("glusterfs: service [%s] already exist in namespace [%s]", service, namespace)
		err = nil
	}
	if err != nil {
		glog.Errorf("glusterfs: failed to create service: %v", err)
		return nil, nil, fmt.Errorf("error creating service: %v", err)
	}
	return endpoint, service, nil
}

func (d *glusterfsVolumeDeleter) deleteEndpointService(namespace string, epServiceName string) (err error) {
	kubeClient := d.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("glusterfs: failed to get kube client when deleting endpoint service")
	}
	err = kubeClient.Core().Services(namespace).Delete(epServiceName, nil)
	if err != nil {
		glog.Errorf("glusterfs: error deleting service %s/%s: %v", namespace, epServiceName, err)
		return fmt.Errorf("error deleting service %s/%s: %v", namespace, epServiceName, err)
	}
	glog.V(1).Infof("glusterfs: service/endpoint %s/%s deleted successfully", namespace, epServiceName)
	return nil
}

// parseSecret finds a given Secret instance and reads user password from it.
func parseSecret(namespace, secretName string, kubeClient clientset.Interface) (string, error) {
	secretMap, err := volutil.GetSecretForPV(namespace, secretName, glusterfsPluginName, kubeClient)
	if err != nil {
		glog.Errorf("failed to get secret %s/%s: %v", namespace, secretName, err)
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

// parseClassParameters parses StorageClass.Parameters
func parseClassParameters(params map[string]string, kubeClient clientset.Interface) (*provisionerConfig, error) {
	var cfg provisionerConfig
	var err error

	cfg.gidMin = defaultGidMin
	cfg.gidMax = defaultGidMax

	authEnabled := true
	parseVolumeType := ""
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
				cfg.clusterId = v
			}
		case "restauthenabled":
			authEnabled = dstrings.ToLower(v) == "true"
		case "gidmin":
			parseGidMin, err := convertGid(v)
			if err != nil {
				return nil, fmt.Errorf("glusterfs: invalid value %q for volume plugin %s", k, glusterfsPluginName)
			}
			if parseGidMin < absoluteGidMin {
				return nil, fmt.Errorf("glusterfs: gidMin must be >= %v", absoluteGidMin)
			}
			if parseGidMin > absoluteGidMax {
				return nil, fmt.Errorf("glusterfs: gidMin must be <= %v", absoluteGidMax)
			}
			cfg.gidMin = parseGidMin
		case "gidmax":
			parseGidMax, err := convertGid(v)
			if err != nil {
				return nil, fmt.Errorf("glusterfs: invalid value %q for volume plugin %s", k, glusterfsPluginName)
			}
			if parseGidMax < absoluteGidMin {
				return nil, fmt.Errorf("glusterfs: gidMax must be >= %v", absoluteGidMin)
			}
			if parseGidMax > absoluteGidMax {
				return nil, fmt.Errorf("glusterfs: gidMax must be <= %v", absoluteGidMax)
			}
			cfg.gidMax = parseGidMax
		case "volumetype":
			parseVolumeType = v

		default:
			return nil, fmt.Errorf("glusterfs: invalid option %q for volume plugin %s", k, glusterfsPluginName)
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
					return nil, fmt.Errorf("error [%v] when parsing value %q of option '%s' for volume plugin %s.", err, parseVolumeTypeInfo[1], "volumetype", glusterfsPluginName)
				}
				cfg.volumeType = gapi.VolumeDurabilityInfo{Type: gapi.DurabilityReplicate, Replicate: gapi.ReplicaDurability{Replica: newReplicaCount}}
			} else {
				cfg.volumeType = gapi.VolumeDurabilityInfo{Type: gapi.DurabilityReplicate, Replicate: gapi.ReplicaDurability{Replica: replicaCount}}
			}
		case "disperse":
			if len(parseVolumeTypeInfo) >= 3 {
				newDisperseData, err := convertVolumeParam(parseVolumeTypeInfo[1])
				if err != nil {
					return nil, fmt.Errorf("error [%v] when parsing value %q of option '%s' for volume plugin %s.", parseVolumeTypeInfo[1], err, "volumetype", glusterfsPluginName)
				}
				newDisperseRedundancy, err := convertVolumeParam(parseVolumeTypeInfo[2])
				if err != nil {
					return nil, fmt.Errorf("error [%v] when parsing value %q of option '%s' for volume plugin %s.", err, parseVolumeTypeInfo[2], "volumetype", glusterfsPluginName)
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

	return &cfg, nil
}
