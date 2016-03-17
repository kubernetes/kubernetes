/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"os"
	"path"
	dstrings "strings"

	"github.com/golang/glog"
	gapp "github.com/heketi/heketi/apps/glusterfs"
	gcli "github.com/heketi/heketi/client/api/go-client"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&glusterfsPlugin{nil, exec.New()}}
}

type glusterfsPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &glusterfsPlugin{}
var _ volume.PersistentVolumePlugin = &glusterfsPlugin{}
var _ volume.DeletableVolumePlugin = &glusterfsPlugin{}
var _ volume.ProvisionableVolumePlugin = &glusterfsPlugin{}
var _ volume.Provisioner = &glusterfsVolumeProvisioner{}
var _ volume.Deleter = &glusterfsVolumeDeleter{}

const (
	glusterfsPluginName = "kubernetes.io/glusterfs"
)

func (plugin *glusterfsPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *glusterfsPlugin) Name() string {
	return glusterfsPluginName
}

func (plugin *glusterfsPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Glusterfs == nil) ||
		(spec.Volume != nil && spec.Volume.Glusterfs == nil) {
		return false
	}

	return true

}

func (plugin *glusterfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *glusterfsPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Builder, error) {
	source, _ := plugin.getGlusterVolumeSource(spec)
	ep_name := source.EndpointsName
	ns := pod.Namespace
	ep, err := plugin.host.GetKubeClient().Core().Endpoints(ns).Get(ep_name)
	if err != nil {
		glog.Errorf("glusterfs: failed to get endpoints %s[%v]", ep_name, err)
		return nil, err
	}
	glog.V(1).Infof("glusterfs: endpoints %v", ep)
	return plugin.newBuilderInternal(spec, ep, pod, plugin.host.GetMounter(), exec.New())
}

func (plugin *glusterfsPlugin) getGlusterVolumeSource(spec *volume.Spec) (*api.GlusterfsVolumeSource, bool) {
	// Glusterfs volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// Glusterfs volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	if spec.Volume != nil && spec.Volume.Glusterfs != nil {
		return spec.Volume.Glusterfs, spec.Volume.Glusterfs.ReadOnly
	} else {
		return spec.PersistentVolume.Spec.Glusterfs, spec.ReadOnly
	}
}

func (plugin *glusterfsPlugin) newBuilderInternal(spec *volume.Spec, ep *api.Endpoints, pod *api.Pod, mounter mount.Interface, exe exec.Interface) (volume.Builder, error) {
	source, readOnly := plugin.getGlusterVolumeSource(spec)
	return &glusterfsBuilder{
		glusterfs: &glusterfs{
			volName: spec.Name(),
			mounter: mounter,
			pod:     pod,
			plugin:  plugin,
		},
		hosts:    ep,
		path:     source.Path,
		readOnly: readOnly,
		exe:      exe}, nil
}

func (plugin *glusterfsPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *glusterfsPlugin) newCleanerInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	return &glusterfsCleaner{&glusterfs{
		volName: volName,
		mounter: mounter,
		pod:     &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
		plugin:  plugin,
	}}, nil
}

func (plugin *glusterfsPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

// Glusterfs volumes represent a bare host file or directory mount of an Glusterfs export.
type glusterfs struct {
	volName string
	pod     *api.Pod
	mounter mount.Interface
	plugin  *glusterfsPlugin
	volume.MetricsNil
}

type glusterfsBuilder struct {
	*glusterfs
	hosts    *api.Endpoints
	path     string
	readOnly bool
	exe      exec.Interface
}

var _ volume.Builder = &glusterfsBuilder{}

func (b *glusterfsBuilder) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *glusterfsBuilder) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *glusterfsBuilder) SetUpAt(dir string, fsGroup *int64) error {
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
	c := &glusterfsCleaner{b.glusterfs}
	c.cleanup(dir)
	return err
}

func (glusterfsVolume *glusterfs) GetPath() string {
	name := glusterfsPluginName
	return glusterfsVolume.plugin.host.GetPodVolumeDir(glusterfsVolume.pod.UID, strings.EscapeQualifiedNameForDisk(name), glusterfsVolume.volName)
}

type glusterfsCleaner struct {
	*glusterfs
}

var _ volume.Cleaner = &glusterfsCleaner{}

func (c *glusterfsCleaner) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *glusterfsCleaner) TearDownAt(dir string) error {
	return c.cleanup(dir)
}

func (c *glusterfsCleaner) cleanup(dir string) error {
	notMnt, err := c.mounter.IsLikelyNotMountPoint(dir)
	if err != nil {
		return fmt.Errorf("glusterfs: Error checking IsLikelyNotMountPoint: %v", err)
	}
	if notMnt {
		return os.RemoveAll(dir)
	}

	if err := c.mounter.Unmount(dir); err != nil {
		return fmt.Errorf("glusterfs: Unmounting failed: %v", err)
	}
	notMnt, mntErr := c.mounter.IsLikelyNotMountPoint(dir)
	if mntErr != nil {
		return fmt.Errorf("glusterfs: IsLikelyNotMountPoint check failed: %v", mntErr)
	}
	if notMnt {
		if err := os.RemoveAll(dir); err != nil {
			return fmt.Errorf("glusterfs: RemoveAll failed: %v", err)
		}
	}

	return nil
}

func (b *glusterfsBuilder) setUpAtInternal(dir string) error {
	var errs error

	options := []string{}
	if b.readOnly {
		options = append(options, "ro")
	}

	p := path.Join(b.glusterfs.plugin.host.GetPluginDir(glusterfsPluginName), b.glusterfs.volName)
	if err := os.MkdirAll(p, 0750); err != nil {
		return fmt.Errorf("glusterfs: mkdir failed: %v", err)
	}
	log := path.Join(p, "glusterfs.log")
	options = append(options, "log-file="+log)

	addr := make(map[string]struct{})
	for _, s := range b.hosts.Subsets {
		for _, a := range s.Addresses {
			addr[a.IP] = struct{}{}
		}
	}

	// Avoid mount storm, pick a host randomly.
	// Iterate all hosts until mount succeeds.
	for hostIP := range addr {
		errs = b.mounter.Mount(hostIP+":"+b.path, dir, "glusterfs", options)
		if errs == nil {
			return nil
		}
	}
	return fmt.Errorf("glusterfs: mount failed: %v", errs)
}

func (plugin *glusterfsPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	if len(options.AccessModes) == 0 {
		options.AccessModes = plugin.GetAccessModes()
	}
	return plugin.newProvisionerInternal(options)
}

func (plugin *glusterfsPlugin) newProvisionerInternal(options volume.VolumeOptions) (volume.Provisioner, error) {
	return &glusterfsVolumeProvisioner{
		glusterfsBuilder: &glusterfsBuilder{
			glusterfs: &glusterfs{
				plugin: plugin,
			},
		},
		options: options,
	}, nil
}

type glusterfsClusterConf struct {
	Glusterep          string `json:"endpoint"`
	GlusterRestvolpath string `json:"path"`
	GlusterRestUrl     string `json:"resturl"`
	GlusterRestAuth    bool   `json:"restauthenabled"`
	GlusterRestUser    string `json:"restuser"`
	GlusterRestUserKey string `json:"restuserkey"`
}

type glusterfsVolumeProvisioner struct {
	*glusterfsBuilder
	*glusterfsClusterConf
	options volume.VolumeOptions
}

var clusterconf = new(glusterfsClusterConf)

func (plugin *glusterfsPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec)
}

func (plugin *glusterfsPlugin) newDeleterInternal(spec *volume.Spec) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Glusterfs == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Spec.Glusterfs is nil")
	}
	return &glusterfsVolumeDeleter{
		glusterfsBuilder: &glusterfsBuilder{
			glusterfs: &glusterfs{

				volName: spec.Name(),
				plugin:  plugin,
			},

			path: spec.PersistentVolume.Spec.Glusterfs.Path,
		}}, nil
}

type glusterfsVolumeDeleter struct {
	*glusterfsBuilder
	*glusterfsClusterConf
}

func (d *glusterfsVolumeDeleter) GetPath() string {
	name := glusterfsPluginName
	return d.plugin.host.GetPodVolumeDir(d.glusterfsBuilder.glusterfs.pod.UID, strings.EscapeQualifiedNameForDisk(name), d.glusterfsBuilder.glusterfs.volName)
}

func (d *glusterfsVolumeDeleter) Delete() error {
	return d.DeleteVolume()
}

func (d *glusterfsVolumeDeleter) DeleteVolume() error {
	glog.V(1).Infof("glusterfs: delete volume :%s ", d.glusterfsBuilder.path)
	volumetodel := d.glusterfsBuilder.path
	d.glusterfsClusterConf = clusterconf
	newvolumetodel := dstrings.TrimPrefix(volumetodel, "vol_")
	cli := gcli.NewClient(d.GlusterRestUrl, d.GlusterRestUser, d.GlusterRestUserKey)
	if cli == nil {
		glog.V(1).Infof("glusterfs: failed to create gluster rest client")
		return fmt.Errorf("glusterfs: failed to create gluster rest client")
	}
	err := cli.VolumeDelete(newvolumetodel)
	if err != nil {
		glog.V(1).Infof("glusterfs: error when deleting the volume :%s", err)
		return err
	}
	glog.V(1).Infof("glusterfs: volume %s deleted successfully", volumetodel)
	return nil

}

func (r *glusterfsVolumeProvisioner) Provision(pv *api.PersistentVolume) error {

	config := r.glusterfsBuilder.plugin.host.GetStorageConfigDir()
	if config == "" {
		glog.V(1).Infof("glusterfs: no cluster storage config file provided")
		return fmt.Errorf("glusterfs: no gluster cluster storage config provided")
	}
	file := path.Join(config, "gluster.json")
	fp, err := os.Open(file)
	if err != nil {
		glog.V(1).Infof("glusterfs cluster configuration file open err %s/%s", file, err)
		return fmt.Errorf("glusterfs cluster configuration file open err %s/%s", file, err)
	}
	defer fp.Close()

	decoder := json.NewDecoder(fp)
	if err = decoder.Decode(clusterconf); err != nil {
		glog.V(1).Infof("glusterfs: decode err: %s.", err)
		return fmt.Errorf("glusterfs: decode err: %s.", err)
	}

	r.glusterfsClusterConf = clusterconf
	glusterfs, sizeGB, err := r.CreateVolume()
	if err != nil {
		glog.V(1).Infof("glusterfs: create volume err: %s.", err)
		return fmt.Errorf("glusterfs: create volume err: %s.", err)
	}

	pv.Spec.PersistentVolumeSource.Glusterfs = glusterfs

	pv.Spec.Capacity = api.ResourceList{
		api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return nil
}

func (c *glusterfsVolumeProvisioner) NewPersistentVolumeTemplate() (*api.PersistentVolume, error) {
	// Provide glusterfs api.PersistentVolume.Spec, it will be filled in
	// glusterfsVolumeProvisioner.Provision()

	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pv-glusterfs-",
			Labels:       map[string]string{},
			Annotations: map[string]string{
				"kubernetes.io/createdby": "glusterfs-dynamic-provisioner",
			},
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: c.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   c.options.AccessModes,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): c.options.Capacity,
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				Glusterfs: &api.GlusterfsVolumeSource{

					EndpointsName: "test",
					Path:          "foo",
					ReadOnly:      true,
				},
			},
		},
	}, nil

}

func (p *glusterfsVolumeProvisioner) CreateVolume() (r *api.GlusterfsVolumeSource, size int, err error) {
	volSizeBytes := p.options.Capacity.Value()
	const gb = 1024 * 1024 * 1024
	sz := int((volSizeBytes + gb - 1) / gb)
	glog.V(1).Infof("glusterfs: create volume of size:%s ", volSizeBytes)
	if *(p.glusterfsClusterConf) == (glusterfsClusterConf{}) {
		glog.V(1).Infof("glusterfs : rest server endpoint is empty")
	}
	cli := gcli.NewClient(p.GlusterRestUrl, p.GlusterRestUser, p.GlusterRestUserKey)
	if cli == nil {
		glog.V(1).Infof("glusterfs: failed to create gluster rest client")
	}
	volumeReq := &gapp.VolumeCreateRequest{Size: sz}
	volume, err := cli.VolumeCreate(volumeReq)
	if err != nil {
		glog.V(1).Infof("glusterfs: error [%s] when creating the volume", err)
	}
	glog.V(1).Infof("\n glusterfs: volume with size :%d and name:%s created", volume.Size, volume.Name)
	return &api.GlusterfsVolumeSource{
		EndpointsName: p.glusterfsClusterConf.Glusterep,
		Path:          volume.Name,
		ReadOnly:      false,
	}, sz, nil
}
