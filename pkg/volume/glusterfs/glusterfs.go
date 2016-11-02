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
	"os"
	"path"
	dstrings "strings"

	"github.com/golang/glog"
	gcli "github.com/heketi/heketi/client/api/go-client"
	gapi "github.com/heketi/heketi/pkg/glusterfs/api"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
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
	glusterfsPluginName       = "kubernetes.io/glusterfs"
	volPrefix                 = "vol_"
	dynamicEpSvcPrefix        = "gluster-dynamic-"
	replicaCount              = 3
	durabilityType            = "replicate"
	secretKeyName             = "key" // key name used in secret
	annGlusterURL             = "glusterfs.kubernetes.io/url"
	annGlusterSecretName      = "glusterfs.kubernetes.io/secretname"
	annGlusterSecretNamespace = "glusterfs.kubernetes.io/secretnamespace"
	annGlusterUserKey         = "glusterfs.kubernetes.io/userkey"
	annGlusterUser            = "glusterfs.kubernetes.io/userid"
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

func (plugin *glusterfsPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func (plugin *glusterfsPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	source, _ := plugin.getGlusterVolumeSource(spec)
	ep_name := source.EndpointsName
	// PVC/POD is in same ns.
	ns := pod.Namespace
	ep, err := plugin.host.GetKubeClient().Core().Endpoints(ns).Get(ep_name)
	if err != nil {
		glog.Errorf("glusterfs: failed to get endpoints %s[%v]", ep_name, err)
		return nil, err
	}
	glog.V(1).Infof("glusterfs: endpoints %v", ep)
	return plugin.newMounterInternal(spec, ep, pod, plugin.host.GetMounter(), exec.New())
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

func (plugin *glusterfsPlugin) newMounterInternal(spec *volume.Spec, ep *api.Endpoints, pod *api.Pod, mounter mount.Interface, exe exec.Interface) (volume.Mounter, error) {
	source, readOnly := plugin.getGlusterVolumeSource(spec)
	return &glusterfsMounter{
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

func (plugin *glusterfsPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *glusterfsPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &glusterfsUnmounter{&glusterfs{
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

func (plugin *glusterfsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	glusterfsVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			Glusterfs: &api.GlusterfsVolumeSource{
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
	pod     *api.Pod
	mounter mount.Interface
	plugin  *glusterfsPlugin
	volume.MetricsNil
}

type glusterfsMounter struct {
	*glusterfs
	hosts    *api.Endpoints
	path     string
	readOnly bool
	exe      exec.Interface
}

var _ volume.Mounter = &glusterfsMounter{}

func (b *glusterfsMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
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
	c := &glusterfsUnmounter{b.glusterfs}
	c.cleanup(dir)
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
	return c.cleanup(dir)
}

func (c *glusterfsUnmounter) cleanup(dir string) error {
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
			errs = b.mounter.Mount(ip+":"+b.path, dir, "glusterfs", options)
			if errs == nil {
				glog.Infof("glusterfs: successfully mounted %s", dir)
				return nil
			}
		}
	}

	// Failed mount scenario.
	// Since gluster does not return error text
	// it all goes in a log file, we will read the log file
	logerror := readGlusterLog(log, b.pod.Name)
	if logerror != nil {
		// return fmt.Errorf("glusterfs: mount failed: %v", logerror)
		return fmt.Errorf("glusterfs: mount failed: %v the following error information was pulled from the glusterfs log to help diagnose this issue: %v", errs, logerror)
	}
	return fmt.Errorf("glusterfs: mount failed: %v", errs)
}

func getVolumeSource(
	spec *volume.Spec) (*api.GlusterfsVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Glusterfs != nil {
		return spec.Volume.Glusterfs, spec.Volume.Glusterfs.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Glusterfs != nil {
		return spec.PersistentVolume.Spec.Glusterfs, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Gluster volume type")
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

type provisioningConfig struct {
	url             string
	user            string
	userKey         string
	secretNamespace string
	secretName      string
	secretValue     string
}

type glusterfsVolumeProvisioner struct {
	*glusterfsMounter
	provisioningConfig
	options volume.VolumeOptions
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
	provisioningConfig
	spec *api.PersistentVolume
}

func (d *glusterfsVolumeDeleter) GetPath() string {
	name := glusterfsPluginName
	return d.plugin.host.GetPodVolumeDir(d.glusterfsMounter.glusterfs.pod.UID, strings.EscapeQualifiedNameForDisk(name), d.glusterfsMounter.glusterfs.volName)
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
	d.provisioningConfig = *cfg

	glog.V(4).Infof("glusterfs: deleting volume %q with configuration %+v", volumeId, d.provisioningConfig)

	cli := gcli.NewClient(d.url, d.user, d.secretValue)
	if cli == nil {
		glog.Errorf("glusterfs: failed to create gluster rest client")
		return fmt.Errorf("glusterfs: failed to create gluster rest client, REST server authentication failed")
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

func (r *glusterfsVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	var err error
	if r.options.PVC.Spec.Selector != nil {
		glog.V(4).Infof("glusterfs: not able to parse your claim Selector")
		return nil, fmt.Errorf("glusterfs: not able to parse your claim Selector")
	}
	glog.V(4).Infof("glusterfs: Provison VolumeOptions %v", r.options)

	cfg, err := parseClassParameters(r.options.Parameters, r.plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}
	r.provisioningConfig = *cfg

	glog.V(4).Infof("glusterfs: creating volume with configuration %+v", r.provisioningConfig)
	glusterfs, sizeGB, err := r.CreateVolume()
	if err != nil {
		glog.Errorf("glusterfs: create volume err: %v.", err)
		return nil, fmt.Errorf("glusterfs: create volume err: %v.", err)
	}
	pv := new(api.PersistentVolume)
	pv.Spec.PersistentVolumeSource.Glusterfs = glusterfs
	pv.Spec.PersistentVolumeReclaimPolicy = r.options.PersistentVolumeReclaimPolicy
	pv.Spec.AccessModes = r.options.PVC.Spec.AccessModes
	if len(pv.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = r.plugin.GetAccessModes()
	}
	pv.Spec.Capacity = api.ResourceList{
		api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return pv, nil
}

func (p *glusterfsVolumeProvisioner) CreateVolume() (r *api.GlusterfsVolumeSource, size int, err error) {
	capacity := p.options.PVC.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
	volSizeBytes := capacity.Value()
	sz := int(volume.RoundUpSize(volSizeBytes, 1024*1024*1024))
	glog.V(2).Infof("glusterfs: create volume of size: %d bytes and configuration %+v", volSizeBytes, p.provisioningConfig)
	if p.url == "" {
		glog.Errorf("glusterfs : rest server endpoint is empty")
		return nil, 0, fmt.Errorf("failed to create gluster REST client, REST URL is empty")
	}
	cli := gcli.NewClient(p.url, p.user, p.secretValue)
	if cli == nil {
		glog.Errorf("glusterfs: failed to create gluster rest client")
		return nil, 0, fmt.Errorf("failed to create gluster REST client, REST server authentication failed")
	}
	volumeReq := &gapi.VolumeCreateRequest{Size: sz, Durability: gapi.VolumeDurabilityInfo{Type: durabilityType, Replicate: gapi.ReplicaDurability{Replica: replicaCount}}}
	volume, err := cli.VolumeCreate(volumeReq)
	if err != nil {
		glog.Errorf("glusterfs: error creating volume %v ", err)
		return nil, 0, fmt.Errorf("error creating volume %v", err)
	}
	glog.V(1).Infof("glusterfs: volume with size: %d and name: %s created", volume.Size, volume.Name)
	clusterinfo, err := cli.ClusterInfo(volume.Cluster)
	if err != nil {
		glog.Errorf("glusterfs: failed to get cluster details")
		return nil, 0, fmt.Errorf("failed to get cluster details %v", err)
	}
	// For the above dynamically provisioned volume, we gather the list of node IPs
	// of the cluster on which provisioned volume belongs to, as there can be multiple
	// clusters.
	var dynamicHostIps []string
	for _, node := range clusterinfo.Nodes {
		nodei, err := cli.NodeInfo(string(node))
		if err != nil {
			glog.Errorf("glusterfs: failed to get hostip %v ", err)
			return nil, 0, fmt.Errorf("failed to get hostip %v", err)
		}
		ipaddr := dstrings.Join(nodei.NodeAddRequest.Hostnames.Storage, "")
		dynamicHostIps = append(dynamicHostIps, ipaddr)
	}
	glog.V(3).Infof("glusterfs: hostlist :%v", dynamicHostIps)
	if len(dynamicHostIps) == 0 {
		glog.Errorf("glusterfs: no hosts found %v ", err)
		return nil, 0, fmt.Errorf("no hosts found %v", err)
	}

	// The 'endpointname' is created in form of 'gluster-dynamic-<claimname>'.
	// createEndpointService() checks for this 'endpoint' existence in PVC's namespace and
	// If not found, it create an endpoint and svc using the IPs we dynamically picked at time
	// of volume creation.
	epServiceName := dynamicEpSvcPrefix + p.options.PVC.Name
	epNamespace := p.options.PVC.Namespace
	endpoint, service, err := p.createEndpointService(epNamespace, epServiceName, dynamicHostIps, p.options.PVC.Name)
	if err != nil {
		glog.Errorf("glusterfs: failed to create endpoint/service")
		err = cli.VolumeDelete(volume.Id)
		if err != nil {
			glog.Errorf("glusterfs: error when deleting the volume :%v , manual deletion required", err)
		}
		return nil, 0, fmt.Errorf("failed to create endpoint/service %v", err)
	}
	glog.V(3).Infof("glusterfs: dynamic ep %v and svc : %v ", endpoint, service)
	return &api.GlusterfsVolumeSource{
		EndpointsName: endpoint.Name,
		Path:          volume.Name,
		ReadOnly:      false,
	}, sz, nil
}

func (p *glusterfsVolumeProvisioner) createEndpointService(namespace string, epServiceName string, hostips []string, pvcname string) (endpoint *api.Endpoints, service *api.Service, err error) {

	addrlist := make([]api.EndpointAddress, len(hostips))
	for i, v := range hostips {
		addrlist[i].IP = v
	}
	endpoint = &api.Endpoints{
		ObjectMeta: api.ObjectMeta{
			Namespace: namespace,
			Name:      epServiceName,
			Labels: map[string]string{
				"gluster.kubernetes.io/provisioned-for-pvc": pvcname,
			},
		},
		Subsets: []api.EndpointSubset{{
			Addresses: addrlist,
			Ports:     []api.EndpointPort{{Port: 1, Protocol: "TCP"}},
		}},
	}
	_, err = p.plugin.host.GetKubeClient().Core().Endpoints(namespace).Create(endpoint)
	if err != nil && errors.IsAlreadyExists(err) {
		glog.V(1).Infof("glusterfs: endpoint [%s] already exist in namespace [%s]", endpoint, namespace)
		err = nil
	}
	if err != nil {
		glog.Errorf("glusterfs: failed to create endpoint %v", err)
		return nil, nil, fmt.Errorf("error creating endpoint %v", err)
	}
	service = &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name:      epServiceName,
			Namespace: namespace,
			Labels: map[string]string{
				"gluster.kubernetes.io/provisioned-for-pvc": pvcname,
			},
		},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{
				{Protocol: "TCP", Port: 1}}}}
	_, err = p.plugin.host.GetKubeClient().Core().Services(namespace).Create(service)
	if err != nil && errors.IsAlreadyExists(err) {
		glog.V(1).Infof("glusterfs: service [%s] already exist in namespace [%s]", service, namespace)
		err = nil
	}
	if err != nil {
		glog.Errorf("glusterfs: failed to create service %v", err)
		return nil, nil, fmt.Errorf("error creating service %v", err)
	}
	return endpoint, service, nil
}

func (d *glusterfsVolumeDeleter) deleteEndpointService(namespace string, epServiceName string) (err error) {
	err = d.plugin.host.GetKubeClient().Core().Services(namespace).Delete(epServiceName, nil)
	if err != nil {
		glog.Errorf("glusterfs: failed to delete service %s in namespace %s error %v", epServiceName, namespace, err)
		return fmt.Errorf("error deleting service %v in namespace [%s]", err, namespace)
	}
	glog.V(1).Infof("glusterfs: service/endpoint [%s] in namespace [%s] deleted successfully", epServiceName, namespace)
	return nil
}

// parseSecret finds a given Secret instance and reads user password from it.
func parseSecret(namespace, secretName string, kubeClient clientset.Interface) (string, error) {
	secretMap, err := volutil.GetSecretForPV(namespace, secretName, glusterfsPluginName, kubeClient)
	if err != nil {
		glog.Errorf("failed to get secret from [%q/%q]", namespace, secretName)
		return "", fmt.Errorf("failed to get secret from [%q/%q]", namespace, secretName)
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
func parseClassParameters(params map[string]string, kubeClient clientset.Interface) (*provisioningConfig, error) {
	var cfg provisioningConfig
	var err error

	authEnabled := true
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
		case "restauthenabled":
			authEnabled = dstrings.ToLower(v) == "true"
		default:
			return nil, fmt.Errorf("glusterfs: invalid option %q for volume plugin %s", k, glusterfsPluginName)
		}
	}

	if len(cfg.url) == 0 {
		return nil, fmt.Errorf("StorageClass for provisioner %s must contain 'resturl' parameter", glusterfsPluginName)
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
	return &cfg, nil
}
