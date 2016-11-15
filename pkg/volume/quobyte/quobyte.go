/*
Copyright 2016 The Kubernetes Authors.

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

package quobyte

import (
	"fmt"
	"os"
	"path"
	goStrings "strings"

	"github.com/golang/glog"
	"github.com/pborman/uuid"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&quobytePlugin{nil}}
}

type quobytePlugin struct {
	host volume.VolumeHost
}

// This user is used to authenticate against the
// Quobyte API server and holds all information
type quobyteAPIConfig struct {
	quobyteUser      string
	quobytePassword  string
	quobyteAPIServer string
}

var _ volume.VolumePlugin = &quobytePlugin{}
var _ volume.PersistentVolumePlugin = &quobytePlugin{}
var _ volume.DeletableVolumePlugin = &quobytePlugin{}
var _ volume.ProvisionableVolumePlugin = &quobytePlugin{}
var _ volume.Provisioner = &quobyteVolumeProvisioner{}
var _ volume.Deleter = &quobyteVolumeDeleter{}

const (
	quobytePluginName = "kubernetes.io/quobyte"

	annotationQuobyteAPIServer          = "quobyte.kubernetes.io/api"
	annotationQuobyteAPISecret          = "quobyte.kubernetes.io/apiuser"
	annotationQuobyteAPISecretNamespace = "quobyte.kubernetes.io/apipassword"
)

func (plugin *quobytePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *quobytePlugin) GetPluginName() string {
	return quobytePluginName
}

func (plugin *quobytePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v:%v",
		volumeSource.Registry,
		volumeSource.Volume), nil
}

func (plugin *quobytePlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Quobyte == nil) ||
		(spec.Volume != nil && spec.Volume.Quobyte == nil) {
		return false
	}

	// If Quobyte is already mounted we don't need to check if the binary is installed
	if mounter, err := plugin.newMounterInternal(spec, nil, plugin.host.GetMounter()); err == nil {
		qm, _ := mounter.(*quobyteMounter)
		pluginDir := plugin.host.GetPluginDir(strings.EscapeQualifiedNameForDisk(quobytePluginName))
		if mounted, err := qm.pluginDirIsMounted(pluginDir); mounted && err == nil {
			glog.V(4).Infof("quobyte: can support")
			return true
		}
	} else {
		glog.V(4).Infof("quobyte: Error: %v", err)
	}

	if out, err := exec.New().Command("ls", "/sbin/mount.quobyte").CombinedOutput(); err == nil {
		glog.V(4).Infof("quobyte: can support: %s", string(out))
		return true
	}

	return false
}

func (plugin *quobytePlugin) RequiresRemount() bool {
	return false
}

func (plugin *quobytePlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
		api.ReadWriteMany,
	}
}

func getVolumeSource(spec *volume.Spec) (*api.QuobyteVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.Quobyte != nil {
		return spec.Volume.Quobyte, spec.Volume.Quobyte.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.Quobyte != nil {
		return spec.PersistentVolume.Spec.Quobyte, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a Quobyte volume type")
}

func (plugin *quobytePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	quobyteVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			Quobyte: &api.QuobyteVolumeSource{
				Volume: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(quobyteVolume), nil
}

func (plugin *quobytePlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod, plugin.host.GetMounter())
}

func (plugin *quobytePlugin) newMounterInternal(spec *volume.Spec, pod *api.Pod, mounter mount.Interface) (volume.Mounter, error) {
	source, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &quobyteMounter{
		quobyte: &quobyte{
			volName: spec.Name(),
			user:    source.User,
			group:   source.Group,
			mounter: mounter,
			pod:     pod,
			volume:  source.Volume,
			plugin:  plugin,
		},
		registry: source.Registry,
		readOnly: readOnly,
	}, nil
}

func (plugin *quobytePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter())
}

func (plugin *quobytePlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &quobyteUnmounter{
		&quobyte{
			volName: volName,
			mounter: mounter,
			pod:     &api.Pod{ObjectMeta: api.ObjectMeta{UID: podUID}},
			plugin:  plugin,
		},
	}, nil
}

// Quobyte volumes represent a bare host directory mount of an quobyte export.
type quobyte struct {
	volName string
	pod     *api.Pod
	user    string
	group   string
	volume  string
	tenant  string
	config  string
	mounter mount.Interface
	plugin  *quobytePlugin
	volume.MetricsNil
}

type quobyteMounter struct {
	*quobyte
	registry string
	readOnly bool
}

var _ volume.Mounter = &quobyteMounter{}

func (mounter *quobyteMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        mounter.readOnly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (mounter *quobyteMounter) CanMount() error {
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (mounter *quobyteMounter) SetUp(fsGroup *int64) error {
	pluginDir := mounter.plugin.host.GetPluginDir(strings.EscapeQualifiedNameForDisk(quobytePluginName))
	return mounter.SetUpAt(pluginDir, fsGroup)
}

func (mounter *quobyteMounter) SetUpAt(dir string, fsGroup *int64) error {
	// Check if Quobyte is already mounted on the host in the Plugin Dir
	// if so we can use this mountpoint instead of creating a new one
	// IsLikelyNotMountPoint wouldn't check the mount type
	if mounted, err := mounter.pluginDirIsMounted(dir); err != nil {
		return err
	} else if mounted {
		return nil
	}

	os.MkdirAll(dir, 0750)
	var options []string
	if mounter.readOnly {
		options = append(options, "ro")
	}

	//if a trailing slash is missing we add it here
	if err := mounter.mounter.Mount(mounter.correctTraillingSlash(mounter.registry), dir, "quobyte", options); err != nil {
		return fmt.Errorf("quobyte: mount failed: %v", err)
	}

	glog.V(4).Infof("quobyte: mount set up: %s", dir)

	return nil
}

// GetPath returns the path to the user specific mount of a Quobyte volume
// Returns a path in the format ../user#group@volume
func (quobyteVolume *quobyte) GetPath() string {
	user := quobyteVolume.user
	if len(user) == 0 {
		user = "root"
	}

	group := quobyteVolume.group
	if len(group) == 0 {
		group = "nfsnobody"
	}

	// Quobyte has only one mount in the PluginDir where all Volumes are mounted
	// The Quobyte client does a fixed-user mapping
	pluginDir := quobyteVolume.plugin.host.GetPluginDir(strings.EscapeQualifiedNameForDisk(quobytePluginName))
	return path.Join(pluginDir, fmt.Sprintf("%s#%s@%s", user, group, quobyteVolume.volume))
}

type quobyteUnmounter struct {
	*quobyte
}

var _ volume.Unmounter = &quobyteUnmounter{}

func (unmounter *quobyteUnmounter) TearDown() error {
	return unmounter.TearDownAt(unmounter.GetPath())
}

// We don't need to unmount on the host because only one mount exists
func (unmounter *quobyteUnmounter) TearDownAt(dir string) error {
	return nil
}

type quobyteVolumeDeleter struct {
	*quobyteMounter
	pv *api.PersistentVolume
}

func (plugin *quobytePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.Quobyte == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Spec.Quobyte is nil")
	}

	return plugin.newDeleterInternal(spec)
}

func (plugin *quobytePlugin) newDeleterInternal(spec *volume.Spec) (volume.Deleter, error) {
	source, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	return &quobyteVolumeDeleter{
		quobyteMounter: &quobyteMounter{
			quobyte: &quobyte{
				volName: spec.Name(),
				user:    source.User,
				group:   source.Group,
				volume:  source.Volume,
				plugin:  plugin,
			},
			registry: source.Registry,
			readOnly: readOnly,
		},
		pv: spec.PersistentVolume,
	}, nil
}

func (plugin *quobytePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options)
}

func (plugin *quobytePlugin) newProvisionerInternal(options volume.VolumeOptions) (volume.Provisioner, error) {
	return &quobyteVolumeProvisioner{
		quobyteMounter: &quobyteMounter{
			quobyte: &quobyte{
				plugin: plugin,
			},
		},
		options: options,
	}, nil
}

type quobyteVolumeProvisioner struct {
	*quobyteMounter
	options volume.VolumeOptions
}

func (provisioner *quobyteVolumeProvisioner) Provision() (*api.PersistentVolume, error) {
	if provisioner.options.PVC.Spec.Selector != nil {
		return nil, fmt.Errorf("claim Selector is not supported")
	}
	provisioner.config = "BASE"
	provisioner.tenant = "DEFAULT"

	cfg, err := parseAPIConfig(provisioner.plugin, provisioner.options.Parameters)
	if err != nil {
		return nil, err
	}
	for k, v := range provisioner.options.Parameters {
		switch goStrings.ToLower(k) {
		case "registry":
			provisioner.registry = v
		case "user":
			provisioner.user = v
		case "group":
			provisioner.group = v
		case "quobytetenant":
			provisioner.tenant = v
		case "quobyteconfig":
			provisioner.config = v
		case "adminsecretname",
			"adminsecretnamespace",
			"quobyteapiserver":
			continue
		default:
			return nil, fmt.Errorf("invalid option %q for volume plugin %s", k, provisioner.plugin.GetPluginName())
		}
	}

	if !validateRegistry(provisioner.registry) {
		return nil, fmt.Errorf("Quoybte registry missing or malformed: must be a host:port pair or multiple pairs separated by commas")
	}

	// create random image name
	provisioner.volume = fmt.Sprintf("kubernetes-dynamic-pvc-%s", uuid.NewUUID())

	manager := &quobyteVolumeManager{
		config: cfg,
	}

	vol, sizeGB, err := manager.createVolume(provisioner)
	if err != nil {
		return nil, err
	}
	pv := new(api.PersistentVolume)
	pv.Spec.PersistentVolumeSource.Quobyte = vol
	pv.Spec.PersistentVolumeReclaimPolicy = provisioner.options.PersistentVolumeReclaimPolicy
	pv.Spec.AccessModes = provisioner.options.PVC.Spec.AccessModes
	if len(pv.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = provisioner.plugin.GetAccessModes()
	}
	pv.Spec.Capacity = api.ResourceList{
		api.ResourceName(api.ResourceStorage): resource.MustParse(fmt.Sprintf("%dGi", sizeGB)),
	}
	return pv, nil
}

func (deleter *quobyteVolumeDeleter) GetPath() string {
	return deleter.quobyte.GetPath()
}

func (deleter *quobyteVolumeDeleter) Delete() error {
	class, err := util.GetClassForVolume(deleter.plugin.host.GetKubeClient(), deleter.pv)
	if err != nil {
		return err
	}

	cfg, err := parseAPIConfig(deleter.plugin, class.Parameters)
	if err != nil {
		return err
	}
	manager := &quobyteVolumeManager{
		config: cfg,
	}
	return manager.deleteVolume(deleter)
}

// Parse API configuration (url, username and password) out of class.Parameters.
func parseAPIConfig(plugin *quobytePlugin, params map[string]string) (*quobyteAPIConfig, error) {
	var apiServer, secretName string
	secretNamespace := "default"

	deleteKeys := []string{}

	for k, v := range params {
		switch goStrings.ToLower(k) {
		case "adminsecretname":
			secretName = v
			deleteKeys = append(deleteKeys, k)
		case "adminsecretnamespace":
			secretNamespace = v
			deleteKeys = append(deleteKeys, k)
		case "quobyteapiserver":
			apiServer = v
			deleteKeys = append(deleteKeys, k)
		}
	}

	if len(apiServer) == 0 {
		return nil, fmt.Errorf("Quoybte API server missing or malformed: must be a http(s)://host:port pair or multiple pairs separated by commas")
	}

	secretMap, err := util.GetSecretForPV(secretNamespace, secretName, quobytePluginName, plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}

	cfg := &quobyteAPIConfig{
		quobyteAPIServer: apiServer,
	}

	var ok bool
	if cfg.quobyteUser, ok = secretMap["user"]; !ok {
		return nil, fmt.Errorf("Missing \"user\" in secret %s/%s", secretNamespace, secretName)
	}

	if cfg.quobytePassword, ok = secretMap["password"]; !ok {
		return nil, fmt.Errorf("Missing \"password\" in secret %s/%s", secretNamespace, secretName)
	}

	return cfg, nil
}
