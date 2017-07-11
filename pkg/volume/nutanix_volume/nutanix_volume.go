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

package nutanix_volume

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	dstrings "strings"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/keymutex"
	"k8s.io/kubernetes/pkg/util/mount"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&nutanixVolumePlugin{host: nil, exe: exec.New()}}
}

type nutanixVolumePlugin struct {
	host volume.VolumeHost
	exe  exec.Interface

	// Singleton key mutex for keeping attach/detach operations for the same volume atomic.
	volMutex      keymutex.KeyMutex
	prismEndPoint string
	secretValue   string
}

var _ volume.VolumePlugin = &nutanixVolumePlugin{}
var _ volume.PersistentVolumePlugin = &nutanixVolumePlugin{}
var _ volume.DeletableVolumePlugin = &nutanixVolumePlugin{}
var _ volume.ProvisionableVolumePlugin = &nutanixVolumePlugin{}
var _ volume.Provisioner = &nutanixVolumeProvisioner{}
var _ volume.Deleter = &nutanixVolumeDeleter{}

const (
	nutanixVolumePluginName = "kubernetes.io/nutanix-volume"
	secretKeyName           = "key"
)

func (plugin *nutanixVolumePlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	plugin.volMutex = keymutex.NewKeyMutex()
	return nil
}

func (plugin *nutanixVolumePlugin) GetPluginName() string {
	return nutanixVolumePluginName
}

func (plugin *nutanixVolumePlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getNutanixVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return volumeSource.VolumeName, nil
}

func (plugin *nutanixVolumePlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.NutanixVolume == nil) ||
		(spec.Volume != nil && spec.Volume.NutanixVolume == nil) {
		return false
	}

	src, _, err := getNutanixVolumeSource(spec)
	if err != nil {
		glog.Errorf("nutanix_volume: error getting volume source from spec: %v.", err)
		return false
	}

	// Set cluster credential if volumeSource is present. NewUnmounter call does not provide
	// volumeSource. We need cluster credential during unmount to remove host's IQN from
	// volume group in nutanix cluster.
	plugin.prismEndPoint = src.PrismEndPoint
	plugin.secretValue, err = getSecretValue(src.User, src.Password, src.SecretName,
		src.SecretNamespace, plugin.host.GetKubeClient())
	if err != nil {
		glog.Errorf("nutanix_volume: error getting credential for cluster: %v.", err)
		return false
	}

	return true
}

func (plugin *nutanixVolumePlugin) RequiresRemount() bool {
	return false
}

func (plugin *nutanixVolumePlugin) SupportsMountOption() bool {
	return true
}

func (plugin *nutanixVolumePlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *nutanixVolumePlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *nutanixVolumePlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	return plugin.newMounterInternal(spec, pod.UID, &NutanixVolumeUtil{}, plugin.host.GetMounter())
}

func getNutanixVolumeSource(spec *volume.Spec) (*v1.NutanixVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.NutanixVolume != nil {
		return spec.Volume.NutanixVolume, spec.Volume.NutanixVolume.ReadOnly, nil
	}
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.NutanixVolume != nil {
		return spec.PersistentVolume.Spec.NutanixVolume, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference nutanix volume type")
}

func (plugin *nutanixVolumePlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Mounter, error) {
	src, readOnly, err := getNutanixVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	secretValue, err := getSecretValue(src.User, src.Password, src.SecretName, src.SecretNamespace,
		plugin.host.GetKubeClient())
	if err != nil {
		// Do not bail out as NewMounter can be called with partial volume.spec.
		glog.Warningf("nutanix_volume: error getting credential for cluster: %v.", err)
	}
	return &nutanixVolumeMounter{
		nutanixVolume: &nutanixVolume{
			podUID:              podUID,
			volName:             spec.Name(),
			dataServiceEndPoint: src.DataServiceEndPoint,
			iscsiTarget:         src.IscsiTarget,
			prismEndPoint:       src.PrismEndPoint,
			secretValue:         secretValue,
			volumeUUID:          src.VolumeUUID,
			manager:             manager,
			plugin:              plugin,
		},
		fsType:       src.FSType,
		readOnly:     readOnly,
		mounter:      &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()},
		deviceUtil:   volutil.NewDeviceHandler(volutil.NewIOHandler()),
		mountOptions: volume.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *nutanixVolumePlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, &NutanixVolumeUtil{}, plugin.host.GetMounter())
}

func (plugin *nutanixVolumePlugin) newUnmounterInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &nutanixVolumeUnmounter{
		nutanixVolume: &nutanixVolume{
			volName: volName,
			podUID:  podUID,
			manager: manager,
			plugin:  plugin,
		},
		mounter: mounter,
	}, nil
}

func (plugin *nutanixVolumePlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

func (plugin *nutanixVolumePlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	nutanixVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			NutanixVolume: &v1.NutanixVolumeSource{
				VolumeName: volumeName,
			},
		},
	}
	return volume.NewSpecFromVolume(nutanixVolume), nil
}

type nutanixVolume struct {
	podUID              types.UID
	volName             string
	prismEndPoint       string
	secretValue         string
	dataServiceEndPoint string
	iscsiTarget         string
	volumeUUID          string
	plugin              *nutanixVolumePlugin
	manager             diskManager
	volume.MetricsNil
}

type nutanixVolumeMounter struct {
	*nutanixVolume
	fsType       string
	readOnly     bool
	mounter      *mount.SafeFormatAndMount
	deviceUtil   volutil.DeviceUtil
	mountOptions []string
}

var _ volume.Mounter = &nutanixVolumeMounter{}

func (b *nutanixVolumeMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.readOnly,
		Managed:         false,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *nutanixVolumeMounter) CanMount() error {
	_, err := b.plugin.exe.LookPath("iscsiadm")
	if err != nil {
		return fmt.Errorf("nuatnix_volume: ISCSI is not installed in the node")
	}
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (b *nutanixVolumeMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *nutanixVolumeMounter) SetUpAt(dir string, fsGroup *int64) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(b.manager, *b, dir, b.mounter, fsGroup)
	if err != nil {
		glog.Errorf("nutanix_volume: failed to setup nutanix volume, error: %v", err)
	}
	return err
}

func (nutanixVolume *nutanixVolume) GetPath() string {
	name := nutanixVolumePluginName
	return nutanixVolume.plugin.host.GetPodVolumeDir(nutanixVolume.podUID, utilstrings.EscapeQualifiedNameForDisk(name), nutanixVolume.volName)
}

type nutanixVolumeUnmounter struct {
	*nutanixVolume
	mounter mount.Interface
}

var _ volume.Unmounter = &nutanixVolumeUnmounter{}

func (c *nutanixVolumeUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *nutanixVolumeUnmounter) TearDownAt(dir string) error {
	if pathExists, pathErr := volutil.PathExists(dir); pathErr != nil {
		return fmt.Errorf("nutanix_volume: Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("nutanix_volume: Warning: Unmount skipped because path does not exist: %v", dir)
		return nil
	}
	err := diskTearDown(c.manager, *c, dir, c.mounter)
	if err != nil {
		glog.Errorf("nutanix_volume: failed to teardown nutanix volume, error: %v", err)
	}
	return err
}

func (plugin *nutanixVolumePlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options)
}

func (plugin *nutanixVolumePlugin) newProvisionerInternal(options volume.VolumeOptions) (volume.Provisioner, error) {
	return &nutanixVolumeProvisioner{
		plugin:  plugin,
		options: options,
	}, nil
}

type provisionerParams struct {
	prismEndPoint       string
	dataServiceEndPoint string
	user                string
	password            string
	secretNamespace     string
	secretName          string
	secretValue         string
	name                string
	storageContainer    string
	// StorageClass name.
	scName    string
	isShared  bool
	sizeBytes int64
	fsType    string
}

type nutanixVolumeProvisioner struct {
	plugin  *nutanixVolumePlugin
	params  provisionerParams
	options volume.VolumeOptions
}

func (plugin *nutanixVolumePlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	return plugin.newDeleterInternal(spec)
}

func (plugin *nutanixVolumePlugin) newDeleterInternal(spec *volume.Spec) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.NutanixVolume == nil {
		return nil, fmt.Errorf("nutanix_volume: spec.PersistentVolumeSource.Spec.NutanixVolume is nil")
	}
	return &nutanixVolumeDeleter{
		nutanixVolume: &nutanixVolume{
			volName: spec.Name(),
			plugin:  plugin,
		},
		pv: spec.PersistentVolume,
	}, nil
}

type nutanixVolumeDeleter struct {
	*nutanixVolume
	pv *v1.PersistentVolume
}

func (d *nutanixVolumeDeleter) GetPath() string {
	name := nutanixVolumePluginName
	return d.plugin.host.GetPodVolumeDir(d.podUID, utilstrings.EscapeQualifiedNameForDisk(name), d.volName)
}

func (d *nutanixVolumeDeleter) Delete() error {
	var err error

	volume := d.pv.Spec.NutanixVolume
	glog.V(2).Infof("nutanix_volume: deleting volume with configuration %+v", volume)
	secretValue, err := getSecretValue(volume.User, volume.Password, volume.SecretName,
		volume.SecretNamespace, d.plugin.host.GetKubeClient())
	if err != nil {
		return fmt.Errorf("nutanix_volume: failed to get cluster credential, error: %v", err)
	}

	cli, err := NewNutanixClient(volume.PrismEndPoint, secretValue)
	if err != nil {
		return fmt.Errorf("nutanix_volume: failed to create REST client, error: %v", err)
	}

	// Remove host's iqn from nutanix volume.
	err = cli.DetachAllVG(volume.VolumeUUID)
	if err != nil {
		return err
	}

	err = cli.DeleteVG(volume.VolumeUUID)
	if err != nil {
		glog.Errorf("nutanix_volume: error when deleting the volume :%v", err)
		return err
	}
	glog.V(2).Infof("nutanix_volume: volume %s deleted successfully", volume.VolumeName)

	return nil
}

func (r *nutanixVolumeProvisioner) Provision() (*v1.PersistentVolume, error) {
	var err error

	glog.V(2).Infof("nutanix_volume: Provision VolumeOptions %v", r.options)
	// Currently nutanix_volume does not support any label selector in PVC.
	if r.options.PVC.Spec.Selector != nil {
		return nil, fmt.Errorf("nutanix_volume: not able to parse claim Selector")
	}

	// nutanix_volume does not support ReadWriteMany access mode.
	if isAccessModeSupported(r.options.PVC.Spec.AccessModes) == false {
		return nil, fmt.Errorf("nutanix_volume: requested AccessModes for volume is not supported")
	}

	// Default volume access mode is ReadWriteOnce.
	if len(r.options.PVC.Spec.AccessModes) == 0 {
		r.options.PVC.Spec.AccessModes[0] = v1.ReadWriteOnce
	}

	capacity := r.options.PVC.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	volSizeBytes := capacity.Value()

	// Generate unique PV name based on the claim name.
	pvName := generatePVName(r.options.PVC)
	scName := *r.options.PVC.Spec.StorageClassName
	cfg, err := parseClassParameters(r.options.Parameters, r.plugin.host.GetKubeClient())
	if err != nil {
		return nil, err
	}
	cfg.scName = scName
	cfg.isShared = isSharedPVRequested(r.options.PVC.Spec.AccessModes)
	cfg.name = pvName
	cfg.sizeBytes = volSizeBytes
	r.params = *cfg

	volumeSource, size, err := r.CreateVolume()
	if err != nil {
		return nil, fmt.Errorf("nutanix_volume: create volume err: %v", err)
	}

	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: pvName,
			Annotations: map[string]string{
				"kubernetes.io/createdby": "nutanix-volume-dynamic-provisioner",
			},
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: r.options.PersistentVolumeReclaimPolicy,
			AccessModes:                   r.options.PVC.Spec.AccessModes,
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dKi", size/1024)),
			},
			PersistentVolumeSource: v1.PersistentVolumeSource{
				NutanixVolume: volumeSource,
			},
		},
	}

	glog.V(2).Infof("Returning pv %+v", pv)
	return pv, nil
}

func (p *nutanixVolumeProvisioner) CreateVolume() (r *v1.NutanixVolumeSource, size int64, err error) {
	var volume *VGInfoDTO
	var vgConfig *VGCreateDTO

	cli, err := NewNutanixClient(p.params.prismEndPoint, p.params.secretValue)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create REST client, error: %v", err)
	}

	vgConfig, err = volumeCreateParams(cli, &p.params)
	volume, err = cli.CreateVG(vgConfig)
	if err != nil {
		return nil, 0, fmt.Errorf("error creating volume %v, cfg %v", err, vgConfig)
	}
	glog.V(2).Infof("nutanix_volume: volume with size: %d and name: %s created", p.params.sizeBytes, volume.Name)

	return &v1.NutanixVolumeSource{
		User:                p.params.user,
		Password:            p.params.password,
		SecretName:          p.params.secretName,
		SecretNamespace:     p.params.secretNamespace,
		PrismEndPoint:       p.params.prismEndPoint,
		DataServiceEndPoint: p.params.dataServiceEndPoint,
		VolumeName:          volume.Name,
		VolumeUUID:          volume.UUID,
		IscsiTarget:         volume.IscsiTarget,
		FSType:              p.params.fsType,
		ReadOnly:            p.params.isShared,
	}, p.params.sizeBytes, nil
}

//Create volume_group create rest API config parameters.
func volumeCreateParams(cli *NutanixClient, params *provisionerParams) (*VGCreateDTO, error) {
	storageContainerUUID, err := cli.getStorageContainerUUID(params.storageContainer)
	if err != nil {
		return nil, err
	}
	glog.V(4).Info("nutanix_volume: Got Container uuid: ", storageContainerUUID)

	flashModeEnabled := false
	if params.scName == "gold" {
		flashModeEnabled = true
	}

	vgConfig := &VGCreateDTO{
		Name:             params.name,
		IsShared:         params.isShared,
		IscsiTarget:      "",
		FlashModeEnabled: flashModeEnabled,
		DiskList:         []*VGDiskDTO{},
	}
	diskConfig := VGDiskCreateDTO{
		Size:                 params.sizeBytes,
		StorageContainerUUID: storageContainerUUID,
	}
	disk := &VGDiskDTO{CreateConfig: diskConfig}
	vgConfig.DiskList = append(vgConfig.DiskList, disk)
	return vgConfig, nil
}

// parseSecret finds a given Secret instance and reads user password from it.
func parseSecret(namespace, secretName string, kubeClient clientset.Interface) (string, error) {
	secretMap, err := volutil.GetSecretForPV(namespace, secretName, nutanixVolumePluginName, kubeClient)
	if err != nil {
		return "", fmt.Errorf("failed to get secret %s/%s: %v", namespace, secretName, err)
	}
	if len(secretMap) == 0 {
		return "", fmt.Errorf("empty secret map in secret %s/%s", namespace, secretName)
	}
	for k, v := range secretMap {
		if k == secretKeyName {
			return v, nil
		}
	}
	return "", fmt.Errorf("Secret %s/%s does not have key %s", namespace, secretName, secretKeyName)
}

// parseClassParameters parses StorageClass.Parameters
func parseClassParameters(params map[string]string, kubeClient clientset.Interface) (*provisionerParams, error) {
	var cfg provisionerParams
	var err error

	for k, v := range params {
		switch dstrings.ToLower(k) {
		case "prismendpoint":
			cfg.prismEndPoint = v
		case "dataserviceendpoint":
			cfg.dataServiceEndPoint = v
		case "user":
			cfg.user = v
		case "password":
			cfg.password = v
		case "secretname":
			cfg.secretName = v
		case "secretnamespace":
			cfg.secretNamespace = v
		case "storagecontainer":
			cfg.storageContainer = v
		case "fstype":
			cfg.fsType = v
		default:
			return nil, fmt.Errorf("nutanix_volume: invalid option %q for volume plugin %s", k, nutanixVolumePluginName)
		}
	}

	if (len(cfg.user) == 0 || len(cfg.password) == 0) && (len(cfg.secretName) == 0) {
		return nil, fmt.Errorf("StorageClass for provisioner %s must contain either 'user/password' or 'secretName' parameter", nutanixVolumePluginName)
	}
	if len(cfg.dataServiceEndPoint) == 0 {
		return nil, fmt.Errorf("StorageClass for provisioner %s must contain 'dataServiceEndPoint' parameter", nutanixVolumePluginName)
	}

	if len(cfg.prismEndPoint) == 0 {
		return nil, fmt.Errorf("StorageClass for provisioner %s must contain 'prismEndPoint' parameter", nutanixVolumePluginName)
	}

	if len(cfg.storageContainer) == 0 {
		return nil, fmt.Errorf("StorageClass for provisioner %s must contain 'storageContainer' parameter", nutanixVolumePluginName)
	}

	cfg.secretValue, err = getSecretValue(cfg.user, cfg.password, cfg.secretName, cfg.secretNamespace, kubeClient)

	return &cfg, err
}

func getSecretValue(user, password, secretName, secretNamespace string, kubeClient clientset.Interface) (string, error) {
	var err error
	var secretValue string

	if len(secretName) != 0 {
		if len(secretNamespace) == 0 {
			secretNamespace = "default"
		}
		// secretName + Namespace has precedence over userKey
		secretValue, err = parseSecret(secretNamespace, secretName, kubeClient)
		if err != nil {
			return "", err
		}
		creds := dstrings.SplitN(secretValue, ":", 2)
		if len(creds) != 2 {
			return "", fmt.Errorf("Secret value is not encoded using '<user>:<password>' format")
		}
		user = creds[0]
		password = creds[1]
	}

	if len(user) != 0 && len(password) != 0 {
		secretValue = base64.StdEncoding.EncodeToString([]byte(user + ":" + password))
		return secretValue, err
	}

	return "", fmt.Errorf("Invalid user and password for the cluster")
}

func generatePVName(PVC *v1.PersistentVolumeClaim) string {
	return fmt.Sprintf("%x-nutanix-k8-volume", sha256.Sum256([]byte(PVC.Name+PVC.Namespace)))
}

// Return true if ReadOnlyMany access mode is requested.
func isSharedPVRequested(AccessModes []v1.PersistentVolumeAccessMode) bool {
	for _, mode := range AccessModes {
		if mode == v1.ReadOnlyMany {
			return true
		}
	}
	return false
}

// Return true if requested AccessMode is supported.
func isAccessModeSupported(AccessModes []v1.PersistentVolumeAccessMode) bool {
	for _, mode := range AccessModes {
		if mode != v1.ReadWriteOnce && mode != v1.ReadOnlyMany {
			return false
		}
	}
	return true
}
