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

package rbd

import (
	"fmt"
	dstrings "strings"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumehelper"
)

var (
	supportedFeatures = sets.NewString("layering")
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&rbdPlugin{}}
}

// rbdPlugin implements Volume.VolumePlugin.
type rbdPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &rbdPlugin{}
var _ volume.PersistentVolumePlugin = &rbdPlugin{}
var _ volume.DeletableVolumePlugin = &rbdPlugin{}
var _ volume.ProvisionableVolumePlugin = &rbdPlugin{}
var _ volume.AttachableVolumePlugin = &rbdPlugin{}

const (
	rbdPluginName                  = "kubernetes.io/rbd"
	secretKeyName                  = "key" // key name used in secret
	rbdImageFormat1                = "1"
	rbdImageFormat2                = "2"
	rbdDefaultAdminId              = "admin"
	rbdDefaultAdminSecretNamespace = "default"
	rbdDefaultPool                 = "rbd"
	rbdDefaultUserId               = rbdDefaultAdminId
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, strings.EscapeQualifiedNameForDisk(rbdPluginName), volName)
}

func (plugin *rbdPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *rbdPlugin) GetPluginName() string {
	return rbdPluginName
}

func (plugin *rbdPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	mon, err := getVolumeSourceMonitors(spec)
	if err != nil {
		return "", err
	}
	img, err := getVolumeSourceImage(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v:%v",
		mon,
		img), nil
}

func (plugin *rbdPlugin) CanSupport(spec *volume.Spec) bool {
	if (spec.Volume != nil && spec.Volume.RBD == nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.RBD == nil) {
		return false
	}

	return true
}

func (plugin *rbdPlugin) RequiresRemount() bool {
	return false
}

func (plugin *rbdPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *rbdPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *rbdPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
	}
}

func (plugin *rbdPlugin) createMounterFromVolumeSpecAndPod(spec *volume.Spec, pod *v1.Pod) (*rbdMounter, error) {
	var err error
	mon, err := getVolumeSourceMonitors(spec)
	if err != nil {
		return nil, err
	}
	img, err := getVolumeSourceImage(spec)
	if err != nil {
		return nil, err
	}
	fstype, err := getVolumeSourceFSType(spec)
	if err != nil {
		return nil, err
	}
	pool, err := getVolumeSourcePool(spec)
	if err != nil {
		return nil, err
	}
	id, err := getVolumeSourceUser(spec)
	if err != nil {
		return nil, err
	}
	keyring, err := getVolumeSourceKeyRing(spec)
	if err != nil {
		return nil, err
	}
	ro, err := getVolumeSourceReadOnly(spec)
	if err != nil {
		return nil, err
	}

	secretName, secretNs, err := getSecretNameAndNamespace(spec, pod.Namespace)
	if err != nil {
		return nil, err
	}
	secret := ""
	if len(secretName) > 0 && len(secretNs) > 0 {
		// if secret is provideded, retrieve it
		kubeClient := plugin.host.GetKubeClient()
		if kubeClient == nil {
			return nil, fmt.Errorf("Cannot get kube client")
		}
		secrets, err := kubeClient.Core().Secrets(secretNs).Get(secretName, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("Couldn't get secret %v/%v err: %v", secretNs, secretName, err)
			return nil, err
		}
		for _, data := range secrets.Data {
			secret = string(data)
		}
	}

	return &rbdMounter{
		rbd:     newRBD("", spec.Name(), img, pool, ro, plugin, &RBDUtil{}),
		Mon:     mon,
		Id:      id,
		Keyring: keyring,
		Secret:  secret,
		fsType:  fstype,
	}, nil
}

func (plugin *rbdPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	secretName, secretNs, err := getSecretNameAndNamespace(spec, pod.Namespace)
	if err != nil {
		return nil, err
	}
	secret := ""
	if len(secretName) > 0 && len(secretNs) > 0 {
		// if secret is provideded, retrieve it
		kubeClient := plugin.host.GetKubeClient()
		if kubeClient == nil {
			return nil, fmt.Errorf("Cannot get kube client")
		}
		secrets, err := kubeClient.CoreV1().Secrets(secretNs).Get(secretName, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("Couldn't get secret %v/%v err: %v", secretNs, secretName, err)
			return nil, err
		}
		for _, data := range secrets.Data {
			secret = string(data)
		}
	}

	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &RBDUtil{}, secret)
}

func (plugin *rbdPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager diskManager, secret string) (volume.Mounter, error) {
	mon, err := getVolumeSourceMonitors(spec)
	if err != nil {
		return nil, err
	}
	img, err := getVolumeSourceImage(spec)
	if err != nil {
		return nil, err
	}
	fstype, err := getVolumeSourceFSType(spec)
	if err != nil {
		return nil, err
	}
	pool, err := getVolumeSourcePool(spec)
	if err != nil {
		return nil, err
	}
	id, err := getVolumeSourceUser(spec)
	if err != nil {
		return nil, err
	}
	keyring, err := getVolumeSourceKeyRing(spec)
	if err != nil {
		return nil, err
	}
	ro, err := getVolumeSourceReadOnly(spec)
	if err != nil {
		return nil, err
	}

	return &rbdMounter{
		rbd:          newRBD(podUID, spec.Name(), img, pool, ro, plugin, manager),
		Mon:          mon,
		Id:           id,
		Keyring:      keyring,
		Secret:       secret,
		fsType:       fstype,
		mountOptions: volume.MountOptionFromSpec(spec),
	}, nil
}

func (plugin *rbdPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &RBDUtil{})
}

func (plugin *rbdPlugin) newUnmounterInternal(volName string, podUID types.UID, manager diskManager) (volume.Unmounter, error) {
	return &rbdUnmounter{
		rbdMounter: &rbdMounter{
			rbd: newRBD(podUID, volName, "", "", false, plugin, manager),
			Mon: make([]string, 0),
		},
	}, nil
}

func (plugin *rbdPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	rbdVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			RBD: &v1.RBDVolumeSource{
				CephMonitors: []string{},
			},
		},
	}
	return volume.NewSpecFromVolume(rbdVolume), nil
}

func (plugin *rbdPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.RBD == nil {
		return nil, fmt.Errorf("spec.PersistentVolumeSource.Spec.RBD is nil")
	}
	class, err := volutil.GetClassForVolume(plugin.host.GetKubeClient(), spec.PersistentVolume)
	if err != nil {
		return nil, err
	}
	adminSecretName := ""
	adminSecretNamespace := rbdDefaultAdminSecretNamespace
	admin := ""

	for k, v := range class.Parameters {
		switch dstrings.ToLower(k) {
		case "adminid":
			admin = v
		case "adminsecretname":
			adminSecretName = v
		case "adminsecretnamespace":
			adminSecretNamespace = v
		}
	}

	if admin == "" {
		admin = rbdDefaultAdminId
	}
	secret, err := parsePVSecret(adminSecretNamespace, adminSecretName, plugin.host.GetKubeClient())
	if err != nil {
		return nil, fmt.Errorf("failed to get admin secret from [%q/%q]: %v", adminSecretNamespace, adminSecretName, err)
	}
	return plugin.newDeleterInternal(spec, admin, secret, &RBDUtil{})
}

func (plugin *rbdPlugin) newDeleterInternal(spec *volume.Spec, admin, secret string, manager diskManager) (volume.Deleter, error) {
	return &rbdVolumeDeleter{
		rbdMounter: &rbdMounter{
			rbd:         newRBD("", spec.Name(), spec.PersistentVolume.Spec.RBD.RBDImage, spec.PersistentVolume.Spec.RBD.RBDPool, false, plugin, manager),
			Mon:         spec.PersistentVolume.Spec.RBD.CephMonitors,
			adminId:     admin,
			adminSecret: secret,
		}}, nil
}

func (plugin *rbdPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &RBDUtil{})
}

func (plugin *rbdPlugin) newProvisionerInternal(options volume.VolumeOptions, manager diskManager) (volume.Provisioner, error) {
	return &rbdVolumeProvisioner{
		rbdMounter: &rbdMounter{
			rbd: newRBD("", "", "", "", false, plugin, manager),
		},
		options: options,
	}, nil
}

// rbdVolumeProvisioner implements volume.Provisioner interface.
type rbdVolumeProvisioner struct {
	*rbdMounter
	options volume.VolumeOptions
}

var _ volume.Provisioner = &rbdVolumeProvisioner{}

func (r *rbdVolumeProvisioner) Provision() (*v1.PersistentVolume, error) {
	if !volume.AccessModesContainedInAll(r.plugin.GetAccessModes(), r.options.PVC.Spec.AccessModes) {
		return nil, fmt.Errorf("invalid AccessModes %v: only AccessModes %v are supported", r.options.PVC.Spec.AccessModes, r.plugin.GetAccessModes())
	}

	if r.options.PVC.Spec.Selector != nil {
		return nil, fmt.Errorf("claim Selector is not supported")
	}
	var err error
	adminSecretName := ""
	adminSecretNamespace := rbdDefaultAdminSecretNamespace
	secret := ""
	secretName := ""
	secretNamespace := ""
	imageFormat := rbdImageFormat2
	fstype := ""

	for k, v := range r.options.Parameters {
		switch dstrings.ToLower(k) {
		case "monitors":
			arr := dstrings.Split(v, ",")
			for _, m := range arr {
				r.Mon = append(r.Mon, m)
			}
		case "adminid":
			r.adminId = v
		case "adminsecretname":
			adminSecretName = v
		case "adminsecretnamespace":
			adminSecretNamespace = v
		case "userid":
			r.Id = v
		case "pool":
			r.Pool = v
		case "usersecretname":
			secretName = v
		case "usersecretnamespace":
			secretNamespace = v
		case "imageformat":
			imageFormat = v
		case "imagefeatures":
			arr := dstrings.Split(v, ",")
			for _, f := range arr {
				if !supportedFeatures.Has(f) {
					return nil, fmt.Errorf("invalid feature %q for volume plugin %s, supported features are: %v", f, r.plugin.GetPluginName(), supportedFeatures)
				} else {
					r.imageFeatures = append(r.imageFeatures, f)
				}
			}
		case volume.VolumeParameterFSType:
			fstype = v
		default:
			return nil, fmt.Errorf("invalid option %q for volume plugin %s", k, r.plugin.GetPluginName())
		}
	}
	// sanity check
	if imageFormat != rbdImageFormat1 && imageFormat != rbdImageFormat2 {
		return nil, fmt.Errorf("invalid ceph imageformat %s, expecting %s or %s",
			imageFormat, rbdImageFormat1, rbdImageFormat2)
	}
	r.imageFormat = imageFormat
	if adminSecretName == "" {
		return nil, fmt.Errorf("missing Ceph admin secret name")
	}
	if secret, err = parsePVSecret(adminSecretNamespace, adminSecretName, r.plugin.host.GetKubeClient()); err != nil {
		return nil, fmt.Errorf("failed to get admin secret from [%q/%q]: %v", adminSecretNamespace, adminSecretName, err)
	}
	r.adminSecret = secret
	if len(r.Mon) < 1 {
		return nil, fmt.Errorf("missing Ceph monitors")
	}
	if secretName == "" {
		return nil, fmt.Errorf("missing user secret name")
	}
	if r.adminId == "" {
		r.adminId = rbdDefaultAdminId
	}
	if r.Pool == "" {
		r.Pool = rbdDefaultPool
	}
	if r.Id == "" {
		r.Id = r.adminId
	}

	// create random image name
	image := fmt.Sprintf("kubernetes-dynamic-pvc-%s", uuid.NewUUID())
	r.rbdMounter.Image = image
	rbd, sizeMB, err := r.manager.CreateImage(r)
	if err != nil {
		glog.Errorf("rbd: create volume failed, err: %v", err)
		return nil, err
	}
	glog.Infof("successfully created rbd image %q", image)
	pv := new(v1.PersistentVolume)
	metav1.SetMetaDataAnnotation(&pv.ObjectMeta, volumehelper.VolumeDynamicallyCreatedByKey, "rbd-dynamic-provisioner")
	rbd.SecretRef = new(v1.SecretReference)
	rbd.SecretRef.Name = secretName
	rbd.SecretRef.Namespace = secretNamespace
	rbd.RadosUser = r.Id
	rbd.FSType = fstype
	pv.Spec.PersistentVolumeSource.RBD = rbd
	pv.Spec.PersistentVolumeReclaimPolicy = r.options.PersistentVolumeReclaimPolicy
	pv.Spec.AccessModes = r.options.PVC.Spec.AccessModes
	if len(pv.Spec.AccessModes) == 0 {
		pv.Spec.AccessModes = r.plugin.GetAccessModes()
	}
	pv.Spec.Capacity = v1.ResourceList{
		v1.ResourceName(v1.ResourceStorage): resource.MustParse(fmt.Sprintf("%dMi", sizeMB)),
	}
	pv.Spec.MountOptions = r.options.MountOptions
	return pv, nil
}

// rbdVolumeDeleter implements volume.Deleter interface.
type rbdVolumeDeleter struct {
	*rbdMounter
}

var _ volume.Deleter = &rbdVolumeDeleter{}

func (r *rbdVolumeDeleter) GetPath() string {
	return getPath(r.podUID, r.volName, r.plugin.host)
}

func (r *rbdVolumeDeleter) Delete() error {
	return r.manager.DeleteImage(r)
}

// rbd implmenets volume.Volume interface.
// It's embedded in Mounter/Unmounter/Deleter.
type rbd struct {
	volName  string
	podUID   types.UID
	Pool     string
	Image    string
	ReadOnly bool
	plugin   *rbdPlugin
	mounter  *mount.SafeFormatAndMount
	exec     mount.Exec
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager                diskManager
	volume.MetricsProvider `json:"-"`
}

var _ volume.Volume = &rbd{}

func (rbd *rbd) GetPath() string {
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return getPath(rbd.podUID, rbd.volName, rbd.plugin.host)
}

// newRBD creates a new rbd.
func newRBD(podUID types.UID, volName string, image string, pool string, readOnly bool, plugin *rbdPlugin, manager diskManager) *rbd {
	return &rbd{
		podUID:          podUID,
		volName:         volName,
		Image:           image,
		Pool:            pool,
		ReadOnly:        readOnly,
		plugin:          plugin,
		mounter:         volumehelper.NewSafeFormatAndMountFromHost(plugin.GetPluginName(), plugin.host),
		exec:            plugin.host.GetExec(plugin.GetPluginName()),
		manager:         manager,
		MetricsProvider: volume.NewMetricsStatFS(getPath(podUID, volName, plugin.host)),
	}
}

// rbdMounter implements volume.Mounter interface.
// It contains information which need to be persisted in whole life cycle of PV
// on the node. It is persisted at the very beginning in the pod mount point
// directory.
// Note: Capitalized field names of this struct determines the information
// persisted on the disk, DO NOT change them. (TODO: refactoring to use a dedicated struct?)
type rbdMounter struct {
	*rbd
	// capitalized so they can be exported in persistRBD()
	Mon           []string
	Id            string
	Keyring       string
	Secret        string
	fsType        string
	adminSecret   string
	adminId       string
	mountOptions  []string
	imageFormat   string
	imageFeatures []string
}

var _ volume.Mounter = &rbdMounter{}

func (b *rbd) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.ReadOnly,
		Managed:         !b.ReadOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *rbdMounter) CanMount() error {
	return nil
}

func (b *rbdMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *rbdMounter) SetUpAt(dir string, fsGroup *int64) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	glog.V(4).Infof("rbd: attempting to setup at %s", dir)
	err := diskSetUp(b.manager, *b, dir, b.mounter, fsGroup)
	if err != nil {
		glog.Errorf("rbd: failed to setup at %s %v", dir, err)
	}
	glog.V(3).Infof("rbd: successfully setup at %s", dir)
	return err
}

// rbdUnmounter implements volume.Unmounter interface.
type rbdUnmounter struct {
	*rbdMounter
}

var _ volume.Unmounter = &rbdUnmounter{}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (c *rbdUnmounter) TearDown() error {
	return c.TearDownAt(c.GetPath())
}

func (c *rbdUnmounter) TearDownAt(dir string) error {
	glog.V(4).Infof("rbd: attempting to teardown at %s", dir)
	if pathExists, pathErr := volutil.PathExists(dir); pathErr != nil {
		return fmt.Errorf("Error checking if path exists: %v", pathErr)
	} else if !pathExists {
		glog.Warningf("Warning: Unmount skipped because path does not exist: %v", dir)
		return nil
	}
	err := diskTearDown(c.manager, *c, dir, c.mounter)
	if err != nil {
		return err
	}
	glog.V(3).Infof("rbd: successfully teardown at %s", dir)
	return nil
}

func getVolumeSourceMonitors(spec *volume.Spec) ([]string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.CephMonitors, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.CephMonitors, nil
	}

	return nil, fmt.Errorf("Spec does not reference a RBD volume type")
}

func getVolumeSourceImage(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.RBDImage, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.RBDImage, nil
	}

	return "", fmt.Errorf("Spec does not reference a RBD volume type")
}

func getVolumeSourceFSType(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.FSType, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.FSType, nil
	}

	return "", fmt.Errorf("Spec does not reference a RBD volume type")
}

func getVolumeSourcePool(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.RBDPool, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.RBDPool, nil
	}

	return "", fmt.Errorf("Spec does not reference a RBD volume type")
}

func getVolumeSourceUser(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.RadosUser, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.RadosUser, nil
	}

	return "", fmt.Errorf("Spec does not reference a RBD volume type")
}

func getVolumeSourceKeyRing(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.Keyring, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.Keyring, nil
	}

	return "", fmt.Errorf("Spec does not reference a RBD volume type")
}

func getVolumeSourceReadOnly(spec *volume.Spec) (bool, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		// rbd volumes used as a PersistentVolume gets the ReadOnly flag indirectly through
		// the persistent-claim volume used to mount the PV
		return spec.ReadOnly, nil
	}

	return false, fmt.Errorf("Spec does not reference a RBD volume type")
}

func parsePodSecret(pod *v1.Pod, secretName string, kubeClient clientset.Interface) (string, error) {
	secret, err := volutil.GetSecretForPod(pod, secretName, kubeClient)
	if err != nil {
		glog.Errorf("failed to get secret from [%q/%q]", pod.Namespace, secretName)
		return "", fmt.Errorf("failed to get secret from [%q/%q]", pod.Namespace, secretName)
	}
	return parseSecretMap(secret)
}

func parsePVSecret(namespace, secretName string, kubeClient clientset.Interface) (string, error) {
	secret, err := volutil.GetSecretForPV(namespace, secretName, rbdPluginName, kubeClient)
	if err != nil {
		glog.Errorf("failed to get secret from [%q/%q]", namespace, secretName)
		return "", fmt.Errorf("failed to get secret from [%q/%q]", namespace, secretName)
	}
	return parseSecretMap(secret)
}

// parseSecretMap locates the secret by key name.
func parseSecretMap(secretMap map[string]string) (string, error) {
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

func getSecretNameAndNamespace(spec *volume.Spec, defaultNamespace string) (string, string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		localSecretRef := spec.Volume.RBD.SecretRef
		if localSecretRef != nil {
			return localSecretRef.Name, defaultNamespace, nil
		}
		return "", "", nil

	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		secretRef := spec.PersistentVolume.Spec.RBD.SecretRef
		secretNs := defaultNamespace
		if secretRef != nil {
			if len(secretRef.Namespace) != 0 {
				secretNs = secretRef.Namespace
			}
			return secretRef.Name, secretNs, nil
		}
		return "", "", nil
	}
	return "", "", fmt.Errorf("Spec does not reference an RBD volume type")
}
