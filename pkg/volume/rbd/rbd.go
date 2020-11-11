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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	dstrings "strings"

	"k8s.io/klog/v2"
	"k8s.io/mount-utils"
	utilexec "k8s.io/utils/exec"
	utilstrings "k8s.io/utils/strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
)

var (
	supportedFeatures = sets.NewString("layering")
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
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
var _ volume.ExpandableVolumePlugin = &rbdPlugin{}
var _ volume.BlockVolumePlugin = &rbdPlugin{}
var _ volume.DeviceMountableVolumePlugin = &rbdPlugin{}

const (
	rbdPluginName                  = "kubernetes.io/rbd"
	secretKeyName                  = "key" // key name used in secret
	rbdImageFormat1                = "1"
	rbdImageFormat2                = "2"
	rbdDefaultAdminID              = "admin"
	rbdDefaultAdminSecretNamespace = "default"
	rbdDefaultPool                 = "rbd"
)

func getPath(uid types.UID, volName string, host volume.VolumeHost) string {
	return host.GetPodVolumeDir(uid, utilstrings.EscapeQualifiedName(rbdPluginName), volName)
}

func (plugin *rbdPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *rbdPlugin) GetPluginName() string {
	return rbdPluginName
}

func (plugin *rbdPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	pool, err := getVolumeSourcePool(spec)
	if err != nil {
		return "", err
	}
	img, err := getVolumeSourceImage(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v:%v",
		pool,
		img), nil
}

func (plugin *rbdPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.Volume != nil && spec.Volume.RBD != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.RBD != nil)
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

type rbdVolumeExpander struct {
	*rbdMounter
}

func (plugin *rbdPlugin) getAdminAndSecret(spec *volume.Spec) (string, string, error) {
	class, err := volutil.GetClassForVolume(plugin.host.GetKubeClient(), spec.PersistentVolume)
	if err != nil {
		return "", "", err
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
		admin = rbdDefaultAdminID
	}
	secret, err := parsePVSecret(adminSecretNamespace, adminSecretName, plugin.host.GetKubeClient())
	if err != nil {
		return admin, "", fmt.Errorf("failed to get admin secret from [%q/%q]: %v", adminSecretNamespace, adminSecretName, err)
	}

	return admin, secret, nil
}

func (plugin *rbdPlugin) ExpandVolumeDevice(spec *volume.Spec, newSize resource.Quantity, oldSize resource.Quantity) (resource.Quantity, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.RBD == nil {
		return oldSize, fmt.Errorf("spec.PersistentVolume.Spec.RBD is nil")
	}

	// get admin and secret
	admin, secret, err := plugin.getAdminAndSecret(spec)
	if err != nil {
		return oldSize, err
	}

	expander := &rbdVolumeExpander{
		rbdMounter: &rbdMounter{
			rbd: &rbd{
				volName: spec.Name(),
				Image:   spec.PersistentVolume.Spec.RBD.RBDImage,
				Pool:    spec.PersistentVolume.Spec.RBD.RBDPool,
				plugin:  plugin,
				manager: &rbdUtil{},
				mounter: &mount.SafeFormatAndMount{Interface: plugin.host.GetMounter(plugin.GetPluginName())},
				exec:    plugin.host.GetExec(plugin.GetPluginName()),
			},
			Mon:         spec.PersistentVolume.Spec.RBD.CephMonitors,
			adminID:     admin,
			adminSecret: secret,
		},
	}

	expandedSize, err := expander.ResizeImage(oldSize, newSize)
	if err != nil {
		return oldSize, err
	}
	return expandedSize, nil

}

func (plugin *rbdPlugin) NodeExpand(resizeOptions volume.NodeResizeOptions) (bool, error) {
	fsVolume, err := volutil.CheckVolumeModeFilesystem(resizeOptions.VolumeSpec)
	if err != nil {
		return false, fmt.Errorf("error checking VolumeMode: %v", err)
	}
	// if volume is not a fs file system, there is nothing for us to do here.
	if !fsVolume {
		return true, nil
	}
	_, err = volutil.GenericResizeFS(plugin.host, plugin.GetPluginName(), resizeOptions.DevicePath, resizeOptions.DeviceMountPath)
	if err != nil {
		return false, err
	}
	return true, nil
}

var _ volume.NodeExpandableVolumePlugin = &rbdPlugin{}

func (expander *rbdVolumeExpander) ResizeImage(oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error) {
	return expander.manager.ExpandImage(expander, oldSize, newSize)
}

func (plugin *rbdPlugin) RequiresFSResize() bool {
	return true
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
	ams, err := getVolumeAccessModes(spec)
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
			return nil, fmt.Errorf("cannot get kube client")
		}
		secrets, err := kubeClient.CoreV1().Secrets(secretNs).Get(context.TODO(), secretName, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("couldn't get secret %v/%v err: %v", secretNs, secretName, err)
			return nil, err
		}
		for _, data := range secrets.Data {
			secret = string(data)
		}
	}

	return &rbdMounter{
		rbd:         newRBD("", spec.Name(), img, pool, ro, plugin, &rbdUtil{}),
		Mon:         mon,
		ID:          id,
		Keyring:     keyring,
		Secret:      secret,
		fsType:      fstype,
		accessModes: ams,
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
			return nil, fmt.Errorf("cannot get kube client")
		}
		secrets, err := kubeClient.CoreV1().Secrets(secretNs).Get(context.TODO(), secretName, metav1.GetOptions{})
		if err != nil {
			err = fmt.Errorf("couldn't get secret %v/%v err: %v", secretNs, secretName, err)
			return nil, err
		}
		for _, data := range secrets.Data {
			secret = string(data)
		}
	}

	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &rbdUtil{}, secret)
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
	ams, err := getVolumeAccessModes(spec)
	if err != nil {
		return nil, err
	}

	return &rbdMounter{
		rbd:          newRBD(podUID, spec.Name(), img, pool, ro, plugin, manager),
		Mon:          mon,
		ID:           id,
		Keyring:      keyring,
		Secret:       secret,
		fsType:       fstype,
		mountOptions: volutil.MountOptionFromSpec(spec),
		accessModes:  ams,
	}, nil
}

func (plugin *rbdPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &rbdUtil{})
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
	mounter := plugin.host.GetMounter(plugin.GetPluginName())
	kvh, ok := plugin.host.(volume.KubeletVolumeHost)
	if !ok {
		return nil, fmt.Errorf("plugin volume host does not implement KubeletVolumeHost interface")
	}
	hu := kvh.GetHostUtil()
	pluginMntDir := volutil.GetPluginMountDir(plugin.host, plugin.GetPluginName())
	sourceName, err := hu.GetDeviceNameFromMount(mounter, mountPath, pluginMntDir)
	if err != nil {
		return nil, err
	}
	s := dstrings.Split(sourceName, "-image-")
	if len(s) != 2 {
		// The mountPath parameter is the volume mount path for a specific pod, its format
		// is /var/lib/kubelet/pods/{podUID}/volumes/{volumePluginName}/{volumeName}.
		// mounter.GetDeviceNameFromMount will find the device path(such as /dev/rbd0) by
		// mountPath first, and then try to find the global device mount path from the mounted
		// path list of this device. sourceName is extracted from this global device mount path.
		// mounter.GetDeviceNameFromMount expects the global device mount path conforms to canonical
		// format: /var/lib/kubelet/plugins/kubernetes.io/rbd/mounts/{pool}-image-{image}.
		// If this assertion failed, it means that the global device mount path is created by
		// the deprecated format: /var/lib/kubelet/plugins/kubernetes.io/rbd/rbd/{pool}-image-{image}.
		// So we will try to check whether this old style global device mount path exist or not.
		// If existed, extract the sourceName from this old style path, otherwise return an error.
		klog.V(3).Infof("SourceName %s wrong, fallback to old format", sourceName)
		sourceName, err = plugin.getDeviceNameFromOldMountPath(mounter, mountPath)
		if err != nil {
			return nil, err
		}
		s = dstrings.Split(sourceName, "-image-")
		if len(s) != 2 {
			return nil, fmt.Errorf("sourceName %s wrong, should be pool+\"-image-\"+imageName", sourceName)
		}
	}
	rbdVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			RBD: &v1.RBDVolumeSource{
				RBDPool:  s[0],
				RBDImage: s[1],
			},
		},
	}
	return volume.NewSpecFromVolume(rbdVolume), nil
}

func (plugin *rbdPlugin) ConstructBlockVolumeSpec(podUID types.UID, volumeName, mapPath string) (*volume.Spec, error) {
	pluginDir := plugin.host.GetVolumeDevicePluginDir(rbdPluginName)
	blkutil := volumepathhandler.NewBlockVolumePathHandler()

	globalMapPathUUID, err := blkutil.FindGlobalMapPathUUIDFromPod(pluginDir, mapPath, podUID)
	if err != nil {
		return nil, err
	}
	klog.V(5).Infof("globalMapPathUUID: %v, err: %v", globalMapPathUUID, err)
	globalMapPath := filepath.Dir(globalMapPathUUID)
	if len(globalMapPath) == 1 {
		return nil, fmt.Errorf("failed to retrieve volume plugin information from globalMapPathUUID: %v", globalMapPathUUID)
	}
	return getVolumeSpecFromGlobalMapPath(globalMapPath, volumeName)
}

func getVolumeSpecFromGlobalMapPath(globalMapPath, volumeName string) (*volume.Spec, error) {
	// Retrieve volume spec information from globalMapPath
	// globalMapPath example:
	//   plugins/kubernetes.io/{PluginName}/{DefaultKubeletVolumeDevicesDirName}/{volumePluginDependentPath}
	pool, image, err := getPoolAndImageFromMapPath(globalMapPath)
	if err != nil {
		return nil, err
	}
	block := v1.PersistentVolumeBlock
	rbdVolume := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: volumeName,
		},
		Spec: v1.PersistentVolumeSpec{
			PersistentVolumeSource: v1.PersistentVolumeSource{
				RBD: &v1.RBDPersistentVolumeSource{
					RBDImage: image,
					RBDPool:  pool,
				},
			},
			VolumeMode: &block,
		},
	}

	return volume.NewSpecFromPersistentVolume(rbdVolume, true), nil
}

func (plugin *rbdPlugin) NewBlockVolumeMapper(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.BlockVolumeMapper, error) {

	var uid types.UID
	if pod != nil {
		uid = pod.UID
	}
	secret := ""
	if pod != nil {
		secretName, secretNs, err := getSecretNameAndNamespace(spec, pod.Namespace)
		if err != nil {
			return nil, err
		}
		if len(secretName) > 0 && len(secretNs) > 0 {
			// if secret is provideded, retrieve it
			kubeClient := plugin.host.GetKubeClient()
			if kubeClient == nil {
				return nil, fmt.Errorf("cannot get kube client")
			}
			secrets, err := kubeClient.CoreV1().Secrets(secretNs).Get(context.TODO(), secretName, metav1.GetOptions{})
			if err != nil {
				err = fmt.Errorf("couldn't get secret %v/%v err: %v", secretNs, secretName, err)
				return nil, err
			}
			for _, data := range secrets.Data {
				secret = string(data)
			}
		}
	}

	return plugin.newBlockVolumeMapperInternal(spec, uid, &rbdUtil{}, secret, plugin.host.GetMounter(plugin.GetPluginName()), plugin.host.GetExec(plugin.GetPluginName()))
}

func (plugin *rbdPlugin) newBlockVolumeMapperInternal(spec *volume.Spec, podUID types.UID, manager diskManager, secret string, mounter mount.Interface, exec utilexec.Interface) (volume.BlockVolumeMapper, error) {
	mon, err := getVolumeSourceMonitors(spec)
	if err != nil {
		return nil, err
	}
	img, err := getVolumeSourceImage(spec)
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

	return &rbdDiskMapper{
		rbd:     newRBD(podUID, spec.Name(), img, pool, ro, plugin, manager),
		mon:     mon,
		id:      id,
		keyring: keyring,
		secret:  secret,
	}, nil
}

func (plugin *rbdPlugin) NewBlockVolumeUnmapper(volName string, podUID types.UID) (volume.BlockVolumeUnmapper, error) {
	return plugin.newUnmapperInternal(volName, podUID, &rbdUtil{})
}

func (plugin *rbdPlugin) newUnmapperInternal(volName string, podUID types.UID, manager diskManager) (volume.BlockVolumeUnmapper, error) {
	return &rbdDiskUnmapper{
		rbdDiskMapper: &rbdDiskMapper{
			rbd: newRBD(podUID, volName, "", "", false, plugin, manager),
			mon: make([]string, 0),
		},
	}, nil
}

func (plugin *rbdPlugin) getDeviceNameFromOldMountPath(mounter mount.Interface, mountPath string) (string, error) {
	refs, err := mounter.GetMountRefs(mountPath)
	if err != nil {
		return "", err
	}
	// baseMountPath is the prefix of deprecated device global mounted path,
	// such as: /var/lib/kubelet/plugins/kubernetes.io/rbd/rbd
	baseMountPath := filepath.Join(plugin.host.GetPluginDir(rbdPluginName), "rbd")
	for _, ref := range refs {
		if dstrings.HasPrefix(ref, baseMountPath) {
			return filepath.Rel(baseMountPath, ref)
		}
	}
	return "", fmt.Errorf("can't find source name from mounted path: %s", mountPath)
}

func (plugin *rbdPlugin) NewDeleter(spec *volume.Spec) (volume.Deleter, error) {
	if spec.PersistentVolume != nil && spec.PersistentVolume.Spec.RBD == nil {
		return nil, fmt.Errorf("spec.PersistentVolume.Spec.RBD is nil")
	}

	admin, secret, err := plugin.getAdminAndSecret(spec)
	if err != nil {
		return nil, err
	}

	return plugin.newDeleterInternal(spec, admin, secret, &rbdUtil{})
}

func (plugin *rbdPlugin) newDeleterInternal(spec *volume.Spec, admin, secret string, manager diskManager) (volume.Deleter, error) {
	return &rbdVolumeDeleter{
		rbdMounter: &rbdMounter{
			rbd:         newRBD("", spec.Name(), spec.PersistentVolume.Spec.RBD.RBDImage, spec.PersistentVolume.Spec.RBD.RBDPool, false, plugin, manager),
			Mon:         spec.PersistentVolume.Spec.RBD.CephMonitors,
			adminID:     admin,
			adminSecret: secret,
		}}, nil
}

func (plugin *rbdPlugin) NewProvisioner(options volume.VolumeOptions) (volume.Provisioner, error) {
	return plugin.newProvisionerInternal(options, &rbdUtil{})
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

func (r *rbdVolumeProvisioner) Provision(selectedNode *v1.Node, allowedTopologies []v1.TopologySelectorTerm) (*v1.PersistentVolume, error) {
	if !volutil.AccessModesContainedInAll(r.plugin.GetAccessModes(), r.options.PVC.Spec.AccessModes) {
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
	keyring := ""
	imageFormat := rbdImageFormat2
	fstype := ""

	for k, v := range r.options.Parameters {
		switch dstrings.ToLower(k) {
		case "monitors":
			arr := dstrings.Split(v, ",")
			r.Mon = append(r.Mon, arr...)
		case "adminid":
			r.adminID = v
		case "adminsecretname":
			adminSecretName = v
		case "adminsecretnamespace":
			adminSecretNamespace = v
		case "userid":
			r.ID = v
		case "pool":
			r.Pool = v
		case "usersecretname":
			secretName = v
		case "usersecretnamespace":
			secretNamespace = v
		case "keyring":
			keyring = v
		case "imageformat":
			imageFormat = v
		case "imagefeatures":
			arr := dstrings.Split(v, ",")
			for _, f := range arr {
				if !supportedFeatures.Has(f) {
					return nil, fmt.Errorf("invalid feature %q for volume plugin %s, supported features are: %v", f, r.plugin.GetPluginName(), supportedFeatures)
				}
				r.imageFeatures = append(r.imageFeatures, f)
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
	if secretName == "" && keyring == "" {
		return nil, fmt.Errorf("must specify either keyring or user secret name")
	}
	if r.adminID == "" {
		r.adminID = rbdDefaultAdminID
	}
	if r.Pool == "" {
		r.Pool = rbdDefaultPool
	}
	if r.ID == "" {
		r.ID = r.adminID
	}

	// create random image name
	image := fmt.Sprintf("kubernetes-dynamic-pvc-%s", uuid.NewUUID())
	r.rbdMounter.Image = image
	rbd, sizeMB, err := r.manager.CreateImage(r)
	if err != nil {
		klog.Errorf("rbd: create volume failed, err: %v", err)
		return nil, err
	}
	klog.Infof("successfully created rbd image %q", image)
	pv := new(v1.PersistentVolume)
	metav1.SetMetaDataAnnotation(&pv.ObjectMeta, volutil.VolumeDynamicallyCreatedByKey, "rbd-dynamic-provisioner")

	if secretName != "" {
		rbd.SecretRef = new(v1.SecretReference)
		rbd.SecretRef.Name = secretName
		rbd.SecretRef.Namespace = secretNamespace
	} else {
		var filePathRegex = regexp.MustCompile(`^(?:/[^/!;` + "`" + ` ]+)+$`)
		if keyring != "" && !filePathRegex.MatchString(keyring) {
			return nil, fmt.Errorf("keyring field must contain a path to a file")
		}
		rbd.Keyring = keyring
	}

	volumeMode := r.options.PVC.Spec.VolumeMode
	if volumeMode != nil && *volumeMode == v1.PersistentVolumeBlock {
		// Block volumes should not have any FSType
		fstype = ""
	}

	rbd.RadosUser = r.ID
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
	pv.Spec.VolumeMode = volumeMode

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
	exec     utilexec.Interface
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
		mounter:         volutil.NewSafeFormatAndMountFromHost(plugin.GetPluginName(), plugin.host),
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
	ID            string
	Keyring       string
	Secret        string `datapolicy:"token"`
	fsType        string
	adminSecret   string `datapolicy:"token"`
	adminID       string
	mountOptions  []string
	imageFormat   string
	imageFeatures []string
	accessModes   []v1.PersistentVolumeAccessMode
}

var _ volume.Mounter = &rbdMounter{}

func (rbd *rbd) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        rbd.ReadOnly,
		Managed:         !rbd.ReadOnly,
		SupportsSELinux: true,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (b *rbdMounter) CanMount() error {
	return nil
}

func (b *rbdMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return b.SetUpAt(b.GetPath(), mounterArgs)
}

func (b *rbdMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	klog.V(4).Infof("rbd: attempting to setup at %s", dir)
	err := diskSetUp(b.manager, *b, dir, b.mounter, mounterArgs.FsGroup, mounterArgs.FSGroupChangePolicy, b.plugin)
	if err != nil {
		klog.Errorf("rbd: failed to setup at %s %v", dir, err)
	}
	klog.V(3).Infof("rbd: successfully setup at %s", dir)
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
	klog.V(4).Infof("rbd: attempting to teardown at %s", dir)
	if pathExists, pathErr := mount.PathExists(dir); pathErr != nil {
		return fmt.Errorf("error checking if path exists: %v", pathErr)
	} else if !pathExists {
		klog.Warningf("Warning: Unmount skipped because path does not exist: %v", dir)
		return nil
	}
	err := diskTearDown(c.manager, *c, dir, c.mounter)
	if err != nil {
		return err
	}
	klog.V(3).Infof("rbd: successfully teardown at %s", dir)
	return nil
}

var _ volume.BlockVolumeMapper = &rbdDiskMapper{}

type rbdDiskMapper struct {
	*rbd
	mon     []string
	id      string
	keyring string
	secret  string
}

var _ volume.BlockVolumeUnmapper = &rbdDiskUnmapper{}
var _ volume.CustomBlockVolumeUnmapper = &rbdDiskUnmapper{}

// GetGlobalMapPath returns global map path and error
// path: plugins/kubernetes.io/{PluginName}/volumeDevices/{rbd pool}-image-{rbd image-name}/{podUid}
func (rbd *rbd) GetGlobalMapPath(spec *volume.Spec) (string, error) {
	return rbd.rbdGlobalMapPath(spec)
}

// GetPodDeviceMapPath returns pod device map path and volume name
// path: pods/{podUid}/volumeDevices/kubernetes.io~rbd
// volumeName: pv0001
func (rbd *rbd) GetPodDeviceMapPath() (string, string) {
	return rbd.rbdPodDeviceMapPath()
}

func (rbd *rbd) rbdGlobalMapPath(spec *volume.Spec) (string, error) {
	mon, err := getVolumeSourceMonitors(spec)
	if err != nil {
		return "", err
	}
	img, err := getVolumeSourceImage(spec)
	if err != nil {
		return "", err
	}
	pool, err := getVolumeSourcePool(spec)
	if err != nil {
		return "", err
	}
	ro, err := getVolumeSourceReadOnly(spec)
	if err != nil {
		return "", err
	}

	mounter := &rbdMounter{
		rbd: newRBD("", spec.Name(), img, pool, ro, rbd.plugin, &rbdUtil{}),
		Mon: mon,
	}
	return rbd.manager.MakeGlobalVDPDName(*mounter.rbd), nil
}

func (rbd *rbd) rbdPodDeviceMapPath() (string, string) {
	name := rbdPluginName
	return rbd.plugin.host.GetPodVolumeDeviceDir(rbd.podUID, utilstrings.EscapeQualifiedName(name)), rbd.volName
}

type rbdDiskUnmapper struct {
	*rbdDiskMapper
}

func getPoolAndImageFromMapPath(mapPath string) (string, string, error) {

	pathParts := dstrings.Split(mapPath, "/")
	if len(pathParts) < 2 {
		return "", "", fmt.Errorf("corrupted mapPath")
	}
	rbdParts := dstrings.Split(pathParts[len(pathParts)-1], "-image-")

	if len(rbdParts) < 2 {
		return "", "", fmt.Errorf("corrupted mapPath")
	}
	return string(rbdParts[0]), string(rbdParts[1]), nil
}

func getBlockVolumeDevice(mapPath string) (string, error) {
	pool, image, err := getPoolAndImageFromMapPath(mapPath)
	if err != nil {
		return "", err
	}
	// Getting full device path
	device, found := getDevFromImageAndPool(pool, image)
	if !found {
		return "", err
	}
	return device, nil
}

func (rbd *rbdDiskUnmapper) TearDownDevice(mapPath, _ string) error {

	device, err := getBlockVolumeDevice(mapPath)
	if err != nil {
		return fmt.Errorf("rbd: failed to get loopback for device: %v, err: %v", device, err)
	}

	err = rbd.manager.DetachBlockDisk(*rbd, mapPath)
	if err != nil {
		return fmt.Errorf("rbd: failed to detach disk: %s\nError: %v", mapPath, err)
	}
	klog.V(4).Infof("rbd: %q is unmapped, deleting the directory", mapPath)

	err = os.RemoveAll(mapPath)
	if err != nil {
		return fmt.Errorf("rbd: failed to delete the directory: %s\nError: %v", mapPath, err)
	}
	klog.V(4).Infof("rbd: successfully detached disk: %s", mapPath)

	return nil
}

func (rbd *rbdDiskUnmapper) UnmapPodDevice() error {
	return nil
}

func getVolumeSourceMonitors(spec *volume.Spec) ([]string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.CephMonitors, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.CephMonitors, nil
	}

	return nil, fmt.Errorf("spec does not reference a RBD volume type")
}

func getVolumeSourceImage(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.RBDImage, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.RBDImage, nil
	}

	return "", fmt.Errorf("spec does not reference a RBD volume type")
}

func getVolumeSourceFSType(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.FSType, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.FSType, nil
	}

	return "", fmt.Errorf("spec does not reference a RBD volume type")
}

func getVolumeSourcePool(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.RBDPool, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.RBDPool, nil
	}

	return "", fmt.Errorf("spec does not reference a RBD volume type")
}

func getVolumeSourceUser(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.RadosUser, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.RadosUser, nil
	}

	return "", fmt.Errorf("spec does not reference a RBD volume type")
}

func getVolumeSourceKeyRing(spec *volume.Spec) (string, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD.Keyring, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD.Keyring, nil
	}

	return "", fmt.Errorf("spec does not reference a RBD volume type")
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

	return false, fmt.Errorf("spec does not reference a RBD volume type")
}

func getVolumeAccessModes(spec *volume.Spec) ([]v1.PersistentVolumeAccessMode, error) {
	// Only PersistentVolumeSpec has AccessModes
	if spec.PersistentVolume != nil {
		if spec.PersistentVolume.Spec.RBD != nil {
			return spec.PersistentVolume.Spec.AccessModes, nil
		}
		return nil, fmt.Errorf("spec does not reference a RBD volume type")
	}

	return nil, nil
}

func parsePVSecret(namespace, secretName string, kubeClient clientset.Interface) (string, error) {
	secret, err := volutil.GetSecretForPV(namespace, secretName, rbdPluginName, kubeClient)
	if err != nil {
		klog.Errorf("failed to get secret from [%q/%q]: %+v", namespace, secretName, err)
		return "", fmt.Errorf("failed to get secret from [%q/%q]: %+v", namespace, secretName, err)
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
	return "", "", fmt.Errorf("spec does not reference an RBD volume type")
}
