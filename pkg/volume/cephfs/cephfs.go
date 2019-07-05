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

package cephfs

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
	utilstrings "k8s.io/utils/strings"
)

// ProbeVolumePlugins is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&cephfsPlugin{nil}}
}

type cephfsPlugin struct {
	host volume.VolumeHost
}

var _ volume.VolumePlugin = &cephfsPlugin{}

const (
	cephfsPluginName = "kubernetes.io/cephfs"
)

func (plugin *cephfsPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *cephfsPlugin) GetPluginName() string {
	return cephfsPluginName
}

func (plugin *cephfsPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	mon, _, _, _, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%v", mon), nil
}

func (plugin *cephfsPlugin) CanSupport(spec *volume.Spec) bool {
	return (spec.Volume != nil && spec.Volume.CephFS != nil) || (spec.PersistentVolume != nil && spec.PersistentVolume.Spec.CephFS != nil)
}

func (plugin *cephfsPlugin) IsMigratedToCSI() bool {
	return false
}

func (plugin *cephfsPlugin) RequiresRemount() bool {
	return false
}

func (plugin *cephfsPlugin) SupportsMountOption() bool {
	return true
}

func (plugin *cephfsPlugin) SupportsBulkVolumeVerification() bool {
	return false
}

func (plugin *cephfsPlugin) GetAccessModes() []v1.PersistentVolumeAccessMode {
	return []v1.PersistentVolumeAccessMode{
		v1.ReadWriteOnce,
		v1.ReadOnlyMany,
		v1.ReadWriteMany,
	}
}

func (plugin *cephfsPlugin) NewMounter(spec *volume.Spec, pod *v1.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
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
		for name, data := range secrets.Data {
			secret = string(data)
			klog.V(4).Infof("found ceph secret info: %s", name)
		}
	}
	return plugin.newMounterInternal(spec, pod.UID, plugin.host.GetMounter(plugin.GetPluginName()), secret)
}

func (plugin *cephfsPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, mounter mount.Interface, secret string) (volume.Mounter, error) {
	mon, path, id, secretFile, readOnly, err := getVolumeSource(spec)
	if err != nil {
		return nil, err
	}

	if id == "" {
		id = "admin"
	}
	if path == "" {
		path = "/"
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}

	if secretFile == "" {
		secretFile = "/etc/ceph/" + id + ".secret"
	}

	return &cephfsMounter{
		cephfs: &cephfs{
			podUID:       podUID,
			volName:      spec.Name(),
			mon:          mon,
			path:         path,
			secret:       secret,
			id:           id,
			secretFile:   secretFile,
			readonly:     readOnly,
			mounter:      mounter,
			plugin:       plugin,
			mountOptions: util.MountOptionFromSpec(spec),
		},
	}, nil
}

func (plugin *cephfsPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	return plugin.newUnmounterInternal(volName, podUID, plugin.host.GetMounter(plugin.GetPluginName()))
}

func (plugin *cephfsPlugin) newUnmounterInternal(volName string, podUID types.UID, mounter mount.Interface) (volume.Unmounter, error) {
	return &cephfsUnmounter{
		cephfs: &cephfs{
			podUID:  podUID,
			volName: volName,
			mounter: mounter,
			plugin:  plugin},
	}, nil
}

func (plugin *cephfsPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	cephfsVolume := &v1.Volume{
		Name: volumeName,
		VolumeSource: v1.VolumeSource{
			CephFS: &v1.CephFSVolumeSource{
				Monitors: []string{},
				Path:     mountPath,
			},
		},
	}
	return volume.NewSpecFromVolume(cephfsVolume), nil
}

// CephFS volumes represent a bare host file or directory mount of an CephFS export.
type cephfs struct {
	volName    string
	podUID     types.UID
	mon        []string
	path       string
	id         string
	secret     string
	secretFile string
	readonly   bool
	mounter    mount.Interface
	plugin     *cephfsPlugin
	volume.MetricsNil
	mountOptions []string
}

type cephfsMounter struct {
	*cephfs
}

var _ volume.Mounter = &cephfsMounter{}

func (cephfsVolume *cephfsMounter) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        cephfsVolume.readonly,
		Managed:         false,
		SupportsSELinux: false,
	}
}

// Checks prior to mount operations to verify that the required components (binaries, etc.)
// to mount the volume are available on the underlying node.
// If not, it returns an error
func (cephfsVolume *cephfsMounter) CanMount() error {
	return nil
}

// SetUp attaches the disk and bind mounts to the volume path.
func (cephfsVolume *cephfsMounter) SetUp(mounterArgs volume.MounterArgs) error {
	return cephfsVolume.SetUpAt(cephfsVolume.GetPath(), mounterArgs)
}

// SetUpAt attaches the disk and bind mounts to the volume path.
func (cephfsVolume *cephfsMounter) SetUpAt(dir string, mounterArgs volume.MounterArgs) error {
	notMnt, err := cephfsVolume.mounter.IsLikelyNotMountPoint(dir)
	klog.V(4).Infof("CephFS mount set up: %s %v %v", dir, !notMnt, err)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if !notMnt {
		return nil
	}

	if err := os.MkdirAll(dir, 0750); err != nil {
		return err
	}

	// check whether it belongs to fuse, if not, default to use kernel mount.
	if cephfsVolume.checkFuseMount() {
		klog.V(4).Info("CephFS fuse mount.")
		err = cephfsVolume.execFuseMount(dir)
		// cleanup no matter if fuse mount fail.
		keyringPath := cephfsVolume.GetKeyringPath()
		_, StatErr := os.Stat(keyringPath)
		if !os.IsNotExist(StatErr) {
			os.RemoveAll(keyringPath)
		}
		if err == nil {
			// cephfs fuse mount succeeded.
			return nil
		}
		// if cephfs fuse mount failed, fallback to kernel mount.
		klog.V(2).Infof("CephFS fuse mount failed: %v, fallback to kernel mount.", err)

	}
	klog.V(4).Info("CephFS kernel mount.")

	err = cephfsVolume.execMount(dir)
	if err != nil {
		// cleanup upon failure.
		mount.CleanupMountPoint(dir, cephfsVolume.mounter, false)
		return err
	}
	return nil
}

type cephfsUnmounter struct {
	*cephfs
}

var _ volume.Unmounter = &cephfsUnmounter{}

// TearDown unmounts the bind mount
func (cephfsVolume *cephfsUnmounter) TearDown() error {
	return cephfsVolume.TearDownAt(cephfsVolume.GetPath())
}

// TearDownAt unmounts the bind mount
func (cephfsVolume *cephfsUnmounter) TearDownAt(dir string) error {
	return mount.CleanupMountPoint(dir, cephfsVolume.mounter, false)
}

// GetPath creates global mount path
func (cephfsVolume *cephfs) GetPath() string {
	name := cephfsPluginName
	return cephfsVolume.plugin.host.GetPodVolumeDir(cephfsVolume.podUID, utilstrings.EscapeQualifiedName(name), cephfsVolume.volName)
}

// GetKeyringPath creates cephfuse keyring path
func (cephfsVolume *cephfs) GetKeyringPath() string {
	name := cephfsPluginName
	volumeDir := cephfsVolume.plugin.host.GetPodVolumeDir(cephfsVolume.podUID, utilstrings.EscapeQualifiedName(name), cephfsVolume.volName)
	volumeKeyringDir := volumeDir + "~keyring"
	return volumeKeyringDir
}

func (cephfsVolume *cephfs) execMount(mountpoint string) error {
	// cephfs mount option
	cephOpt := ""
	// override secretfile if secret is provided
	if cephfsVolume.secret != "" {
		cephOpt = "name=" + cephfsVolume.id + ",secret=" + cephfsVolume.secret
	} else {
		cephOpt = "name=" + cephfsVolume.id + ",secretfile=" + cephfsVolume.secretFile
	}
	// build option array
	opt := []string{}
	if cephfsVolume.readonly {
		opt = append(opt, "ro")
	}
	opt = append(opt, cephOpt)

	// build src like mon1:6789,mon2:6789,mon3:6789:/
	hosts := cephfsVolume.mon
	l := len(hosts)
	// pass all monitors and let ceph randomize and fail over
	i := 0
	src := ""
	for i = 0; i < l-1; i++ {
		src += hosts[i] + ","
	}
	src += hosts[i] + ":" + cephfsVolume.path

	opt = util.JoinMountOptions(cephfsVolume.mountOptions, opt)
	if err := cephfsVolume.mounter.Mount(src, mountpoint, "ceph", opt); err != nil {
		return fmt.Errorf("CephFS: mount failed: %v", err)
	}

	return nil
}

func (cephfsVolume *cephfsMounter) checkFuseMount() bool {
	execute := cephfsVolume.plugin.host.GetExec(cephfsVolume.plugin.GetPluginName())
	switch runtime.GOOS {
	case "linux":
		if _, err := execute.Run("/usr/bin/test", "-x", "/sbin/mount.fuse.ceph"); err == nil {
			klog.V(4).Info("/sbin/mount.fuse.ceph exists, it should be fuse mount.")
			return true
		}
		return false
	}
	return false
}

func (cephfsVolume *cephfs) execFuseMount(mountpoint string) error {
	// cephfs keyring file
	keyringFile := ""
	// override secretfile if secret is provided
	if cephfsVolume.secret != "" {
		// TODO: cephfs fuse currently doesn't support secret option,
		// remove keyring file create once secret option is supported.
		klog.V(4).Info("cephfs mount begin using fuse.")

		keyringPath := cephfsVolume.GetKeyringPath()
		os.MkdirAll(keyringPath, 0750)

		payload := make(map[string]util.FileProjection, 1)
		var fileProjection util.FileProjection

		keyring := fmt.Sprintf("[client.%s]\nkey = %s\n", cephfsVolume.id, cephfsVolume.secret)

		fileProjection.Data = []byte(keyring)
		fileProjection.Mode = int32(0644)
		fileName := cephfsVolume.id + ".keyring"

		payload[fileName] = fileProjection

		writerContext := fmt.Sprintf("cephfuse:%v.keyring", cephfsVolume.id)
		writer, err := util.NewAtomicWriter(keyringPath, writerContext)
		if err != nil {
			klog.Errorf("failed to create atomic writer: %v", err)
			return err
		}

		err = writer.Write(payload)
		if err != nil {
			klog.Errorf("failed to write payload to dir: %v", err)
			return err
		}

		keyringFile = filepath.Join(keyringPath, fileName)

	} else {
		keyringFile = cephfsVolume.secretFile
	}

	// build src like mon1:6789,mon2:6789,mon3:6789:/
	hosts := cephfsVolume.mon
	l := len(hosts)
	// pass all monitors and let ceph randomize and fail over
	i := 0
	src := ""
	for i = 0; i < l-1; i++ {
		src += hosts[i] + ","
	}
	src += hosts[i]

	mountArgs := []string{}
	mountArgs = append(mountArgs, "-k")
	mountArgs = append(mountArgs, keyringFile)
	mountArgs = append(mountArgs, "-m")
	mountArgs = append(mountArgs, src)
	mountArgs = append(mountArgs, mountpoint)
	mountArgs = append(mountArgs, "-r")
	mountArgs = append(mountArgs, cephfsVolume.path)
	mountArgs = append(mountArgs, "--id")
	mountArgs = append(mountArgs, cephfsVolume.id)

	// build option array
	opt := []string{}
	if cephfsVolume.readonly {
		opt = append(opt, "ro")
	}
	opt = util.JoinMountOptions(cephfsVolume.mountOptions, opt)
	if len(opt) > 0 {
		mountArgs = append(mountArgs, "-o")
		mountArgs = append(mountArgs, strings.Join(opt, ","))
	}

	klog.V(4).Infof("Mounting cmd ceph-fuse with arguments (%s)", mountArgs)
	command := exec.Command("ceph-fuse", mountArgs...)
	output, err := command.CombinedOutput()
	if err != nil || !(strings.Contains(string(output), "starting fuse")) {
		return fmt.Errorf("Ceph-fuse failed: %v\narguments: %s\nOutput: %s", err, mountArgs, string(output))
	}

	return nil
}

func getVolumeSource(spec *volume.Spec) ([]string, string, string, string, bool, error) {
	if spec.Volume != nil && spec.Volume.CephFS != nil {
		mon := spec.Volume.CephFS.Monitors
		path := spec.Volume.CephFS.Path
		user := spec.Volume.CephFS.User
		secretFile := spec.Volume.CephFS.SecretFile
		readOnly := spec.Volume.CephFS.ReadOnly
		return mon, path, user, secretFile, readOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.CephFS != nil {
		mon := spec.PersistentVolume.Spec.CephFS.Monitors
		path := spec.PersistentVolume.Spec.CephFS.Path
		user := spec.PersistentVolume.Spec.CephFS.User
		secretFile := spec.PersistentVolume.Spec.CephFS.SecretFile
		readOnly := spec.PersistentVolume.Spec.CephFS.ReadOnly
		return mon, path, user, secretFile, readOnly, nil
	}

	return nil, "", "", "", false, fmt.Errorf("Spec does not reference a CephFS volume type")
}

func getSecretNameAndNamespace(spec *volume.Spec, defaultNamespace string) (string, string, error) {
	if spec.Volume != nil && spec.Volume.CephFS != nil {
		localSecretRef := spec.Volume.CephFS.SecretRef
		if localSecretRef != nil {
			return localSecretRef.Name, defaultNamespace, nil
		}
		return "", "", nil

	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.CephFS != nil {
		secretRef := spec.PersistentVolume.Spec.CephFS.SecretRef
		secretNs := defaultNamespace
		if secretRef != nil {
			if len(secretRef.Namespace) != 0 {
				secretNs = secretRef.Namespace
			}
			return secretRef.Name, secretNs, nil
		}
		return "", "", nil
	}
	return "", "", fmt.Errorf("Spec does not reference an CephFS volume type")
}
