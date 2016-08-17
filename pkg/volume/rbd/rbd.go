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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&rbdPlugin{nil, exec.New()}}
}

type rbdPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &rbdPlugin{}
var _ volume.PersistentVolumePlugin = &rbdPlugin{}

const (
	rbdPluginName = "kubernetes.io/rbd"
)

func (plugin *rbdPlugin) Init(host volume.VolumeHost) error {
	plugin.host = host
	return nil
}

func (plugin *rbdPlugin) GetPluginName() string {
	return rbdPluginName
}

func (plugin *rbdPlugin) GetVolumeName(spec *volume.Spec) (string, error) {
	volumeSource, _, err := getVolumeSource(spec)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf(
		"%v:%v",
		volumeSource.CephMonitors,
		volumeSource.RBDImage), nil
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

func (plugin *rbdPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

func (plugin *rbdPlugin) NewMounter(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions) (volume.Mounter, error) {
	secret := ""
	source, _ := plugin.getRBDVolumeSource(spec)

	if source.SecretRef != nil {
		kubeClient := plugin.host.GetKubeClient()
		if kubeClient == nil {
			return nil, fmt.Errorf("Cannot get kube client")
		}

		secretName, err := kubeClient.Core().Secrets(pod.Namespace).Get(source.SecretRef.Name)
		if err != nil {
			glog.Errorf("Couldn't get secret %v/%v", pod.Namespace, source.SecretRef)
			return nil, err
		}
		for name, data := range secretName.Data {
			secret = string(data)
			glog.V(1).Infof("ceph secret info: %s/%s", name, secret)
		}

	}
	// Inject real implementations here, test through the internal function.
	return plugin.newMounterInternal(spec, pod.UID, &RBDUtil{}, plugin.host.GetMounter(), secret)
}

func (plugin *rbdPlugin) getRBDVolumeSource(spec *volume.Spec) (*api.RBDVolumeSource, bool) {
	// rbd volumes used directly in a pod have a ReadOnly flag set by the pod author.
	// rbd volumes used as a PersistentVolume gets the ReadOnly flag indirectly through the persistent-claim volume used to mount the PV
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD, spec.Volume.RBD.ReadOnly
	} else {
		return spec.PersistentVolume.Spec.RBD, spec.ReadOnly
	}
}

func (plugin *rbdPlugin) newMounterInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface, secret string) (volume.Mounter, error) {
	source, readOnly := plugin.getRBDVolumeSource(spec)
	pool := source.RBDPool
	id := source.RadosUser
	keyring := source.Keyring

	return &rbdMounter{
		rbd: &rbd{
			podUID:   podUID,
			volName:  spec.Name(),
			Image:    source.RBDImage,
			Pool:     pool,
			ReadOnly: readOnly,
			manager:  manager,
			mounter:  &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()},
			plugin:   plugin,
		},
		Mon:     source.CephMonitors,
		Id:      id,
		Keyring: keyring,
		Secret:  secret,
		fsType:  source.FSType,
	}, nil
}

func (plugin *rbdPlugin) NewUnmounter(volName string, podUID types.UID) (volume.Unmounter, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newUnmounterInternal(volName, podUID, &RBDUtil{}, plugin.host.GetMounter())
}

func (plugin *rbdPlugin) newUnmounterInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Unmounter, error) {
	return &rbdUnmounter{
		rbdMounter: &rbdMounter{
			rbd: &rbd{
				podUID:  podUID,
				volName: volName,
				manager: manager,
				mounter: &mount.SafeFormatAndMount{Interface: mounter, Runner: exec.New()},
				plugin:  plugin,
			},
			Mon: make([]string, 0),
		},
	}, nil
}

func (plugin *rbdPlugin) ConstructVolumeSpec(volumeName, mountPath string) (*volume.Spec, error) {
	rbdVolume := &api.Volume{
		Name: volumeName,
		VolumeSource: api.VolumeSource{
			RBD: &api.RBDVolumeSource{
				CephMonitors: []string{},
			},
		},
	}
	return volume.NewSpecFromVolume(rbdVolume), nil
}

type rbd struct {
	volName  string
	podUID   types.UID
	Pool     string
	Image    string
	ReadOnly bool
	plugin   *rbdPlugin
	mounter  *mount.SafeFormatAndMount
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
	volume.MetricsNil
}

func (rbd *rbd) GetPath() string {
	name := rbdPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return rbd.plugin.host.GetPodVolumeDir(rbd.podUID, strings.EscapeQualifiedNameForDisk(name), rbd.volName)
}

type rbdMounter struct {
	*rbd
	// capitalized so they can be exported in persistRBD()
	Mon     []string
	Id      string
	Keyring string
	Secret  string
	fsType  string
}

var _ volume.Mounter = &rbdMounter{}

func (b *rbd) GetAttributes() volume.Attributes {
	return volume.Attributes{
		ReadOnly:        b.ReadOnly,
		Managed:         !b.ReadOnly,
		SupportsSELinux: true,
	}
}

func (b *rbdMounter) SetUp(fsGroup *int64) error {
	return b.SetUpAt(b.GetPath(), fsGroup)
}

func (b *rbdMounter) SetUpAt(dir string, fsGroup *int64) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	glog.V(4).Infof("rbd: attempting to SetUp and mount %s", dir)
	err := diskSetUp(b.manager, *b, dir, b.mounter, fsGroup)
	if err != nil {
		glog.Errorf("rbd: failed to setup mount %s %v", dir, err)
	}
	return err
}

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
	return diskTearDown(c.manager, *c, dir, c.mounter)
}

func (plugin *rbdPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}

func getVolumeSource(
	spec *volume.Spec) (*api.RBDVolumeSource, bool, error) {
	if spec.Volume != nil && spec.Volume.RBD != nil {
		return spec.Volume.RBD, spec.Volume.RBD.ReadOnly, nil
	} else if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.RBD != nil {
		return spec.PersistentVolume.Spec.RBD, spec.ReadOnly, nil
	}

	return nil, false, fmt.Errorf("Spec does not reference a RBD volume type")
}
