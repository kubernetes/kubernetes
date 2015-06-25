/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/golang/glog"
)

// This is the primary entrypoint for volume plugins.
func ProbeVolumePlugins() []volume.VolumePlugin {
	return []volume.VolumePlugin{&RBDPlugin{nil, exec.New()}}
}

type RBDPlugin struct {
	host volume.VolumeHost
	exe  exec.Interface
}

var _ volume.VolumePlugin = &RBDPlugin{}

const (
	RBDPluginName = "kubernetes.io/rbd"
)

func (plugin *RBDPlugin) Init(host volume.VolumeHost) {
	plugin.host = host
}

func (plugin *RBDPlugin) Name() string {
	return RBDPluginName
}

func (plugin *RBDPlugin) CanSupport(spec *volume.Spec) bool {
	if spec.VolumeSource.RBD == nil && spec.PersistentVolumeSource.RBD == nil {
		return false
	}
	// see if rbd is there
	_, err := plugin.execCommand("rbd", []string{"-h"})
	if err == nil {
		return true
	}

	return false
}

func (plugin *RBDPlugin) GetAccessModes() []api.PersistentVolumeAccessMode {
	return []api.PersistentVolumeAccessMode{
		api.ReadWriteOnce,
		api.ReadOnlyMany,
	}
}

func (plugin *RBDPlugin) NewBuilder(spec *volume.Spec, pod *api.Pod, _ volume.VolumeOptions, mounter mount.Interface) (volume.Builder, error) {
	secret := ""
	source := plugin.getRBDVolumeSource(spec)

	if source.SecretRef != nil {
		kubeClient := plugin.host.GetKubeClient()
		if kubeClient == nil {
			return nil, fmt.Errorf("Cannot get kube client")
		}

		secretName, err := kubeClient.Secrets(pod.Namespace).Get(source.SecretRef.Name)
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
	return plugin.newBuilderInternal(spec, pod.UID, &RBDUtil{}, mounter, secret)
}

func (plugin *RBDPlugin) getRBDVolumeSource(spec *volume.Spec) *api.RBDVolumeSource {
	if spec.VolumeSource.RBD != nil {
		return spec.VolumeSource.RBD
	} else {
		return spec.PersistentVolumeSource.RBD
	}
}

func (plugin *RBDPlugin) newBuilderInternal(spec *volume.Spec, podUID types.UID, manager diskManager, mounter mount.Interface, secret string) (volume.Builder, error) {
	source := plugin.getRBDVolumeSource(spec)
	pool := source.RBDPool
	if pool == "" {
		pool = "rbd"
	}
	id := source.RadosUser
	if id == "" {
		id = "admin"
	}
	keyring := source.Keyring
	if keyring == "" {
		keyring = "/etc/ceph/keyring"
	}

	return &rbd{
		podUID:   podUID,
		volName:  spec.Name,
		mon:      source.CephMonitors,
		image:    source.RBDImage,
		pool:     pool,
		id:       id,
		keyring:  keyring,
		secret:   secret,
		fsType:   source.FSType,
		readOnly: source.ReadOnly,
		manager:  manager,
		mounter:  mounter,
		plugin:   plugin,
	}, nil
}

func (plugin *RBDPlugin) NewCleaner(volName string, podUID types.UID, mounter mount.Interface) (volume.Cleaner, error) {
	// Inject real implementations here, test through the internal function.
	return plugin.newCleanerInternal(volName, podUID, &RBDUtil{}, mounter)
}

func (plugin *RBDPlugin) newCleanerInternal(volName string, podUID types.UID, manager diskManager, mounter mount.Interface) (volume.Cleaner, error) {
	return &rbd{
		podUID:  podUID,
		volName: volName,
		manager: manager,
		mounter: mounter,
		plugin:  plugin,
	}, nil
}

type rbd struct {
	volName  string
	podUID   types.UID
	mon      []string
	pool     string
	id       string
	image    string
	keyring  string
	secret   string
	fsType   string
	readOnly bool
	plugin   *RBDPlugin
	mounter  mount.Interface
	// Utility interface that provides API calls to the provider to attach/detach disks.
	manager diskManager
}

func (rbd *rbd) GetPath() string {
	name := RBDPluginName
	// safe to use PodVolumeDir now: volume teardown occurs before pod is cleaned up
	return rbd.plugin.host.GetPodVolumeDir(rbd.podUID, util.EscapeQualifiedNameForDisk(name), rbd.volName)
}

func (rbd *rbd) SetUp() error {
	return rbd.SetUpAt(rbd.GetPath())
}

func (rbd *rbd) SetUpAt(dir string) error {
	// diskSetUp checks mountpoints and prevent repeated calls
	err := diskSetUp(rbd.manager, *rbd, dir, rbd.mounter)
	if err != nil {
		glog.Errorf("rbd: failed to setup")
		return err
	}
	globalPDPath := rbd.manager.MakeGlobalPDName(*rbd)
	// make mountpoint rw/ro work as expected
	//FIXME revisit pkg/util/mount and ensure rw/ro is implemented as expected
	mode := "rw"
	if rbd.readOnly {
		mode = "ro"
	}
	rbd.plugin.execCommand("mount", []string{"-o", "remount," + mode, globalPDPath, dir})

	return nil
}

// Unmounts the bind mount, and detaches the disk only if the disk
// resource was the last reference to that disk on the kubelet.
func (rbd *rbd) TearDown() error {
	return rbd.TearDownAt(rbd.GetPath())
}

func (rbd *rbd) TearDownAt(dir string) error {
	return diskTearDown(rbd.manager, *rbd, dir, rbd.mounter)
}

func (plugin *RBDPlugin) execCommand(command string, args []string) ([]byte, error) {
	cmd := plugin.exe.Command(command, args...)
	return cmd.CombinedOutput()
}
