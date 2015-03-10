/*
Copyright 2015 Google Inc. All rights reserved.

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

package secret

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/golang/glog"
)

// ProbeVolumePlugin is the entry point for plugin detection in a package.
func ProbeVolumePlugins() []volume.Plugin {
	return []volume.Plugin{&secretPlugin{}}
}

const (
	secretPluginName = "kubernetes.io/secret"
)

// secretPlugin implements the VolumePlugin interface.
type secretPlugin struct {
	host volume.Host
}

func (plugin *secretPlugin) Init(host volume.Host) {
	plugin.host = host
}

func (plugin *secretPlugin) Name() string {
	return secretPluginName
}

func (plugin *secretPlugin) CanSupport(spec *api.Volume) bool {
	if spec.Secret != nil {
		return true
	}

	return false
}

func (plugin *secretPlugin) NewBuilder(spec *api.Volume, podRef *api.ObjectReference) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, podRef)
}

func (plugin *secretPlugin) newBuilderInternal(spec *api.Volume, podRef *api.ObjectReference) (volume.Builder, error) {
	return &secretVolume{spec.Name, podRef, plugin, &spec.Secret.Target}, nil
}

func (plugin *secretPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID)
}

func (plugin *secretPlugin) newCleanerInternal(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &secretVolume{volName, &api.ObjectReference{UID: podUID}, plugin, nil}, nil
}

// secretVolume handles retrieving secrets from the API server
// and placing them into the volume on the host.
type secretVolume struct {
	volName   string
	podRef    *api.ObjectReference
	plugin    *secretPlugin
	secretRef *api.ObjectReference
}

func (sv *secretVolume) SetUp() error {
	// TODO: explore tmpfs for secret volumes
	hostPath := sv.GetPath()
	glog.V(3).Infof("Setting up volume %v for pod %v at %v", sv.volName, sv.podRef.UID, hostPath)
	err := os.MkdirAll(hostPath, 0777)
	if err != nil {
		return err
	}

	kubeClient := sv.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("Cannot setup secret volume %v because kube client is not configured", sv)
	}

	secret, err := kubeClient.Secrets(sv.podRef.Namespace).Get(sv.secretRef.Name)
	if err != nil {
		glog.Errorf("Couldn't get secret %v/%v", sv.secretRef.Namespace, sv.secretRef.Name)
		return err
	}

	for name, data := range secret.Data {
		hostFilePath := path.Join(hostPath, name)
		err := ioutil.WriteFile(hostFilePath, data, 0777)
		if err != nil {
			glog.Errorf("Error writing secret data to host path: %v, %v", hostFilePath, err)
			return err
		}
	}

	return nil
}

func (sv *secretVolume) GetPath() string {
	return sv.plugin.host.GetPodVolumeDir(sv.podRef.UID, volume.EscapePluginName(secretPluginName), sv.volName)
}

func (sv *secretVolume) TearDown() error {
	glog.V(3).Infof("Tearing down volume %v for pod %v at %v", sv.volName, sv.podRef.UID, sv.GetPath())
	tmpDir, err := volume.RenameDirectory(sv.GetPath(), sv.volName+".deleting~")
	if err != nil {
		return err
	}
	err = os.RemoveAll(tmpDir)
	if err != nil {
		return err
	}
	return nil
}
