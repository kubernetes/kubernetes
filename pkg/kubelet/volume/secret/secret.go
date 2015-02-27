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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
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
	if spec.Source.Secret != nil {
		return true
	}

	return false
}

func (plugin *secretPlugin) NewBuilder(spec *api.Volume, podUID types.UID) (volume.Builder, error) {
	return plugin.newBuilderInternal(spec, podUID)
}

func (plugin *secretPlugin) newBuilderInternal(spec *api.Volume, podUID types.UID) (volume.Builder, error) {
	return &secretVolume{spec.Name, podUID, plugin, spec.Source.Secret}, nil
}

func (plugin *secretPlugin) NewCleaner(volName string, podUID types.UID) (volume.Cleaner, error) {
	return plugin.newCleanerInternal(volName, podUID)
}

func (plugin *secretPlugin) newCleanerInternal(volName string, podUID types.UID) (volume.Cleaner, error) {
	return &secretVolume{volName, podUID, plugin, nil}, nil
}

// secretVolume handles retrieving secrets from the API server
// and placing them into the volume on the host.
type secretVolume struct {
	volName string
	podUID  types.UID
	plugin  *secretPlugin
	source  *api.SecretVolumeSource
}

func (sv *secretVolume) SetUp() error {
	// TODO: explore tmpfs for secret volumes
	hostPath := sv.GetPath()
	glog.V(3).Infof("Setting up volume %v for pod %v at %v", sv.volName, sv.podUID, hostPath)
	err := os.MkdirAll(hostPath, 0777)
	if err != nil {
		return err
	}

	kubeClient := sv.plugin.host.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("Cannot setup secret volume %v because kube client is not configured", sv)
	}

	secret, err := kubeClient.Secrets(sv.source.Target.Namespace).Get(sv.source.Target.Name)
	if err != nil {
		glog.Errorf("Couldn't get secret %v/%v", sv.source.Target.Namespace, sv.source.Target.Name)
		return err
	}

	if sv.source.EnvAdaptations != nil {
		envFileContent := makeEnvFileContent(secret, sv.source.EnvAdaptations)
		hostFilePath := path.Join(hostPath, sv.source.EnvAdaptations.Name)

		err := ioutil.WriteFile(hostFilePath, []byte(envFileContent), 0777)
		if err != nil {
			glog.Errorf("Error writing secret data to env file at host path: %v, %v", hostFilePath, err)
		}
	} else {
		for name, data := range secret.Data {
			hostFilePath := path.Join(hostPath, name)
			err := ioutil.WriteFile(hostFilePath, data, 0777)
			if err != nil {
				glog.Errorf("Error writing secret data to host path: %v, %v", hostFilePath, err)
				return err
			}
		}
	}

	return nil
}

func makeEnvFileContent(secret *api.Secret, env *api.SecretEnv) string {
	adaptedKeys := util.NewStringSet()
	content := ""

	for _, adaptation := range env.Adaptations {
		data, ok := secret.Data[adaptation.From]
		if !ok {
			glog.Errorf("Could't apply adaptation of secret data with non-existent key: %v", adaptation.From)
			continue
		}
		// TODO: should there be a validation here for the size of the secret data?

		adaptedKeys.Insert(adaptation.From)
		content += fmt.Sprintf("export %v=\"%v\"\n", adaptation.To, string(data))
	}

	for key, data := range secret.Data {
		if adaptedKeys.Has(key) {
			continue
		}

		content += fmt.Sprintf("export %v=\"%v\"\n", convertKeyToVar(key), string(data))
	}

	return content
}

func convertKeyToVar(key string) string {
	key = strings.ToUpper(key)
	key = strings.Replace(key, "-", "_", -1)
	return strings.Replace(key, ".", "_", -1)
}

func (sv *secretVolume) GetPath() string {
	return sv.plugin.host.GetPodVolumeDir(sv.podUID, volume.EscapePluginName(secretPluginName), sv.volName)
}

func (sv *secretVolume) TearDown() error {
	glog.V(3).Infof("Tearing down volume %v for pod %v at %v", sv.volName, sv.podUID, sv.GetPath())
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
