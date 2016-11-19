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

package scaleio

import (
	"errors"
	"fmt"
	"path"
	"strconv"

	"github.com/golang/glog"

	api "k8s.io/kubernetes/pkg/api/v1"
	meta "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

var (
	confKey = struct {
		gateway,
		sslEnabled,
		secretRef,
		system,
		protectionDomain,
		storagePool,
		storageMode,
		sdcRootPath,
		volumeName,
		fsType,
		readOnly,
		username,
		password string
	}{
		gateway:          "gateway",
		sslEnabled:       "sslEnabled",
		secretRef:        "secretRef",
		system:           "system",
		protectionDomain: "protectionDomain",
		storagePool:      "storagePool",
		storageMode:      "storageMode",
		sdcRootPath:      "sdcRootPath",
		volumeName:       "volumeName",
		fsType:           "fsType",
		readOnly:         "readOnly",
		username:         "username",
		password:         "password",
	}
)

// mapScaleIOVolumeSource maps attributes from a ScaleIOVolumeSource to config
func mapVolumeSource(config map[string]string, source *api.ScaleIOVolumeSource) {
	config[confKey.gateway] = source.Gateway
	config[confKey.secretRef] = string(source.SecretRef.Name)
	config[confKey.system] = source.System
	config[confKey.volumeName] = source.VolumeName

	//optionals
	assertConfigDefaults(config)
}

func validateConfigs(config map[string]string) error {
	if config[confKey.gateway] == "" {
		return errors.New("missing gateway")
	}
	if config[confKey.secretRef] == "" {
		return errors.New("missing secret ref")
	}
	if config[confKey.system] == "" {
		return errors.New("missing system")
	}

	return nil
}

func assertConfigDefaults(config map[string]string) {
	b, _ := strconv.ParseBool(config[confKey.sslEnabled])
	config[confKey.sslEnabled] = strconv.FormatBool(b)
	config[confKey.protectionDomain] = defaultString(config[confKey.protectionDomain], "default")
	config[confKey.storagePool] = defaultString(config[confKey.storagePool], "default")
	config[confKey.storageMode] = defaultString(config[confKey.storageMode], "ThinProvisioned")
	config[confKey.sdcRootPath] = defaultString(config[confKey.sdcRootPath], "/opt/emc/scaleio/sdc/bin/")
	config[confKey.fsType] = defaultString(config[confKey.fsType], "ext4")
	b, _ = strconv.ParseBool(config[confKey.readOnly])
	config[confKey.readOnly] = strconv.FormatBool(b)
}

func defaultString(val, defVal string) string {
	if val == "" {
		return defVal
	}
	return val
}

func getSecret(plug *sioPlugin, secretRefName string) (*api.Secret, error) {
	if secretRefName == "" {
		return nil, errors.New("no SecretRef found")
	}

	kubeClient := plug.host.GetKubeClient()
	secret, err := kubeClient.Core().Secrets(sioDefaultNamespace).Get(secretRefName, meta.GetOptions{})
	if err != nil {
		return nil, err
	}
	return secret, nil
}

func mapSecret(config map[string]string, secret *api.Secret) {
	for key, val := range secret.Data {
		config[key] = string(val) // save base64 bytes as string
	}
}

// saveConfigMap creates/stores a new configMap from data
func saveConfigMap(plug *sioPlugin, data map[string]string) error {
	configName := data[confKey.volumeName]

	kubeClient := plug.host.GetKubeClient()
	var configMap *api.ConfigMap
	configMap, err := kubeClient.Core().ConfigMaps(sioDefaultNamespace).Get(configName, meta.GetOptions{})
	if err != nil {
		glog.Warningf("sio: failed to get ConfigMap %s (will attempt to create new): %v", configName, err)
		configMap = &api.ConfigMap{
			ObjectMeta: api.ObjectMeta{
				Name:      configName,
				Namespace: sioDefaultNamespace,
			},
			Data: data,
		}
		configMap, err = kubeClient.Core().ConfigMaps(sioDefaultNamespace).Create(configMap)
	} else {
		configMap.Data = data
		configMap, err = kubeClient.Core().ConfigMaps(sioDefaultNamespace).Update(configMap)
	}

	return err
}

// setupConfigData configures and saves configuration data as internal configMap.
func setupConfigData(
	plug *sioPlugin,
	sioSource *api.ScaleIOVolumeSource,
) (map[string]string, error) {

	configData := make(map[string]string)
	mapVolumeSource(configData, sioSource)

	secret, err := getSecret(plug, sioSource.SecretRef.Name)
	if err != nil {
		glog.Errorf("sio: failed to get secret: %v", err)
		return nil, err
	}
	mapSecret(configData, secret)

	// ensure secret is provided
	validateConfigs(configData)

	// persist/update configMap
	err = saveConfigMap(plug, configData)
	if err != nil {
		glog.Errorf("sio: unable to save/update configMap: %v", err)
		return nil, err
	}
	return configData, nil
}

// getVolumeSourceFromSpec safely extracts ScaleIOVolumeSource from spec
func getVolumeSourceFromSpec(spec *volume.Spec) (*api.ScaleIOVolumeSource, error) {
	if spec.Volume != nil && spec.Volume.ScaleIO != nil {
		return spec.Volume.ScaleIO, nil
	}
	if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ScaleIO != nil {
		return spec.PersistentVolume.Spec.ScaleIO, nil
	}

	return nil, fmt.Errorf("ScaleIO not defined in spec")
}

// getNodeVolumeDir returns the volume mount location for the host
func getNodeVolumeDir(p *sioPlugin, service, volName string) string {
	return path.Join(
		p.host.GetPluginDir(sioPluginName),
		"mounts",
		fmt.Sprintf("%s/%s", service, volName))
}

func verifyDevicePath(path string) (string, error) {
	if pathExists, err := volutil.PathExists(path); err != nil {
		glog.Errorf("sio: failed to verify device path: %v", err)
		return "", err
	} else if pathExists {
		return path, nil
	}
	return "", nil
}
