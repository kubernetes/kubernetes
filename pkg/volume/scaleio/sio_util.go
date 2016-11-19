/*
Copyright 2017 The Kubernetes Authors.

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

	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	api "k8s.io/kubernetes/pkg/api/v1"
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

	secretNotFoundErr       = errors.New("secret not found")
	configMapNotFoundErr    = errors.New("configMap not found")
	gatewayNotProvidedErr   = errors.New("gateway not provided")
	secretRefNotProvidedErr = errors.New("secret ref not provided")
	systemNotProvidedErr    = errors.New("secret not provided")
)

// mapScaleIOVolumeSource maps attributes from a ScaleIOVolumeSource to config
func mapVolumeSource(config map[string]string, source *api.ScaleIOVolumeSource) {
	config[confKey.gateway] = source.Gateway
	config[confKey.secretRef] = func() string {
		if source.SecretRef != nil {
			return string(source.SecretRef.Name)
		}
		return ""
	}()
	config[confKey.system] = source.System
	config[confKey.volumeName] = source.VolumeName
	config[confKey.sslEnabled] = strconv.FormatBool(source.SSLEnabled)
	config[confKey.protectionDomain] = source.ProtectionDomain
	config[confKey.storagePool] = source.StoragePool
	config[confKey.storageMode] = source.StorageMode
	config[confKey.sdcRootPath] = source.SDCRootPath
	config[confKey.fsType] = source.FSType
	config[confKey.readOnly] = strconv.FormatBool(source.ReadOnly)

	//optionals
	applyConfigDefaults(config)
}

func validateConfigs(config map[string]string) error {
	if config[confKey.gateway] == "" {
		return gatewayNotProvidedErr
	}
	if config[confKey.secretRef] == "" {
		return secretRefNotProvidedErr
	}
	if config[confKey.system] == "" {
		return systemNotProvidedErr
	}

	return nil
}

//TODO remove, default check is done by k8s type checks
func applyConfigDefaults(config map[string]string) {
	b, _ := strconv.ParseBool(config[confKey.sslEnabled])
	config[confKey.sslEnabled] = strconv.FormatBool(b)
	config[confKey.protectionDomain] = defaultString(config[confKey.protectionDomain], "default")
	config[confKey.storagePool] = defaultString(config[confKey.storagePool], "default")
	config[confKey.storageMode] = defaultString(config[confKey.storageMode], "ThinProvisioned")
	config[confKey.sdcRootPath] = defaultString(config[confKey.sdcRootPath], "/opt/emc/scaleio/sdc/bin/")
	config[confKey.fsType] = defaultString(config[confKey.fsType], "xfs")
	b, _ = strconv.ParseBool(config[confKey.readOnly])
	config[confKey.readOnly] = strconv.FormatBool(b)
}

func defaultString(val, defVal string) string {
	if val == "" {
		return defVal
	}
	return val
}

// saveConfigMap creates/stores a configMap from the config data
func saveConfigMap(plug *sioPlugin, data map[string]string) error {
	configName := data[confKey.volumeName]
	glog.V(4).Info(log("saving confing data as %v configMap", configName))

	kubeClient := plug.host.GetKubeClient()
	var configMap *api.ConfigMap
	configMap, err := kubeClient.Core().ConfigMaps(sioDefaultNamespace).Get(configName, meta.GetOptions{})
	if err != nil {
		glog.Warning(log("failed to get existing configMap [%s] will attempt to create new one", err))
		configMap = &api.ConfigMap{
			ObjectMeta: meta.ObjectMeta{
				Name:      configName,
				Namespace: sioDefaultNamespace,
			},
			Data: data,
		}
		configMap, err = kubeClient.Core().ConfigMaps(sioDefaultNamespace).Create(configMap)
		glog.V(4).Info(log("creating new configMap  %s/%s", sioDefaultNamespace, configName))
	} else {
		configMap.Data = data
		configMap, err = kubeClient.Core().ConfigMaps(sioDefaultNamespace).Update(configMap)
		glog.V(4).Info(log("updating existing configMap %s/%s", sioDefaultNamespace, configName))
	}
	if err != nil {
		glog.Errorf(log("failed to save configMap: %v", err))
		return err
	}
	glog.V(4).Info(log("config map %s saved successfully", configMap.Name))
	return nil
}

// setupConfigData setups the needed config data as internal configMap.
func setupConfigData(plug *sioPlugin, configData map[string]string) error {
	glog.V(4).Info(log("setting up config data"))
	// ensure needed config is provided
	if err := validateConfigs(configData); err != nil {
		glog.Errorf(log("config data setup failed: %s", err))
		return err
	}

	// load secret
	secretRefName := configData[confKey.secretRef]
	glog.V(4).Info(log("loading secret %s in config data", secretRefName))
	kubeClient := plug.host.GetKubeClient()
	secretMap, err := volutil.GetSecretForPV(sioDefaultNamespace, secretRefName, sioPluginName, kubeClient)
	if err != nil {
		glog.Error(log("failed to get secret: %v", err))
		return secretNotFoundErr
	}
	// merge secret data
	for key, val := range secretMap {
		configData[key] = val
	}

	return nil
}

// loadConfigData retrieves configuration data from storage
func loadConfigMap(plug *sioPlugin, configName string) (map[string]string, error) {
	glog.V(4).Info(log("loading configMap %s/%s", sioDefaultNamespace, configName))
	kubeClient := plug.host.GetKubeClient()
	confMap, err := kubeClient.Core().ConfigMaps(sioDefaultNamespace).Get(configName, meta.GetOptions{})
	if err != nil {
		glog.Errorf(log("failed to load ConfigMap %s: %v", configName, err))
		return nil, configMapNotFoundErr
	}

	configs := confMap.Data
	applyConfigDefaults(configs)
	if err := validateConfigs(configs); err != nil {
		glog.Error(log("failed to load ConfigMap %s: %v", err))
		return nil, err
	}

	return configs, nil
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
		glog.Errorf("scaleio: failed to verify device path: %v", err)
		return "", err
	} else if pathExists {
		return path, nil
	}
	return "", nil
}

func log(msg string, parts ...interface{}) string {
	return fmt.Sprintf(fmt.Sprintf("scaleio: %s", msg), parts...)
}
