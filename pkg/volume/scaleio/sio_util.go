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
	"encoding/gob"
	"errors"
	"fmt"
	"os"
	"path"
	"strconv"

	"github.com/golang/glog"

	api "k8s.io/api/core/v1"
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
		volSpecName,
		fsType,
		readOnly,
		username,
		password,
		namespace string
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
		volSpecName:      "volSpecName",
		fsType:           "fsType",
		readOnly:         "readOnly",
		username:         "username",
		password:         "password",
		namespace:        "namespace",
	}
	nsSep       = "%"
	sdcRootPath = "/opt/emc/scaleio/sdc/bin"

	secretNotFoundErr              = errors.New("secret not found")
	configMapNotFoundErr           = errors.New("configMap not found")
	gatewayNotProvidedErr          = errors.New("ScaleIO gateway not provided")
	secretRefNotProvidedErr        = errors.New("secret ref not provided")
	systemNotProvidedErr           = errors.New("ScaleIO system not provided")
	storagePoolNotProvidedErr      = errors.New("ScaleIO storage pool not provided")
	protectionDomainNotProvidedErr = errors.New("ScaleIO protection domain not provided")
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
	if config[confKey.storagePool] == "" {
		return storagePoolNotProvidedErr
	}
	if config[confKey.protectionDomain] == "" {
		return protectionDomainNotProvidedErr
	}

	return nil
}

// applyConfigDefaults apply known defaults to incoming spec for dynamic PVCs.
func applyConfigDefaults(config map[string]string) {
	b, err := strconv.ParseBool(config[confKey.sslEnabled])
	if err != nil {
		glog.Warning(log("failed to parse param sslEnabled, setting it to false"))
		b = false
	}
	config[confKey.sslEnabled] = strconv.FormatBool(b)
	config[confKey.storageMode] = defaultString(config[confKey.storageMode], "ThinProvisioned")
	config[confKey.fsType] = defaultString(config[confKey.fsType], "xfs")
	b, err = strconv.ParseBool(config[confKey.readOnly])
	if err != nil {
		glog.Warning(log("failed to parse param readOnly, setting it to false"))
		b = false
	}
	config[confKey.readOnly] = strconv.FormatBool(b)
}

func defaultString(val, defVal string) string {
	if val == "" {
		return defVal
	}
	return val
}

// loadConfig loads configuration data from a file on disk
func loadConfig(configName string) (map[string]string, error) {
	glog.V(4).Info(log("loading config file %s", configName))
	file, err := os.Open(configName)
	if err != nil {
		glog.Error(log("failed to open config file %s: %v", configName, err))
		return nil, err
	}
	defer file.Close()
	data := map[string]string{}
	if err := gob.NewDecoder(file).Decode(&data); err != nil {
		glog.Error(log("failed to parse config data %s: %v", configName, err))
		return nil, err
	}
	applyConfigDefaults(data)
	if err := validateConfigs(data); err != nil {
		glog.Error(log("failed to load ConfigMap %s: %v", err))
		return nil, err
	}

	return data, nil
}

// saveConfig saves the configuration data to local disk
func saveConfig(configName string, data map[string]string) error {
	glog.V(4).Info(log("saving config file %s", configName))

	dir := path.Dir(configName)
	if _, err := os.Stat(dir); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		glog.V(4).Info(log("creating config dir for config data: %s", dir))
		if err := os.MkdirAll(dir, 0750); err != nil {
			glog.Error(log("failed to create config data dir %v", err))
			return err
		}
	}

	file, err := os.Create(configName)
	if err != nil {
		glog.V(4).Info(log("failed to save config data file %s: %v", configName, err))
		return err
	}
	defer file.Close()
	if err := gob.NewEncoder(file).Encode(data); err != nil {
		glog.Error(log("failed to save config %s: %v", configName, err))
		return err
	}
	glog.V(4).Info(log("config data file saved successfully as %s", configName))
	return nil
}

// attachSecret loads secret object and attaches to configData
func attachSecret(plug *sioPlugin, namespace string, configData map[string]string) error {
	// load secret
	secretRefName := configData[confKey.secretRef]
	kubeClient := plug.host.GetKubeClient()
	secretMap, err := volutil.GetSecretForPV(namespace, secretRefName, sioPluginName, kubeClient)
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

func log(msg string, parts ...interface{}) string {
	return fmt.Sprintf(fmt.Sprintf("scaleio: %s", msg), parts...)
}
