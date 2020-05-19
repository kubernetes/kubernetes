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

	"k8s.io/klog/v2"

	api "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
	volutil "k8s.io/kubernetes/pkg/volume/util"
)

type volSourceAttribs struct {
	volName,
	fsType string
	readOnly bool
}

var (
	confKey = struct {
		gateway,
		sslEnabled,
		secretName,
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
		secretNamespace,
		sdcGUID string
	}{
		gateway:          "gateway",
		sslEnabled:       "sslEnabled",
		secretName:       "secretRef",
		secretNamespace:  "secretNamespace",
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
		sdcGUID:          "sdcGUID",
	}
	sdcGUIDLabelName = "scaleio.sdcGUID"
	sdcRootPath      = "/opt/emc/scaleio/sdc/bin"

	errSecretNotFound              = errors.New("secret not found")
	errGatewayNotProvided          = errors.New("ScaleIO gateway not provided")
	errSecretRefNotProvided        = errors.New("secret ref not provided")
	errSystemNotProvided           = errors.New("ScaleIO system not provided")
	errStoragePoolNotProvided      = errors.New("ScaleIO storage pool not provided")
	errProtectionDomainNotProvided = errors.New("ScaleIO protection domain not provided")
)

// mapVolumeSpec maps attributes from either ScaleIOVolumeSource  or ScaleIOPersistentVolumeSource to config
func mapVolumeSpec(config map[string]string, spec *volume.Spec) {

	if source, err := getScaleIOPersistentVolumeSourceFromSpec(spec); err == nil {
		config[confKey.gateway] = source.Gateway
		config[confKey.system] = source.System
		config[confKey.volumeName] = source.VolumeName
		config[confKey.sslEnabled] = strconv.FormatBool(source.SSLEnabled)
		config[confKey.protectionDomain] = source.ProtectionDomain
		config[confKey.storagePool] = source.StoragePool
		config[confKey.storageMode] = source.StorageMode
		config[confKey.fsType] = source.FSType
		config[confKey.readOnly] = strconv.FormatBool(source.ReadOnly)
	}

	if source, err := getScaleIOVolumeSourceFromSpec(spec); err == nil {
		config[confKey.gateway] = source.Gateway
		config[confKey.system] = source.System
		config[confKey.volumeName] = source.VolumeName
		config[confKey.sslEnabled] = strconv.FormatBool(source.SSLEnabled)
		config[confKey.protectionDomain] = source.ProtectionDomain
		config[confKey.storagePool] = source.StoragePool
		config[confKey.storageMode] = source.StorageMode
		config[confKey.fsType] = source.FSType
		config[confKey.readOnly] = strconv.FormatBool(source.ReadOnly)
	}

	//optionals
	applyConfigDefaults(config)
}

func validateConfigs(config map[string]string) error {
	if config[confKey.gateway] == "" {
		return errGatewayNotProvided
	}
	if config[confKey.secretName] == "" {
		return errSecretRefNotProvided
	}
	if config[confKey.system] == "" {
		return errSystemNotProvided
	}
	if config[confKey.storagePool] == "" {
		return errStoragePoolNotProvided
	}
	if config[confKey.protectionDomain] == "" {
		return errProtectionDomainNotProvided
	}

	return nil
}

// applyConfigDefaults apply known defaults to incoming spec for dynamic PVCs.
func applyConfigDefaults(config map[string]string) {
	b, err := strconv.ParseBool(config[confKey.sslEnabled])
	if err != nil {
		klog.Warning(log("failed to parse param sslEnabled, setting it to false"))
		b = false
	}
	config[confKey.sslEnabled] = strconv.FormatBool(b)
	config[confKey.storageMode] = defaultString(config[confKey.storageMode], "ThinProvisioned")
	config[confKey.fsType] = defaultString(config[confKey.fsType], "xfs")
	b, err = strconv.ParseBool(config[confKey.readOnly])
	if err != nil {
		klog.Warning(log("failed to parse param readOnly, setting it to false"))
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
	klog.V(4).Info(log("loading config file %s", configName))
	file, err := os.Open(configName)
	if err != nil {
		klog.Error(log("failed to open config file %s: %v", configName, err))
		return nil, err
	}
	defer file.Close()
	data := map[string]string{}
	if err := gob.NewDecoder(file).Decode(&data); err != nil {
		klog.Error(log("failed to parse config data %s: %v", configName, err))
		return nil, err
	}
	applyConfigDefaults(data)
	if err := validateConfigs(data); err != nil {
		klog.Error(log("failed to load ConfigMap %s: %v", err))
		return nil, err
	}

	return data, nil
}

// saveConfig saves the configuration data to local disk
func saveConfig(configName string, data map[string]string) error {
	klog.V(4).Info(log("saving config file %s", configName))

	dir := path.Dir(configName)
	if _, err := os.Stat(dir); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		klog.V(4).Info(log("creating config dir for config data: %s", dir))
		if err := os.MkdirAll(dir, 0750); err != nil {
			klog.Error(log("failed to create config data dir %v", err))
			return err
		}
	}

	file, err := os.Create(configName)
	if err != nil {
		klog.V(4).Info(log("failed to save config data file %s: %v", configName, err))
		return err
	}
	defer file.Close()
	if err := gob.NewEncoder(file).Encode(data); err != nil {
		klog.Error(log("failed to save config %s: %v", configName, err))
		return err
	}
	klog.V(4).Info(log("config data file saved successfully as %s", configName))
	return nil
}

// attachSecret loads secret object and attaches to configData
func attachSecret(plug *sioPlugin, namespace string, configData map[string]string) error {
	// load secret
	secretRefName := configData[confKey.secretName]
	kubeClient := plug.host.GetKubeClient()
	secretMap, err := volutil.GetSecretForPV(namespace, secretRefName, sioPluginName, kubeClient)
	if err != nil {
		klog.Error(log("failed to get secret: %v", err))
		return errSecretNotFound
	}
	// merge secret data
	for key, val := range secretMap {
		configData[key] = val
	}

	return nil
}

// attachSdcGUID injects the sdc guid node label value into config
func attachSdcGUID(plug *sioPlugin, conf map[string]string) error {
	guid, err := getSdcGUIDLabel(plug)
	if err != nil {
		return err
	}
	conf[confKey.sdcGUID] = guid
	return nil
}

// getSdcGUIDLabel fetches the scaleio.sdcGuid node label
// associated with the node executing this code.
func getSdcGUIDLabel(plug *sioPlugin) (string, error) {
	nodeLabels, err := plug.host.GetNodeLabels()
	if err != nil {
		return "", err
	}
	label, ok := nodeLabels[sdcGUIDLabelName]
	if !ok {
		klog.V(4).Info(log("node label %s not found", sdcGUIDLabelName))
		return "", nil
	}

	klog.V(4).Info(log("found node label %s=%s", sdcGUIDLabelName, label))
	return label, nil
}

// getVolumeSourceFromSpec safely extracts ScaleIOVolumeSource or ScaleIOPersistentVolumeSource from spec
func getVolumeSourceFromSpec(spec *volume.Spec) (interface{}, error) {
	if spec.Volume != nil && spec.Volume.ScaleIO != nil {
		return spec.Volume.ScaleIO, nil
	}
	if spec.PersistentVolume != nil &&
		spec.PersistentVolume.Spec.ScaleIO != nil {
		return spec.PersistentVolume.Spec.ScaleIO, nil
	}

	return nil, fmt.Errorf("ScaleIO not defined in spec")
}

func getVolumeSourceAttribs(spec *volume.Spec) (*volSourceAttribs, error) {
	attribs := new(volSourceAttribs)
	if pvSource, err := getScaleIOPersistentVolumeSourceFromSpec(spec); err == nil {
		attribs.volName = pvSource.VolumeName
		attribs.fsType = pvSource.FSType
		attribs.readOnly = pvSource.ReadOnly
	} else if pSource, err := getScaleIOVolumeSourceFromSpec(spec); err == nil {
		attribs.volName = pSource.VolumeName
		attribs.fsType = pSource.FSType
		attribs.readOnly = pSource.ReadOnly
	} else {
		msg := log("failed to get ScaleIOVolumeSource or ScaleIOPersistentVolumeSource from spec")
		klog.Error(msg)
		return nil, errors.New(msg)
	}
	return attribs, nil
}

func getScaleIOPersistentVolumeSourceFromSpec(spec *volume.Spec) (*api.ScaleIOPersistentVolumeSource, error) {
	source, err := getVolumeSourceFromSpec(spec)
	if err != nil {
		return nil, err
	}
	if val, ok := source.(*api.ScaleIOPersistentVolumeSource); ok {
		return val, nil
	}
	return nil, fmt.Errorf("spec is not a valid ScaleIOPersistentVolume type")
}

func getScaleIOVolumeSourceFromSpec(spec *volume.Spec) (*api.ScaleIOVolumeSource, error) {
	source, err := getVolumeSourceFromSpec(spec)
	if err != nil {
		return nil, err
	}
	if val, ok := source.(*api.ScaleIOVolumeSource); ok {
		return val, nil
	}
	return nil, fmt.Errorf("spec is not a valid ScaleIOVolume type")
}

func getSecretAndNamespaceFromSpec(spec *volume.Spec, pod *api.Pod) (secretName string, secretNS string, err error) {
	if source, err := getScaleIOVolumeSourceFromSpec(spec); err == nil {
		secretName = source.SecretRef.Name
		if pod != nil {
			secretNS = pod.Namespace
		}
	} else if source, err := getScaleIOPersistentVolumeSourceFromSpec(spec); err == nil {
		if source.SecretRef != nil {
			secretName = source.SecretRef.Name
			secretNS = source.SecretRef.Namespace
			if secretNS == "" && pod != nil {
				secretNS = pod.Namespace
			}
		}
	} else {
		return "", "", errors.New("failed to get ScaleIOVolumeSource or ScaleIOPersistentVolumeSource")
	}
	return secretName, secretNS, nil
}

func log(msg string, parts ...interface{}) string {
	return fmt.Sprintf(fmt.Sprintf("scaleio: %s", msg), parts...)
}
