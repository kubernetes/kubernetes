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
	"os"
	"path"
	"reflect"
	"testing"

	utiltesting "k8s.io/client-go/util/testing"
	api "k8s.io/kubernetes/pkg/api/v1"
)

var (
	vol = &api.Volume{
		Name: testSioVolName,
		VolumeSource: api.VolumeSource{
			ScaleIO: &api.ScaleIOVolumeSource{
				Gateway:          "http://test.scaleio:1111",
				System:           "sio",
				ProtectionDomain: "defaultPD",
				StoragePool:      "defaultSP",
				VolumeName:       "test-vol",
				FSType:           "ext4",
				SecretRef:        &api.LocalObjectReference{Name: "test-secret"},
			},
		},
	}

	config = map[string]string{
		confKey.system:           "sio",
		confKey.gateway:          "http://sio/",
		confKey.volSpecName:      testSioVolName,
		confKey.volumeName:       "sio-vol",
		confKey.secretRef:        "sio-secret",
		confKey.protectionDomain: "defaultPD",
		confKey.storagePool:      "deraultSP",
		confKey.fsType:           "xfs",
		confKey.readOnly:         "true",
	}

	testConfigFile = "conf.dat"
)

func TestUtilMapVolumeSource(t *testing.T) {
	data := make(map[string]string)
	mapVolumeSource(data, vol.VolumeSource.ScaleIO)
	if data[confKey.gateway] != "http://test.scaleio:1111" {
		t.Error("Unexpected gateway value")
	}
	if data[confKey.system] != "sio" {
		t.Error("Unexpected system value")
	}
	if data[confKey.protectionDomain] != "defaultPD" {
		t.Error("Unexpected protection domain value")
	}
	if data[confKey.storagePool] != "defaultSP" {
		t.Error("Unexpected storage pool value")
	}
	if data[confKey.volumeName] != "test-vol" {
		t.Error("Unexpected volume name value")
	}
	if data[confKey.fsType] != "ext4" {
		t.Error("Unexpected fstype value")
	}
	if data[confKey.secretRef] != "test-secret" {
		t.Error("Unexpected secret ref value")
	}
	if data[confKey.sslEnabled] != "false" {
		t.Error("Unexpected sslEnabled value")
	}
	if data[confKey.readOnly] != "false" {
		t.Error("Unexpected readOnly value: ", data[confKey.readOnly])
	}
}

func TestUtilValidateConfigs(t *testing.T) {
	data := map[string]string{
		confKey.secretRef: "sio-secret",
		confKey.system:    "sio",
	}
	if err := validateConfigs(data); err != gatewayNotProvidedErr {
		t.Error("Expecting error for missing gateway, but did not get it")
	}
}

func TestUtilApplyConfigDefaults(t *testing.T) {
	data := map[string]string{
		confKey.system:     "sio",
		confKey.gateway:    "http://sio/",
		confKey.volumeName: "sio-vol",
		confKey.secretRef:  "test-secret",
	}
	applyConfigDefaults(data)

	if data[confKey.gateway] != "http://sio/" {
		t.Error("Unexpected gateway value")
	}
	if data[confKey.system] != "sio" {
		t.Error("Unexpected system value")
	}
	if data[confKey.protectionDomain] != "default" {
		t.Error("Unexpected protection domain value")
	}
	if data[confKey.storagePool] != "default" {
		t.Error("Unexpected storage pool value")
	}
	if data[confKey.volumeName] != "sio-vol" {
		t.Error("Unexpected volume name value")
	}
	if data[confKey.fsType] != "xfs" {
		t.Error("Unexpected fstype value")
	}
	if data[confKey.storageMode] != "ThinProvisioned" {
		t.Error("Unexpected storage mode value")
	}
	if data[confKey.secretRef] != "test-secret" {
		t.Error("Unexpected secret ref value")
	}
	if data[confKey.sslEnabled] != "false" {
		t.Error("Unexpected sslEnabled value")
	}
	if data[confKey.readOnly] != "false" {
		t.Error("Unexpected readOnly value: ", data[confKey.readOnly])
	}
}

func TestUtilDefaultString(t *testing.T) {
	if defaultString("", "foo") != "foo" {
		t.Error("Unexpected value for default value")
	}
}

func TestUtilSaveConfig(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("scaleio-test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	config := path.Join(tmpDir, testConfigFile)
	data := map[string]string{
		confKey.gateway:    "https://test-gateway/",
		confKey.secretRef:  "sio-secret",
		confKey.sslEnabled: "false",
	}
	if err := saveConfig(config, data); err != nil {
		t.Fatal("failed while saving data", err)
	}
	file, err := os.Open(config)
	if err != nil {
		t.Fatal("failed to open conf file: ", file)
	}
	dataRcvd := map[string]string{}
	if err := gob.NewDecoder(file).Decode(&dataRcvd); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(data, dataRcvd) {
		t.Error("we got problem, config data not the same")
	}
}

func TestUtilAttachSecret(t *testing.T) {
	plugMgr, tmpDir := newPluginMgr(t)
	defer os.RemoveAll(tmpDir)

	plug, err := plugMgr.FindPluginByName(sioPluginName)
	if err != nil {
		t.Errorf("Can't find the plugin %v", sioPluginName)
	}
	sioPlug, ok := plug.(*sioPlugin)
	if !ok {
		t.Errorf("Cannot assert plugin to be type sioPlugin")
	}

	data := make(map[string]string)
	for k, v := range config {
		data[k] = v
	}
	if err := attachSecret(sioPlug, "default", data); err != nil {
		t.Errorf("failed to setupConfigData %v", err)
	}
	if data[confKey.username] == "" {
		t.Errorf("failed to merge secret")
	}
}

func TestUtilLoadConfig(t *testing.T) {
	tmpDir, err := utiltesting.MkTmpdir("scaleio-test")
	if err != nil {
		t.Fatalf("can't make a temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	configFile := path.Join(tmpDir, sioConfigFileName)

	if err := saveConfig(configFile, config); err != nil {
		t.Fatal("failed while saving data", err)
	}

	dataRcvd, err := loadConfig(configFile)
	if dataRcvd[confKey.gateway] != config[confKey.gateway] ||
		dataRcvd[confKey.system] != config[confKey.system] {
		t.Fatal("loaded config data not matching saved config data")
	}
}
