/*
Copyright 2020 The Kubernetes Authors.

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

package config

import (
	"fmt"
	"io"
	"io/ioutil"
	"path"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
)

// LoadStaticAndKubeConfig extracts the StaticConfigFile from configFile
func LoadStaticAndKubeConfig(configFile io.Reader) (string, string, error) {
	var staticConfigFile, kubeConfigFile string
	if configFile != nil {
		// we have a config so parse it.
		data, err := ioutil.ReadAll(configFile)
		if err != nil {
			return "", "", err
		}
		decoder := codecs.UniversalDecoder()
		decodedObj, err := runtime.Decode(decoder, data)
		if err != nil {
			return "", "", err
		}
		config, ok := decodedObj.(*webhookadmission.WebhookAdmission)
		if !ok {
			return "", "", fmt.Errorf("unexpected type: %T", decodedObj)
		}

		if config.KubeConfigFile != "" && !path.IsAbs(config.KubeConfigFile) {
			return "", "", field.Invalid(field.NewPath("kubeConfigFile"), config.KubeConfigFile, "must be an absolute file path")
		}

		if config.StaticConfigFile != "" && !path.IsAbs(config.StaticConfigFile) {
			return "", "", field.Invalid(field.NewPath("staticConfigFile"), config.StaticConfigFile, "must be an absolute file path")
		}

		kubeConfigFile = config.KubeConfigFile
		staticConfigFile = config.StaticConfigFile
	}
	return kubeConfigFile, staticConfigFile, nil
}
