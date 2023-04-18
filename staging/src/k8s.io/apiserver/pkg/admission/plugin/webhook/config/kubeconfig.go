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

package config

import (
	"fmt"
	"io"
	"path"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission/v1"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission/v1alpha1"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	utilruntime.Must(webhookadmission.AddToScheme(scheme))
	utilruntime.Must(v1.AddToScheme(scheme))
	utilruntime.Must(v1alpha1.AddToScheme(scheme))
}

// LoadConfig extract the KubeConfigFile from configFile
func LoadConfig(configFile io.Reader) (string, error) {
	var kubeconfigFile string
	if configFile != nil {
		// we have a config so parse it.
		data, err := io.ReadAll(configFile)
		if err != nil {
			return "", err
		}
		decoder := codecs.UniversalDecoder()
		decodedObj, err := runtime.Decode(decoder, data)
		if err != nil {
			return "", err
		}
		config, ok := decodedObj.(*webhookadmission.WebhookAdmission)
		if !ok {
			return "", fmt.Errorf("unexpected type: %T", decodedObj)
		}

		if !path.IsAbs(config.KubeConfigFile) {
			return "", field.Invalid(field.NewPath("kubeConfigFile"), config.KubeConfigFile, "must be an absolute file path")
		}

		kubeconfigFile = config.KubeConfigFile
	}
	return kubeconfigFile, nil
}
