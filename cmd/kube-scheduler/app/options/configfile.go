/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"errors"
	"io/ioutil"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	kubeschedulerconfigv1alpha1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1alpha1"
)

func loadConfigFromFile(file string) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return loadConfig(data)
}

func loadConfig(data []byte) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	configObj := &kubeschedulerconfig.KubeSchedulerConfiguration{}
	if err := runtime.DecodeInto(kubeschedulerscheme.Codecs.UniversalDecoder(), data, configObj); err != nil {
		return nil, err
	}

	return configObj, nil
}

// WriteConfigFile writes the config into the given file name as YAML.
func WriteConfigFile(fileName string, cfg *kubeschedulerconfig.KubeSchedulerConfiguration) error {
	var encoder runtime.Encoder
	mediaTypes := kubeschedulerscheme.Codecs.SupportedMediaTypes()
	for _, info := range mediaTypes {
		if info.MediaType == "application/yaml" {
			encoder = info.Serializer
			break
		}
	}
	if encoder == nil {
		return errors.New("unable to locate yaml encoder")
	}
	encoder = json.NewYAMLSerializer(json.DefaultMetaFactory, kubeschedulerscheme.Scheme, kubeschedulerscheme.Scheme)
	encoder = kubeschedulerscheme.Codecs.EncoderForVersion(encoder, kubeschedulerconfigv1alpha1.SchemeGroupVersion)

	configFile, err := os.Create(fileName)
	if err != nil {
		return err
	}
	defer configFile.Close()
	if err := encoder.Encode(cfg, configFile); err != nil {
		return err
	}

	return nil
}
