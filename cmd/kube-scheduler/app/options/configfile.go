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
	"fmt"
	"io/ioutil"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
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
	const mediaType = runtime.ContentTypeYAML
	info, ok := runtime.SerializerInfoForMediaType(kubeschedulerscheme.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return fmt.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
	}

	encoder := kubeschedulerscheme.Codecs.EncoderForVersion(info.Serializer, kubeschedulerconfigv1alpha1.SchemeGroupVersion)

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
