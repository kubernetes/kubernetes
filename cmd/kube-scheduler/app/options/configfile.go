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
	"sigs.k8s.io/yaml"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	kubeschedulerconfigv1beta1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta1"
)

func loadConfigFromFile(file, instanceFile string) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	// If instance configuration exists try to merge into the shared configuration
	// before final object conversion.
	if len(instanceFile) > 0 {
		instanceData, err := ioutil.ReadFile(instanceFile)
		if err != nil {
			return nil, err
		}

		if data, err = mergeInstanceConfiguration(data, instanceData); err != nil {
			return nil, err
		}
	}

	return loadConfig(data)
}

// mergeInstanceConfiguration merge a shared and instance specific configuration.
func mergeInstanceConfiguration(data, instanceData []byte) ([]byte, error) {
	obj := &kubeschedulerconfig.KubeSchedulerConfiguration{}

	// Convert the shared configuration from YAML to JSON.
	jsonData, err := yaml.YAMLToJSON(data)
	if err != nil {
		return nil, err
	}

	// Convert the instance configuration from YAML to JSON.
	jsonInstanceData, err := yaml.YAMLToJSON(instanceData)
	if err != nil {
		return nil, err
	}

	// Merge both configuration and returns the final patch.
	return strategicpatch.StrategicMergePatch(jsonData, jsonInstanceData, obj)
}

func loadConfig(data []byte) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	// The UniversalDecoder runs defaulting and returns the internal type by default.
	obj, gvk, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	if cfgObj, ok := obj.(*kubeschedulerconfig.KubeSchedulerConfiguration); ok {
		return cfgObj, nil
	}
	return nil, fmt.Errorf("couldn't decode as KubeSchedulerConfiguration, got %s: ", gvk)
}

// WriteConfigFile writes the config into the given file name as YAML.
func WriteConfigFile(fileName string, cfg *kubeschedulerconfig.KubeSchedulerConfiguration) error {
	const mediaType = runtime.ContentTypeYAML
	info, ok := runtime.SerializerInfoForMediaType(kubeschedulerscheme.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return fmt.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
	}

	encoder := kubeschedulerscheme.Codecs.EncoderForVersion(info.Serializer, kubeschedulerconfigv1beta1.SchemeGroupVersion)

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
