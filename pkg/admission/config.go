/*
Copyright 2016 The Kubernetes Authors.

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

package admission

import (
	"fmt"
	"io/ioutil"

	"github.com/ghodss/yaml"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/runtime"
)

// ReadAdmissionConfiguration reads the admission configuration at the specified path.
// It returns the loaded admission configuration if the input file aligns with the required syntax.
// If it does not align with the provided syntax, it returns a default configuration.
// It does this to preserve backward compatibility when admission control files were opaque.
// It returns an error if the file did not exist.
func ReadAdmissionConfiguration(configFilePath string) (*componentconfig.AdmissionConfiguration, error) {
	externalConfig := &componentconfigv1alpha1.AdmissionConfiguration{}
	internalConfig := &componentconfig.AdmissionConfiguration{}

	// a file was provided, so we just read it.
	if configFilePath != "" {
		data, err := ioutil.ReadFile(configFilePath)
		if err != nil {
			return internalConfig, fmt.Errorf("unable to read admission control configuration from %q [%v]", configFilePath, err)
		}
		// if we are unable to decode, we assume this is legacy input that is unable to do multiple opaque configurations.
		if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), data, externalConfig); err != nil {
			return internalConfig, nil
		}
	}
	// apply defaulting
	api.Scheme.Default(externalConfig)

	// convert to internal form
	if err := api.Scheme.Convert(externalConfig, internalConfig, nil); err != nil {
		return internalConfig, err
	}
	return internalConfig, nil
}

// GetAdmissionPluginConfigurationFileNameFor returns a file that holds the admission plugin configuration.
func GetAdmissionPluginConfigurationFileNameFor(pluginCfg componentconfig.AdmissionPluginConfiguration) (string, error) {
	// if there is nothing nested in the object, we return the named location
	obj := pluginCfg.Configuration
	if obj == nil {
		return pluginCfg.Location, nil
	}

	// to maintain compatibility, we encode the nested configuration to a temp file
	configFile, err := ioutil.TempFile("", "admission-plugin-config")
	if err != nil {
		return "", err
	}
	if err = configFile.Close(); err != nil {
		return "", err
	}

	content, err := writeYAML(obj)
	if err != nil {
		return "", err
	}
	if err = ioutil.WriteFile(configFile.Name(), content, 0644); err != nil {
		return "", err
	}
	return configFile.Name(), nil
}

// GetAdmissionPluginConfigurationFileName takes the admission configuration and returns the config file
// for the specified plugin.  If no specific configuration file is provided, we provide the default file
// to preserve backward compatibility.
func GetAdmissionPluginConfigurationFileName(cfg *componentconfig.AdmissionConfiguration, pluginName string, defaultConfigFilePath string) (string, error) {
	for _, pluginCfg := range cfg.PluginConfigurations {
		if pluginName != pluginCfg.Name {
			continue
		}
		configFilePath, err := GetAdmissionPluginConfigurationFileNameFor(pluginCfg)
		if err != nil {
			return "", err
		}
		return configFilePath, nil
	}
	return defaultConfigFilePath, nil
}

// writeYAML writes the specified object to a byte array as yaml.
func writeYAML(obj runtime.Object) ([]byte, error) {
	json, err := runtime.Encode(api.Codecs.LegacyCodec(), obj)
	if err != nil {
		return nil, err
	}

	content, err := yaml.JSONToYAML(json)
	if err != nil {
		return nil, err
	}
	return content, err
}
