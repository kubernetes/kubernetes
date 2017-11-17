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

package admission

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"

	"bytes"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/apis/apiserver"
	apiserverv1alpha1 "k8s.io/apiserver/pkg/apis/apiserver/v1alpha1"
)

func makeAbs(path, base string) (string, error) {
	if filepath.IsAbs(path) {
		return path, nil
	}
	if len(base) == 0 || base == "." {
		cwd, err := os.Getwd()
		if err != nil {
			return "", err
		}
		base = cwd
	}
	return filepath.Join(base, path), nil
}

// ReadAdmissionConfiguration reads the admission configuration at the specified path.
// It returns the loaded admission configuration if the input file aligns with the required syntax.
// If it does not align with the provided syntax, it returns a default configuration for the enumerated
// set of pluginNames whose config location references the specified configFilePath.
// It does this to preserve backward compatibility when admission control files were opaque.
// It returns an error if the file did not exist.
func ReadAdmissionConfiguration(pluginNames []string, configFilePath string, configScheme *runtime.Scheme) (ConfigProvider, error) {
	if configFilePath == "" {
		return configProvider{config: &apiserver.AdmissionConfiguration{}}, nil
	}
	// a file was provided, so we just read it.
	data, err := ioutil.ReadFile(configFilePath)
	if err != nil {
		return nil, fmt.Errorf("unable to read admission control configuration from %q [%v]", configFilePath, err)
	}
	codecs := serializer.NewCodecFactory(configScheme)
	decoder := codecs.UniversalDecoder()
	decodedObj, err := runtime.Decode(decoder, data)
	// we were able to decode the file successfully
	if err == nil {
		decodedConfig, ok := decodedObj.(*apiserver.AdmissionConfiguration)
		if !ok {
			return nil, fmt.Errorf("unexpected type: %T", decodedObj)
		}
		baseDir := path.Dir(configFilePath)
		for i := range decodedConfig.Plugins {
			if decodedConfig.Plugins[i].Path == "" {
				continue
			}
			// we update relative file paths to absolute paths
			absPath, err := makeAbs(decodedConfig.Plugins[i].Path, baseDir)
			if err != nil {
				return nil, err
			}
			decodedConfig.Plugins[i].Path = absPath
		}
		return configProvider{
			config: decodedConfig,
			scheme: configScheme,
		}, nil
	}
	// we got an error where the decode wasn't related to a missing type
	if !(runtime.IsMissingVersion(err) || runtime.IsMissingKind(err) || runtime.IsNotRegisteredError(err)) {
		return nil, err
	}
	// convert the legacy format to the new admission control format
	// in order to preserve backwards compatibility, we set plugins that
	// previously read input from a non-versioned file configuration to the
	// current input file.
	legacyPluginsWithUnversionedConfig := sets.NewString("ImagePolicyWebhook", "PodNodeSelector")
	externalConfig := &apiserverv1alpha1.AdmissionConfiguration{}
	for _, pluginName := range pluginNames {
		if legacyPluginsWithUnversionedConfig.Has(pluginName) {
			externalConfig.Plugins = append(externalConfig.Plugins,
				apiserverv1alpha1.AdmissionPluginConfiguration{
					Name: pluginName,
					Path: configFilePath})
		}
	}
	configScheme.Default(externalConfig)
	internalConfig := &apiserver.AdmissionConfiguration{}
	if err := configScheme.Convert(externalConfig, internalConfig, nil); err != nil {
		return nil, err
	}
	return configProvider{
		config: internalConfig,
		scheme: configScheme,
	}, nil
}

type configProvider struct {
	config *apiserver.AdmissionConfiguration
	scheme *runtime.Scheme
}

// GetAdmissionPluginConfigurationFor returns a reader that holds the admission plugin configuration.
func GetAdmissionPluginConfigurationFor(pluginCfg apiserver.AdmissionPluginConfiguration, scheme *runtime.Scheme) (io.Reader, error) {
	// if there is nothing nested in the object, we return the named location
	obj := pluginCfg.Configuration
	if obj != nil {
		// serialize the configuration and build a reader for it
		content, err := writeYAML(obj, scheme)
		if err != nil {
			return nil, err
		}
		return bytes.NewBuffer(content), nil
	}
	// there is nothing nested, so we delegate to path
	if pluginCfg.Path != "" {
		content, err := ioutil.ReadFile(pluginCfg.Path)
		if err != nil {
			glog.Fatalf("Couldn't open admission plugin configuration %s: %#v", pluginCfg.Path, err)
			return nil, err
		}
		return bytes.NewBuffer(content), nil
	}
	// there is no special config at all
	return nil, nil
}

// GetAdmissionPluginConfiguration takes the admission configuration and returns a reader
// for the specified plugin.  If no specific configuration is present, we return a nil reader.
func (p configProvider) ConfigFor(pluginName string) (io.Reader, error) {
	// there is no config, so there is no potential config
	if p.config == nil {
		return nil, nil
	}
	// look for matching plugin and get configuration
	for _, pluginCfg := range p.config.Plugins {
		if pluginName != pluginCfg.Name {
			continue
		}
		pluginConfig, err := GetAdmissionPluginConfigurationFor(pluginCfg, p.scheme)
		if err != nil {
			return nil, err
		}
		return pluginConfig, nil
	}
	// there is no registered config that matches on plugin name.
	return nil, nil
}

// writeYAML writes the specified object to a byte array as yaml.
func writeYAML(obj runtime.Object, scheme *runtime.Scheme) ([]byte, error) {
	gvks, _, err := scheme.ObjectKinds(obj)
	if err != nil {
		return nil, err
	}
	gvs := []schema.GroupVersion{}
	for _, gvk := range gvks {
		gvs = append(gvs, gvk.GroupVersion())
	}
	codecs := serializer.NewCodecFactory(scheme)
	json, err := runtime.Encode(codecs.LegacyCodec(gvs...), obj)
	if err != nil {
		return nil, err
	}

	content, err := yaml.JSONToYAML(json)
	if err != nil {
		return nil, err
	}
	return content, err
}
