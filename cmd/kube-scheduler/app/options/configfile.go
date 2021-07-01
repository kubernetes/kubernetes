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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	configv1beta1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta1"
	configv1beta2 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1beta2"
)

func loadConfigFromFile(file string) (*config.KubeSchedulerConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return loadConfig(data)
}

func loadConfig(data []byte) (*config.KubeSchedulerConfiguration, error) {
	// The UniversalDecoder runs defaulting and returns the internal type by default.
	obj, gvk, err := scheme.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		return nil, err
	}
	if cfgObj, ok := obj.(*config.KubeSchedulerConfiguration); ok {
		// We don't set this field in pkg/scheduler/apis/config/{version}/conversion.go
		// because the field will be cleared later by API machinery during
		// conversion. See KubeSchedulerConfiguration internal type definition for
		// more details.
		cfgObj.TypeMeta.APIVersion = gvk.GroupVersion().String()
		return cfgObj, nil
	}
	return nil, fmt.Errorf("couldn't decode as KubeSchedulerConfiguration, got %s: ", gvk)
}

func encodeConfig(cfg *config.KubeSchedulerConfiguration) (*bytes.Buffer, error) {
	buf := new(bytes.Buffer)
	const mediaType = runtime.ContentTypeYAML
	info, ok := runtime.SerializerInfoForMediaType(scheme.Codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return buf, fmt.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
	}

	var encoder runtime.Encoder
	switch cfg.TypeMeta.APIVersion {
	case configv1beta1.SchemeGroupVersion.String():
		encoder = scheme.Codecs.EncoderForVersion(info.Serializer, configv1beta1.SchemeGroupVersion)
	case configv1beta2.SchemeGroupVersion.String():
		encoder = scheme.Codecs.EncoderForVersion(info.Serializer, configv1beta2.SchemeGroupVersion)
	default:
		encoder = scheme.Codecs.EncoderForVersion(info.Serializer, configv1beta2.SchemeGroupVersion)
	}
	if err := encoder.Encode(cfg, buf); err != nil {
		return buf, err
	}
	return buf, nil
}

// LogOrWriteConfig logs the completed component config and writes it into the given file name as YAML, if either is enabled
func LogOrWriteConfig(fileName string, cfg *config.KubeSchedulerConfiguration, completedProfiles []config.KubeSchedulerProfile) error {
	if !(klog.V(2).Enabled() || len(fileName) > 0) {
		return nil
	}
	cfg.Profiles = completedProfiles

	if len(fileName) > 0 {
		// Since the default component config lists all the default plugins as enabled
		// without disabling them, we must explicitly disable all the plugins for each
		// extension point using the "*" expression to ensure that the generated config
		// file is usable
		disabledPlugins := []config.Plugin{{Name: "*"}}
		for i := range cfg.Profiles {
			if cfg.Profiles[i].Plugins == nil {
				continue
			}
			cfg.Profiles[i].Plugins.QueueSort.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.PreFilter.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.Filter.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.PostFilter.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.PreScore.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.Score.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.Reserve.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.Permit.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.PreBind.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.Bind.Disabled = disabledPlugins
			cfg.Profiles[i].Plugins.PostBind.Disabled = disabledPlugins
		}
	}

	buf, err := encodeConfig(cfg)
	if err != nil {
		return err
	}

	if klog.V(2).Enabled() {
		klog.InfoS("Using component config", "config", buf.String())
	}

	if len(fileName) > 0 {
		configFile, err := os.Create(fileName)
		if err != nil {
			return err
		}
		defer configFile.Close()
		if _, err := io.Copy(configFile, buf); err != nil {
			return err
		}
		klog.InfoS("Wrote configuration", "file", fileName)
		os.Exit(0)
	}
	return nil
}
