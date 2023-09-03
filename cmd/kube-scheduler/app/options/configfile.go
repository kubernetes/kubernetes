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
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	configv1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1"
)

// LoadConfigFromFile loads scheduler config from the specified file path
func LoadConfigFromFile(logger klog.Logger, file string) (*config.KubeSchedulerConfiguration, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return loadConfig(logger, data)
}

func loadConfig(logger klog.Logger, data []byte) (*config.KubeSchedulerConfiguration, error) {
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
	case configv1.SchemeGroupVersion.String():
		encoder = scheme.Codecs.EncoderForVersion(info.Serializer, configv1.SchemeGroupVersion)
	default:
		encoder = scheme.Codecs.EncoderForVersion(info.Serializer, configv1.SchemeGroupVersion)
	}
	if err := encoder.Encode(cfg, buf); err != nil {
		return buf, err
	}
	return buf, nil
}

// LogOrWriteConfig logs the completed component config and writes it into the given file name as YAML, if either is enabled
func LogOrWriteConfig(logger klog.Logger, fileName string, cfg *config.KubeSchedulerConfiguration, completedProfiles []config.KubeSchedulerProfile) error {
	loggerV := logger.V(2)
	if !loggerV.Enabled() && len(fileName) == 0 {
		return nil
	}
	cfg.Profiles = completedProfiles

	buf, err := encodeConfig(cfg)
	if err != nil {
		return err
	}

	if loggerV.Enabled() {
		loggerV.Info("Using component config", "config", buf.String())
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
		logger.Info("Wrote configuration", "file", fileName)
		os.Exit(0)
	}
	return nil
}
