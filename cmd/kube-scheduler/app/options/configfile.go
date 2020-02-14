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

	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-base/codec"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerscheme "k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	kubeschedulerconfigv1alpha1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1alpha1"
	kubeschedulerconfigv1alpha2 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1alpha2"
)

func loadConfigFromFile(file string) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}

	return loadConfig(data)
}

func loadConfig(data []byte) (*kubeschedulerconfig.KubeSchedulerConfiguration, error) {
	// The UniversalDecoder runs defaulting and returns the internal type by default.
	obj, gvk, err := kubeschedulerscheme.Codecs.UniversalDecoder().Decode(data, nil, nil)
	if err != nil {
		// Try strict decoding first. If that fails decode with a lenient
		// decoder, which has only v1alpha1 registered, and log a warning.
		// The lenient path is to be dropped when support for v1alpha1 is dropped.
		if !runtime.IsStrictDecodingError(err) {
			return nil, err
		}

		var lenientErr error
		_, lenientCodecs, lenientErr := codec.NewLenientSchemeAndCodecs(
			kubeschedulerconfig.AddToScheme,
			kubeschedulerconfigv1alpha1.AddToScheme,
		)
		if lenientErr != nil {
			return nil, lenientErr
		}
		obj, gvk, lenientErr = lenientCodecs.UniversalDecoder().Decode(data, nil, nil)
		if lenientErr != nil {
			return nil, err
		}
		klog.Warningf("using lenient decoding as strict decoding failed: %v", err)
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

	encoder := kubeschedulerscheme.Codecs.EncoderForVersion(info.Serializer, kubeschedulerconfigv1alpha2.SchemeGroupVersion)

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
