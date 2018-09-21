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
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
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

		// if this is a componentconfig/v1alpha1 KubeSchedulerConfiguration object, coerce it to kubescheduler.config.k8s.io/v1alpha1 with a warning
		// TODO: drop this block in 1.13
		if runtime.IsNotRegisteredError(err) {
			originalErr := err
			var (
				u               = &unstructured.Unstructured{}
				codec           = json.NewYAMLSerializer(json.DefaultMetaFactory, kubeschedulerscheme.Scheme, kubeschedulerscheme.Scheme)
				legacyConfigGVK = schema.GroupVersionKind{Group: "componentconfig", Version: "v1alpha1", Kind: "KubeSchedulerConfiguration"}
			)
			// attempt to decode to an unstructured object
			obj, gvk, err := codec.Decode(data, nil, u)

			// if this errored, or the object we read was not the legacy alpha gvk, return the original error
			if err != nil || gvk == nil || *gvk != legacyConfigGVK {
				return nil, originalErr
			}

			fmt.Printf("WARNING: the provided config file is an unsupported apiVersion (%q), which will be removed in future releases\n\n", legacyConfigGVK.GroupVersion().String())
			fmt.Printf("WARNING: switch to command-line flags or update your config file apiVersion to %q\n\n", kubeschedulerconfigv1alpha1.SchemeGroupVersion.String())
			fmt.Printf("WARNING: apiVersions at alpha-level are not guaranteed to be supported in future releases\n\n")

			// attempt to coerce to the new alpha gvk
			if err := meta.NewAccessor().SetAPIVersion(obj, kubeschedulerconfigv1alpha1.SchemeGroupVersion.String()); err != nil {
				// return the original error on failure
				return nil, originalErr
			}

			// attempt to encode the coerced apiVersion back to bytes
			buffer := bytes.NewBuffer([]byte{})
			if err := codec.Encode(obj, buffer); err != nil {
				// return the original error on failure
				return nil, originalErr
			}

			// re-attempt to load the coerced apiVersion
			return loadConfig(buffer.Bytes())
		}

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
