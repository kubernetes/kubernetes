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

package resourcequota

import (
	"fmt"
	"io"
	"io/ioutil"
	"regexp"

	runtime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	componentconfigv1alpha1 "k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
)

// LoadResourceQuotaConfiguration loads the provided configuration.
func LoadResourceQuotaConfiguration(config io.Reader) (*componentconfig.ResourceQuotaConfiguration, error) {
	// if no config is provided, return a default configuration
	if config == nil {
		externalConfig := &componentconfigv1alpha1.ResourceQuotaConfiguration{}
		api.Scheme.Default(externalConfig)
		internalConfig := &componentconfig.ResourceQuotaConfiguration{}
		if err := api.Scheme.Convert(externalConfig, internalConfig, nil); err != nil {
			return nil, err
		}
		return internalConfig, nil
	}
	// we have a config so parse it.
	data, err := ioutil.ReadAll(config)
	if err != nil {
		return nil, err
	}
	decoder := api.Codecs.UniversalDecoder()
	decodedObj, err := runtime.Decode(decoder, data)
	if err != nil {
		return nil, err
	}
	resourceQuotaConfiguration, ok := decodedObj.(*componentconfig.ResourceQuotaConfiguration)
	if !ok {
		return nil, fmt.Errorf("unexpected type: %T", decodedObj)
	}
	return resourceQuotaConfiguration, nil
}

// ValidateResourceQuotaConfiguration validates the configuration.
func ValidateResourceQuotaConfiguration(config *componentconfig.ResourceQuotaConfiguration) error {
	if config == nil {
		return nil
	}
	for _, limitedResource := range config.LimitedResources {
		if _, err := regexp.Compile(limitedResource.MatchExpression); err != nil {
			return fmt.Errorf("invalid resource quota configuration: %v", err)
		}
	}
	return nil
}
