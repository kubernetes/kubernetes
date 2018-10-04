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

package config

import (
	"fmt"
	"io/ioutil"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// SetJoinDynamicDefaults checks and sets configuration values for the JoinConfiguration object
func SetJoinDynamicDefaults(cfg *kubeadmapi.JoinConfiguration) error {

	if err := SetNodeRegistrationDynamicDefaults(&cfg.NodeRegistration, cfg.ControlPlane); err != nil {
		return err
	}

	if err := SetAPIEndpointDynamicDefaults(&cfg.APIEndpoint); err != nil {
		return err
	}

	return nil
}

// NodeConfigFileAndDefaultsToInternalConfig
func NodeConfigFileAndDefaultsToInternalConfig(cfgPath string, defaultversionedcfg *kubeadmapiv1alpha3.JoinConfiguration) (*kubeadmapi.JoinConfiguration, error) {
	internalcfg := &kubeadmapi.JoinConfiguration{}

	if cfgPath != "" {
		// Loads configuration from config file, if provided
		// Nb. --config overrides command line flags, TODO: fix this
		glog.V(1).Infoln("loading configuration from the given file")

		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}

		if err := DetectUnsupportedVersion(b); err != nil {
			return nil, err
		}

		gvkmap, err := kubeadmutil.SplitYAMLDocuments(b)
		if err != nil {
			return nil, err
		}

		joinBytes := []byte{}
		for gvk, bytes := range gvkmap {
			if gvk.Kind == constants.JoinConfigurationKind || gvk.Kind == constants.NodeConfigurationKind {
				joinBytes = bytes
			}
		}

		if len(joinBytes) == 0 {
			return nil, fmt.Errorf("no %s found in config file %q", constants.JoinConfigurationKind, cfgPath)
		}

		if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), joinBytes, internalcfg); err != nil {
			return nil, err
		}
	} else {
		// Takes passed flags into account; the defaulting is executed once again enforcing assignement of
		// static default values to cfg only for values not provided with flags
		kubeadmscheme.Scheme.Default(defaultversionedcfg)
		kubeadmscheme.Scheme.Convert(defaultversionedcfg, internalcfg, nil)
	}

	// Applies dynamic defaults to settings not provided with flags
	if err := SetJoinDynamicDefaults(internalcfg); err != nil {
		return nil, err
	}
	// Validates cfg (flags/configs + defaults)
	if err := validation.ValidateJoinConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}

	return internalcfg, nil
}
