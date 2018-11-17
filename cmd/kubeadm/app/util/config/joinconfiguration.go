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
	"io/ioutil"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict"
)

// SetJoinDynamicDefaults checks and sets configuration values for the JoinConfiguration object
func SetJoinDynamicDefaults(cfg *kubeadmapi.JoinConfiguration) error {
	addMasterTaint := false
	if cfg.ControlPlane != nil {
		addMasterTaint = true
	}
	if err := SetNodeRegistrationDynamicDefaults(&cfg.NodeRegistration, addMasterTaint); err != nil {
		return err
	}

	if err := SetJoinControlPlaneDefaults(cfg.ControlPlane); err != nil {
		return err
	}

	return nil
}

// SetJoinControlPlaneDefaults checks and sets configuration values for the JoinControlPlane object
func SetJoinControlPlaneDefaults(cfg *kubeadmapi.JoinControlPlane) error {
	if cfg != nil {
		if err := SetAPIEndpointDynamicDefaults(&cfg.LocalAPIEndpoint); err != nil {
			return err
		}
	}
	return nil
}

// JoinConfigFileAndDefaultsToInternalConfig takes a path to a config file and a versioned configuration that can serve as the default config
// If cfgPath is specified, defaultversionedcfg will always get overridden. Otherwise, the default config (often populated by flags) will be used.
// Then the external, versioned configuration is defaulted and converted to the internal type.
// Right thereafter, the configuration is defaulted again with dynamic values (like IP addresses of a machine, etc)
// Lastly, the internal config is validated and returned.
func JoinConfigFileAndDefaultsToInternalConfig(cfgPath string, defaultversionedcfg *kubeadmapiv1beta1.JoinConfiguration) (*kubeadmapi.JoinConfiguration, error) {
	internalcfg := &kubeadmapi.JoinConfiguration{}

	if cfgPath != "" {
		// Loads configuration from config file, if provided
		// Nb. --config overrides command line flags, TODO: fix this
		klog.V(1).Infoln("loading configuration from the given file")

		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, errors.Wrapf(err, "unable to read config from %q ", cfgPath)
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
			if gvk.Kind == constants.JoinConfigurationKind {
				joinBytes = bytes
				// verify the validity of the YAML
				strict.VerifyUnmarshalStrict(bytes, gvk)
			}
		}

		if len(joinBytes) == 0 {
			return nil, errors.Errorf("no %s found in config file %q", constants.JoinConfigurationKind, cfgPath)
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
