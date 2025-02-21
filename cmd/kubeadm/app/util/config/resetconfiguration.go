/*
Copyright 2023 The Kubernetes Authors.

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
	"os"
	"strings"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict"
	kubeadmruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
)

// SetResetDynamicDefaults checks and sets configuration values for the ResetConfiguration object
func SetResetDynamicDefaults(cfg *kubeadmapi.ResetConfiguration, skipCRIDetect bool) error {
	var err error
	if cfg.CRISocket == "" {
		if skipCRIDetect {
			klog.V(4).Infof("skip CRI socket detection, fill with the default CRI socket %s", constants.DefaultCRISocket)
			cfg.CRISocket = constants.DefaultCRISocket
			return nil
		}
		cfg.CRISocket, err = kubeadmruntime.DetectCRISocket()
		if err != nil {
			return err
		}
		klog.V(1).Infof("detected and using CRI socket: %s", cfg.CRISocket)
	} else {
		if !strings.HasPrefix(cfg.CRISocket, kubeadmapiv1.DefaultContainerRuntimeURLScheme) {
			klog.Warningf("Usage of CRI endpoints without URL scheme is deprecated and can cause kubelet errors "+
				"in the future. Automatically prepending scheme %q to the \"criSocket\" with value %q. "+
				"Please update your configuration!", kubeadmapiv1.DefaultContainerRuntimeURLScheme, cfg.CRISocket)
			cfg.CRISocket = kubeadmapiv1.DefaultContainerRuntimeURLScheme + "://" + cfg.CRISocket
		}
	}
	return nil
}

// LoadOrDefaultResetConfiguration takes a path to a config file and a versioned configuration that can serve as the default config
// If cfgPath is specified, defaultversionedcfg will always get overridden. Otherwise, the default config (often populated by flags) will be used.
// Then the external, versioned configuration is defaulted and converted to the internal type.
// Right thereafter, the configuration is defaulted again with dynamic values
// Lastly, the internal config is validated and returned.
func LoadOrDefaultResetConfiguration(cfgPath string, defaultversionedcfg *kubeadmapiv1.ResetConfiguration, opts LoadOrDefaultConfigurationOptions) (*kubeadmapi.ResetConfiguration, error) {
	var (
		config *kubeadmapi.ResetConfiguration
		err    error
	)
	if cfgPath != "" {
		// Loads configuration from config file, if provided
		config, err = LoadResetConfigurationFromFile(cfgPath, opts)
	} else {
		config, err = DefaultedResetConfiguration(defaultversionedcfg, opts)
	}
	if err == nil {
		prepareStaticVariables(config)
	}
	return config, err
}

// LoadResetConfigurationFromFile loads versioned ResetConfiguration from file, converts it to internal, defaults and validates it
func LoadResetConfigurationFromFile(cfgPath string, opts LoadOrDefaultConfigurationOptions) (*kubeadmapi.ResetConfiguration, error) {
	klog.V(1).Infof("loading configuration from %q", cfgPath)

	b, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to read config from %q ", cfgPath)
	}

	gvkmap, err := kubeadmutil.SplitYAMLDocuments(b)
	if err != nil {
		return nil, err
	}

	return documentMapToResetConfiguration(gvkmap, false, opts.AllowExperimental, false, opts.SkipCRIDetect)
}

// documentMapToResetConfiguration takes a map between GVKs and YAML/JSON documents (as returned by SplitYAMLDocuments),
// finds a ResetConfiguration, decodes it, dynamically defaults it and then validates it prior to return.
func documentMapToResetConfiguration(gvkmap kubeadmapi.DocumentMap, allowDeprecated, allowExperimental bool, strictErrors bool, skipCRIDetect bool) (*kubeadmapi.ResetConfiguration, error) {
	resetBytes := []byte{}
	for gvk, bytes := range gvkmap {
		// not interested in anything other than ResetConfiguration
		if gvk.Kind != constants.ResetConfigurationKind {
			klog.Warningf("[config] WARNING: Ignored configuration document with GroupVersionKind %v\n", gvk)
			continue
		}

		// check if this version is supported and possibly not deprecated
		if err := validateSupportedVersion(gvk, allowDeprecated, allowExperimental); err != nil {
			return nil, err
		}

		// verify the validity of the YAML/JSON
		if err := strict.VerifyUnmarshalStrict([]*runtime.Scheme{kubeadmscheme.Scheme}, gvk, bytes); err != nil {
			if !strictErrors {
				klog.Warning(err.Error())
			} else {
				return nil, err
			}
		}

		resetBytes = bytes
	}

	if len(resetBytes) == 0 {
		return nil, errors.Errorf("no %s found in the supplied config", constants.JoinConfigurationKind)
	}

	internalcfg := &kubeadmapi.ResetConfiguration{}
	if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), resetBytes, internalcfg); err != nil {
		return nil, err
	}

	// Applies dynamic defaults to settings not provided with flags
	if err := SetResetDynamicDefaults(internalcfg, skipCRIDetect); err != nil {
		return nil, err
	}
	// Validates cfg
	if err := validation.ValidateResetConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}

	return internalcfg, nil
}

// DefaultedResetConfiguration takes a versioned ResetConfiguration (usually filled in by command line parameters), defaults it, converts it to internal and validates it
func DefaultedResetConfiguration(defaultversionedcfg *kubeadmapiv1.ResetConfiguration, opts LoadOrDefaultConfigurationOptions) (*kubeadmapi.ResetConfiguration, error) {
	internalcfg := &kubeadmapi.ResetConfiguration{}

	// Takes passed flags into account; the defaulting is executed once again enforcing assignment of
	// static default values to cfg only for values not provided with flags
	kubeadmscheme.Scheme.Default(defaultversionedcfg)
	if err := kubeadmscheme.Scheme.Convert(defaultversionedcfg, internalcfg, nil); err != nil {
		return nil, err
	}

	// Applies dynamic defaults to settings not provided with flags
	if err := SetResetDynamicDefaults(internalcfg, opts.SkipCRIDetect); err != nil {
		return nil, err
	}
	// Validates cfg
	if err := validation.ValidateResetConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}

	return internalcfg, nil
}
