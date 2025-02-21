/*
Copyright 2024 The Kubernetes Authors.

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
)

// documentMapToUpgradeConfiguration takes a map between GVKs and YAML/JSON documents (as returned by SplitYAMLDocuments),
// finds a UpgradeConfiguration, decodes it, dynamically defaults it and then validates it prior to return.
func documentMapToUpgradeConfiguration(gvkmap kubeadmapi.DocumentMap, allowDeprecated bool) (*kubeadmapi.UpgradeConfiguration, error) {
	upgradeBytes := []byte{}

	for gvk, bytes := range gvkmap {
		if gvk.Kind != constants.UpgradeConfigurationKind {
			klog.Warningf("[config] WARNING: Ignored configuration document with GroupVersionKind %v\n", gvk)
			continue
		}

		// check if this version is supported and possibly not deprecated
		if err := validateSupportedVersion(gvk, allowDeprecated, true); err != nil {
			return nil, err
		}

		// verify the validity of the YAML/JSON
		if err := strict.VerifyUnmarshalStrict([]*runtime.Scheme{kubeadmscheme.Scheme}, gvk, bytes); err != nil {
			klog.Warning(err.Error())
		}

		upgradeBytes = bytes
	}

	if len(upgradeBytes) == 0 {
		return nil, errors.Errorf("no %s found in the supplied config", constants.UpgradeConfigurationKind)
	}

	// Set internalcfg to an empty struct value the deserializer will populate
	internalcfg := &kubeadmapi.UpgradeConfiguration{}
	// Decode the bytes into the internal struct. Under the hood, the bytes will be unmarshalled into the
	// right external version, defaulted, and converted into the internal version.
	if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), upgradeBytes, internalcfg); err != nil {
		return nil, err
	}

	// Validates cfg
	if err := validation.ValidateUpgradeConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}

	return internalcfg, nil
}

// DocMapToUpgradeConfiguration converts documentMap to an internal, defaulted and validated UpgradeConfiguration object.
func DocMapToUpgradeConfiguration(gvkmap kubeadmapi.DocumentMap) (*kubeadmapi.UpgradeConfiguration, error) {
	return documentMapToUpgradeConfiguration(gvkmap, false)
}

// LoadUpgradeConfigurationFromFile loads UpgradeConfiguration from a file.
func LoadUpgradeConfigurationFromFile(cfgPath string, _ LoadOrDefaultConfigurationOptions) (*kubeadmapi.UpgradeConfiguration, error) {
	var err error
	var upgradeCfg *kubeadmapi.UpgradeConfiguration

	// Otherwise, we have a config file. Let's load it.
	configBytes, err := os.ReadFile(cfgPath)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to load config from file %q", cfgPath)
	}

	// Split the YAML/JSON documents in the file into a DocumentMap
	docmap, err := kubeadmutil.SplitConfigDocuments(configBytes)
	if err != nil {
		return nil, err
	}

	// Convert documentMap to internal UpgradeConfiguration, InitConfiguration and ClusterConfiguration from config file will be ignored.
	// Upgrade should respect the cluster configuration from the existing cluster, re-configure the cluster with a InitConfiguration and
	// ClusterConfiguration from the config file is not allowed for upgrade.
	if upgradeCfg, err = DocMapToUpgradeConfiguration(docmap); err != nil {
		return nil, err
	}

	return upgradeCfg, nil
}

// LoadOrDefaultUpgradeConfiguration takes a path to a config file and a versioned configuration that can serve as the default config
// If cfgPath is specified, defaultversionedcfg will always get overridden. Otherwise, the default config (often populated by flags) will be used.
// Then the external, versioned configuration is defaulted and converted to the internal type.
// Right thereafter, the configuration is defaulted again with dynamic values
// Lastly, the internal config is validated and returned.
func LoadOrDefaultUpgradeConfiguration(cfgPath string, defaultversionedcfg *kubeadmapiv1.UpgradeConfiguration, opts LoadOrDefaultConfigurationOptions) (*kubeadmapi.UpgradeConfiguration, error) {
	var (
		config *kubeadmapi.UpgradeConfiguration
		err    error
	)
	if cfgPath != "" {
		// Loads configuration from config file, if provided
		config, err = LoadUpgradeConfigurationFromFile(cfgPath, opts)
	} else {
		config, err = DefaultedUpgradeConfiguration(defaultversionedcfg, opts)
	}
	if err == nil {
		prepareStaticVariables(config)
	}
	return config, err
}

// DefaultedUpgradeConfiguration takes a versioned UpgradeConfiguration (usually filled in by command line parameters), defaults it, converts it to internal and validates it
func DefaultedUpgradeConfiguration(defaultversionedcfg *kubeadmapiv1.UpgradeConfiguration, _ LoadOrDefaultConfigurationOptions) (*kubeadmapi.UpgradeConfiguration, error) {
	internalcfg := &kubeadmapi.UpgradeConfiguration{}

	// Takes passed flags into account; the defaulting is executed once again enforcing assignment of
	// static default values to cfg only for values not provided with flags
	kubeadmscheme.Scheme.Default(defaultversionedcfg)
	if err := kubeadmscheme.Scheme.Convert(defaultversionedcfg, internalcfg, nil); err != nil {
		return nil, err
	}

	// Validates cfg
	if err := validation.ValidateUpgradeConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}

	return internalcfg, nil
}
