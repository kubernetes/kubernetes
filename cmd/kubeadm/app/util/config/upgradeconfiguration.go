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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfig "k8s.io/kubelet/config/v1beta1"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict"
)

var componentCfgGV = sets.New(kubeproxyconfig.GroupName, kubeletconfig.GroupName)

// documentMapToUpgradeConfiguration takes a map between GVKs and YAML documents (as returned by SplitYAMLDocuments),
// finds a UpgradeConfiguration, decodes it, dynamically defaults it and then validates it prior to return.
func documentMapToUpgradeConfiguration(gvkmap kubeadmapi.DocumentMap, allowDeprecated bool) (*kubeadmapi.UpgradeConfiguration, error) {
	var internalcfg *kubeadmapi.UpgradeConfiguration

	for gvk, bytes := range gvkmap {
		// check if this version is supported and possibly not deprecated
		if err := validateSupportedVersion(gvk, allowDeprecated, true); err != nil {
			return nil, err
		}

		// verify the validity of the YAML
		if err := strict.VerifyUnmarshalStrict([]*runtime.Scheme{kubeadmscheme.Scheme}, gvk, bytes); err != nil {
			klog.Warning(err.Error())
		}

		if kubeadmutil.GroupVersionKindsHasInitConfiguration(gvk) || kubeadmutil.GroupVersionKindsHasClusterConfiguration(gvk) {
			klog.Warningf("[config] WARNING: YAML document with GroupVersionKind %v is deprecated for upgrade, please use config file with kind of UpgradeConfiguration instead \n", gvk)
			continue
		}

		if kubeadmutil.GroupVersionKindsHasUpgradeConfiguration(gvk) {
			// Set internalcfg to an empty struct value the deserializer will populate
			internalcfg = &kubeadmapi.UpgradeConfiguration{}
			// Decode the bytes into the internal struct. Under the hood, the bytes will be unmarshalled into the
			// right external version, defaulted, and converted into the internal version.
			if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), bytes, internalcfg); err != nil {
				return nil, err
			}
			continue
		}

		// If the group is neither a kubeadm core type or of a supported component config group, we dump a warning about it being ignored
		if !componentconfigs.Scheme.IsGroupRegistered(gvk.Group) {
			klog.Warningf("[config] WARNING: Ignored YAML document with GroupVersionKind %v\n", gvk)
		}
	}

	// If UpgradeConfiguration wasn't given, default it by creating an external struct instance, default it and convert into the internal type
	if internalcfg == nil {
		extinitcfg := &kubeadmapiv1.UpgradeConfiguration{}
		kubeadmscheme.Scheme.Default(extinitcfg)
		// Set upgradeCfg to an empty struct value the deserializer will populate
		internalcfg = &kubeadmapi.UpgradeConfiguration{}
		if err := kubeadmscheme.Scheme.Convert(extinitcfg, internalcfg, nil); err != nil {
			return nil, err
		}
	}

	// Validates cfg
	if err := validation.ValidateUpgradeConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}

	return internalcfg, nil
}

// DocMapToUpgradeConfiguration converts documentMap to an internal, defaulted and validated UpgradeConfiguration object.
// The map may contain many different YAML documents. These YAML documents are parsed one-by-one
// and well-known ComponentConfig GroupVersionKinds are stored inside of the internal UpgradeConfiguration struct.
// The resulting UpgradeConfiguration is then dynamically defaulted and validated prior to return.
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

	// Split the YAML documents in the file into a DocumentMap
	docmap, err := kubeadmutil.SplitYAMLDocuments(configBytes)
	if err != nil {
		return nil, err
	}

	// Convert documentMap to internal UpgradeConfiguration, InitConfiguration and ClusterConfiguration from config file will be ignored.
	// Upgrade should respect the cluster configuration from the existing cluster, re-configure the cluster with a InitConfiguration and
	// ClusterConfiguration from the config file is not allowed for upgrade.
	if isKubeadmConfigPresent(docmap) {
		if upgradeCfg, err = DocMapToUpgradeConfiguration(docmap); err != nil {
			return nil, err
		}
	}

	// Check is there any component configs defined in the config file.
	for gvk := range docmap {
		if componentCfgGV.Has(gvk.Group) {
			klog.Warningf("[config] WARNING: YAML document with Component Configs %v is deprecated for upgrade and will be ignored \n", gvk.Group)
			continue
		}
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
