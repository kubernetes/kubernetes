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

package config

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net"
	"reflect"
	"sort"
	"strconv"

	"github.com/pkg/errors"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	bootstraputil "k8s.io/cluster-bootstrap/token/util"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

// SetInitDynamicDefaults checks and sets configuration values for the InitConfiguration object
func SetInitDynamicDefaults(cfg *kubeadmapi.InitConfiguration) error {
	if err := SetBootstrapTokensDynamicDefaults(&cfg.BootstrapTokens); err != nil {
		return err
	}
	if err := SetNodeRegistrationDynamicDefaults(&cfg.NodeRegistration, true); err != nil {
		return err
	}
	if err := SetAPIEndpointDynamicDefaults(&cfg.LocalAPIEndpoint); err != nil {
		return err
	}
	if err := SetClusterDynamicDefaults(&cfg.ClusterConfiguration, cfg.LocalAPIEndpoint.AdvertiseAddress, cfg.LocalAPIEndpoint.BindPort); err != nil {
		return err
	}
	return nil
}

// SetBootstrapTokensDynamicDefaults checks and sets configuration values for the BootstrapTokens object
func SetBootstrapTokensDynamicDefaults(cfg *[]kubeadmapi.BootstrapToken) error {
	// Populate the .Token field with a random value if unset
	// We do this at this layer, and not the API defaulting layer
	// because of possible security concerns, and more practically
	// because we can't return errors in the API object defaulting
	// process but here we can.
	for i, bt := range *cfg {
		if bt.Token != nil && len(bt.Token.String()) > 0 {
			continue
		}

		tokenStr, err := bootstraputil.GenerateBootstrapToken()
		if err != nil {
			return errors.Wrap(err, "couldn't generate random token")
		}
		token, err := kubeadmapi.NewBootstrapTokenString(tokenStr)
		if err != nil {
			return err
		}
		(*cfg)[i].Token = token
	}

	return nil
}

// SetNodeRegistrationDynamicDefaults checks and sets configuration values for the NodeRegistration object
func SetNodeRegistrationDynamicDefaults(cfg *kubeadmapi.NodeRegistrationOptions, masterTaint bool) error {
	var err error
	cfg.Name, err = nodeutil.GetHostname(cfg.Name)
	if err != nil {
		return err
	}

	// Only if the slice is nil, we should append the master taint. This allows the user to specify an empty slice for no default master taint
	if masterTaint && cfg.Taints == nil {
		cfg.Taints = []v1.Taint{kubeadmconstants.MasterTaint}
	}

	return nil
}

// SetAPIEndpointDynamicDefaults checks and sets configuration values for the APIEndpoint object
func SetAPIEndpointDynamicDefaults(cfg *kubeadmapi.APIEndpoint) error {
	// validate cfg.API.AdvertiseAddress.
	addressIP := net.ParseIP(cfg.AdvertiseAddress)
	if addressIP == nil && cfg.AdvertiseAddress != "" {
		return errors.Errorf("couldn't use \"%s\" as \"apiserver-advertise-address\", must be ipv4 or ipv6 address", cfg.AdvertiseAddress)
	}
	// This is the same logic as the API Server uses, except that if no interface is found the address is set to 0.0.0.0, which is invalid and cannot be used
	// for bootstrapping a cluster.
	ip, err := ChooseAPIServerBindAddress(addressIP)
	if err != nil {
		return err
	}
	cfg.AdvertiseAddress = ip.String()

	return nil
}

// SetClusterDynamicDefaults checks and sets values for the ClusterConfiguration object
func SetClusterDynamicDefaults(cfg *kubeadmapi.ClusterConfiguration, advertiseAddress string, bindPort int32) error {
	// Default all the embedded ComponentConfig structs
	componentconfigs.Known.Default(cfg)

	ip := net.ParseIP(advertiseAddress)
	if ip.To4() != nil {
		cfg.ComponentConfigs.KubeProxy.BindAddress = kubeadmapiv1beta1.DefaultProxyBindAddressv4
	} else {
		cfg.ComponentConfigs.KubeProxy.BindAddress = kubeadmapiv1beta1.DefaultProxyBindAddressv6
	}

	// Resolve possible version labels and validate version string
	if err := NormalizeKubernetesVersion(cfg); err != nil {
		return err
	}

	// If ControlPlaneEndpoint is specified without a port number defaults it to
	// the bindPort number of the APIEndpoint.
	// This will allow join of additional control plane instances with different bindPort number
	if cfg.ControlPlaneEndpoint != "" {
		host, port, err := kubeadmutil.ParseHostPort(cfg.ControlPlaneEndpoint)
		if err != nil {
			return err
		}
		if port == "" {
			cfg.ControlPlaneEndpoint = net.JoinHostPort(host, strconv.FormatInt(int64(bindPort), 10))
		}
	}

	// Downcase SANs. Some domain names (like ELBs) have capitals in them.
	LowercaseSANs(cfg.APIServer.CertSANs)
	return nil
}

// ConfigFileAndDefaultsToInternalConfig takes a path to a config file and a versioned configuration that can serve as the default config
// If cfgPath is specified, defaultversionedcfg will always get overridden. Otherwise, the default config (often populated by flags) will be used.
// Then the external, versioned configuration is defaulted and converted to the internal type.
// Right thereafter, the configuration is defaulted again with dynamic values (like IP addresses of a machine, etc)
// Lastly, the internal config is validated and returned.
func ConfigFileAndDefaultsToInternalConfig(cfgPath string, defaultversionedcfg *kubeadmapiv1beta1.InitConfiguration) (*kubeadmapi.InitConfiguration, error) {
	internalcfg := &kubeadmapi.InitConfiguration{}

	if cfgPath != "" {
		// Loads configuration from config file, if provided
		// Nb. --config overrides command line flags
		klog.V(1).Infoln("loading configuration from the given file")

		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, errors.Wrapf(err, "unable to read config from %q ", cfgPath)
		}
		internalcfg, err = BytesToInternalConfig(b)
		if err != nil {
			return nil, err
		}
	} else {
		// Takes passed flags into account; the defaulting is executed once again enforcing assignment of
		// static default values to cfg only for values not provided with flags
		kubeadmscheme.Scheme.Default(defaultversionedcfg)
		kubeadmscheme.Scheme.Convert(defaultversionedcfg, internalcfg, nil)
	}

	// Applies dynamic defaults to settings not provided with flags
	if err := SetInitDynamicDefaults(internalcfg); err != nil {
		return nil, err
	}
	// Validates cfg (flags/configs + defaults + dynamic defaults)
	if err := validation.ValidateInitConfiguration(internalcfg).ToAggregate(); err != nil {
		return nil, err
	}
	return internalcfg, nil
}

// BytesToInternalConfig converts a byte slice to an internal, defaulted and validated configuration object.
// The byte slice may contain one or many different YAML documents. These YAML documents are parsed one-by-one
// and well-known ComponentConfig GroupVersionKinds are stored inside of the internal InitConfiguration struct
func BytesToInternalConfig(b []byte) (*kubeadmapi.InitConfiguration, error) {
	var initcfg *kubeadmapi.InitConfiguration
	var clustercfg *kubeadmapi.ClusterConfiguration
	decodedComponentConfigObjects := map[componentconfigs.RegistrationKind]runtime.Object{}

	if err := DetectUnsupportedVersion(b); err != nil {
		return nil, err
	}

	gvkmap, err := kubeadmutil.SplitYAMLDocuments(b)
	if err != nil {
		return nil, err
	}

	for gvk, fileContent := range gvkmap {
		// verify the validity of the YAML
		strict.VerifyUnmarshalStrict(fileContent, gvk)

		// Try to get the registration for the ComponentConfig based on the kind
		regKind := componentconfigs.RegistrationKind(gvk.Kind)
		registration, found := componentconfigs.Known[regKind]
		if found {
			// Unmarshal the bytes from the YAML document into a runtime.Object containing the ComponentConfiguration struct
			obj, err := registration.Unmarshal(fileContent)
			if err != nil {
				return nil, err
			}
			decodedComponentConfigObjects[regKind] = obj
			continue
		}

		if kubeadmutil.GroupVersionKindsHasInitConfiguration(gvk) {
			// Set initcfg to an empty struct value the deserializer will populate
			initcfg = &kubeadmapi.InitConfiguration{}
			// Decode the bytes into the internal struct. Under the hood, the bytes will be unmarshalled into the
			// right external version, defaulted, and converted into the internal version.
			if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), fileContent, initcfg); err != nil {
				return nil, err
			}
			continue
		}
		if kubeadmutil.GroupVersionKindsHasClusterConfiguration(gvk) {
			// Set clustercfg to an empty struct value the deserializer will populate
			clustercfg = &kubeadmapi.ClusterConfiguration{}
			// Decode the bytes into the internal struct. Under the hood, the bytes will be unmarshalled into the
			// right external version, defaulted, and converted into the internal version.
			if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), fileContent, clustercfg); err != nil {
				return nil, err
			}
			continue
		}

		fmt.Printf("[config] WARNING: Ignored YAML document with GroupVersionKind %v\n", gvk)
	}

	// Enforce that InitConfiguration and/or ClusterConfiguration has to exist among the YAML documents
	if initcfg == nil && clustercfg == nil {
		return nil, errors.New("no InitConfiguration or ClusterConfiguration kind was found in the YAML file")
	}

	// If InitConfiguration wasn't given, default it by creating an external struct instance, default it and convert into the internal type
	if initcfg == nil {
		extinitcfg := &kubeadmapiv1beta1.InitConfiguration{}
		kubeadmscheme.Scheme.Default(extinitcfg)
		// Set initcfg to an empty struct value the deserializer will populate
		initcfg = &kubeadmapi.InitConfiguration{}
		kubeadmscheme.Scheme.Convert(extinitcfg, initcfg, nil)
	}
	// If ClusterConfiguration was given, populate it in the InitConfiguration struct
	if clustercfg != nil {
		initcfg.ClusterConfiguration = *clustercfg
	}

	// Save the loaded ComponentConfig objects in the initcfg object
	for kind, obj := range decodedComponentConfigObjects {
		registration, found := componentconfigs.Known[kind]
		if found {
			if ok := registration.SetToInternalConfig(obj, &initcfg.ClusterConfiguration); !ok {
				return nil, errors.Errorf("couldn't save componentconfig value for kind %q", string(kind))
			}
		} else {
			// This should never happen in practice
			fmt.Printf("[config] WARNING: Decoded a kind that couldn't be saved to the internal configuration: %q\n", string(kind))
		}
	}
	return initcfg, nil
}

func defaultedInternalConfig() *kubeadmapi.ClusterConfiguration {
	externalcfg := &kubeadmapiv1beta1.ClusterConfiguration{}
	internalcfg := &kubeadmapi.ClusterConfiguration{}

	kubeadmscheme.Scheme.Default(externalcfg)
	kubeadmscheme.Scheme.Convert(externalcfg, internalcfg, nil)

	// Default the embedded ComponentConfig structs
	componentconfigs.Known.Default(internalcfg)
	return internalcfg
}

// MarshalInitConfigurationToBytes marshals the internal InitConfiguration object to bytes. It writes the embedded
// ClusterConfiguration object with ComponentConfigs out as separate YAML documents
func MarshalInitConfigurationToBytes(cfg *kubeadmapi.InitConfiguration, gv schema.GroupVersion) ([]byte, error) {
	initbytes, err := kubeadmutil.MarshalToYamlForCodecs(cfg, gv, kubeadmscheme.Codecs)
	if err != nil {
		return []byte{}, err
	}
	allFiles := [][]byte{initbytes}

	// Exception: If the specified groupversion is targeting the internal type, don't print embedded ClusterConfiguration contents
	// This is mostly used for unit testing. In a real scenario the internal version of the API is never marshalled as-is.
	if gv.Version != runtime.APIVersionInternal {
		clusterbytes, err := MarshalClusterConfigurationToBytes(&cfg.ClusterConfiguration, gv)
		if err != nil {
			return []byte{}, err
		}
		allFiles = append(allFiles, clusterbytes)
	}
	return bytes.Join(allFiles, []byte(kubeadmconstants.YAMLDocumentSeparator)), nil
}

// MarshalClusterConfigurationToBytes marshals the internal ClusterConfiguration object to bytes. It writes the embedded
// ComponentConfiguration objects out as separate YAML documents
func MarshalClusterConfigurationToBytes(clustercfg *kubeadmapi.ClusterConfiguration, gv schema.GroupVersion) ([]byte, error) {
	clusterbytes, err := kubeadmutil.MarshalToYamlForCodecs(clustercfg, gv, kubeadmscheme.Codecs)
	if err != nil {
		return []byte{}, err
	}
	allFiles := [][]byte{clusterbytes}
	componentConfigContent := map[string][]byte{}
	defaultedcfg := defaultedInternalConfig()

	for kind, registration := range componentconfigs.Known {
		// If the ComponentConfig struct for the current registration is nil, skip it when marshalling
		realobj, ok := registration.GetFromInternalConfig(clustercfg)
		if !ok {
			continue
		}

		defaultedobj, ok := registration.GetFromInternalConfig(defaultedcfg)
		// Invalid: The caller asked to not print the componentconfigs if defaulted, but defaultComponentConfigs() wasn't able to create default objects to use for reference
		if !ok {
			return []byte{}, errors.New("couldn't create a default componentconfig object")
		}

		// If the real ComponentConfig object differs from the default, print it out. If not, there's no need to print it out, so skip it
		if !reflect.DeepEqual(realobj, defaultedobj) {
			contentBytes, err := registration.Marshal(realobj)
			if err != nil {
				return []byte{}, err
			}
			componentConfigContent[string(kind)] = contentBytes
		}
	}

	// Sort the ComponentConfig files by kind when marshalling
	sortedComponentConfigFiles := consistentOrderByteSlice(componentConfigContent)
	allFiles = append(allFiles, sortedComponentConfigFiles...)
	return bytes.Join(allFiles, []byte(kubeadmconstants.YAMLDocumentSeparator)), nil
}

// consistentOrderByteSlice takes a map of a string key and a byte slice, and returns a byte slice of byte slices
// with consistent ordering, where the keys in the map determine the ordering of the return value. This has to be
// done as the order of a for...range loop over a map in go is undeterministic, and could otherwise lead to flakes
// in e.g. unit tests when marshalling content with a random order
func consistentOrderByteSlice(content map[string][]byte) [][]byte {
	keys := []string{}
	sortedContent := [][]byte{}
	for key := range content {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		sortedContent = append(sortedContent, content[key])
	}
	return sortedContent
}
