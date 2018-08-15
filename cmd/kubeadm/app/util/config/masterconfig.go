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

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	netutil "k8s.io/apimachinery/pkg/util/net"
	bootstraputil "k8s.io/client-go/tools/bootstrap/token/util"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
)

// SetInitDynamicDefaults checks and sets configuration values for the InitConfiguration object
func SetInitDynamicDefaults(cfg *kubeadmapi.InitConfiguration) error {

	// Default all the embedded ComponentConfig structs
	componentconfigs.Known.Default(cfg)

	// validate cfg.API.AdvertiseAddress.
	addressIP := net.ParseIP(cfg.API.AdvertiseAddress)
	if addressIP == nil && cfg.API.AdvertiseAddress != "" {
		return fmt.Errorf("couldn't use \"%s\" as \"apiserver-advertise-address\", must be ipv4 or ipv6 address", cfg.API.AdvertiseAddress)
	}
	// Choose the right address for the API Server to advertise. If the advertise address is localhost or 0.0.0.0, the default interface's IP address is used
	// This is the same logic as the API Server uses
	ip, err := netutil.ChooseBindAddress(addressIP)
	if err != nil {
		return err
	}
	cfg.API.AdvertiseAddress = ip.String()
	ip = net.ParseIP(cfg.API.AdvertiseAddress)
	if ip.To4() != nil {
		cfg.ComponentConfigs.KubeProxy.BindAddress = kubeadmapiv1alpha3.DefaultProxyBindAddressv4
	} else {
		cfg.ComponentConfigs.KubeProxy.BindAddress = kubeadmapiv1alpha3.DefaultProxyBindAddressv6
	}
	// Resolve possible version labels and validate version string
	if err := NormalizeKubernetesVersion(cfg); err != nil {
		return err
	}

	// Downcase SANs. Some domain names (like ELBs) have capitals in them.
	LowercaseSANs(cfg.APIServerCertSANs)

	// Populate the .Token field with a random value if unset
	// We do this at this layer, and not the API defaulting layer
	// because of possible security concerns, and more practically
	// because we can't return errors in the API object defaulting
	// process but here we can.
	for i, bt := range cfg.BootstrapTokens {
		if bt.Token != nil && len(bt.Token.String()) > 0 {
			continue
		}

		tokenStr, err := bootstraputil.GenerateBootstrapToken()
		if err != nil {
			return fmt.Errorf("couldn't generate random token: %v", err)
		}
		token, err := kubeadmapi.NewBootstrapTokenString(tokenStr)
		if err != nil {
			return err
		}
		cfg.BootstrapTokens[i].Token = token
	}

	cfg.NodeRegistration.Name, err = nodeutil.GetHostname(cfg.NodeRegistration.Name)
	if err != nil {
		return err
	}

	// Only if the slice is nil, we should append the master taint. This allows the user to specify an empty slice for no default master taint
	if cfg.NodeRegistration.Taints == nil {
		cfg.NodeRegistration.Taints = []v1.Taint{kubeadmconstants.MasterTaint}
	}

	return nil
}

// ConfigFileAndDefaultsToInternalConfig takes a path to a config file and a versioned configuration that can serve as the default config
// If cfgPath is specified, defaultversionedcfg will always get overridden. Otherwise, the default config (often populated by flags) will be used.
// Then the external, versioned configuration is defaulted and converted to the internal type.
// Right thereafter, the configuration is defaulted again with dynamic values (like IP addresses of a machine, etc)
// Lastly, the internal config is validated and returned.
func ConfigFileAndDefaultsToInternalConfig(cfgPath string, defaultversionedcfg *kubeadmapiv1alpha3.InitConfiguration) (*kubeadmapi.InitConfiguration, error) {
	internalcfg := &kubeadmapi.InitConfiguration{}

	if cfgPath != "" {
		// Loads configuration from config file, if provided
		// Nb. --config overrides command line flags
		glog.V(1).Infoln("loading configuration from the given file")

		b, err := ioutil.ReadFile(cfgPath)
		if err != nil {
			return nil, fmt.Errorf("unable to read config from %q [%v]", cfgPath, err)
		}
		return BytesToInternalConfig(b)
	}

	// Takes passed flags into account; the defaulting is executed once again enforcing assignment of
	// static default values to cfg only for values not provided with flags
	kubeadmscheme.Scheme.Default(defaultversionedcfg)
	kubeadmscheme.Scheme.Convert(defaultversionedcfg, internalcfg, nil)

	return defaultAndValidate(internalcfg)
}

// BytesToInternalConfig converts a byte slice to an internal, defaulted and validated configuration object.
// The byte slice may contain one or many different YAML documents. These YAML documents are parsed one-by-one
// and well-known ComponentConfig GroupVersionKinds are stored inside of the internal InitConfiguration struct
func BytesToInternalConfig(b []byte) (*kubeadmapi.InitConfiguration, error) {
	internalcfg := &kubeadmapi.InitConfiguration{}
	decodedObjs := map[componentconfigs.RegistrationKind]runtime.Object{}
	masterConfigFound := false

	if err := DetectUnsupportedVersion(b); err != nil {
		return nil, err
	}

	gvkmap, err := kubeadmutil.SplitYAMLDocuments(b)
	if err != nil {
		return nil, err
	}

	for gvk, fileContent := range gvkmap {
		// Try to get the registration for the ComponentConfig based on the kind
		regKind := componentconfigs.RegistrationKind(gvk.Kind)
		registration, found := componentconfigs.Known[regKind]
		if found {
			// Unmarshal the bytes from the YAML document into a runtime.Object containing the ComponentConfiguration struct
			obj, err := registration.Unmarshal(fileContent)
			if err != nil {
				return nil, err
			}
			decodedObjs[regKind] = obj
			continue
		}

		if kubeadmutil.GroupVersionKindsHasInitConfiguration(gvk) {
			if err := runtime.DecodeInto(kubeadmscheme.Codecs.UniversalDecoder(), fileContent, internalcfg); err != nil {
				return nil, err
			}
			masterConfigFound = true
			continue
		}

		fmt.Printf("[config] WARNING: Ignored YAML document with GroupVersionKind %v\n", gvk)
	}
	// Just as an extra safety check, don't proceed if a InitConfiguration object wasn't found
	if !masterConfigFound {
		return nil, fmt.Errorf("no InitConfiguration kind was found in the YAML file")
	}

	// Save the loaded ComponentConfig objects in the internalcfg object
	for kind, obj := range decodedObjs {
		registration, found := componentconfigs.Known[kind]
		if found {
			if ok := registration.SetToInternalConfig(obj, internalcfg); !ok {
				return nil, fmt.Errorf("couldn't save componentconfig value for kind %q", string(kind))
			}
		} else {
			// This should never happen in practice
			fmt.Printf("[config] WARNING: Decoded a kind that couldn't be saved to the internal configuration: %q\n", string(kind))
		}
	}

	return defaultAndValidate(internalcfg)
}

func defaultAndValidate(cfg *kubeadmapi.InitConfiguration) (*kubeadmapi.InitConfiguration, error) {
	// Applies dynamic defaults to settings not provided with flags
	if err := SetInitDynamicDefaults(cfg); err != nil {
		return nil, err
	}
	// Validates cfg (flags/configs + defaults + dynamic defaults)
	if err := validation.ValidateInitConfiguration(cfg).ToAggregate(); err != nil {
		return nil, err
	}

	return cfg, nil
}

func defaultedInternalConfig() *kubeadmapi.InitConfiguration {
	externalcfg := &kubeadmapiv1alpha3.InitConfiguration{}
	internalcfg := &kubeadmapi.InitConfiguration{}

	kubeadmscheme.Scheme.Default(externalcfg)
	kubeadmscheme.Scheme.Convert(externalcfg, internalcfg, nil)

	// Default the embedded ComponentConfig structs
	componentconfigs.Known.Default(internalcfg)
	return internalcfg
}

// MarshalInitConfigurationToBytes marshals the internal InitConfiguration object to bytes. It writes the embedded
// ComponentConfiguration objects out as separate YAML documents
func MarshalInitConfigurationToBytes(cfg *kubeadmapi.InitConfiguration, gv schema.GroupVersion) ([]byte, error) {
	masterbytes, err := kubeadmutil.MarshalToYamlForCodecs(cfg, gv, kubeadmscheme.Codecs)
	if err != nil {
		return []byte{}, err
	}
	allFiles := [][]byte{masterbytes}
	componentConfigContent := map[string][]byte{}

	// If the specified groupversion is targeting the internal type, don't print the extra componentconfig YAML documents
	if gv.Version != runtime.APIVersionInternal {

		defaultedcfg := defaultedInternalConfig()

		for kind, registration := range componentconfigs.Known {
			// If the ComponentConfig struct for the current registration is nil, skip it when marshalling
			realobj, ok := registration.GetFromInternalConfig(cfg)
			if !ok {
				continue
			}

			defaultedobj, ok := registration.GetFromInternalConfig(defaultedcfg)
			// Invalid: The caller asked to not print the componentconfigs if defaulted, but defaultComponentConfigs() wasn't able to create default objects to use for reference
			if !ok {
				return []byte{}, fmt.Errorf("couldn't create a default componentconfig object")
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
