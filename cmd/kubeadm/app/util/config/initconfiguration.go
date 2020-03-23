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
	"strconv"

	"github.com/pkg/errors"
	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	netutil "k8s.io/apimachinery/pkg/util/net"
	bootstraputil "k8s.io/cluster-bootstrap/token/util"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/config/strict"
	kubeadmruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
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
	return SetClusterDynamicDefaults(&cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint)
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
func SetNodeRegistrationDynamicDefaults(cfg *kubeadmapi.NodeRegistrationOptions, ControlPlaneTaint bool) error {
	var err error
	cfg.Name, err = kubeadmutil.GetHostname(cfg.Name)
	if err != nil {
		return err
	}

	// Only if the slice is nil, we should append the control-plane taint. This allows the user to specify an empty slice for no default control-plane taint
	if ControlPlaneTaint && cfg.Taints == nil {
		cfg.Taints = []v1.Taint{kubeadmconstants.ControlPlaneTaint}
	}

	if cfg.CRISocket == "" {
		cfg.CRISocket, err = kubeadmruntime.DetectCRISocket()
		if err != nil {
			return err
		}
		klog.V(1).Infof("detected and using CRI socket: %s", cfg.CRISocket)
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

	// kubeadm allows users to specify address=Loopback as a selector for global unicast IP address that can be found on loopback interface.
	// e.g. This is required for network setups where default routes are present, but network interfaces use only link-local addresses (e.g. as described in RFC5549).
	if addressIP.IsLoopback() {
		loopbackIP, err := netutil.ChooseBindAddressForInterface(netutil.LoopbackInterfaceName)
		if err != nil {
			return err
		}
		if loopbackIP != nil {
			klog.V(4).Infof("Found active IP %v on loopback interface", loopbackIP.String())
			cfg.AdvertiseAddress = loopbackIP.String()
			return nil
		}
		return errors.New("unable to resolve link-local addresses")
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
func SetClusterDynamicDefaults(cfg *kubeadmapi.ClusterConfiguration, LocalAPIEndpoint *kubeadmapi.APIEndpoint) error {
	// Default all the embedded ComponentConfig structs
	componentconfigs.Default(cfg, LocalAPIEndpoint)

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
			cfg.ControlPlaneEndpoint = net.JoinHostPort(host, strconv.FormatInt(int64(LocalAPIEndpoint.BindPort), 10))
		}
	}

	// Downcase SANs. Some domain names (like ELBs) have capitals in them.
	LowercaseSANs(cfg.APIServer.CertSANs)
	return nil
}

// DefaultedInitConfiguration takes a versioned init config (often populated by flags), defaults it and converts it into internal InitConfiguration
func DefaultedInitConfiguration(versionedInitCfg *kubeadmapiv1beta2.InitConfiguration, versionedClusterCfg *kubeadmapiv1beta2.ClusterConfiguration) (*kubeadmapi.InitConfiguration, error) {
	internalcfg := &kubeadmapi.InitConfiguration{}

	// Takes passed flags into account; the defaulting is executed once again enforcing assignment of
	// static default values to cfg only for values not provided with flags
	kubeadmscheme.Scheme.Default(versionedInitCfg)
	if err := kubeadmscheme.Scheme.Convert(versionedInitCfg, internalcfg, nil); err != nil {
		return nil, err
	}

	kubeadmscheme.Scheme.Default(versionedClusterCfg)
	if err := kubeadmscheme.Scheme.Convert(versionedClusterCfg, &internalcfg.ClusterConfiguration, nil); err != nil {
		return nil, err
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

// LoadInitConfigurationFromFile loads a supported versioned InitConfiguration from a file, converts it into internal config, defaults it and verifies it.
func LoadInitConfigurationFromFile(cfgPath string) (*kubeadmapi.InitConfiguration, error) {
	klog.V(1).Infof("loading configuration from %q", cfgPath)

	b, err := ioutil.ReadFile(cfgPath)
	if err != nil {
		return nil, errors.Wrapf(err, "unable to read config from %q ", cfgPath)
	}

	return BytesToInitConfiguration(b)
}

// LoadOrDefaultInitConfiguration takes a path to a config file and a versioned configuration that can serve as the default config
// If cfgPath is specified, the versioned configs will always get overridden with the one in the file (specified by cfgPath).
// The external, versioned configuration is defaulted and converted to the internal type.
// Right thereafter, the configuration is defaulted again with dynamic values (like IP addresses of a machine, etc)
// Lastly, the internal config is validated and returned.
func LoadOrDefaultInitConfiguration(cfgPath string, versionedInitCfg *kubeadmapiv1beta2.InitConfiguration, versionedClusterCfg *kubeadmapiv1beta2.ClusterConfiguration) (*kubeadmapi.InitConfiguration, error) {
	if cfgPath != "" {
		// Loads configuration from config file, if provided
		// Nb. --config overrides command line flags
		return LoadInitConfigurationFromFile(cfgPath)
	}

	return DefaultedInitConfiguration(versionedInitCfg, versionedClusterCfg)
}

// BytesToInitConfiguration converts a byte slice to an internal, defaulted and validated InitConfiguration object.
// The map may contain many different YAML documents. These YAML documents are parsed one-by-one
// and well-known ComponentConfig GroupVersionKinds are stored inside of the internal InitConfiguration struct.
// The resulting InitConfiguration is then dynamically defaulted and validated prior to return.
func BytesToInitConfiguration(b []byte) (*kubeadmapi.InitConfiguration, error) {
	gvkmap, err := kubeadmutil.SplitYAMLDocuments(b)
	if err != nil {
		return nil, err
	}

	return documentMapToInitConfiguration(gvkmap, false)
}

// documentMapToInitConfiguration converts a map of GVKs and YAML documents to defaulted and validated configuration object.
func documentMapToInitConfiguration(gvkmap kubeadmapi.DocumentMap, allowDeprecated bool) (*kubeadmapi.InitConfiguration, error) {
	var initcfg *kubeadmapi.InitConfiguration
	var clustercfg *kubeadmapi.ClusterConfiguration

	for gvk, fileContent := range gvkmap {
		// first, check if this GVK is supported and possibly not deprecated
		if err := validateSupportedVersion(gvk.GroupVersion(), allowDeprecated); err != nil {
			return nil, err
		}

		// verify the validity of the YAML
		strict.VerifyUnmarshalStrict(fileContent, gvk)

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

		// If the group is neither a kubeadm core type or of a supported component config group, we dump a warning about it being ignored
		if !componentconfigs.Scheme.IsGroupRegistered(gvk.Group) {
			fmt.Printf("[config] WARNING: Ignored YAML document with GroupVersionKind %v\n", gvk)
		}
	}

	// Enforce that InitConfiguration and/or ClusterConfiguration has to exist among the YAML documents
	if initcfg == nil && clustercfg == nil {
		return nil, errors.New("no InitConfiguration or ClusterConfiguration kind was found in the YAML file")
	}

	// If InitConfiguration wasn't given, default it by creating an external struct instance, default it and convert into the internal type
	if initcfg == nil {
		extinitcfg := &kubeadmapiv1beta2.InitConfiguration{}
		kubeadmscheme.Scheme.Default(extinitcfg)
		// Set initcfg to an empty struct value the deserializer will populate
		initcfg = &kubeadmapi.InitConfiguration{}
		if err := kubeadmscheme.Scheme.Convert(extinitcfg, initcfg, nil); err != nil {
			return nil, err
		}
	}
	// If ClusterConfiguration was given, populate it in the InitConfiguration struct
	if clustercfg != nil {
		initcfg.ClusterConfiguration = *clustercfg
	}

	// Load any component configs
	if err := componentconfigs.FetchFromDocumentMap(&initcfg.ClusterConfiguration, gvkmap); err != nil {
		return nil, err
	}

	// Applies dynamic defaults to settings not provided with flags
	if err := SetInitDynamicDefaults(initcfg); err != nil {
		return nil, err
	}

	// Validates cfg (flags/configs + defaults + dynamic defaults)
	if err := validation.ValidateInitConfiguration(initcfg).ToAggregate(); err != nil {
		return nil, err
	}

	return initcfg, nil
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
		clusterbytes, err := kubeadmutil.MarshalToYamlForCodecs(&cfg.ClusterConfiguration, gv, kubeadmscheme.Codecs)
		if err != nil {
			return []byte{}, err
		}
		allFiles = append(allFiles, clusterbytes)
	}
	return bytes.Join(allFiles, []byte(kubeadmconstants.YAMLDocumentSeparator)), nil
}
