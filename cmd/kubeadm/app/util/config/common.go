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
	"bytes"
	"fmt"
	"net"
	"reflect"
	"strings"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
	componentversion "k8s.io/component-base/version"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmscheme "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/scheme"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// MarshalKubeadmConfigObject marshals an Object registered in the kubeadm scheme. If the object is a InitConfiguration or ClusterConfiguration, some extra logic is run
func MarshalKubeadmConfigObject(obj runtime.Object) ([]byte, error) {
	switch internalcfg := obj.(type) {
	case *kubeadmapi.InitConfiguration:
		return MarshalInitConfigurationToBytes(internalcfg, kubeadmapiv1.SchemeGroupVersion)
	default:
		return kubeadmutil.MarshalToYamlForCodecs(obj, kubeadmapiv1.SchemeGroupVersion, kubeadmscheme.Codecs)
	}
}

// validateSupportedVersion checks if the supplied GroupVersion is not on the lists of old unsupported or deprecated GVs.
// If it is, an error is returned.
func validateSupportedVersion(gv schema.GroupVersion, allowDeprecated bool) error {
	// The support matrix will look something like this now and in the future:
	// v1.10 and earlier: v1alpha1
	// v1.11: v1alpha1 read-only, writes only v1alpha2 config
	// v1.12: v1alpha2 read-only, writes only v1alpha3 config. Errors if the user tries to use v1alpha1
	// v1.13: v1alpha3 read-only, writes only v1beta1 config. Errors if the user tries to use v1alpha1 or v1alpha2
	// v1.14: v1alpha3 convert only, writes only v1beta1 config. Errors if the user tries to use v1alpha1 or v1alpha2
	// v1.15: v1beta1 read-only, writes only v1beta2 config. Errors if the user tries to use v1alpha1, v1alpha2 or v1alpha3
	// v1.22: v1beta2 read-only, writes only v1beta3 config. Errors if the user tries to use v1beta1 and older
	oldKnownAPIVersions := map[string]string{
		"kubeadm.k8s.io/v1alpha1": "v1.11",
		"kubeadm.k8s.io/v1alpha2": "v1.12",
		"kubeadm.k8s.io/v1alpha3": "v1.14",
		"kubeadm.k8s.io/v1beta1":  "v1.15",
	}

	// Deprecated API versions are supported by us, but can only be used for migration.
	deprecatedAPIVersions := map[string]struct{}{
		"kubeadm.k8s.io/v1beta2": {},
	}

	gvString := gv.String()

	if useKubeadmVersion := oldKnownAPIVersions[gvString]; useKubeadmVersion != "" {
		return errors.Errorf("your configuration file uses an old API spec: %q. Please use kubeadm %s instead and run 'kubeadm config migrate --old-config old.yaml --new-config new.yaml', which will write the new, similar spec using a newer API version.", gv.String(), useKubeadmVersion)
	}

	if _, present := deprecatedAPIVersions[gvString]; present && !allowDeprecated {
		klog.Warningf("your configuration file uses a deprecated API spec: %q. Please use 'kubeadm config migrate --old-config old.yaml --new-config new.yaml', which will write the new, similar spec using a newer API version.", gv)
	}

	return nil
}

// NormalizeKubernetesVersion resolves version labels, sets alternative
// image registry if requested for CI builds, and validates minimal
// version that kubeadm SetInitDynamicDefaultssupports.
func NormalizeKubernetesVersion(cfg *kubeadmapi.ClusterConfiguration) error {
	isCIVersion := kubeadmutil.KubernetesIsCIVersion(cfg.KubernetesVersion)

	// Requested version is automatic CI build, thus use KubernetesCI Image Repository for core images
	if isCIVersion && cfg.ImageRepository == kubeadmapiv1.DefaultImageRepository {
		cfg.CIImageRepository = constants.DefaultCIImageRepository
	}

	// Parse and validate the version argument and resolve possible CI version labels
	ver, err := kubeadmutil.KubernetesReleaseVersion(cfg.KubernetesVersion)
	if err != nil {
		return err
	}

	// Requested version is automatic CI build, thus mark CIKubernetesVersion as `ci/<resolved-version>`
	if isCIVersion {
		cfg.CIKubernetesVersion = fmt.Sprintf("%s%s", constants.CIKubernetesVersionPrefix, ver)
	}

	cfg.KubernetesVersion = ver

	// Parse the given kubernetes version and make sure it's higher than the lowest supported
	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return errors.Wrapf(err, "couldn't parse Kubernetes version %q", cfg.KubernetesVersion)
	}

	// During the k8s release process, a kubeadm version in the main branch could be 1.23.0-pre,
	// while the 1.22.0 version is not released yet. The MinimumControlPlaneVersion validation
	// in such a case will not pass, since the value of MinimumControlPlaneVersion would be
	// calculated as kubeadm version - 1 (1.22) and k8sVersion would still be at 1.21.x
	// (fetched from the 'stable' marker). Handle this case by only showing a warning.
	mcpVersion := constants.MinimumControlPlaneVersion
	versionInfo := componentversion.Get()
	if isKubeadmPrereleaseVersion(&versionInfo, k8sVersion, mcpVersion) {
		klog.V(1).Infof("WARNING: tolerating control plane version %s, assuming that k8s version %s is not released yet",
			cfg.KubernetesVersion, mcpVersion)
		return nil
	}
	// If not a pre-release version, handle the validation normally.
	if k8sVersion.LessThan(mcpVersion) {
		return errors.Errorf("this version of kubeadm only supports deploying clusters with the control plane version >= %s. Current version: %s",
			mcpVersion, cfg.KubernetesVersion)
	}
	return nil
}

// LowercaseSANs can be used to force all SANs to be lowercase so it passes IsDNS1123Subdomain
func LowercaseSANs(sans []string) {
	for i, san := range sans {
		lowercase := strings.ToLower(san)
		if lowercase != san {
			klog.V(1).Infof("lowercasing SAN %q to %q", san, lowercase)
			sans[i] = lowercase
		}
	}
}

// VerifyAPIServerBindAddress can be used to verify if a bind address for the API Server is 0.0.0.0,
// in which case this address is not valid and should not be used.
func VerifyAPIServerBindAddress(address string) error {
	ip := netutils.ParseIPSloppy(address)
	if ip == nil {
		return errors.Errorf("cannot parse IP address: %s", address)
	}
	// There are users with network setups where default routes are present, but network interfaces
	// use only link-local addresses (e.g. as described in RFC5549).
	// In many cases that matching global unicast IP address can be found on loopback interface,
	// so kubeadm allows users to specify address=Loopback for handling supporting the scenario above.
	// Nb. SetAPIEndpointDynamicDefaults will try to translate loopback to a valid address afterwards
	if ip.IsLoopback() {
		return nil
	}
	if !ip.IsGlobalUnicast() {
		return errors.Errorf("cannot use %q as the bind address for the API Server", address)
	}
	return nil
}

// ChooseAPIServerBindAddress is a wrapper for netutil.ResolveBindAddress that also handles
// the case where no default routes were found and an IP for the API server could not be obtained.
func ChooseAPIServerBindAddress(bindAddress net.IP) (net.IP, error) {
	ip, err := netutil.ResolveBindAddress(bindAddress)
	if err != nil {
		if netutil.IsNoRoutesError(err) {
			klog.Warningf("WARNING: could not obtain a bind address for the API Server: %v; using: %s", err, constants.DefaultAPIServerBindAddress)
			defaultIP := netutils.ParseIPSloppy(constants.DefaultAPIServerBindAddress)
			if defaultIP == nil {
				return nil, errors.Errorf("cannot parse default IP address: %s", constants.DefaultAPIServerBindAddress)
			}
			return defaultIP, nil
		}
		return nil, err
	}
	if bindAddress != nil && !bindAddress.IsUnspecified() && !reflect.DeepEqual(ip, bindAddress) {
		klog.Warningf("WARNING: overriding requested API server bind address: requested %q, actual %q", bindAddress, ip)
	}
	return ip, nil
}

// MigrateOldConfig migrates an old configuration from a byte slice into a new one (returned again as a byte slice).
// Only kubeadm kinds are migrated. Others are silently ignored.
func MigrateOldConfig(oldConfig []byte) ([]byte, error) {
	newConfig := [][]byte{}

	gvkmap, err := kubeadmutil.SplitYAMLDocuments(oldConfig)
	if err != nil {
		return []byte{}, err
	}

	gvks := []schema.GroupVersionKind{}
	for gvk := range gvkmap {
		gvks = append(gvks, gvk)
	}

	// Migrate InitConfiguration and ClusterConfiguration if there are any in the config
	if kubeadmutil.GroupVersionKindsHasInitConfiguration(gvks...) || kubeadmutil.GroupVersionKindsHasClusterConfiguration(gvks...) {
		o, err := documentMapToInitConfiguration(gvkmap, true)
		if err != nil {
			return []byte{}, err
		}
		b, err := MarshalKubeadmConfigObject(o)
		if err != nil {
			return []byte{}, err
		}
		newConfig = append(newConfig, b)
	}

	// Migrate JoinConfiguration if there is any
	if kubeadmutil.GroupVersionKindsHasJoinConfiguration(gvks...) {
		o, err := documentMapToJoinConfiguration(gvkmap, true)
		if err != nil {
			return []byte{}, err
		}
		b, err := MarshalKubeadmConfigObject(o)
		if err != nil {
			return []byte{}, err
		}
		newConfig = append(newConfig, b)
	}

	return bytes.Join(newConfig, []byte(constants.YAMLDocumentSeparator)), nil
}

// isKubeadmPrereleaseVersion returns true if the kubeadm version is a pre-release version and
// the minimum control plane version is N+2 MINOR version of the given k8sVersion.
func isKubeadmPrereleaseVersion(versionInfo *apimachineryversion.Info, k8sVersion, mcpVersion *version.Version) bool {
	if len(versionInfo.Major) != 0 { // Make sure the component version is populated
		kubeadmVersion := version.MustParseSemantic(versionInfo.String())
		if len(kubeadmVersion.PreRelease()) != 0 { // Only handle this if the kubeadm binary is a pre-release
			// After incrementing the k8s MINOR version by one, if this version is equal or greater than the
			// MCP version, return true.
			v := k8sVersion.WithMinor(k8sVersion.Minor() + 1)
			if comp, _ := v.Compare(mcpVersion.String()); comp != -1 {
				return true
			}
		}
	}
	return false
}
