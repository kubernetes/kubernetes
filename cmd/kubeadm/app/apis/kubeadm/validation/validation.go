/*
Copyright 2016 The Kubernetes Authors.

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

package validation

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	bootstrapapi "k8s.io/client-go/tools/bootstrap/token/api"
	bootstraputil "k8s.io/client-go/tools/bootstrap/token/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// ValidateInitConfiguration validates an InitConfiguration object and collects all encountered errors
func ValidateInitConfiguration(c *kubeadm.InitConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateNodeRegistrationOptions(&c.NodeRegistration, field.NewPath("nodeRegistration"))...)
	allErrs = append(allErrs, ValidateBootstrapTokens(c.BootstrapTokens, field.NewPath("bootstrapTokens"))...)
	allErrs = append(allErrs, ValidateClusterConfiguration(&c.ClusterConfiguration)...)
	allErrs = append(allErrs, ValidateAPIEndpoint(&c.APIEndpoint, field.NewPath("apiEndpoint"))...)
	return allErrs
}

// ValidateClusterConfiguration validates an ClusterConfiguration object and collects all encountered errors
func ValidateClusterConfiguration(c *kubeadm.ClusterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateNetworking(&c.Networking, field.NewPath("networking"))...)
	allErrs = append(allErrs, ValidateCertSANs(c.APIServerCertSANs, field.NewPath("apiServerCertSANs"))...)
	allErrs = append(allErrs, ValidateAbsolutePath(c.CertificatesDir, field.NewPath("certificatesDir"))...)
	allErrs = append(allErrs, ValidateFeatureGates(c.FeatureGates, field.NewPath("featureGates"))...)
	allErrs = append(allErrs, ValidateHostPort(c.ControlPlaneEndpoint, field.NewPath("controlPlaneEndpoint"))...)
	allErrs = append(allErrs, ValidateEtcd(&c.Etcd, field.NewPath("etcd"))...)
	allErrs = append(allErrs, componentconfigs.Known.Validate(c)...)
	return allErrs
}

// ValidateJoinConfiguration validates node configuration and collects all encountered errors
func ValidateJoinConfiguration(c *kubeadm.JoinConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(c)...)
	allErrs = append(allErrs, ValidateNodeRegistrationOptions(&c.NodeRegistration, field.NewPath("nodeRegistration"))...)
	allErrs = append(allErrs, ValidateAPIEndpoint(&c.APIEndpoint, field.NewPath("apiEndpoint"))...)

	if !filepath.IsAbs(c.CACertPath) || !strings.HasSuffix(c.CACertPath, ".crt") {
		allErrs = append(allErrs, field.Invalid(field.NewPath("caCertPath"), c.CACertPath, "the ca certificate path must be an absolute path"))
	}
	return allErrs
}

// ValidateNodeRegistrationOptions validates the NodeRegistrationOptions object
func ValidateNodeRegistrationOptions(nro *kubeadm.NodeRegistrationOptions, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(nro.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "--node-name or .nodeRegistration.name in the config file is a required value. It seems like this value couldn't be automatically detected in your environment, please specify the desired value using the CLI or config file."))
	} else {
		allErrs = append(allErrs, apivalidation.ValidateDNS1123Subdomain(nro.Name, field.NewPath("name"))...)
	}
	allErrs = append(allErrs, ValidateSocketPath(nro.CRISocket, fldPath.Child("criSocket"))...)
	// TODO: Maybe validate .Taints as well in the future using something like validateNodeTaints() in pkg/apis/core/validation
	return allErrs
}

// ValidateDiscovery validates discovery related configuration and collects all encountered errors
func ValidateDiscovery(c *kubeadm.JoinConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(c.DiscoveryToken) != 0 {
		allErrs = append(allErrs, ValidateToken(c.DiscoveryToken, field.NewPath("discoveryToken"))...)
	}
	if len(c.DiscoveryFile) != 0 {
		allErrs = append(allErrs, ValidateDiscoveryFile(c.DiscoveryFile, field.NewPath("discoveryFile"))...)
		if len(c.TLSBootstrapToken) != 0 {
			allErrs = append(allErrs, ValidateToken(c.TLSBootstrapToken, field.NewPath("tlsBootstrapToken"))...)
		}
	} else {
		allErrs = append(allErrs, ValidateToken(c.TLSBootstrapToken, field.NewPath("tlsBootstrapToken"))...)
	}
	allErrs = append(allErrs, ValidateArgSelection(c, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateJoinDiscoveryTokenAPIServer(c.DiscoveryTokenAPIServers, field.NewPath("discoveryTokenAPIServers"))...)

	return allErrs
}

// ValidateArgSelection validates discovery related configuration and collects all encountered errors
func ValidateArgSelection(cfg *kubeadm.JoinConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.DiscoveryToken) != 0 && len(cfg.DiscoveryFile) != 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "discoveryToken and discoveryFile cannot both be set"))
	}
	if len(cfg.DiscoveryToken) == 0 && len(cfg.DiscoveryFile) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "discoveryToken or discoveryFile must be set"))
	}
	if len(cfg.DiscoveryTokenAPIServers) < 1 && len(cfg.DiscoveryToken) != 0 {
		allErrs = append(allErrs, field.Required(fldPath, "discoveryTokenAPIServers not set"))
	}

	if len(cfg.DiscoveryFile) != 0 && len(cfg.DiscoveryTokenCACertHashes) != 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "discoveryTokenCACertHashes cannot be used with discoveryFile"))
	}

	if len(cfg.DiscoveryFile) == 0 && len(cfg.DiscoveryToken) != 0 &&
		len(cfg.DiscoveryTokenCACertHashes) == 0 && !cfg.DiscoveryTokenUnsafeSkipCAVerification {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "using token-based discovery without discoveryTokenCACertHashes can be unsafe. set --discovery-token-unsafe-skip-ca-verification to continue"))
	}

	// TODO remove once we support multiple api servers
	if len(cfg.DiscoveryTokenAPIServers) > 1 {
		fmt.Println("[validation] WARNING: kubeadm doesn't fully support multiple API Servers yet")
	}
	return allErrs
}

// ValidateJoinDiscoveryTokenAPIServer validates discovery token for API server
func ValidateJoinDiscoveryTokenAPIServer(apiServers []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, m := range apiServers {
		_, _, err := net.SplitHostPort(m)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, m, err.Error()))
		}
	}
	return allErrs
}

// ValidateDiscoveryFile validates location of a discovery file
func ValidateDiscoveryFile(discoveryFile string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	u, err := url.Parse(discoveryFile)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, discoveryFile, "not a valid HTTPS URL or a file on disk"))
		return allErrs
	}

	if u.Scheme == "" {
		// URIs with no scheme should be treated as files
		if _, err := os.Stat(discoveryFile); os.IsNotExist(err) {
			allErrs = append(allErrs, field.Invalid(fldPath, discoveryFile, "not a valid HTTPS URL or a file on disk"))
		}
		return allErrs
	}

	if u.Scheme != "https" {
		allErrs = append(allErrs, field.Invalid(fldPath, discoveryFile, "if a URL is used, the scheme must be https"))
	}
	return allErrs
}

// ValidateBootstrapTokens validates a slice of BootstrapToken objects
func ValidateBootstrapTokens(bts []kubeadm.BootstrapToken, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for i, bt := range bts {
		btPath := fldPath.Child(fmt.Sprintf("%d", i))
		allErrs = append(allErrs, ValidateToken(bt.Token.String(), btPath.Child("token"))...)
		allErrs = append(allErrs, ValidateTokenUsages(bt.Usages, btPath.Child("usages"))...)
		allErrs = append(allErrs, ValidateTokenGroups(bt.Usages, bt.Groups, btPath.Child("groups"))...)

		if bt.Expires != nil && bt.TTL != nil {
			allErrs = append(allErrs, field.Invalid(btPath, "", "the BootstrapToken .TTL and .Expires fields are mutually exclusive"))
		}
	}
	return allErrs
}

// ValidateToken validates a Bootstrap Token
func ValidateToken(token string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if !bootstraputil.IsValidBootstrapToken(token) {
		allErrs = append(allErrs, field.Invalid(fldPath, token, "the bootstrap token is invalid"))
	}

	return allErrs
}

// ValidateTokenGroups validates token groups
func ValidateTokenGroups(usages []string, groups []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	// adding groups only makes sense for authentication
	usagesSet := sets.NewString(usages...)
	usageAuthentication := strings.TrimPrefix(bootstrapapi.BootstrapTokenUsageAuthentication, bootstrapapi.BootstrapTokenUsagePrefix)
	if len(groups) > 0 && !usagesSet.Has(usageAuthentication) {
		allErrs = append(allErrs, field.Invalid(fldPath, groups, fmt.Sprintf("token groups cannot be specified unless --usages includes %q", usageAuthentication)))
	}

	// validate any extra group names
	for _, group := range groups {
		if err := bootstraputil.ValidateBootstrapGroupName(group); err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, groups, err.Error()))
		}
	}

	return allErrs
}

// ValidateTokenUsages validates token usages
func ValidateTokenUsages(usages []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	// validate usages
	if err := bootstraputil.ValidateUsages(usages); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, usages, err.Error()))
	}

	return allErrs
}

// ValidateEtcd validates the .Etcd sub-struct.
func ValidateEtcd(e *kubeadm.Etcd, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	localPath := fldPath.Child("local")
	externalPath := fldPath.Child("external")

	if e.Local == nil && e.External == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "either .Etcd.Local or .Etcd.External is required"))
		return allErrs
	}
	if e.Local != nil && e.External != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, "", ".Etcd.Local and .Etcd.External are mutually exclusive"))
		return allErrs
	}
	if e.Local != nil {
		allErrs = append(allErrs, ValidateAbsolutePath(e.Local.DataDir, localPath.Child("dataDir"))...)
		allErrs = append(allErrs, ValidateCertSANs(e.Local.ServerCertSANs, localPath.Child("serverCertSANs"))...)
		allErrs = append(allErrs, ValidateCertSANs(e.Local.PeerCertSANs, localPath.Child("peerCertSANs"))...)
	}
	if e.External != nil {
		requireHTTPS := true
		// Only allow the http scheme if no certs/keys are passed
		if e.External.CAFile == "" && e.External.CertFile == "" && e.External.KeyFile == "" {
			requireHTTPS = false
		}
		// Require either none or both of the cert/key pair
		if (e.External.CertFile == "" && e.External.KeyFile != "") || (e.External.CertFile != "" && e.External.KeyFile == "") {
			allErrs = append(allErrs, field.Invalid(externalPath, "", "either both or none of .Etcd.External.CertFile and .Etcd.External.KeyFile must be set"))
		}
		// If the cert and key are specified, require the VA as well
		if e.External.CertFile != "" && e.External.KeyFile != "" && e.External.CAFile == "" {
			allErrs = append(allErrs, field.Invalid(externalPath, "", "setting .Etcd.External.CertFile and .Etcd.External.KeyFile requires .Etcd.External.CAFile"))
		}

		allErrs = append(allErrs, ValidateURLs(e.External.Endpoints, requireHTTPS, externalPath.Child("endpoints"))...)
		if e.External.CAFile != "" {
			allErrs = append(allErrs, ValidateAbsolutePath(e.External.CAFile, externalPath.Child("caFile"))...)
		}
		if e.External.CertFile != "" {
			allErrs = append(allErrs, ValidateAbsolutePath(e.External.CertFile, externalPath.Child("certFile"))...)
		}
		if e.External.KeyFile != "" {
			allErrs = append(allErrs, ValidateAbsolutePath(e.External.KeyFile, externalPath.Child("keyFile"))...)
		}
	}
	return allErrs
}

// ValidateCertSANs validates alternative names
func ValidateCertSANs(altnames []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, altname := range altnames {
		if len(validation.IsDNS1123Subdomain(altname)) != 0 && net.ParseIP(altname) == nil {
			allErrs = append(allErrs, field.Invalid(fldPath, altname, "altname is not a valid dns label or ip address"))
		}
	}
	return allErrs
}

// ValidateURLs validates the URLs given in the string slice, makes sure they are parseable. Optionally, it can enforcs HTTPS usage.
func ValidateURLs(urls []string, requireHTTPS bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, urlstr := range urls {
		u, err := url.Parse(urlstr)
		if err != nil || u.Scheme == "" {
			allErrs = append(allErrs, field.Invalid(fldPath, urlstr, "not a valid URL"))
		}
		if requireHTTPS && u.Scheme != "https" {
			allErrs = append(allErrs, field.Invalid(fldPath, urlstr, "the URL must be using the HTTPS scheme"))
		}
	}
	return allErrs
}

// ValidateIPFromString validates ip address
func ValidateIPFromString(ipaddr string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if net.ParseIP(ipaddr) == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, ipaddr, "ip address is not valid"))
	}
	return allErrs
}

// ValidatePort validates port numbers
func ValidatePort(port int32, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if _, err := kubeadmutil.ParsePort(strconv.Itoa(int(port))); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, port, "port number is not valid"))
	}
	return allErrs
}

// ValidateHostPort validates host[:port] endpoints
func ValidateHostPort(endpoint string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if _, _, err := kubeadmutil.ParseHostPort(endpoint); endpoint != "" && err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, endpoint, "endpoint is not valid"))
	}
	return allErrs
}

// ValidateIPNetFromString validates network portion of ip address
func ValidateIPNetFromString(subnet string, minAddrs int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	_, svcSubnet, err := net.ParseCIDR(subnet)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, subnet, "couldn't parse subnet"))
		return allErrs
	}
	numAddresses := ipallocator.RangeSize(svcSubnet)
	if numAddresses < minAddrs {
		allErrs = append(allErrs, field.Invalid(fldPath, subnet, "subnet is too small"))
	}
	return allErrs
}

// ValidateNetworking validates networking configuration
func ValidateNetworking(c *kubeadm.Networking, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateDNS1123Subdomain(c.DNSDomain, field.NewPath("dnsDomain"))...)
	allErrs = append(allErrs, ValidateIPNetFromString(c.ServiceSubnet, constants.MinimumAddressesInServiceSubnet, field.NewPath("serviceSubnet"))...)
	if len(c.PodSubnet) != 0 {
		allErrs = append(allErrs, ValidateIPNetFromString(c.PodSubnet, constants.MinimumAddressesInServiceSubnet, field.NewPath("podSubnet"))...)
	}
	return allErrs
}

// ValidateAbsolutePath validates whether provided path is absolute or not
func ValidateAbsolutePath(path string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !filepath.IsAbs(path) {
		allErrs = append(allErrs, field.Invalid(fldPath, path, "path is not absolute"))
	}
	return allErrs
}

// ValidateMixedArguments validates passed arguments
func ValidateMixedArguments(flag *pflag.FlagSet) error {
	// If --config isn't set, we have nothing to validate
	if !flag.Changed("config") {
		return nil
	}

	mixedInvalidFlags := []string{}
	flag.Visit(func(f *pflag.Flag) {
		if f.Name == "config" || f.Name == "ignore-preflight-errors" || strings.HasPrefix(f.Name, "skip-") || f.Name == "dry-run" || f.Name == "kubeconfig" || f.Name == "v" || f.Name == "rootfs" {
			// "--skip-*" flags or other whitelisted flags can be set with --config
			return
		}
		mixedInvalidFlags = append(mixedInvalidFlags, f.Name)
	})

	if len(mixedInvalidFlags) != 0 {
		return fmt.Errorf("can not mix '--config' with arguments %v", mixedInvalidFlags)
	}
	return nil
}

// ValidateFeatureGates validates provided feature gates
func ValidateFeatureGates(featureGates map[string]bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	validFeatures := features.Keys(features.InitFeatureGates)

	// check valid feature names are provided
	for k := range featureGates {
		if !features.Supports(features.InitFeatureGates, k) {
			allErrs = append(allErrs, field.Invalid(fldPath, featureGates,
				fmt.Sprintf("%s is not a valid feature name. Valid features are: %s", k, validFeatures)))
		}
	}

	return allErrs
}

// ValidateAPIEndpoint validates API server's endpoint
func ValidateAPIEndpoint(c *kubeadm.APIEndpoint, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateIPFromString(c.AdvertiseAddress, fldPath.Child("advertiseAddress"))...)
	allErrs = append(allErrs, ValidatePort(c.BindPort, fldPath.Child("bindPort"))...)
	return allErrs
}

// ValidateIgnorePreflightErrors validates duplicates in ignore-preflight-errors flag.
func ValidateIgnorePreflightErrors(ignorePreflightErrors []string) (sets.String, error) {
	ignoreErrors := sets.NewString()
	allErrs := field.ErrorList{}

	for _, item := range ignorePreflightErrors {
		ignoreErrors.Insert(strings.ToLower(item)) // parameters are case insensitive
	}

	if ignoreErrors.Has("all") && ignoreErrors.Len() > 1 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("ignore-preflight-errors"), strings.Join(ignoreErrors.List(), ","), "don't specify individual checks if 'all' is used"))
	}

	return ignoreErrors, allErrs.ToAggregate()
}

// ValidateSocketPath validates format of socket path or url
func ValidateSocketPath(socket string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	u, err := url.Parse(socket)
	if err != nil {
		return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("url parsing error: %v", err)))
	}

	if u.Scheme == "" {
		if !filepath.IsAbs(u.Path) {
			return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("path is not absolute: %s", socket)))
		}
	} else if u.Scheme != kubeadmapiv1alpha3.DefaultUrlScheme {
		return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("url scheme %s is not supported", u.Scheme)))
	}

	return allErrs
}
