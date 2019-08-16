/*
Copyright 2019 The Kubernetes Authors.

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

	"github.com/pkg/errors"
	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	bootstraputil "k8s.io/cluster-bootstrap/token/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	kubeadmcmdoptions "k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	utilnet "k8s.io/utils/net"
)

// ValidateInitConfiguration validates an InitConfiguration object and collects all encountered errors
func ValidateInitConfiguration(c *kubeadm.InitConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateNodeRegistrationOptions(&c.NodeRegistration, field.NewPath("nodeRegistration"))...)
	allErrs = append(allErrs, ValidateBootstrapTokens(c.BootstrapTokens, field.NewPath("bootstrapTokens"))...)
	allErrs = append(allErrs, ValidateClusterConfiguration(&c.ClusterConfiguration)...)
	// TODO(Arvinderpal): update advertiseAddress validation for dual-stack once it's implemented.
	allErrs = append(allErrs, ValidateAPIEndpoint(&c.LocalAPIEndpoint, field.NewPath("localAPIEndpoint"))...)
	// TODO: Maybe validate that .CertificateKey is a valid hex encoded AES key
	return allErrs
}

// ValidateClusterConfiguration validates an ClusterConfiguration object and collects all encountered errors
func ValidateClusterConfiguration(c *kubeadm.ClusterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateNetworking(c, field.NewPath("networking"))...)
	allErrs = append(allErrs, ValidateAPIServer(&c.APIServer, field.NewPath("apiServer"))...)
	allErrs = append(allErrs, ValidateAbsolutePath(c.CertificatesDir, field.NewPath("certificatesDir"))...)
	allErrs = append(allErrs, ValidateFeatureGates(c.FeatureGates, field.NewPath("featureGates"))...)
	allErrs = append(allErrs, ValidateHostPort(c.ControlPlaneEndpoint, field.NewPath("controlPlaneEndpoint"))...)
	allErrs = append(allErrs, ValidateEtcd(&c.Etcd, field.NewPath("etcd"))...)
	allErrs = append(allErrs, componentconfigs.Known.Validate(c)...)
	return allErrs
}

// ValidateAPIServer validates a APIServer object and collects all encountered errors
func ValidateAPIServer(a *kubeadm.APIServer, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateCertSANs(a.CertSANs, fldPath.Child("certSANs"))...)
	return allErrs
}

// ValidateJoinConfiguration validates node configuration and collects all encountered errors
func ValidateJoinConfiguration(c *kubeadm.JoinConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateNodeRegistrationOptions(&c.NodeRegistration, field.NewPath("nodeRegistration"))...)
	allErrs = append(allErrs, ValidateJoinControlPlane(c.ControlPlane, field.NewPath("controlPlane"))...)

	if !filepath.IsAbs(c.CACertPath) || !strings.HasSuffix(c.CACertPath, ".crt") {
		allErrs = append(allErrs, field.Invalid(field.NewPath("caCertPath"), c.CACertPath, "the ca certificate path must be an absolute path"))
	}
	return allErrs
}

// ValidateJoinControlPlane validates joining control plane configuration and collects all encountered errors
func ValidateJoinControlPlane(c *kubeadm.JoinControlPlane, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if c != nil {
		allErrs = append(allErrs, ValidateAPIEndpoint(&c.LocalAPIEndpoint, fldPath.Child("localAPIEndpoint"))...)
		// TODO: Maybe validate that .CertificateKey is a valid hex encoded AES key
	}
	return allErrs
}

// ValidateNodeRegistrationOptions validates the NodeRegistrationOptions object
func ValidateNodeRegistrationOptions(nro *kubeadm.NodeRegistrationOptions, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(nro.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "--node-name or .nodeRegistration.name in the config file is a required value. It seems like this value couldn't be automatically detected in your environment, please specify the desired value using the CLI or config file."))
	} else {
		nameFldPath := fldPath.Child("name")
		for _, err := range validation.IsDNS1123Subdomain(nro.Name) {
			allErrs = append(allErrs, field.Invalid(nameFldPath, nro.Name, err))
		}
	}
	allErrs = append(allErrs, ValidateSocketPath(nro.CRISocket, fldPath.Child("criSocket"))...)
	// TODO: Maybe validate .Taints as well in the future using something like validateNodeTaints() in pkg/apis/core/validation
	return allErrs
}

// ValidateDiscovery validates discovery related configuration and collects all encountered errors
func ValidateDiscovery(d *kubeadm.Discovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if d.BootstrapToken == nil && d.File == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "bootstrapToken or file must be set"))
	}

	if d.BootstrapToken != nil && d.File != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "bootstrapToken and file cannot both be set"))
	}

	if d.BootstrapToken != nil {
		allErrs = append(allErrs, ValidateDiscoveryBootstrapToken(d.BootstrapToken, fldPath.Child("bootstrapToken"))...)
		allErrs = append(allErrs, ValidateToken(d.TLSBootstrapToken, fldPath.Child("tlsBootstrapToken"))...)
	}

	if d.File != nil {
		allErrs = append(allErrs, ValidateDiscoveryFile(d.File, fldPath.Child("file"))...)
		if len(d.TLSBootstrapToken) != 0 {
			allErrs = append(allErrs, ValidateToken(d.TLSBootstrapToken, fldPath.Child("tlsBootstrapToken"))...)
		}
	}

	return allErrs
}

// ValidateDiscoveryBootstrapToken validates bootstrap token discovery configuration
func ValidateDiscoveryBootstrapToken(b *kubeadm.BootstrapTokenDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(b.APIServerEndpoint) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "APIServerEndpoint is not set"))
	}

	if len(b.CACertHashes) == 0 && !b.UnsafeSkipCAVerification {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "using token-based discovery without caCertHashes can be unsafe. Set unsafeSkipCAVerification as true in your kubeadm config file or pass --discovery-token-unsafe-skip-ca-verification flag to continue"))
	}

	allErrs = append(allErrs, ValidateToken(b.Token, fldPath.Child(kubeadmcmdoptions.TokenStr))...)
	allErrs = append(allErrs, ValidateDiscoveryTokenAPIServer(b.APIServerEndpoint, fldPath.Child("apiServerEndpoint"))...)

	return allErrs
}

// ValidateDiscoveryFile validates file discovery configuration
func ValidateDiscoveryFile(f *kubeadm.FileDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, ValidateDiscoveryKubeConfigPath(f.KubeConfigPath, fldPath.Child("kubeConfigPath"))...)

	return allErrs
}

// ValidateDiscoveryTokenAPIServer validates discovery token for API server
func ValidateDiscoveryTokenAPIServer(apiServer string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	_, _, err := net.SplitHostPort(apiServer)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, apiServer, err.Error()))
	}
	return allErrs
}

// ValidateDiscoveryKubeConfigPath validates location of a discovery file
func ValidateDiscoveryKubeConfigPath(discoveryFile string, fldPath *field.Path) field.ErrorList {
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
		allErrs = append(allErrs, ValidateToken(bt.Token.String(), btPath.Child(kubeadmcmdoptions.TokenStr))...)
		allErrs = append(allErrs, ValidateTokenUsages(bt.Usages, btPath.Child(kubeadmcmdoptions.TokenUsages))...)
		allErrs = append(allErrs, ValidateTokenGroups(bt.Usages, bt.Groups, btPath.Child(kubeadmcmdoptions.TokenGroups))...)

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
		if errs := validation.IsDNS1123Subdomain(altname); len(errs) != 0 {
			if errs2 := validation.IsWildcardDNS1123Subdomain(altname); len(errs2) != 0 {
				if net.ParseIP(altname) == nil {
					allErrs = append(allErrs, field.Invalid(fldPath, altname, fmt.Sprintf("altname is not a valid IP address, DNS label or a DNS label with subdomain wildcards: %s; %s", strings.Join(errs, "; "), strings.Join(errs2, "; "))))
				}
			}
		}
	}
	return allErrs
}

// ValidateURLs validates the URLs given in the string slice, makes sure they are parsable. Optionally, it can enforces HTTPS usage.
func ValidateURLs(urls []string, requireHTTPS bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, urlstr := range urls {
		u, err := url.Parse(urlstr)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, urlstr, fmt.Sprintf("URL parse error: %v", err)))
			continue
		}
		if requireHTTPS && u.Scheme != "https" {
			allErrs = append(allErrs, field.Invalid(fldPath, urlstr, "the URL must be using the HTTPS scheme"))
		}
		if u.Scheme == "" {
			allErrs = append(allErrs, field.Invalid(fldPath, urlstr, "the URL without scheme is not allowed"))
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
func ValidateIPNetFromString(subnetStr string, minAddrs int64, isDualStack bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if isDualStack {
		subnets, err := utilnet.ParseCIDRs(strings.Split(subnetStr, ","))
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, err.Error()))
		} else {
			areDualStackCIDRs, err := utilnet.IsDualStackCIDRs(subnets)
			if err != nil {
				allErrs = append(allErrs, field.Invalid(fldPath, subnets, err.Error()))
			} else if !areDualStackCIDRs {
				allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, "expected at least one IP from each family (v4 or v6) for dual-stack networking"))
			}
			for _, s := range subnets {
				numAddresses := ipallocator.RangeSize(s)
				if numAddresses < minAddrs {
					allErrs = append(allErrs, field.Invalid(fldPath, s, "subnet is too small"))
				}
			}
		}
	} else {
		_, svcSubnet, err := net.ParseCIDR(subnetStr)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, "couldn't parse subnet"))
			return allErrs
		}
		numAddresses := ipallocator.RangeSize(svcSubnet)
		if numAddresses < minAddrs {
			allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, "subnet is too small"))
		}
	}
	return allErrs
}

// ValidateNetworking validates networking configuration
func ValidateNetworking(c *kubeadm.ClusterConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	dnsDomainFldPath := field.NewPath("dnsDomain")
	for _, err := range validation.IsDNS1123Subdomain(c.Networking.DNSDomain) {
		allErrs = append(allErrs, field.Invalid(dnsDomainFldPath, c.Networking.DNSDomain, err))
	}
	// check if dual-stack feature-gate is enabled
	isDualStack := features.Enabled(c.FeatureGates, features.IPv6DualStack)
	// TODO(Arvinderpal): use isDualStack flag once list of service CIDRs is supported (PR: #79386)
	allErrs = append(allErrs, ValidateIPNetFromString(c.Networking.ServiceSubnet, constants.MinimumAddressesInServiceSubnet, false /*isDualStack*/, field.NewPath("serviceSubnet"))...)
	if len(c.Networking.PodSubnet) != 0 {
		allErrs = append(allErrs, ValidateIPNetFromString(c.Networking.PodSubnet, constants.MinimumAddressesInServiceSubnet, isDualStack, field.NewPath("podSubnet"))...)
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
		if isAllowedFlag(f.Name) {
			// "--skip-*" flags or other allowed flags can be set with --config
			return
		}
		mixedInvalidFlags = append(mixedInvalidFlags, f.Name)
	})

	if len(mixedInvalidFlags) != 0 {
		return errors.Errorf("can not mix '--config' with arguments %v", mixedInvalidFlags)
	}
	return nil
}

func isAllowedFlag(flagName string) bool {
	knownFlags := sets.NewString(kubeadmcmdoptions.CfgPath,
		kubeadmcmdoptions.IgnorePreflightErrors,
		kubeadmcmdoptions.DryRun,
		kubeadmcmdoptions.KubeconfigPath,
		kubeadmcmdoptions.NodeName,
		kubeadmcmdoptions.NodeCRISocket,
		kubeadmcmdoptions.KubeconfigDir,
		kubeadmcmdoptions.UploadCerts,
		"print-join-command", "rootfs", "v")
	if knownFlags.Has(flagName) {
		return true
	}
	return strings.HasPrefix(flagName, "skip-")
}

// ValidateFeatureGates validates provided feature gates
func ValidateFeatureGates(featureGates map[string]bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	// check valid feature names are provided
	for k := range featureGates {
		if !features.Supports(features.InitFeatureGates, k) {
			allErrs = append(allErrs, field.Invalid(fldPath, featureGates,
				fmt.Sprintf("%s is not a valid feature name.", k)))
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

// ValidateIgnorePreflightErrors validates duplicates in:
// - ignore-preflight-errors flag and
// - ignorePreflightErrors field in {Init,Join}Configuration files.
func ValidateIgnorePreflightErrors(ignorePreflightErrorsFromCLI, ignorePreflightErrorsFromConfigFile []string) (sets.String, error) {
	ignoreErrors := sets.NewString()
	allErrs := field.ErrorList{}

	for _, item := range ignorePreflightErrorsFromConfigFile {
		ignoreErrors.Insert(strings.ToLower(item)) // parameters are case insensitive
	}

	if ignoreErrors.Has("all") {
		// "all" is forbidden in config files. Administrators should use an
		// explicit list of errors they want to ignore, as it can be risky to
		// mask all errors in such a way. Hence, we return an error:
		allErrs = append(allErrs, field.Invalid(field.NewPath("ignorePreflightErrors"), "all", "'all' cannot be used in configuration file"))
	}

	for _, item := range ignorePreflightErrorsFromCLI {
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
		return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("URL parsing error: %v", err)))
	}

	if u.Scheme == "" {
		if !filepath.IsAbs(u.Path) {
			return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("path is not absolute: %s", socket)))
		}
	} else if u.Scheme != kubeadmapiv1beta2.DefaultUrlScheme {
		return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("URL scheme %s is not supported", u.Scheme)))
	}

	return allErrs
}
