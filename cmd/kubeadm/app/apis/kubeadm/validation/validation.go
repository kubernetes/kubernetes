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
	"encoding/hex"
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"

	"github.com/distribution/reference"
	"github.com/pkg/errors"
	"github.com/spf13/pflag"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	bootstraputil "k8s.io/cluster-bootstrap/token/util"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	kubeadmcmdoptions "k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// ValidateInitConfiguration validates an InitConfiguration object and collects all encountered errors
func ValidateInitConfiguration(c *kubeadm.InitConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateNodeRegistrationOptions(&c.NodeRegistration, field.NewPath("nodeRegistration"))...)
	allErrs = append(allErrs, ValidateBootstrapTokens(c.BootstrapTokens, field.NewPath("bootstrapTokens"))...)
	allErrs = append(allErrs, ValidateClusterConfiguration(&c.ClusterConfiguration)...)
	// TODO(Arvinderpal): update advertiseAddress validation for dual-stack once it's implemented.
	allErrs = append(allErrs, ValidateAPIEndpoint(&c.LocalAPIEndpoint, field.NewPath("localAPIEndpoint"))...)
	allErrs = append(allErrs, ValidateCertificateKey(c.CertificateKey, field.NewPath("certificateKey"))...)
	return allErrs
}

// ValidateClusterConfiguration validates an ClusterConfiguration object and collects all encountered errors
func ValidateClusterConfiguration(c *kubeadm.ClusterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDNS(&c.DNS, field.NewPath("dns"))...)
	allErrs = append(allErrs, ValidateNetworking(c, field.NewPath("networking"))...)
	allErrs = append(allErrs, ValidateAPIServer(&c.APIServer, field.NewPath("apiServer"))...)
	allErrs = append(allErrs, ValidateControllerManager(&c.ControllerManager, field.NewPath("controllerManager"))...)
	allErrs = append(allErrs, ValidateScheduler(&c.Scheduler, field.NewPath("scheduler"))...)
	allErrs = append(allErrs, ValidateAbsolutePath(c.CertificatesDir, field.NewPath("certificatesDir"))...)
	allErrs = append(allErrs, ValidateFeatureGates(c.FeatureGates, field.NewPath("featureGates"))...)
	allErrs = append(allErrs, ValidateHostPort(c.ControlPlaneEndpoint, field.NewPath("controlPlaneEndpoint"))...)
	allErrs = append(allErrs, ValidateImageRepository(c.ImageRepository, field.NewPath("imageRepository"))...)
	allErrs = append(allErrs, ValidateEtcd(&c.Etcd, field.NewPath("etcd"))...)
	allErrs = append(allErrs, ValidateEncryptionAlgorithm(c.EncryptionAlgorithm, field.NewPath("encryptionAlgorithm"))...)
	allErrs = append(allErrs, componentconfigs.Validate(c)...)
	return allErrs
}

// ValidateAPIServer validates a APIServer object and collects all encountered errors
func ValidateAPIServer(a *kubeadm.APIServer, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateCertSANs(a.CertSANs, fldPath.Child("certSANs"))...)
	allErrs = append(allErrs, ValidateExtraArgs(a.ExtraArgs, fldPath.Child("extraArgs"))...)
	return allErrs
}

// ValidateControllerManager validates the controller manager object and collects all encountered errors
func ValidateControllerManager(a *kubeadm.ControlPlaneComponent, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateExtraArgs(a.ExtraArgs, fldPath.Child("extraArgs"))...)
	return allErrs
}

// ValidateScheduler validates the scheduler object and collects all encountered errors
func ValidateScheduler(a *kubeadm.ControlPlaneComponent, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateExtraArgs(a.ExtraArgs, fldPath.Child("extraArgs"))...)
	return allErrs
}

// ValidateJoinConfiguration validates node configuration and collects all encountered errors
func ValidateJoinConfiguration(c *kubeadm.JoinConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateNodeRegistrationOptions(&c.NodeRegistration, field.NewPath("nodeRegistration"))...)
	allErrs = append(allErrs, ValidateJoinControlPlane(c.ControlPlane, field.NewPath("controlPlane"))...)

	if !isAbs(c.CACertPath) || !strings.HasSuffix(c.CACertPath, ".crt") {
		allErrs = append(allErrs, field.Invalid(field.NewPath("caCertPath"), c.CACertPath, "the ca certificate path must be an absolute path"))
	}
	return allErrs
}

// ValidateJoinControlPlane validates joining control plane configuration and collects all encountered errors
func ValidateJoinControlPlane(c *kubeadm.JoinControlPlane, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if c != nil {
		allErrs = append(allErrs, ValidateAPIEndpoint(&c.LocalAPIEndpoint, fldPath.Child("localAPIEndpoint"))...)
		allErrs = append(allErrs, ValidateCertificateKey(c.CertificateKey, fldPath.Child("certificateKey"))...)
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
	allErrs = append(allErrs, ValidateExtraArgs(nro.KubeletExtraArgs, fldPath.Child("kubeletExtraArgs"))...)
	allErrs = append(allErrs, ValidateImagePullPolicy(nro.ImagePullPolicy, fldPath.Child("imagePullPolicy"))...)
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
		allErrs = append(allErrs, field.Invalid(fldPath.Child("caCertHashes"), "", "using token-based discovery without caCertHashes can be unsafe. Set unsafeSkipCAVerification as true in your kubeadm config file or pass --discovery-token-unsafe-skip-ca-verification flag to continue"))
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
func ValidateBootstrapTokens(bts []bootstraptokenv1.BootstrapToken, fldPath *field.Path) field.ErrorList {
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
	usagesSet := sets.New(usages...)
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
		if len(e.Local.ImageRepository) > 0 {
			allErrs = append(allErrs, ValidateImageRepository(e.Local.ImageRepository, localPath.Child("imageRepository"))...)
		}
		allErrs = append(allErrs, ValidateExtraArgs(e.Local.ExtraArgs, localPath.Child("extraArgs"))...)
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
		// If the cert and key are specified, require the CA as well
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

// ValidateEncryptionAlgorithm validates the public key algorithm
func ValidateEncryptionAlgorithm(algo kubeadm.EncryptionAlgorithmType, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	knownAlgorithms := sets.New(
		kubeadm.EncryptionAlgorithmECDSAP256,
		kubeadm.EncryptionAlgorithmRSA2048,
		kubeadm.EncryptionAlgorithmRSA3072,
		kubeadm.EncryptionAlgorithmRSA4096,
	)
	if !knownAlgorithms.Has(algo) {
		msg := fmt.Sprintf("Invalid encryption algorithm %q. Must be one of %v", algo, sets.List(knownAlgorithms))
		allErrs = append(allErrs, field.Invalid(fldPath, algo, msg))
	}
	return allErrs
}

// ValidateCertSANs validates alternative names
func ValidateCertSANs(altnames []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, altname := range altnames {
		if errs := validation.IsDNS1123Subdomain(altname); len(errs) != 0 {
			if errs2 := validation.IsWildcardDNS1123Subdomain(altname); len(errs2) != 0 {
				if netutils.ParseIPSloppy(altname) == nil {
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
	if netutils.ParseIPSloppy(ipaddr) == nil {
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
func ValidateIPNetFromString(subnetStr string, minAddrs int64, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	subnets, err := netutils.ParseCIDRs(strings.Split(subnetStr, ","))
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, "couldn't parse subnet"))
		return allErrs
	}
	switch {
	// if DualStack only 2 CIDRs allowed
	case len(subnets) > 2:
		allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, "expected one (IPv4 or IPv6) CIDR or two CIDRs from each family for dual-stack networking"))
	// if DualStack and there are 2 CIDRs validate if there is at least one of each IP family
	case len(subnets) == 2:
		areDualStackCIDRs, err := netutils.IsDualStackCIDRs(subnets)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, err.Error()))
		} else if !areDualStackCIDRs {
			allErrs = append(allErrs, field.Invalid(fldPath, subnetStr, "expected one (IPv4 or IPv6) CIDR or two CIDRs from each family for dual-stack networking"))
		}
	}
	// validate the subnet/s
	for _, s := range subnets {
		numAddresses := netutils.RangeSize(s)
		if numAddresses < minAddrs {
			allErrs = append(allErrs, field.Invalid(fldPath, s.String(), fmt.Sprintf("subnet with %d address(es) is too small, the minimum is %d", numAddresses, minAddrs)))
		}

		// Warn when the subnet is in site-local range - i.e. contains addresses that belong to fec0::/10
		_, siteLocalNet, _ := netutils.ParseCIDRSloppy("fec0::/10")
		if siteLocalNet.Contains(s.IP) || s.Contains(siteLocalNet.IP) {
			klog.Warningf("the subnet %v contains IPv6 site-local addresses that belong to fec0::/10 which has been deprecated by rfc3879", s)
		}
	}
	return allErrs
}

// ValidateServiceSubnetSize validates that the maximum subnet size is not exceeded
// Should be a small cidr due to how it is stored in etcd.
// bigger cidr (specially those offered by IPv6) will add no value
// and significantly increase snapshotting time.
// NOTE: This is identical to validation performed in the apiserver.
func ValidateServiceSubnetSize(subnetStr string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// subnets were already validated
	subnets, _ := netutils.ParseCIDRs(strings.Split(subnetStr, ","))
	for _, serviceSubnet := range subnets {
		ones, bits := serviceSubnet.Mask.Size()
		if bits-ones > constants.MaximumBitsForServiceSubnet {
			errMsg := fmt.Sprintf("specified service subnet is too large; for %d-bit addresses, the mask must be >= %d", bits, bits-constants.MaximumBitsForServiceSubnet)
			allErrs = append(allErrs, field.Invalid(fldPath, serviceSubnet.String(), errMsg))
		}
	}
	return allErrs
}

// ValidatePodSubnetNodeMask validates that the relation between podSubnet and node-masks is correct
func ValidatePodSubnetNodeMask(subnetStr string, c *kubeadm.ClusterConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// subnets were already validated
	subnets, _ := netutils.ParseCIDRs(strings.Split(subnetStr, ","))
	for _, podSubnet := range subnets {
		// obtain podSubnet mask
		mask := podSubnet.Mask
		maskSize, _ := mask.Size()
		// obtain node-cidr-mask
		nodeMask, err := getClusterNodeMask(c, netutils.IsIPv6(podSubnet.IP))
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, podSubnet.String(), err.Error()))
			continue
		}
		// the pod subnet mask needs to allow one or multiple node-masks
		// i.e. if it has a /24 the node mask must be between 24 and 32 for ipv4
		if maskSize > nodeMask {
			allErrs = append(allErrs, field.Invalid(fldPath, podSubnet.String(), fmt.Sprintf("the size of pod subnet with mask %d is smaller than the size of node subnet with mask %d", maskSize, nodeMask)))
		} else if (nodeMask - maskSize) > constants.PodSubnetNodeMaskMaxDiff {
			allErrs = append(allErrs, field.Invalid(fldPath, podSubnet.String(), fmt.Sprintf("pod subnet mask (%d) and node-mask (%d) difference is greater than %d", maskSize, nodeMask, constants.PodSubnetNodeMaskMaxDiff)))
		}
	}
	return allErrs
}

// getClusterNodeMask returns the corresponding node-cidr-mask
// based on the Cluster configuration and the IP family
// Default is 24 for IPv4 and 64 for IPv6
func getClusterNodeMask(c *kubeadm.ClusterConfiguration, isIPv6 bool) (int, error) {
	// defaultNodeMaskCIDRIPv4 is default mask size for IPv4 node cidr for use by the controller manager
	const defaultNodeMaskCIDRIPv4 = 24
	// defaultNodeMaskCIDRIPv6 is default mask size for IPv6 node cidr for use by the controller manager
	const defaultNodeMaskCIDRIPv6 = 64
	var maskSize int
	var maskArg string
	var err error

	if isIPv6 {
		maskArg = "node-cidr-mask-size-ipv6"
	} else {
		maskArg = "node-cidr-mask-size-ipv4"
	}

	maskValue, _ := kubeadm.GetArgValue(c.ControllerManager.ExtraArgs, maskArg, -1)
	if len(maskValue) != 0 {
		// assume it is an integer, if not it will fail later
		maskSize, err = strconv.Atoi(maskValue)
		if err != nil {
			return 0, errors.Wrapf(err, "could not parse the value of the kube-controller-manager flag %s as an integer", maskArg)
		}
	} else if isIPv6 {
		maskSize = defaultNodeMaskCIDRIPv6
	} else {
		maskSize = defaultNodeMaskCIDRIPv4
	}
	return maskSize, nil
}

// ValidateDNS validates the DNS object and collects all encountered errors
func ValidateDNS(dns *kubeadm.DNS, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(dns.ImageRepository) > 0 {
		allErrs = append(allErrs, ValidateImageRepository(dns.ImageRepository, fldPath.Child("imageRepository"))...)
	}

	return allErrs
}

// ValidateNetworking validates networking configuration
func ValidateNetworking(c *kubeadm.ClusterConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	dnsDomainFldPath := fldPath.Child("dnsDomain")
	for _, err := range validation.IsDNS1123Subdomain(c.Networking.DNSDomain) {
		allErrs = append(allErrs, field.Invalid(dnsDomainFldPath, c.Networking.DNSDomain, err))
	}

	if len(c.Networking.ServiceSubnet) != 0 {
		allErrs = append(allErrs, ValidateIPNetFromString(c.Networking.ServiceSubnet, constants.MinimumAddressesInServiceSubnet, fldPath.Child("serviceSubnet"))...)
		// Service subnet was already validated, we need to validate now the subnet size
		allErrs = append(allErrs, ValidateServiceSubnetSize(c.Networking.ServiceSubnet, fldPath.Child("serviceSubnet"))...)
	}
	if len(c.Networking.PodSubnet) != 0 {
		allErrs = append(allErrs, ValidateIPNetFromString(c.Networking.PodSubnet, constants.MinimumAddressesInPodSubnet, fldPath.Child("podSubnet"))...)
		val, _ := kubeadm.GetArgValue(c.ControllerManager.ExtraArgs, "allocate-node-cidrs", -1)
		if val != "false" {
			// Pod subnet was already validated, we need to validate now against the node-mask
			allErrs = append(allErrs, ValidatePodSubnetNodeMask(c.Networking.PodSubnet, c, fldPath.Child("podSubnet"))...)
		}
	}
	return allErrs
}

// ValidateAbsolutePath validates whether provided path is absolute or not
func ValidateAbsolutePath(path string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !isAbs(path) {
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
	allowedFlags := sets.New(kubeadmcmdoptions.CfgPath,
		kubeadmcmdoptions.IgnorePreflightErrors,
		kubeadmcmdoptions.DryRun,
		kubeadmcmdoptions.KubeconfigPath,
		kubeadmcmdoptions.NodeName,
		kubeadmcmdoptions.KubeconfigDir,
		kubeadmcmdoptions.UploadCerts,
		"print-join-command", "rootfs", "v", "log-file")
	if allowedFlags.Has(flagName) {
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

// ValidateCertificateKey validates the certificate key is a valid hex encoded AES key
func ValidateCertificateKey(certificateKey string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(certificateKey) > 0 {
		decodedKey, err := hex.DecodeString(certificateKey)
		if err != nil {
			return append(allErrs, field.Invalid(fldPath, certificateKey, fmt.Sprintf("certificate key decoding error: %v", err)))
		}

		k := len(decodedKey)
		if k != constants.CertificateKeySize {
			allErrs = append(allErrs, field.Invalid(fldPath, certificateKey, fmt.Sprintf("invalid certificate key size %d, the key must be an AES key of size %d", k, constants.CertificateKeySize)))
		}
	}

	return allErrs
}

// ValidateIgnorePreflightErrors validates duplicates in:
// - ignore-preflight-errors flag and
// - ignorePreflightErrors field in {Init,Join}Configuration files.
func ValidateIgnorePreflightErrors(ignorePreflightErrorsFromCLI, ignorePreflightErrorsFromConfigFile []string) (sets.Set[string], error) {
	ignoreErrors := sets.New[string]()
	allErrs := field.ErrorList{}

	for _, item := range ignorePreflightErrorsFromConfigFile {
		ignoreErrors.Insert(strings.ToLower(item)) // parameters are case insensitive
	}

	for _, item := range ignorePreflightErrorsFromCLI {
		ignoreErrors.Insert(strings.ToLower(item)) // parameters are case insensitive
	}

	if ignoreErrors.Has("all") && ignoreErrors.Len() > 1 {
		allErrs = append(allErrs, field.Invalid(field.NewPath("ignore-preflight-errors"), strings.Join(sets.List(ignoreErrors), ","), "don't specify individual checks if 'all' is used"))
	}

	return ignoreErrors, allErrs.ToAggregate()
}

// ValidateSocketPath validates format of socket path or url
func ValidateSocketPath(socket string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(socket) == 0 { // static and dynamic defaulting should have added a value to the field already
		return append(allErrs, field.Invalid(fldPath, socket, "empty CRI socket"))
	}

	u, err := url.Parse(socket)
	if err != nil {
		return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("URL parsing error: %v", err)))
	}

	// static and dynamic defaulting should have ensured that an URL scheme is used
	if u.Scheme != kubeadmapiv1.DefaultContainerRuntimeURLScheme {
		return append(allErrs, field.Invalid(fldPath, socket, fmt.Sprintf("only URL scheme %q is supported, got %q", kubeadmapiv1.DefaultContainerRuntimeURLScheme, u.Scheme)))
	}

	return allErrs
}

// ValidateImageRepository validates the image repository format
func ValidateImageRepository(imageRepository string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	image := fmt.Sprintf("%s/%s:%s", imageRepository, "name", "tag")
	if !reference.ReferenceRegexp.MatchString(image) {
		return append(allErrs, field.Invalid(fldPath, imageRepository, "invalid image repository format"))
	}

	return allErrs
}

// ValidateResetConfiguration validates a ResetConfiguration object and collects all encountered errors
func ValidateResetConfiguration(c *kubeadm.ResetConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateSocketPath(c.CRISocket, field.NewPath("criSocket"))...)
	allErrs = append(allErrs, ValidateAbsolutePath(c.CertificatesDir, field.NewPath("certificatesDir"))...)
	allErrs = append(allErrs, ValidateUnmountFlags(c.UnmountFlags, field.NewPath("unmountFlags"))...)
	return allErrs
}

// ValidateExtraArgs validates a set of arguments and collects all encountered errors
func ValidateExtraArgs(args []kubeadm.Arg, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for idx, arg := range args {
		if len(arg.Name) == 0 {
			allErrs = append(allErrs, field.Invalid(fldPath, fmt.Sprintf("index %d", idx), "argument has no name"))
		}
	}

	return allErrs
}

// ValidateUnmountFlags validates a set of unmount flags and collects all encountered errors
func ValidateUnmountFlags(flags []string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	for idx, flag := range flags {
		switch flag {
		case kubeadm.UnmountFlagMNTForce, kubeadm.UnmountFlagMNTDetach, kubeadm.UnmountFlagMNTExpire, kubeadm.UnmountFlagUmountNoFollow:
			continue
		default:
			allErrs = append(allErrs, field.Invalid(fldPath, fmt.Sprintf("index %d", idx), fmt.Sprintf("unknown unmount flag %s", flag)))
		}
	}

	return allErrs
}

// ValidateImagePullPolicy validates if the user specified pull policy is correct
func ValidateImagePullPolicy(policy corev1.PullPolicy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch policy {
	case "", corev1.PullAlways, corev1.PullIfNotPresent, corev1.PullNever:
		return allErrs
	default:
		allErrs = append(allErrs, field.Invalid(fldPath, policy, "invalid pull policy"))
		return allErrs
	}
}

// ValidateUpgradeConfiguration validates a UpgradeConfiguration object and collects all encountered errors
func ValidateUpgradeConfiguration(c *kubeadm.UpgradeConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	if c.Apply.Patches != nil {
		allErrs = append(allErrs, ValidateAbsolutePath(c.Apply.Patches.Directory, field.NewPath("patches").Child("directory"))...)
	}
	if c.Node.Patches != nil {
		allErrs = append(allErrs, ValidateAbsolutePath(c.Node.Patches.Directory, field.NewPath("patches").Child("directory"))...)
	}
	return allErrs
}
