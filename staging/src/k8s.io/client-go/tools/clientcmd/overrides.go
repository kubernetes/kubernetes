/*
Copyright 2014 The Kubernetes Authors.

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

package clientcmd

import (
	"strconv"
	"strings"

	"github.com/spf13/pflag"

	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

// ConfigOverrides holds values that should override whatever information is pulled from the actual Config object.  You can't
// simply use an actual Config object, because Configs hold maps, but overrides are restricted to "at most one"
type ConfigOverrides struct {
	AuthInfo clientcmdapi.AuthInfo
	// ClusterDefaults are applied before the configured cluster info is loaded.
	ClusterDefaults clientcmdapi.Cluster
	ClusterInfo     clientcmdapi.Cluster
	Context         clientcmdapi.Context
	CurrentContext  string
	Timeout         string
}

// ConfigOverrideFlags holds the flag names to be used for binding command line flags. Notice that this structure tightly
// corresponds to ConfigOverrides
type ConfigOverrideFlags struct {
	AuthOverrideFlags    AuthOverrideFlags
	ClusterOverrideFlags ClusterOverrideFlags
	ContextOverrideFlags ContextOverrideFlags
	CurrentContext       FlagInfo
	Timeout              FlagInfo
}

// AuthOverrideFlags holds the flag names to be used for binding command line flags for AuthInfo objects
type AuthOverrideFlags struct {
	ClientCertificate FlagInfo
	ClientKey         FlagInfo
	Token             FlagInfo
	Impersonate       FlagInfo
	ImpersonateUID    FlagInfo
	ImpersonateGroups FlagInfo
	Username          FlagInfo
	Password          FlagInfo
}

// ContextOverrideFlags holds the flag names to be used for binding command line flags for Cluster objects
type ContextOverrideFlags struct {
	ClusterName  FlagInfo
	AuthInfoName FlagInfo
	Namespace    FlagInfo
}

// ClusterOverride holds the flag names to be used for binding command line flags for Cluster objects
type ClusterOverrideFlags struct {
	APIServer             FlagInfo
	APIVersion            FlagInfo
	CertificateAuthority  FlagInfo
	InsecureSkipTLSVerify FlagInfo
	TLSServerName         FlagInfo
	ProxyURL              FlagInfo
	DisableCompression    FlagInfo
}

// FlagInfo contains information about how to register a flag.  This struct is useful if you want to provide a way for an extender to
// get back a set of recommended flag names, descriptions, and defaults, but allow for customization by an extender.  This makes for
// coherent extension, without full prescription
type FlagInfo struct {
	// LongName is the long string for a flag.  If this is empty, then the flag will not be bound
	LongName string
	// ShortName is the single character for a flag.  If this is empty, then there will be no short flag
	ShortName string
	// Default is the default value for the flag
	Default string
	// Description is the description for the flag
	Description string
}

// AddSecretAnnotation add secret flag to Annotation.
func (f FlagInfo) AddSecretAnnotation(flags *pflag.FlagSet) FlagInfo {
	flags.SetAnnotation(f.LongName, "classified", []string{"true"})
	return f
}

// BindStringFlag binds the flag based on the provided info.  If LongName == "", nothing is registered
func (f FlagInfo) BindStringFlag(flags *pflag.FlagSet, target *string) FlagInfo {
	// you can't register a flag without a long name
	if len(f.LongName) > 0 {
		flags.StringVarP(target, f.LongName, f.ShortName, f.Default, f.Description)
	}
	return f
}

// BindTransformingStringFlag binds the flag based on the provided info.  If LongName == "", nothing is registered
func (f FlagInfo) BindTransformingStringFlag(flags *pflag.FlagSet, target *string, transformer func(string) (string, error)) FlagInfo {
	// you can't register a flag without a long name
	if len(f.LongName) > 0 {
		flags.VarP(newTransformingStringValue(f.Default, target, transformer), f.LongName, f.ShortName, f.Description)
	}
	return f
}

// BindStringSliceFlag binds the flag based on the provided info.  If LongName == "", nothing is registered
func (f FlagInfo) BindStringArrayFlag(flags *pflag.FlagSet, target *[]string) FlagInfo {
	// you can't register a flag without a long name
	if len(f.LongName) > 0 {
		sliceVal := []string{}
		if len(f.Default) > 0 {
			sliceVal = []string{f.Default}
		}
		flags.StringArrayVarP(target, f.LongName, f.ShortName, sliceVal, f.Description)
	}
	return f
}

// BindBoolFlag binds the flag based on the provided info.  If LongName == "", nothing is registered
func (f FlagInfo) BindBoolFlag(flags *pflag.FlagSet, target *bool) FlagInfo {
	// you can't register a flag without a long name
	if len(f.LongName) > 0 {
		// try to parse Default as a bool.  If it fails, assume false
		boolVal, err := strconv.ParseBool(f.Default)
		if err != nil {
			boolVal = false
		}

		flags.BoolVarP(target, f.LongName, f.ShortName, boolVal, f.Description)
	}
	return f
}

const (
	FlagClusterName        = "cluster"
	FlagAuthInfoName       = "user"
	FlagContext            = "context"
	FlagNamespace          = "namespace"
	FlagAPIServer          = "server"
	FlagTLSServerName      = "tls-server-name"
	FlagInsecure           = "insecure-skip-tls-verify"
	FlagCertFile           = "client-certificate"
	FlagKeyFile            = "client-key"
	FlagCAFile             = "certificate-authority"
	FlagEmbedCerts         = "embed-certs"
	FlagBearerToken        = "token"
	FlagImpersonate        = "as"
	FlagImpersonateUID     = "as-uid"
	FlagImpersonateGroup   = "as-group"
	FlagUsername           = "username"
	FlagPassword           = "password"
	FlagTimeout            = "request-timeout"
	FlagProxyURL           = "proxy-url"
	FlagDisableCompression = "disable-compression"
)

// RecommendedConfigOverrideFlags is a convenience method to return recommended flag names prefixed with a string of your choosing
func RecommendedConfigOverrideFlags(prefix string) ConfigOverrideFlags {
	return ConfigOverrideFlags{
		AuthOverrideFlags:    RecommendedAuthOverrideFlags(prefix),
		ClusterOverrideFlags: RecommendedClusterOverrideFlags(prefix),
		ContextOverrideFlags: RecommendedContextOverrideFlags(prefix),

		CurrentContext: FlagInfo{prefix + FlagContext, "", "", "The name of the kubeconfig context to use"},
		Timeout:        FlagInfo{prefix + FlagTimeout, "", "0", "The length of time to wait before giving up on a single server request. Non-zero values should contain a corresponding time unit (e.g. 1s, 2m, 3h). A value of zero means don't timeout requests."},
	}
}

// RecommendedAuthOverrideFlags is a convenience method to return recommended flag names prefixed with a string of your choosing
func RecommendedAuthOverrideFlags(prefix string) AuthOverrideFlags {
	return AuthOverrideFlags{
		ClientCertificate: FlagInfo{prefix + FlagCertFile, "", "", "Path to a client certificate file for TLS"},
		ClientKey:         FlagInfo{prefix + FlagKeyFile, "", "", "Path to a client key file for TLS"},
		Token:             FlagInfo{prefix + FlagBearerToken, "", "", "Bearer token for authentication to the API server"},
		Impersonate:       FlagInfo{prefix + FlagImpersonate, "", "", "Username to impersonate for the operation"},
		ImpersonateUID:    FlagInfo{prefix + FlagImpersonateUID, "", "", "UID to impersonate for the operation"},
		ImpersonateGroups: FlagInfo{prefix + FlagImpersonateGroup, "", "", "Group to impersonate for the operation, this flag can be repeated to specify multiple groups."},
		Username:          FlagInfo{prefix + FlagUsername, "", "", "Username for basic authentication to the API server"},
		Password:          FlagInfo{prefix + FlagPassword, "", "", "Password for basic authentication to the API server"},
	}
}

// RecommendedClusterOverrideFlags is a convenience method to return recommended flag names prefixed with a string of your choosing
func RecommendedClusterOverrideFlags(prefix string) ClusterOverrideFlags {
	return ClusterOverrideFlags{
		APIServer:             FlagInfo{prefix + FlagAPIServer, "", "", "The address and port of the Kubernetes API server"},
		CertificateAuthority:  FlagInfo{prefix + FlagCAFile, "", "", "Path to a cert file for the certificate authority"},
		InsecureSkipTLSVerify: FlagInfo{prefix + FlagInsecure, "", "false", "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure"},
		TLSServerName:         FlagInfo{prefix + FlagTLSServerName, "", "", "If provided, this name will be used to validate server certificate. If this is not provided, hostname used to contact the server is used."},
		ProxyURL:              FlagInfo{prefix + FlagProxyURL, "", "", "If provided, this URL will be used to connect via proxy"},
		DisableCompression:    FlagInfo{prefix + FlagDisableCompression, "", "", "If true, opt-out of response compression for all requests to the server"},
	}
}

// RecommendedContextOverrideFlags is a convenience method to return recommended flag names prefixed with a string of your choosing
func RecommendedContextOverrideFlags(prefix string) ContextOverrideFlags {
	return ContextOverrideFlags{
		ClusterName:  FlagInfo{prefix + FlagClusterName, "", "", "The name of the kubeconfig cluster to use"},
		AuthInfoName: FlagInfo{prefix + FlagAuthInfoName, "", "", "The name of the kubeconfig user to use"},
		Namespace:    FlagInfo{prefix + FlagNamespace, "n", "", "If present, the namespace scope for this CLI request"},
	}
}

// BindOverrideFlags is a convenience method to bind the specified flags to their associated variables
func BindOverrideFlags(overrides *ConfigOverrides, flags *pflag.FlagSet, flagNames ConfigOverrideFlags) {
	BindAuthInfoFlags(&overrides.AuthInfo, flags, flagNames.AuthOverrideFlags)
	BindClusterFlags(&overrides.ClusterInfo, flags, flagNames.ClusterOverrideFlags)
	BindContextFlags(&overrides.Context, flags, flagNames.ContextOverrideFlags)
	flagNames.CurrentContext.BindStringFlag(flags, &overrides.CurrentContext)
	flagNames.Timeout.BindStringFlag(flags, &overrides.Timeout)
}

// BindAuthInfoFlags is a convenience method to bind the specified flags to their associated variables
func BindAuthInfoFlags(authInfo *clientcmdapi.AuthInfo, flags *pflag.FlagSet, flagNames AuthOverrideFlags) {
	flagNames.ClientCertificate.BindStringFlag(flags, &authInfo.ClientCertificate).AddSecretAnnotation(flags)
	flagNames.ClientKey.BindStringFlag(flags, &authInfo.ClientKey).AddSecretAnnotation(flags)
	flagNames.Token.BindStringFlag(flags, &authInfo.Token).AddSecretAnnotation(flags)
	flagNames.Impersonate.BindStringFlag(flags, &authInfo.Impersonate).AddSecretAnnotation(flags)
	flagNames.ImpersonateUID.BindStringFlag(flags, &authInfo.ImpersonateUID).AddSecretAnnotation(flags)
	flagNames.ImpersonateGroups.BindStringArrayFlag(flags, &authInfo.ImpersonateGroups).AddSecretAnnotation(flags)
	flagNames.Username.BindStringFlag(flags, &authInfo.Username).AddSecretAnnotation(flags)
	flagNames.Password.BindStringFlag(flags, &authInfo.Password).AddSecretAnnotation(flags)
}

// BindClusterFlags is a convenience method to bind the specified flags to their associated variables
func BindClusterFlags(clusterInfo *clientcmdapi.Cluster, flags *pflag.FlagSet, flagNames ClusterOverrideFlags) {
	flagNames.APIServer.BindStringFlag(flags, &clusterInfo.Server)
	flagNames.CertificateAuthority.BindStringFlag(flags, &clusterInfo.CertificateAuthority)
	flagNames.InsecureSkipTLSVerify.BindBoolFlag(flags, &clusterInfo.InsecureSkipTLSVerify)
	flagNames.TLSServerName.BindStringFlag(flags, &clusterInfo.TLSServerName)
	flagNames.ProxyURL.BindStringFlag(flags, &clusterInfo.ProxyURL)
	flagNames.DisableCompression.BindBoolFlag(flags, &clusterInfo.DisableCompression)
}

// BindFlags is a convenience method to bind the specified flags to their associated variables
func BindContextFlags(contextInfo *clientcmdapi.Context, flags *pflag.FlagSet, flagNames ContextOverrideFlags) {
	flagNames.ClusterName.BindStringFlag(flags, &contextInfo.Cluster)
	flagNames.AuthInfoName.BindStringFlag(flags, &contextInfo.AuthInfo)
	flagNames.Namespace.BindTransformingStringFlag(flags, &contextInfo.Namespace, RemoveNamespacesPrefix)
}

// RemoveNamespacesPrefix is a transformer that strips "ns/", "namespace/" and "namespaces/" prefixes case-insensitively
func RemoveNamespacesPrefix(value string) (string, error) {
	for _, prefix := range []string{"namespaces/", "namespace/", "ns/"} {
		if len(value) > len(prefix) && strings.EqualFold(value[0:len(prefix)], prefix) {
			value = value[len(prefix):]
			break
		}
	}
	return value, nil
}
