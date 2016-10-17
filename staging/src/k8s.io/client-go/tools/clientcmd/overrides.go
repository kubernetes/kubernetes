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

// ConfigOverrideFlags holds the flag names to be used for binding command line flags.  Notice that this structure tightly
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

// BindStringFlag binds the flag based on the provided info.  If LongName == "", nothing is registered
func (f FlagInfo) BindStringFlag(flags *pflag.FlagSet, target *string) {
	// you can't register a flag without a long name
	if len(f.LongName) > 0 {
		flags.StringVarP(target, f.LongName, f.ShortName, f.Default, f.Description)
	}
}

// BindBoolFlag binds the flag based on the provided info.  If LongName == "", nothing is registered
func (f FlagInfo) BindBoolFlag(flags *pflag.FlagSet, target *bool) {
	// you can't register a flag without a long name
	if len(f.LongName) > 0 {
		// try to parse Default as a bool.  If it fails, assume false
		boolVal, err := strconv.ParseBool(f.Default)
		if err != nil {
			boolVal = false
		}

		flags.BoolVarP(target, f.LongName, f.ShortName, boolVal, f.Description)
	}
}

const (
	FlagClusterName  = "cluster"
	FlagAuthInfoName = "user"
	FlagContext      = "context"
	FlagNamespace    = "namespace"
	FlagAPIServer    = "server"
	FlagAPIVersion   = "api-version"
	FlagInsecure     = "insecure-skip-tls-verify"
	FlagCertFile     = "client-certificate"
	FlagKeyFile      = "client-key"
	FlagCAFile       = "certificate-authority"
	FlagEmbedCerts   = "embed-certs"
	FlagBearerToken  = "token"
	FlagImpersonate  = "as"
	FlagUsername     = "username"
	FlagPassword     = "password"
	FlagTimeout      = "request-timeout"
)

// RecommendedAuthOverrideFlags is a convenience method to return recommended flag names prefixed with a string of your choosing
func RecommendedAuthOverrideFlags(prefix string) AuthOverrideFlags {
	return AuthOverrideFlags{
		ClientCertificate: FlagInfo{prefix + FlagCertFile, "", "", "Path to a client certificate file for TLS"},
		ClientKey:         FlagInfo{prefix + FlagKeyFile, "", "", "Path to a client key file for TLS"},
		Token:             FlagInfo{prefix + FlagBearerToken, "", "", "Bearer token for authentication to the API server"},
		Impersonate:       FlagInfo{prefix + FlagImpersonate, "", "", "Username to impersonate for the operation"},
		Username:          FlagInfo{prefix + FlagUsername, "", "", "Username for basic authentication to the API server"},
		Password:          FlagInfo{prefix + FlagPassword, "", "", "Password for basic authentication to the API server"},
	}
}

// RecommendedClusterOverrideFlags is a convenience method to return recommended flag names prefixed with a string of your choosing
func RecommendedClusterOverrideFlags(prefix string) ClusterOverrideFlags {
	return ClusterOverrideFlags{
		APIServer:             FlagInfo{prefix + FlagAPIServer, "", "", "The address and port of the Kubernetes API server"},
		APIVersion:            FlagInfo{prefix + FlagAPIVersion, "", "", "DEPRECATED: The API version to use when talking to the server"},
		CertificateAuthority:  FlagInfo{prefix + FlagCAFile, "", "", "Path to a cert. file for the certificate authority"},
		InsecureSkipTLSVerify: FlagInfo{prefix + FlagInsecure, "", "false", "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure"},
	}
}

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

// RecommendedContextOverrideFlags is a convenience method to return recommended flag names prefixed with a string of your choosing
func RecommendedContextOverrideFlags(prefix string) ContextOverrideFlags {
	return ContextOverrideFlags{
		ClusterName:  FlagInfo{prefix + FlagClusterName, "", "", "The name of the kubeconfig cluster to use"},
		AuthInfoName: FlagInfo{prefix + FlagAuthInfoName, "", "", "The name of the kubeconfig user to use"},
		Namespace:    FlagInfo{prefix + FlagNamespace, "n", "", "If present, the namespace scope for this CLI request"},
	}
}

// BindAuthInfoFlags is a convenience method to bind the specified flags to their associated variables
func BindAuthInfoFlags(authInfo *clientcmdapi.AuthInfo, flags *pflag.FlagSet, flagNames AuthOverrideFlags) {
	flagNames.ClientCertificate.BindStringFlag(flags, &authInfo.ClientCertificate)
	flagNames.ClientKey.BindStringFlag(flags, &authInfo.ClientKey)
	flagNames.Token.BindStringFlag(flags, &authInfo.Token)
	flagNames.Impersonate.BindStringFlag(flags, &authInfo.Impersonate)
	flagNames.Username.BindStringFlag(flags, &authInfo.Username)
	flagNames.Password.BindStringFlag(flags, &authInfo.Password)
}

// BindClusterFlags is a convenience method to bind the specified flags to their associated variables
func BindClusterFlags(clusterInfo *clientcmdapi.Cluster, flags *pflag.FlagSet, flagNames ClusterOverrideFlags) {
	flagNames.APIServer.BindStringFlag(flags, &clusterInfo.Server)
	// TODO: remove --api-version flag in 1.3.
	flagNames.APIVersion.BindStringFlag(flags, &clusterInfo.APIVersion)
	flags.MarkDeprecated(FlagAPIVersion, "flag is no longer respected and will be deleted in the next release")
	flagNames.CertificateAuthority.BindStringFlag(flags, &clusterInfo.CertificateAuthority)
	flagNames.InsecureSkipTLSVerify.BindBoolFlag(flags, &clusterInfo.InsecureSkipTLSVerify)
}

// BindOverrideFlags is a convenience method to bind the specified flags to their associated variables
func BindOverrideFlags(overrides *ConfigOverrides, flags *pflag.FlagSet, flagNames ConfigOverrideFlags) {
	BindAuthInfoFlags(&overrides.AuthInfo, flags, flagNames.AuthOverrideFlags)
	BindClusterFlags(&overrides.ClusterInfo, flags, flagNames.ClusterOverrideFlags)
	BindContextFlags(&overrides.Context, flags, flagNames.ContextOverrideFlags)
	flagNames.CurrentContext.BindStringFlag(flags, &overrides.CurrentContext)
	flagNames.Timeout.BindStringFlag(flags, &overrides.Timeout)
}

// BindFlags is a convenience method to bind the specified flags to their associated variables
func BindContextFlags(contextInfo *clientcmdapi.Context, flags *pflag.FlagSet, flagNames ContextOverrideFlags) {
	flagNames.ClusterName.BindStringFlag(flags, &contextInfo.Cluster)
	flagNames.AuthInfoName.BindStringFlag(flags, &contextInfo.AuthInfo)
	flagNames.Namespace.BindStringFlag(flags, &contextInfo.Namespace)
}
