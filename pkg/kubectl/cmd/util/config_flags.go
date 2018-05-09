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

package util

import (
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/pflag"

	utilflag "k8s.io/apiserver/pkg/util/flag"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"k8s.io/kubernetes/pkg/kubectl/cmd/util/transport"
)

const (
	flagClusterName      = "cluster"
	flagAuthInfoName     = "user"
	flagContext          = "context"
	flagNamespace        = "namespace"
	flagAPIServer        = "server"
	flagInsecure         = "insecure-skip-tls-verify"
	flagCertFile         = "client-certificate"
	flagKeyFile          = "client-key"
	flagCAFile           = "certificate-authority"
	flagBearerToken      = "token"
	flagImpersonate      = "as"
	flagImpersonateGroup = "as-group"
	flagUsername         = "username"
	flagPassword         = "password"
	flagTimeout          = "request-timeout"
)

// TODO(juanvallejo): move to pkg/kubectl/genericclioptions once
// the dependency on cmdutil is broken here.
// ConfigFlags composes the set of values necessary
// for obtaining a REST client config
type ConfigFlags struct {
	CacheDir   *string
	KubeConfig *string

	// config flags
	ClusterName      *string
	AuthInfoName     *string
	Context          *string
	Namespace        *string
	APIServer        *string
	Insecure         *bool
	CertFile         *string
	KeyFile          *string
	CAFile           *string
	BearerToken      *string
	Impersonate      *string
	ImpersonateGroup *[]string
	Username         *string
	Password         *string
	Timeout          *string
}

// ToRESTConfig implements RESTClientGetter.
// Returns a REST client configuration based on a provided path
// to a .kubeconfig file, loading rules, and config flag overrides.
// Expects the AddFlags method to have been called.
func (f *ConfigFlags) ToRESTConfig() (*rest.Config, error) {
	return f.ToRawKubeConfigLoader().ClientConfig()
}

func (f *ConfigFlags) ToRawKubeConfigLoader() clientcmd.ClientConfig {
	loadingRules := clientcmd.NewDefaultClientConfigLoadingRules()
	// use the standard defaults for this client command
	// DEPRECATED: remove and replace with something more accurate
	loadingRules.DefaultClientConfig = &clientcmd.DefaultClientConfig

	if f.KubeConfig != nil {
		loadingRules.ExplicitPath = *f.KubeConfig
	}

	overrides := &clientcmd.ConfigOverrides{ClusterDefaults: clientcmd.ClusterDefaults}

	// bind auth info flag values to overrides
	if f.CertFile != nil {
		overrides.AuthInfo.ClientCertificate = *f.CertFile
	}
	if f.KeyFile != nil {
		overrides.AuthInfo.ClientKey = *f.KeyFile
	}
	if f.BearerToken != nil {
		overrides.AuthInfo.Token = *f.BearerToken
	}
	if f.Impersonate != nil {
		overrides.AuthInfo.Impersonate = *f.Impersonate
	}
	if f.ImpersonateGroup != nil {
		overrides.AuthInfo.ImpersonateGroups = *f.ImpersonateGroup
	}
	if f.Username != nil {
		overrides.AuthInfo.Username = *f.Username
	}
	if f.Password != nil {
		overrides.AuthInfo.Password = *f.Password
	}

	// bind cluster flags
	if f.APIServer != nil {
		overrides.ClusterInfo.Server = *f.APIServer
	}
	if f.CAFile != nil {
		overrides.ClusterInfo.CertificateAuthority = *f.CAFile
	}
	if f.Insecure != nil {
		overrides.ClusterInfo.InsecureSkipTLSVerify = *f.Insecure
	}

	// bind context flags
	if f.Context != nil {
		overrides.CurrentContext = *f.Context
	}
	if f.ClusterName != nil {
		overrides.Context.Cluster = *f.ClusterName
	}
	if f.AuthInfoName != nil {
		overrides.Context.AuthInfo = *f.AuthInfoName
	}
	if f.Namespace != nil {
		overrides.Context.Namespace = *f.Namespace
	}

	if f.Timeout != nil {
		overrides.Timeout = *f.Timeout
	}

	var clientConfig clientcmd.ClientConfig

	// we only have an interactive prompt when a password is allowed
	if f.Password == nil {
		clientConfig = clientcmd.NewNonInteractiveDeferredLoadingClientConfig(loadingRules, overrides)
	} else {
		clientConfig = clientcmd.NewInteractiveDeferredLoadingClientConfig(loadingRules, overrides, os.Stdin)
	}

	return clientConfig
}

// ToDiscoveryClient implements RESTClientGetter.
// Expects the AddFlags method to have been called.
// Returns a CachedDiscoveryInterface using a computed RESTConfig.
func (f *ConfigFlags) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	config, err := f.ToRESTConfig()
	if err != nil {
		return nil, err
	}

	// The more groups you have, the more discovery requests you need to make.
	// given 25 groups (our groups + a few custom resources) with one-ish version each, discovery needs to make 50 requests
	// double it just so we don't end up here again for a while.  This config is only used for discovery.
	config.Burst = 100

	cacheDir := filepath.Join(homedir.HomeDir(), ".kube", "http-cache")
	if f.CacheDir != nil {
		cacheDir = *f.CacheDir
	}

	if len(cacheDir) > 0 {
		wt := config.WrapTransport
		config.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
			if wt != nil {
				rt = wt(rt)
			}
			return transport.NewCacheRoundTripper(cacheDir, rt)
		}
	}

	discoveryClient, err := discovery.NewDiscoveryClientForConfig(config)
	if err != nil {
		return nil, err
	}
	cacheDir = computeDiscoverCacheDir(filepath.Join(homedir.HomeDir(), ".kube", "cache", "discovery"), config.Host)
	return NewCachedDiscoveryClient(discoveryClient, cacheDir, time.Duration(10*time.Minute)), nil
}

func (f *ConfigFlags) AddFlags(flags *pflag.FlagSet) {
	flags.SetNormalizeFunc(utilflag.WarnWordSepNormalizeFunc) // Warn for "_" flags

	// Normalize all flags that are coming from other packages or pre-configurations
	// a.k.a. change all "_" to "-". e.g. glog package
	flags.SetNormalizeFunc(utilflag.WordSepNormalizeFunc)

	if f.KubeConfig != nil {
		flags.StringVar(f.KubeConfig, "kubeconfig", *f.KubeConfig, "Path to the kubeconfig file to use for CLI requests.")
	}
	if f.CacheDir != nil {
		flags.StringVar(f.CacheDir, FlagHTTPCacheDir, *f.CacheDir, "Default HTTP cache directory")
	}

	// add config options
	if f.CertFile != nil {
		flags.StringVar(f.CertFile, flagCertFile, *f.CertFile, "Path to a client certificate file for TLS")
	}
	if f.KeyFile != nil {
		flags.StringVar(f.KeyFile, flagKeyFile, *f.KeyFile, "Path to a client key file for TLS")
	}
	if f.BearerToken != nil {
		flags.StringVar(f.BearerToken, flagBearerToken, *f.BearerToken, "Bearer token for authentication to the API server")
	}
	if f.Impersonate != nil {
		flags.StringVar(f.Impersonate, flagImpersonate, *f.Impersonate, "Username to impersonate for the operation")
	}
	if f.ImpersonateGroup != nil {
		flags.StringArrayVar(f.ImpersonateGroup, flagImpersonateGroup, *f.ImpersonateGroup, "Group to impersonate for the operation, this flag can be repeated to specify multiple groups.")
	}
	if f.Username != nil {
		flags.StringVar(f.Username, flagUsername, *f.Username, "Username for basic authentication to the API server")
	}
	if f.Password != nil {
		flags.StringVar(f.Password, flagPassword, *f.Password, "Password for basic authentication to the API server")
	}
	if f.ClusterName != nil {
		flags.StringVar(f.ClusterName, flagClusterName, *f.ClusterName, "The name of the kubeconfig cluster to use")
	}
	if f.AuthInfoName != nil {
		flags.StringVar(f.AuthInfoName, flagAuthInfoName, *f.AuthInfoName, "The name of the kubeconfig user to use")
	}
	if f.Namespace != nil {
		flags.StringVarP(f.Namespace, flagNamespace, "n", *f.Namespace, "If present, the namespace scope for this CLI request")
	}
	if f.Context != nil {
		flags.StringVar(f.Context, flagContext, *f.Context, "The name of the kubeconfig context to use")
	}

	if f.APIServer != nil {
		flags.StringVarP(f.APIServer, flagAPIServer, "s", *f.APIServer, "The address and port of the Kubernetes API server")
	}
	if f.Insecure != nil {
		flags.BoolVar(f.Insecure, flagInsecure, *f.Insecure, "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure")
	}
	if f.CAFile != nil {
		flags.StringVar(f.CAFile, flagCAFile, *f.CAFile, "Path to a cert file for the certificate authority")
	}
	if f.Timeout != nil {
		flags.StringVar(f.Timeout, flagTimeout, *f.Timeout, "The length of time to wait before giving up on a single server request. Non-zero values should contain a corresponding time unit (e.g. 1s, 2m, 3h). A value of zero means don't timeout requests.")
	}

}

func (f *ConfigFlags) WithDeprecatedPasswordFlag() *ConfigFlags {
	f.Username = stringptr("")
	f.Password = stringptr("")
	return f
}

func NewConfigFlags() *ConfigFlags {
	impersonateGroup := []string{}
	insecure := false

	return &ConfigFlags{
		Insecure:   &insecure,
		Timeout:    stringptr("0"),
		KubeConfig: stringptr(""),

		ClusterName:      stringptr(""),
		AuthInfoName:     stringptr(""),
		Context:          stringptr(""),
		Namespace:        stringptr(""),
		APIServer:        stringptr(""),
		CertFile:         stringptr(""),
		KeyFile:          stringptr(""),
		CAFile:           stringptr(""),
		BearerToken:      stringptr(""),
		Impersonate:      stringptr(""),
		ImpersonateGroup: &impersonateGroup,
	}
}

func stringptr(val string) *string {
	return &val
}

// TODO(juanvallejo): move to separate file when config_flags are moved
// to pkg/kubectl/genericclioptions
type TestConfigFlags struct {
	clientConfig    clientcmd.ClientConfig
	discoveryClient discovery.CachedDiscoveryInterface
}

func (f *TestConfigFlags) ToRawKubeConfigLoader() clientcmd.ClientConfig {
	if f.clientConfig == nil {
		panic("attempt to obtain a test RawKubeConfigLoader with no clientConfig specified")
	}
	return f.clientConfig
}

func (f *TestConfigFlags) ToRESTConfig() (*rest.Config, error) {
	return f.ToRawKubeConfigLoader().ClientConfig()
}

func (f *TestConfigFlags) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	return f.discoveryClient, nil
}

func (f *TestConfigFlags) WithClientConfig(clientConfig clientcmd.ClientConfig) *TestConfigFlags {
	f.clientConfig = clientConfig
	return f
}

func (f *TestConfigFlags) WithDiscoveryClient(c discovery.CachedDiscoveryInterface) *TestConfigFlags {
	f.discoveryClient = c
	return f
}

func NewTestConfigFlags() *TestConfigFlags {
	return &TestConfigFlags{}
}
