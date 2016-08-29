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
	"fmt"
	"io"
	"io/ioutil"
	"net/url"
	"os"
	"strings"

	"github.com/golang/glog"
	"github.com/imdario/mergo"

	"k8s.io/client-go/1.4/pkg/api"
	"k8s.io/client-go/1.4/rest"
	clientauth "k8s.io/client-go/1.4/tools/auth"
	clientcmdapi "k8s.io/client-go/1.4/tools/clientcmd/api"
)

var (
	// DefaultCluster is the cluster config used when no other config is specified
	// TODO: eventually apiserver should start on 443 and be secure by default
	DefaultCluster = clientcmdapi.Cluster{Server: "http://localhost:8080"}

	// EnvVarCluster allows overriding the DefaultCluster using an envvar for the server name
	EnvVarCluster = clientcmdapi.Cluster{Server: os.Getenv("KUBERNETES_MASTER")}

	DefaultClientConfig = DirectClientConfig{*clientcmdapi.NewConfig(), "", &ConfigOverrides{}, nil, NewDefaultClientConfigLoadingRules()}
)

// ClientConfig is used to make it easy to get an api server client
type ClientConfig interface {
	// RawConfig returns the merged result of all overrides
	RawConfig() (clientcmdapi.Config, error)
	// ClientConfig returns a complete client config
	ClientConfig() (*rest.Config, error)
	// Namespace returns the namespace resulting from the merged
	// result of all overrides and a boolean indicating if it was
	// overridden
	Namespace() (string, bool, error)
	// ConfigAccess returns the rules for loading/persisting the config.
	ConfigAccess() ConfigAccess
}

type PersistAuthProviderConfigForUser func(user string) rest.AuthProviderConfigPersister

// DirectClientConfig is a ClientConfig interface that is backed by a clientcmdapi.Config, options overrides, and an optional fallbackReader for auth information
type DirectClientConfig struct {
	config         clientcmdapi.Config
	contextName    string
	overrides      *ConfigOverrides
	fallbackReader io.Reader
	configAccess   ConfigAccess
}

// NewDefaultClientConfig creates a DirectClientConfig using the config.CurrentContext as the context name
func NewDefaultClientConfig(config clientcmdapi.Config, overrides *ConfigOverrides) ClientConfig {
	return &DirectClientConfig{config, config.CurrentContext, overrides, nil, NewDefaultClientConfigLoadingRules()}
}

// NewNonInteractiveClientConfig creates a DirectClientConfig using the passed context name and does not have a fallback reader for auth information
func NewNonInteractiveClientConfig(config clientcmdapi.Config, contextName string, overrides *ConfigOverrides, configAccess ConfigAccess) ClientConfig {
	return &DirectClientConfig{config, contextName, overrides, nil, configAccess}
}

// NewInteractiveClientConfig creates a DirectClientConfig using the passed context name and a reader in case auth information is not provided via files or flags
func NewInteractiveClientConfig(config clientcmdapi.Config, contextName string, overrides *ConfigOverrides, fallbackReader io.Reader, configAccess ConfigAccess) ClientConfig {
	return &DirectClientConfig{config, contextName, overrides, fallbackReader, configAccess}
}

func (config *DirectClientConfig) RawConfig() (clientcmdapi.Config, error) {
	return config.config, nil
}

// ClientConfig implements ClientConfig
func (config *DirectClientConfig) ClientConfig() (*rest.Config, error) {
	if err := config.ConfirmUsable(); err != nil {
		return nil, err
	}

	configAuthInfo := config.getAuthInfo()
	configClusterInfo := config.getCluster()

	clientConfig := &rest.Config{}
	clientConfig.Host = configClusterInfo.Server
	if u, err := url.ParseRequestURI(clientConfig.Host); err == nil && u.Opaque == "" && len(u.Path) > 1 {
		u.RawQuery = ""
		u.Fragment = ""
		clientConfig.Host = u.String()
	}
	if len(configAuthInfo.Impersonate) > 0 {
		clientConfig.Impersonate = configAuthInfo.Impersonate
	}

	// only try to read the auth information if we are secure
	if rest.IsConfigTransportTLS(*clientConfig) {
		var err error

		// mergo is a first write wins for map value and a last writing wins for interface values
		// NOTE: This behavior changed with https://github.com/imdario/mergo/commit/d304790b2ed594794496464fadd89d2bb266600a.
		//       Our mergo.Merge version is older than this change.
		var persister rest.AuthProviderConfigPersister
		if config.configAccess != nil {
			persister = PersisterForUser(config.configAccess, config.getAuthInfoName())
		}
		userAuthPartialConfig, err := getUserIdentificationPartialConfig(configAuthInfo, config.fallbackReader, persister)
		if err != nil {
			return nil, err
		}
		mergo.Merge(clientConfig, userAuthPartialConfig)

		serverAuthPartialConfig, err := getServerIdentificationPartialConfig(configAuthInfo, configClusterInfo)
		if err != nil {
			return nil, err
		}
		mergo.Merge(clientConfig, serverAuthPartialConfig)
	}

	return clientConfig, nil
}

// clientauth.Info object contain both user identification and server identification.  We want different precedence orders for
// both, so we have to split the objects and merge them separately
// we want this order of precedence for the server identification
// 1.  configClusterInfo (the final result of command line flags and merged .kubeconfig files)
// 2.  configAuthInfo.auth-path (this file can contain information that conflicts with #1, and we want #1 to win the priority)
// 3.  load the ~/.kubernetes_auth file as a default
func getServerIdentificationPartialConfig(configAuthInfo clientcmdapi.AuthInfo, configClusterInfo clientcmdapi.Cluster) (*rest.Config, error) {
	mergedConfig := &rest.Config{}

	// configClusterInfo holds the information identify the server provided by .kubeconfig
	configClientConfig := &rest.Config{}
	configClientConfig.CAFile = configClusterInfo.CertificateAuthority
	configClientConfig.CAData = configClusterInfo.CertificateAuthorityData
	configClientConfig.Insecure = configClusterInfo.InsecureSkipTLSVerify
	mergo.Merge(mergedConfig, configClientConfig)

	return mergedConfig, nil
}

// clientauth.Info object contain both user identification and server identification.  We want different precedence orders for
// both, so we have to split the objects and merge them separately
// we want this order of precedence for user identifcation
// 1.  configAuthInfo minus auth-path (the final result of command line flags and merged .kubeconfig files)
// 2.  configAuthInfo.auth-path (this file can contain information that conflicts with #1, and we want #1 to win the priority)
// 3.  if there is not enough information to idenfity the user, load try the ~/.kubernetes_auth file
// 4.  if there is not enough information to identify the user, prompt if possible
func getUserIdentificationPartialConfig(configAuthInfo clientcmdapi.AuthInfo, fallbackReader io.Reader, persistAuthConfig rest.AuthProviderConfigPersister) (*rest.Config, error) {
	mergedConfig := &rest.Config{}

	// blindly overwrite existing values based on precedence
	if len(configAuthInfo.Token) > 0 {
		mergedConfig.BearerToken = configAuthInfo.Token
	} else if len(configAuthInfo.TokenFile) > 0 {
		tokenBytes, err := ioutil.ReadFile(configAuthInfo.TokenFile)
		if err != nil {
			return nil, err
		}
		mergedConfig.BearerToken = string(tokenBytes)
	}
	if len(configAuthInfo.Impersonate) > 0 {
		mergedConfig.Impersonate = configAuthInfo.Impersonate
	}
	if len(configAuthInfo.ClientCertificate) > 0 || len(configAuthInfo.ClientCertificateData) > 0 {
		mergedConfig.CertFile = configAuthInfo.ClientCertificate
		mergedConfig.CertData = configAuthInfo.ClientCertificateData
		mergedConfig.KeyFile = configAuthInfo.ClientKey
		mergedConfig.KeyData = configAuthInfo.ClientKeyData
	}
	if len(configAuthInfo.Username) > 0 || len(configAuthInfo.Password) > 0 {
		mergedConfig.Username = configAuthInfo.Username
		mergedConfig.Password = configAuthInfo.Password
	}
	if configAuthInfo.AuthProvider != nil {
		mergedConfig.AuthProvider = configAuthInfo.AuthProvider
		mergedConfig.AuthConfigPersister = persistAuthConfig
	}

	// if there still isn't enough information to authenticate the user, try prompting
	if !canIdentifyUser(*mergedConfig) && (fallbackReader != nil) {
		prompter := NewPromptingAuthLoader(fallbackReader)
		promptedAuthInfo := prompter.Prompt()

		promptedConfig := makeUserIdentificationConfig(*promptedAuthInfo)
		previouslyMergedConfig := mergedConfig
		mergedConfig = &rest.Config{}
		mergo.Merge(mergedConfig, promptedConfig)
		mergo.Merge(mergedConfig, previouslyMergedConfig)
	}

	return mergedConfig, nil
}

// makeUserIdentificationFieldsConfig returns a client.Config capable of being merged using mergo for only user identification information
func makeUserIdentificationConfig(info clientauth.Info) *rest.Config {
	config := &rest.Config{}
	config.Username = info.User
	config.Password = info.Password
	config.CertFile = info.CertFile
	config.KeyFile = info.KeyFile
	config.BearerToken = info.BearerToken
	return config
}

// makeUserIdentificationFieldsConfig returns a client.Config capable of being merged using mergo for only server identification information
func makeServerIdentificationConfig(info clientauth.Info) rest.Config {
	config := rest.Config{}
	config.CAFile = info.CAFile
	if info.Insecure != nil {
		config.Insecure = *info.Insecure
	}
	return config
}

func canIdentifyUser(config rest.Config) bool {
	return len(config.Username) > 0 ||
		(len(config.CertFile) > 0 || len(config.CertData) > 0) ||
		len(config.BearerToken) > 0 ||
		config.AuthProvider != nil
}

// Namespace implements ClientConfig
func (config *DirectClientConfig) Namespace() (string, bool, error) {
	if err := config.ConfirmUsable(); err != nil {
		return "", false, err
	}

	configContext := config.getContext()

	if len(configContext.Namespace) == 0 {
		return api.NamespaceDefault, false, nil
	}

	overridden := false
	if config.overrides != nil && config.overrides.Context.Namespace != "" {
		overridden = true
	}
	return configContext.Namespace, overridden, nil
}

// ConfigAccess implements ClientConfig
func (config *DirectClientConfig) ConfigAccess() ConfigAccess {
	return config.configAccess
}

// ConfirmUsable looks a particular context and determines if that particular part of the config is useable.  There might still be errors in the config,
// but no errors in the sections requested or referenced.  It does not return early so that it can find as many errors as possible.
func (config *DirectClientConfig) ConfirmUsable() error {
	validationErrors := make([]error, 0)
	validationErrors = append(validationErrors, validateAuthInfo(config.getAuthInfoName(), config.getAuthInfo())...)
	validationErrors = append(validationErrors, validateClusterInfo(config.getClusterName(), config.getCluster())...)
	// when direct client config is specified, and our only error is that no server is defined, we should
	// return a standard "no config" error
	if len(validationErrors) == 1 && validationErrors[0] == ErrEmptyCluster {
		return newErrConfigurationInvalid([]error{ErrEmptyConfig})
	}
	return newErrConfigurationInvalid(validationErrors)
}

func (config *DirectClientConfig) getContextName() string {
	if len(config.overrides.CurrentContext) != 0 {
		return config.overrides.CurrentContext
	}
	if len(config.contextName) != 0 {
		return config.contextName
	}

	return config.config.CurrentContext
}

func (config *DirectClientConfig) getAuthInfoName() string {
	if len(config.overrides.Context.AuthInfo) != 0 {
		return config.overrides.Context.AuthInfo
	}
	return config.getContext().AuthInfo
}

func (config *DirectClientConfig) getClusterName() string {
	if len(config.overrides.Context.Cluster) != 0 {
		return config.overrides.Context.Cluster
	}
	return config.getContext().Cluster
}

func (config *DirectClientConfig) getContext() clientcmdapi.Context {
	contexts := config.config.Contexts
	contextName := config.getContextName()

	var mergedContext clientcmdapi.Context
	if configContext, exists := contexts[contextName]; exists {
		mergo.Merge(&mergedContext, configContext)
	}
	mergo.Merge(&mergedContext, config.overrides.Context)

	return mergedContext
}

func (config *DirectClientConfig) getAuthInfo() clientcmdapi.AuthInfo {
	authInfos := config.config.AuthInfos
	authInfoName := config.getAuthInfoName()

	var mergedAuthInfo clientcmdapi.AuthInfo
	if configAuthInfo, exists := authInfos[authInfoName]; exists {
		mergo.Merge(&mergedAuthInfo, configAuthInfo)
	}
	mergo.Merge(&mergedAuthInfo, config.overrides.AuthInfo)

	return mergedAuthInfo
}

func (config *DirectClientConfig) getCluster() clientcmdapi.Cluster {
	clusterInfos := config.config.Clusters
	clusterInfoName := config.getClusterName()

	var mergedClusterInfo clientcmdapi.Cluster
	mergo.Merge(&mergedClusterInfo, config.overrides.ClusterDefaults)
	mergo.Merge(&mergedClusterInfo, EnvVarCluster)
	if configClusterInfo, exists := clusterInfos[clusterInfoName]; exists {
		mergo.Merge(&mergedClusterInfo, configClusterInfo)
	}
	mergo.Merge(&mergedClusterInfo, config.overrides.ClusterInfo)
	// An override of --insecure-skip-tls-verify=true and no accompanying CA/CA data should clear already-set CA/CA data
	// otherwise, a kubeconfig containing a CA reference would return an error that "CA and insecure-skip-tls-verify couldn't both be set"
	caLen := len(config.overrides.ClusterInfo.CertificateAuthority)
	caDataLen := len(config.overrides.ClusterInfo.CertificateAuthorityData)
	if config.overrides.ClusterInfo.InsecureSkipTLSVerify && caLen == 0 && caDataLen == 0 {
		mergedClusterInfo.CertificateAuthority = ""
		mergedClusterInfo.CertificateAuthorityData = nil
	}

	return mergedClusterInfo
}

// inClusterClientConfig makes a config that will work from within a kubernetes cluster container environment.
type inClusterClientConfig struct{}

func (inClusterClientConfig) RawConfig() (clientcmdapi.Config, error) {
	return clientcmdapi.Config{}, fmt.Errorf("inCluster environment config doesn't support multiple clusters")
}

func (inClusterClientConfig) ClientConfig() (*rest.Config, error) {
	return rest.InClusterConfig()
}

func (inClusterClientConfig) Namespace() (string, error) {
	// This way assumes you've set the POD_NAMESPACE environment variable using the downward API.
	// This check has to be done first for backwards compatibility with the way InClusterConfig was originally set up
	if ns := os.Getenv("POD_NAMESPACE"); ns != "" {
		return ns, nil
	}

	// Fall back to the namespace associated with the service account token, if available
	if data, err := ioutil.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace"); err == nil {
		if ns := strings.TrimSpace(string(data)); len(ns) > 0 {
			return ns, nil
		}
	}

	return "default", nil
}

func (inClusterClientConfig) ConfigAccess() ConfigAccess {
	return NewDefaultClientConfigLoadingRules()
}

// Possible returns true if loading an inside-kubernetes-cluster is possible.
func (inClusterClientConfig) Possible() bool {
	fi, err := os.Stat("/var/run/secrets/kubernetes.io/serviceaccount/token")
	return os.Getenv("KUBERNETES_SERVICE_HOST") != "" &&
		os.Getenv("KUBERNETES_SERVICE_PORT") != "" &&
		err == nil && !fi.IsDir()
}

// BuildConfigFromFlags is a helper function that builds configs from a master
// url or a kubeconfig filepath. These are passed in as command line flags for cluster
// components. Warnings should reflect this usage. If neither masterUrl or kubeconfigPath
// are passed in we fallback to inClusterConfig. If inClusterConfig fails, we fallback
// to the default config.
func BuildConfigFromFlags(masterUrl, kubeconfigPath string) (*rest.Config, error) {
	if kubeconfigPath == "" && masterUrl == "" {
		glog.Warningf("Neither --kubeconfig nor --master was specified.  Using the inClusterConfig.  This might not work.")
		kubeconfig, err := rest.InClusterConfig()
		if err == nil {
			return kubeconfig, nil
		}
		glog.Warning("error creating inClusterConfig, falling back to default config: ", err)
	}
	return NewNonInteractiveDeferredLoadingClientConfig(
		&ClientConfigLoadingRules{ExplicitPath: kubeconfigPath},
		&ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterUrl}}).ClientConfig()
}

// BuildConfigFromKubeconfigGetter is a helper function that builds configs from a master
// url and a kubeconfigGetter.
func BuildConfigFromKubeconfigGetter(masterUrl string, kubeconfigGetter KubeconfigGetter) (*rest.Config, error) {
	// TODO: We do not need a DeferredLoader here. Refactor code and see if we can use DirectClientConfig here.
	cc := NewNonInteractiveDeferredLoadingClientConfig(
		&ClientConfigGetter{kubeconfigGetter: kubeconfigGetter},
		&ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: masterUrl}})
	return cc.ClientConfig()
}
