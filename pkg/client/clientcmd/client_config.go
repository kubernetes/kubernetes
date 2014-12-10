/*
Copyright 2014 Google Inc. All rights reserved.

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
	"io"
	"os"

	"github.com/imdario/mergo"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

var (
	// TODO: eventually apiserver should start on 443 and be secure by default
	defaultCluster = Cluster{Server: "http://localhost:8080"}
	envVarCluster  = Cluster{Server: os.Getenv("KUBERNETES_MASTER")}
)

// ClientConfig is used to make it easy to get an api server client
type ClientConfig interface {
	// ClientConfig returns a complete client config
	ClientConfig() (*client.Config, error)
}

// DirectClientConfig is a ClientConfig interface that is backed by a Config, options overrides, and an optional fallbackReader for auth information
type DirectClientConfig struct {
	config         Config
	contextName    string
	overrides      *ConfigOverrides
	fallbackReader io.Reader
}

// NewDefaultClientConfig creates a DirectClientConfig using the config.CurrentContext as the context name
func NewDefaultClientConfig(config Config, overrides *ConfigOverrides) ClientConfig {
	return DirectClientConfig{config, config.CurrentContext, overrides, nil}
}

// NewNonInteractiveClientConfig creates a DirectClientConfig using the passed context name and does not have a fallback reader for auth information
func NewNonInteractiveClientConfig(config Config, contextName string, overrides *ConfigOverrides) ClientConfig {
	return DirectClientConfig{config, contextName, overrides, nil}
}

// NewInteractiveClientConfig creates a DirectClientConfig using the passed context name and a reader in case auth information is not provided via files or flags
func NewInteractiveClientConfig(config Config, contextName string, overrides *ConfigOverrides, fallbackReader io.Reader) ClientConfig {
	return DirectClientConfig{config, contextName, overrides, fallbackReader}
}

// ClientConfig implements ClientConfig
func (config DirectClientConfig) ClientConfig() (*client.Config, error) {
	if err := config.ConfirmUsable(); err != nil {
		return nil, err
	}

	configAuthInfo := config.getAuthInfo()
	configClusterInfo := config.getCluster()

	clientConfig := client.Config{}
	clientConfig.Host = configClusterInfo.Server
	clientConfig.Version = configClusterInfo.APIVersion

	// only try to read the auth information if we are secure
	if client.IsConfigTransportTLS(&clientConfig) {
		var authInfo *clientauth.Info
		var err error
		switch {
		case len(configAuthInfo.AuthPath) > 0:
			authInfo, err = NewDefaultAuthLoader().LoadAuth(configAuthInfo.AuthPath)
			if err != nil {
				return nil, err
			}

		case len(configAuthInfo.Token) > 0:
			authInfo = &clientauth.Info{BearerToken: configAuthInfo.Token}

		case len(configAuthInfo.ClientCertificate) > 0:
			authInfo = &clientauth.Info{
				CertFile: configAuthInfo.ClientCertificate,
				KeyFile:  configAuthInfo.ClientKey,
			}

		default:
			authInfo = &clientauth.Info{}
		}

		if !authInfo.Complete() && (config.fallbackReader != nil) {
			prompter := NewPromptingAuthLoader(config.fallbackReader)
			authInfo = prompter.Prompt()
		}

		authInfo.Insecure = &configClusterInfo.InsecureSkipTLSVerify

		clientConfig, err = authInfo.MergeWithConfig(clientConfig)
		if err != nil {
			return nil, err
		}
	}

	return &clientConfig, nil
}

// ConfirmUsable looks a particular context and determines if that particular part of the config is useable.  There might still be errors in the config,
// but no errors in the sections requested or referenced.  It does not return early so that it can find as many errors as possible.
func (config DirectClientConfig) ConfirmUsable() error {
	validationErrors := make([]error, 0)
	validationErrors = append(validationErrors, validateAuthInfo(config.getAuthInfoName(), config.getAuthInfo())...)
	validationErrors = append(validationErrors, validateClusterInfo(config.getClusterName(), config.getCluster())...)

	return util.SliceToError(validationErrors)
}

func (config DirectClientConfig) getContextName() string {
	if len(config.overrides.CurrentContext) != 0 {
		return config.overrides.CurrentContext
	}
	if len(config.contextName) != 0 {
		return config.contextName
	}

	return config.config.CurrentContext
}

func (config DirectClientConfig) getAuthInfoName() string {
	if len(config.overrides.AuthInfoName) != 0 {
		return config.overrides.AuthInfoName
	}
	return config.getContext().AuthInfo
}

func (config DirectClientConfig) getClusterName() string {
	if len(config.overrides.ClusterName) != 0 {
		return config.overrides.ClusterName
	}
	return config.getContext().Cluster
}

func (config DirectClientConfig) getContext() Context {
	return config.config.Contexts[config.getContextName()]
}

func (config DirectClientConfig) getAuthInfo() AuthInfo {
	authInfos := config.config.AuthInfos
	authInfoName := config.getAuthInfoName()

	var mergedAuthInfo AuthInfo
	if configAuthInfo, exists := authInfos[authInfoName]; exists {
		mergo.Merge(&mergedAuthInfo, configAuthInfo)
	}
	mergo.Merge(&mergedAuthInfo, config.overrides.AuthInfo)

	return mergedAuthInfo
}

func (config DirectClientConfig) getCluster() Cluster {
	clusterInfos := config.config.Clusters
	clusterInfoName := config.getClusterName()

	var mergedClusterInfo Cluster
	mergo.Merge(&mergedClusterInfo, defaultCluster)
	mergo.Merge(&mergedClusterInfo, envVarCluster)
	if configClusterInfo, exists := clusterInfos[clusterInfoName]; exists {
		mergo.Merge(&mergedClusterInfo, configClusterInfo)
	}
	mergo.Merge(&mergedClusterInfo, config.overrides.ClusterInfo)

	return mergedClusterInfo
}
